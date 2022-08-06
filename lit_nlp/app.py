# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""LIT backend, as a standard WSGI app."""

import functools
import glob
import os
import random
import time
from typing import Optional, Mapping, Sequence, Union, Callable, Iterable

from absl import logging

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import layout
from lit_nlp.api import model as lit_model
from lit_nlp.api import types
from lit_nlp.components import ablation_flip
from lit_nlp.components import classification_results
from lit_nlp.components import curves
from lit_nlp.components import gradient_maps
from lit_nlp.components import hotflip
from lit_nlp.components import lemon_explainer
from lit_nlp.components import lime_explainer
from lit_nlp.components import metrics
from lit_nlp.components import model_salience
from lit_nlp.components import nearest_neighbors
from lit_nlp.components import pca
from lit_nlp.components import pdp
from lit_nlp.components import projection
from lit_nlp.components import regression_results
from lit_nlp.components import salience_clustering
from lit_nlp.components import scrambler
from lit_nlp.components import selection_state
from lit_nlp.components import shap_explainer
from lit_nlp.components import tcav
from lit_nlp.components import thresholder
from lit_nlp.components import umap
from lit_nlp.components import word_replacer
from lit_nlp.lib import caching
from lit_nlp.lib import serialize
from lit_nlp.lib import utils
from lit_nlp.lib import wsgi_app
import tqdm

  def make_handler(self, fn):
    """Convenience wrapper to handle args and serialization.

    This is a thin shim between server (handler, request, environ) and model
    logic (inputs, args, outputs).

    Args:
      fn: function (JsonDict, **kw) -> JsonDict

    Returns:
      fn wrapped as a request handler
    """

    @functools.wraps(fn)
    def _handler(app: wsgi_app.App, request, environ):
      kw = request.args.to_dict()
      # The frontend needs "simple" data (e.g. NumPy arrays converted to lists),
      # but for requests from Python we may want to use the invertible encoding
      # so that datatypes from remote models are the same as local ones.
      response_simple_json = utils.coerce_bool(
          kw.pop('response_simple_json', True))
      data = serialize.from_json(request.data) if len(request.data) else None
      # Special handling to dereference IDs.
      if data and 'inputs' in data.keys() and 'dataset_name' in kw:
        data['inputs'] = self._reconstitute_inputs(data['inputs'],
                                                   kw['dataset_name'])

      outputs = fn(data, **kw)
      response_body = serialize.to_json(outputs, simple=response_simple_json)
      return app.respond(request, response_body, 'application/json', 200)

    return _handler

  def __init__(
      self,
      models: Mapping[str, lit_model.Model],
      datasets: Mapping[str, lit_dataset.Dataset],
      generators: Optional[Mapping[str, lit_components.Generator]] = None,
      interpreters: Optional[Mapping[str, lit_components.Interpreter]] = None,
      annotators: Optional[list[lit_components.Annotator]] = None,
      layouts: Optional[layout.LitComponentLayouts] = None,
      # General server config; see server_flags.py.
      data_dir: Optional[str] = None,
      warm_start: float = 0.0,
      warm_start_progress_indicator: Optional[ProgressIndicator] = tqdm
      .tqdm,  # not in server_flags
      warm_projections: bool = False,
      client_root: Optional[str] = None,
      demo_mode: bool = False,
      default_layout: Optional[str] = None,
      canonical_url: Optional[str] = None,
      page_title: Optional[str] = None,
      development_demo: bool = False,
      inline_doc: Optional[str] = None,
      onboard_start_doc: Optional[str] = None,
      onboard_end_doc: Optional[str] = None,
      sync_state: bool = False,  # notebook-only; not in server_flags
  ):
    if client_root is None:
      raise ValueError('client_root must be set on application')
    self._demo_mode = demo_mode
    self._development_demo = development_demo
    self._default_layout = default_layout
    self._canonical_url = canonical_url
    self._page_title = page_title
    self._inline_doc = inline_doc
    self._onboard_start_doc = onboard_start_doc
    self._onboard_end_doc = onboard_end_doc
    self._data_dir = data_dir
    if data_dir and not os.path.isdir(data_dir):
      os.mkdir(data_dir)

    print(f'{time.time()} App initializing...')

    # TODO(lit-dev): override layouts instead of merging, to allow clients
    # to opt-out of the default bundled layouts. This will require updating
    # client code to manually merge when this is the desired behavior.
    self._layouts = dict(layout.DEFAULT_LAYOUTS, **(layouts or {}))

    # Wrap models in caching wrapper
    self._models = {
        name: caching.CachingModelWrapper(model, name, cache_dir=data_dir)
        for name, model in models.items()
    }

    self._datasets = dict(datasets)
    # TODO(b/202210900): get rid of this, just dynamically create the empty
    # dataset on the frontend.
    self._datasets['_union_empty'] = lit_dataset.NoneDataset(self._models)

    self._annotators = annotators or []

    # Run annotation on each dataset, creating an annotated dataset and
    # replace the datasets with the annotated versions.
    for ds_key, ds in self._datasets.items():
      self._datasets[ds_key] = self._run_annotators(ds)

    # Index all datasets
    self._datasets = lit_dataset.IndexedDataset.index_all(
        self._datasets, caching.input_hash)

    if generators is not None:
      self._generators = generators
    else:
      self._generators = {
          'Ablation Flip': ablation_flip.AblationFlip(),
          'Hotflip': hotflip.HotFlip(),
          'Scrambler': scrambler.Scrambler(),
          'Word Replacer': word_replacer.WordReplacer(),
      }

    if interpreters is not None:
      self._interpreters = interpreters

    else:
      metrics_group = lit_components.ComponentGroup({
          'regression': metrics.RegressionMetrics(),
          'multiclass': metrics.MulticlassMetrics(),
          'paired': metrics.MulticlassPairedMetrics(),
          'bleu': metrics.CorpusBLEU(),
          'rouge': metrics.RougeL(),
      })
      gradient_map_interpreters = {
          'Grad L2 Norm': gradient_maps.GradientNorm(),
          'Grad ⋅ Input': gradient_maps.GradientDotInput(),
          'Integrated Gradients': gradient_maps.IntegratedGradients(),
          'LIME': lime_explainer.LIME(),
      }
      # pyformat: disable
      self._interpreters: dict[str, lit_components.Interpreter] = {
          'Model-provided salience': model_salience.ModelSalience(self._models),
          'counterfactual explainer': lemon_explainer.LEMON(),
          'tcav': tcav.TCAV(),
          'curves': curves.CurvesInterpreter(),
          'thresholder': thresholder.Thresholder(),
          'metrics': metrics_group,
          'pdp': pdp.PdpInterpreter(),
          'Salience Clustering': salience_clustering.SalienceClustering(
              gradient_map_interpreters),
          'Tabular SHAP': shap_explainer.TabularShapExplainer(),
      }
      # pyformat: enable
      self._interpreters.update(gradient_map_interpreters)

    # Ensure the prediction analysis interpreters are included.
    prediction_analysis_interpreters = {
        'classification': classification_results.ClassificationInterpreter(),
        'regression': regression_results.RegressionInterpreter(),
    }
    # Ensure the embedding-based interpreters are included.
    embedding_based_interpreters = {
        'nearest neighbors': nearest_neighbors.NearestNeighbors(),
        # Embedding projectors expose a standard interface, but get special
        # handling so we can precompute the projections if requested.
        'pca': projection.ProjectionManager(pca.PCAModel),
        'umap': projection.ProjectionManager(umap.UmapModel),
    }
    self._interpreters = dict(**self._interpreters,
                              **prediction_analysis_interpreters,
                              **embedding_based_interpreters)

    # Component to sync state from TS -> Python. Used in notebooks.
    if sync_state:
      self.ui_state_tracker = selection_state.SelectionStateTracker()
      self._interpreters['_sync_state'] = self.ui_state_tracker

    # Information on models, datasets, and other components.
    self._info = self._build_metadata()

    # Optionally, run models to pre-populate cache.
    if warm_projections:
      logging.info(
          'Projection (dimensionality reduction) warm-start requested; '
          'will do full warm-start for all models since predictions are needed.'
      )
      warm_start = 1.0

    if warm_start > 0:
      self._warm_start(
          rate=warm_start, progress_indicator=warm_start_progress_indicator)
      self.save_cache()
      if warm_start >= 1:
        warm_projections = True

    # If you add a new embedding projector that should be warm-started,
    # also add it to the list here.
    # TODO(lit-dev): add some registry mechanism / automation if this grows to
    # more than 2-3 projection types.
    if warm_projections:
      self._warm_projections(['pca', 'umap'])

    handlers = {
        # Metadata endpoints.
        '/get_info': self._get_info,
        # Dataset-related endpoints.
        '/get_dataset': self._get_dataset,
        '/create_dataset': self._create_dataset,
        '/create_model': self._create_model,
        '/get_generated': self._get_generated,
        '/save_datapoints': self._save_datapoints,
        '/load_datapoints': self._load_datapoints,
        '/annotate_new_data': self._annotate_new_data,
        # Model prediction endpoints.
        '/get_preds': self._get_preds,
        '/get_interpretations': self._get_interpretations,
    }

    self._wsgi_app = wsgi_app.App(
        # Wrap endpoint fns to take (handler, request, environ)
        handlers={k: self.make_handler(v) for k, v in handlers.items()},
        project_root=client_root,
        index_file='static/index.html',
    )

  def save_cache(self):
    for m in self._models.values():
      if isinstance(m, caching.CachingModelWrapper):
        m.save_cache()

  def __call__(self, environ, start_response):
    """Implementation of the WSGI interface."""
    return self._wsgi_app(environ, start_response)
