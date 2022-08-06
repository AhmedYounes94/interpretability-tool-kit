# Copyright 2022 Google LLC
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
"""Selection state tracker.

This is a stateful component, intended for use in notebook/Colab contexts
to sync the UI selection state back to Python for further analysis.
"""
from typing import Optional, Sequence, TypedDict

import attr
from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types

IndexedInput = types.IndexedInput
JsonDict = types.JsonDict


@attr.s(auto_attribs=True, kw_only=True)
class SelectionState(object):
  """UI selection state."""
  dataset_name: Optional[str] = None
  dataset: Optional[lit_dataset.IndexedDataset] = None

  primary: Optional[IndexedInput] = None
  selection: list[IndexedInput] = attr.Factory(list)
  pinned: Optional[IndexedInput] = None


class SelectionStateTracker(lit_components.Interpreter):
  """Selection state tracker; mirrors state from frontend SelectionService.

  WARNING: this component is _stateful_, and in current form implements no
  locking or access control. We recommend using this only in a single-user,
  single-threaded context such as IPython or Colab notebooks.
  """

  class ConfigType(TypedDict, total=False):
    dataset_name: Optional[str]
    primary: Optional[str]
    pinned: Optional[str]

  def is_compatible(self, model: lit_model.Model) -> bool:
    """Return false as this is not applicable to specific models."""
    return False

  def run(self, *args, **kw):
    raise NotImplementedError('Use run_with_metadata() instead.')

  def __init__(self):
    self._state = SelectionState()

  @property
  def state(self):
    return self._state

  def run_with_metadata(self,
                        indexed_inputs: Sequence[IndexedInput],
                        model: lit_model.Model,
                        dataset: lit_dataset.IndexedDataset,
                        model_outputs: Optional[list[JsonDict]] = None,
                        config: Optional[ConfigType] = None):
    """Set state according to params."""
    del model
    del model_outputs
    if config is None:
      config = {}

    self._state.dataset_name = config.get('dataset_name')
    self._state.dataset = dataset

    if (primary_id := config.get('primary')):
      self._state.primary = dataset.index[primary_id]

    self._state.selection = list(indexed_inputs)

    if (pinned_id := config.get('pinned')):
      self._state.pinned = dataset.index[pinned_id]
