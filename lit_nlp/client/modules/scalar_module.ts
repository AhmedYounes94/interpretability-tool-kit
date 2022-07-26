/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import '../elements/expansion_panel';

// tslint:disable:no-new-decorators
// taze: ResizeObserver from //third_party/javascript/typings/resize_observer_browser
import * as d3 from 'd3';
import {html} from 'lit';
import {customElement} from 'lit/decorators';
import {Scene, Selection, SpriteView} from 'megaplot';
import {computed, observable} from 'mobx';
// tslint:disable-next-line:ban-module-namespace-object-escape
const seedrandom = require('seedrandom');  // from //third_party/javascript/typings/seedrandom:bundle

import {app} from '../core/app';
import {LitModule} from '../core/lit_module';
import {LegendType} from '../elements/color_legend';
import {ThresholdChange} from '../elements/threshold_slider';
import {getBrandColor, hexToRGBA, RGBATuple} from '../lib/colors';
import {MulticlassPreds} from '../lib/lit_types';
import {styles as sharedStyles} from '../lib/shared_styles.css';
import {formatForDisplay, IndexedInput, ModelInfoMap, ModelSpec} from '../lib/types';
import {doesOutputSpecContain, findSpecKeys, getThresholdFromMargin, isLitSubtype} from '../lib/utils';
import {GLOBAL_FACET} from '../services/classification_service';
import {CalculatedColumnType, CLASSIFICATION_SOURCE_PREFIX, REGRESSION_SOURCE_PREFIX, SCALAR_SOURCE_PREFIX} from '../services/data_service';
import {ClassificationService, ColorService, DataService, FocusService, GroupService, SelectionService} from '../services/services';

import {styles} from './scalar_module.css';

/** The maximum number of scatterplots to render on page load. */
export const MAX_DEFAULT_PLOTS = 2;

const DEFAULT_BORDER_WIDTH = 2;
const DEFAULT_LINE_COLOR = '#cccccc';
const DEFAULT_SCENE_PARAMS = {defaultTransitionTimeMs: 0};
const RGBA_CYEA_700 = hexToRGBA(getBrandColor('cyea', '700').color);
const RGBA_MAGE_400 = hexToRGBA(getBrandColor('mage', '400').color);
const RGBA_MAGE_700 = hexToRGBA(getBrandColor('mage', '700').color);
const RGBA_WHITE: RGBATuple = [1, 1, 1, 1];
const RNG_Y_AXIS = '__LIT_RNG_Y_AXIS';
const SPRITE_SIZE_LG = 6 * window.devicePixelRatio + DEFAULT_BORDER_WIDTH;
const SPRITE_SIZE_MD = 5 * window.devicePixelRatio + DEFAULT_BORDER_WIDTH;
const SPRITE_SIZE_SM = 4 * window.devicePixelRatio + DEFAULT_BORDER_WIDTH;

interface ColumnPositionMap {
  [columnName: string]: number;
}

/** Indexed scalars for an id, inclusive of model input and output scalars. */
interface IndexedScalars {
  id: string;
  data:{
    // Values in this structure can be undefined if the call to
    // DataService.getVal(2) in this.updatePredictions() occurs before the
    // async calls in DataService have returned the scalar values for model's
    // scalar predictions, regressions, and classifications. If these calls have
    // not yet returned, DataService.getVal(2) will try to get the value from
    // the Dataset, but since the key being used is from the model's output spec
    // the function will return undefined. These eventually get sorted out via
    // updates, but can cause hidden/frustrating errors in D3.attr(2) calls if
    // not handled appropriately.
    [key: string]: number | number[] | undefined
  };
  position: {
    x: ColumnPositionMap;
    y: ColumnPositionMap;
  };
}

interface PlotInfo {
  hidden: boolean;
  /** The prediction key in output spec. */
  key: string;
  /** The label of interest in the vocab. */
  label?: string;
  /** A MegaPlot Selection that binds all data to the Scene. */
  defaultPoints?: Selection<IndexedScalars>;
  /** A MegaPlot Selection for selectionService.selectedIds. */
  selectedPoints?: Selection<IndexedScalars>;
  /** A MegaPlot Selection for selectionService.primarySelectedId. */
  pinnedPoint?: Selection<IndexedScalars>;
  /** A MegaPlot Selection for pinnedSelectionService.primarySelectedId. */
  primaryPoint?: Selection<IndexedScalars>;
  /** A MegaPlot Selection for focusService.focusData. */
  hoveredPoint?: Selection<IndexedScalars>;
  /** The MegaPlot Scene into which the scatterplot is rendered. */
  scene?: Scene;
  xScale?: d3.ScaleLinear<number, number>;
  yScale?: d3.ScaleLinear<number, number>;
}

/**
 * A LIT module that visualizes prediction scores and other scalar values.
 */
@customElement('scalar-module')
export class ScalarModule extends LitModule {
  static override title = 'Scalars';
  static override numCols = 4;
  static override template =
      (model: string, selectionServiceIndex: number, shouldReact: number) =>
          html`<scalar-module model=${model} .shouldReact=${shouldReact}
                selectionServiceIndex=${selectionServiceIndex}>
              </scalar-module>`;

  static override get styles() {
    return [sharedStyles, styles];
  }

  static override shouldDisplayModule(modelSpecs: ModelInfoMap) {
    return doesOutputSpecContain(modelSpecs, ['Scalar', 'MulticlassPreds']);
  }

  private readonly colorService = app.getService(ColorService);
  private readonly classificationService =
      app.getService(ClassificationService);
  private readonly groupService = app.getService(GroupService);
  private readonly focusService = app.getService(FocusService);
  private readonly dataService = app.getService(DataService);
  private readonly pinnedSelectionService =
      app.getService(SelectionService, 'pinned');

  // tslint:disable-next-line:no-any ban-module-namespace-object-escape
  private readonly rng = seedrandom('lit');

  private readonly plots = new Map<string, PlotInfo>();
  private readonly resizeObserver = new ResizeObserver(() => {
    const footer =
        this.shadowRoot!.querySelector<HTMLElement>('.module-footer');
    this.legendWidth = footer?.clientWidth || 150;

    for (const info of this.plots.values()) {
      delete info.scene;
    }

    for (const pred of this.preds) {
      pred.position.x = {};
    }

    this.updatePlots();
  });

  private numPlotsRendered: number = 0;
  private legendWidth = 150;
  @observable private preds: IndexedScalars[] = [];


  /**
   * The smaller of 100,000 and the dataset size. Since MegaPlot allocates the
   * next largest perfect square of memory for its texel space, the 100,000
   * sprite limit ensures the GPU shouldn't run out of memory, even when
   * multiple scatterplots are drawn.
   */
  @computed get datasetSize(): number {
    return Math.min(
      this.appState.metadata.datasets[this.appState.currentDataset].size,
      100_000);
  }

  @computed
  private get scalarColumnsToPlot() {
    return this.groupService.numericalFeatureNames.filter(feat => {
      const col = this.dataService.getColumnInfo(feat);
      if (col == null) {
        return true;  // Col will be null for input fields
      } else if (col.source.includes(CLASSIFICATION_SOURCE_PREFIX) ||
                 col.source.includes(REGRESSION_SOURCE_PREFIX) ||
                 col.source.includes(SCALAR_SOURCE_PREFIX)) {
        return col.source.includes(this.model);
      } else {
        return true;
      }
    });
  }

  @computed
  private get classificationKeys() {
    const {output} = this.appState.getModelSpec(this.model);
    return findSpecKeys(output, 'MulticlassPreds');
  }

  override firstUpdated() {
    const getDataChanges = () => [
      this.appState.currentInputData,
      this.dataService.dataVals,
      this.scalarColumnsToPlot
    ];
    this.reactImmediately(getDataChanges, () => {
      for (const info of this.plots.values()) {
        info.defaultPoints?.clear();
        info.selectedPoints?.clear();
        info.primaryPoint?.clear();
        info.pinnedPoint?.clear();
        info.hoveredPoint?.clear();
      }
      this.updatePredictions(this.appState.currentInputData);
    });

    const rebindChanges = () =>
        [this.colorService.selectedColorOption, this.preds];
    this.react(rebindChanges, () => {this.updatePlots();});

    const selectedChange = () => this.selectionService.selectedIds;
    this.react(selectedChange, (selectedIds) => {
      const selectedPreds = this.preds.filter(pred =>
          selectedIds.includes(pred.id) &&
          this.selectionService.primarySelectedId !== pred.id &&
          this.pinnedSelectionService.primarySelectedId !== pred.id &&
          this.focusService.focusData?.datapointId !== pred.id);
      for (const info of this.plots.values()) {
        info.selectedPoints?.clear().bind(selectedPreds);
      }
    });

    const primaryChange = () => this.selectionService.primarySelectedId;
    this.react(primaryChange, (id) => {
      const primaryPred = this.preds.filter(pred => id === pred.id);
      for (const info of this.plots.values()) {
        info.primaryPoint?.clear().bind(primaryPred);
      }
    });

    const pinnedChange = () => this.pinnedSelectionService.primarySelectedId;
    this.react(pinnedChange, (id) => {
      const pinnedPred = this.preds.filter(pred => id === pred.id);
      for (const info of this.plots.values()) {
        info.pinnedPoint?.clear().bind(pinnedPred);
      }
    });

    const hoverChange = () => this.focusService.focusData?.datapointId;
    this.react(hoverChange, (id) => {
      const hoveredPred = this.preds.filter(pred => id === pred.id);
      for (const info of this.plots.values()) {
        info.hoveredPoint?.clear().bind(hoveredPred);
      }
    });

    const container = this.shadowRoot!.getElementById('container')!;
    this.resizeObserver.observe(container);
  }

  /**
   * Get predictions from the backend for all input data and display by
   * prediction score in the plot.
   */
  private async updatePredictions(currentInputData?: IndexedInput[]) {
    if (currentInputData == null) {return;}

    const preds: IndexedScalars[] = [];
    for (const {id} of currentInputData) {
      const pred: IndexedScalars = {id, data: {}, position: {x: {}, y: {}}};
      for (const key of this.classificationKeys) {
        const column = this.dataService.getColumnName(this.model, key);
        pred.data[key] = this.dataService.getVal(id, column);
      }

      for (const scalarKey of this.scalarColumnsToPlot) {
        pred.data[scalarKey] = this.dataService.getVal(id, scalarKey);
      }
      preds.push(pred);
    }

    this.preds = preds;
  }

  /**
   * Returns the scale function for the scatter plot's x axis, for a given
   * key. If the key is for regression, we set the score range to be between the
   * min and max values of the regression scores.
   */
  private getXScale(key: string) {
    let scoreRange = [0, 1];

    const {output} = this.appState.getModelSpec(this.model);
    if (isLitSubtype(output[key], 'Scalar')) {
      const scalarValues = this.preds.map((pred) => pred.data[key]) as number[];
      scoreRange = [Math.min(...scalarValues), Math.max(...scalarValues)];
      // If the range is 0 (all values are identical, then artificially increase
      // the range so that an X-axis is properly displayed.
      if (scoreRange[0] === scoreRange[1]) {
        scoreRange[0] = scoreRange[0] - .1;
        scoreRange[1] = scoreRange[1] + .1;
      }
    } else if (this.scalarColumnsToPlot.indexOf(key) !== -1) {
      scoreRange = this.groupService.numericalFeatureRanges[key];
    }

    return d3.scaleLinear().domain(scoreRange).range([0, 1]);
  }

  /**
   * Returns the scale function for the scatter plot's y axis.
   */
  private getYScale(key: string, isRegression: boolean, errorColumn: string) {
    const scale = d3.scaleLinear().domain([0, 1]).range([0, 1]);

    if (isRegression) {
      const values = this.dataService.getColumn(errorColumn);
      const range = d3.extent(values);
      if (range != null && !range.some(isNaN)) {
        // Make the domain symmetric around 0
        const largest = Math.max(...(range as number[]).map(Math.abs));
        scale.domain([-largest, largest]);
      }
    }

    return scale;   // Regression output field
  }

  private getValue(preds: IndexedScalars, spec: ModelSpec, key: string,
                   label: string): number | undefined {
    // If for a multiclass prediction and the DataService has loaded the
    // classification results, return the label score from the array.
    if (spec.output[key] instanceof MulticlassPreds) {
      const {vocab} = spec.output[key] as MulticlassPreds;
      const index = vocab!.indexOf(label);
      const classPreds = preds.data[key];
      if (Array.isArray(classPreds)) return classPreds[index];
    }
    // Otherwise, return the value of data[key], which may be undefined if the
    // DataService async calls are still pending.
    return preds.data[key] as number | undefined;
  }

  /** Sets up the scatterplot using MegaPlot. */
  private setupPlot(info: PlotInfo, container: HTMLElement) {
    const {key, label} = info;
    const axesDiv = container.querySelector<HTMLDivElement>('.axes')!;
    const sceneDiv = container.querySelector<HTMLDivElement>('.scene')!;
    const {width, height} = sceneDiv.getBoundingClientRect();

    // Clear any existing content, e.g., after a resize
    axesDiv.textContent = '';
    sceneDiv.textContent = '';

    /**
     * The minimum memory that MegaPlot should allocate for its texel space in
     * the GPU. This 3x multiplier ensures this is large enough to account for
     * the entire dataset being in SelectionService.seletedIds plus pinned,
     * primary, and hovered datapoints.
     */
    const desiredSpriteCapacity = this.datasetSize * 3;

    // Determine if this is a RegressionScore column
    const errorColName = `${key}:${CalculatedColumnType.ERROR}`;
    const errFeatInfo = this.dataService.getColumnInfo(errorColName);
    const isRegression = errFeatInfo != null &&
        errFeatInfo.source.includes(REGRESSION_SOURCE_PREFIX);

    // X and Y scales and accessors
    const xScale = info.xScale = this.getXScale(key).range([0, width - 16]);
    const yScale = info.yScale = this.getYScale(key, isRegression, errorColName)
                                     .range([8, height - 8]);

    // Add the axes with D3
    d3.select(axesDiv).style('width', width).style('height', height);
    const axesSVG = d3.select(axesDiv).append<SVGSVGElement>('svg')
                      .style('width', width + 40)     // + for regression labels
                      .style('height', height + 12);  // + for axis labels

    axesSVG.append('g')
           .attr('id', 'xAxis')
           .attr('transform', `translate(40, ${height - 8})`)
           .call(d3.axisBottom(info.xScale));

    axesSVG.append('g')
           .attr('id', 'yAxis')
           .attr('transform', `translate(40, 0)`)
           .call(d3.axisLeft(info.yScale).ticks(isRegression ? 5 : 0 ));

    const lines = axesSVG.append('g')
                         .attr('id', 'lines')
                         .attr('transform', `translate(40, 0)`);

    const [xMin, xMax] = info.xScale.range();
    const [yMin, yMax] = info.yScale.range();

    if (isRegression) {
      const halfHeight = (yMax - yMin) / 2 + yMin;
      lines.append('line')
           .attr('id', 'regression-line')
           .attr('x1', xMin)
           .attr('y1', halfHeight)
           .attr('x2', xMax)
           .attr('y2', halfHeight)
           .style('stroke', DEFAULT_LINE_COLOR);
    }

    const brush = isRegression ? d3.brush() : d3.brushX();
    const brushGroup = axesSVG.append('g')
        .attr('id', 'brushGroup')
        .attr('transform', `translate(40, 0)`)
        .on('mouseenter', () => {this.focusService.clearFocus();});

    brush.extent([[xMin, 0], [xMax, yMax + 12] ]).on('start end', () => {
      const bounds = d3.event.selection;
      if (!d3.event.sourceEvent || !bounds?.length) return;

      const hasYDimension = Array.isArray(bounds[0]);
      const [x, x2] = (hasYDimension ? [bounds[0][0], bounds[1][0]] :
                                        bounds) as number[];
      const [y, y2] = (hasYDimension ? [bounds[0][1], bounds[1][1]] :
                                        [yMin, yMax]) as number[];
      const ids = info.defaultPoints?.hitTest(
          {x, y, width: (x2 - x), height: (y2 - y)}).map((p) => p.id);
      if (ids != null) this.selectionService.selectIds(ids);
      brushGroup.call(brush.move, null);
    });

    brushGroup.call(brush);

    const setCommonSpriteProperties = (sprite: SpriteView,
                                       pred: IndexedScalars) => {
      sprite.BorderRadiusPixel = DEFAULT_BORDER_WIDTH;
      sprite.Sides = 1;

      const modelSpec = this.appState.getModelSpec(this.model);
      const xValue = this.getValue(pred, modelSpec, key, label || '');
      const xScaledValue = xValue != null ? xScale(xValue) : NaN;
      sprite.PositionWorldX = isNaN(xScaledValue) ? xMin : xScaledValue;

      const yColumn = isRegression ? errorColName : RNG_Y_AXIS;
      if (pred.position.y[yColumn] == null) {
        // Store normalized Y value to stop jitter on selection/hover change.
        const yValue = isRegression ?
            this.dataService.getVal(pred.id, yColumn) : this.rng();
        const yPosition = yScale(yValue);
        pred.position.y[yColumn] = isNaN(yPosition) ? yMax : yPosition;
      }
      sprite.PositionWorldY = pred.position.y[yColumn];
    };

    const spriteExit = (sprite: SpriteView) => sprite.SizePixel = 0;

    // Render the scatterplot with MegaPlot
    info.scene = new Scene({
      container: sceneDiv, desiredSpriteCapacity, ...DEFAULT_SCENE_PARAMS
    });
    info.scene.scale.x = 1;
    info.scene.scale.y = 1;
    info.scene.offset.x = 8;
    info.scene.offset.y = height;

    info.defaultPoints = info.scene.createSelection<IndexedScalars>()
        .onExit(spriteExit)
        .onBind((sprite: SpriteView, pred: IndexedScalars) => {
          setCommonSpriteProperties(sprite, pred);

          const indexedInput = this.appState.getCurrentInputDataById(pred.id);
          const color = this.colorService.getDatapointColor(indexedInput);
          sprite.BorderColor = sprite.FillColor = hexToRGBA(color);
          sprite.BorderColorOpacity = sprite.FillColorOpacity = 0.25;
          sprite.SizePixel = SPRITE_SIZE_SM;
        });

    info.selectedPoints = info.scene.createSelection<IndexedScalars>()
        .onExit(spriteExit)
        .onBind((sprite: SpriteView, pred: IndexedScalars) => {
          setCommonSpriteProperties(sprite, pred);
          const indexedInput = this.appState.getCurrentInputDataById(pred.id);
          const color = this.colorService.getDatapointColor(indexedInput);
          sprite.BorderColor = RGBA_WHITE;
          sprite.FillColor = hexToRGBA(color);
          sprite.SizePixel = SPRITE_SIZE_MD;
        });

    info.pinnedPoint = info.scene.createSelection<IndexedScalars>()
        .onExit(spriteExit)
        .onBind((sprite: SpriteView, pred: IndexedScalars) => {
          setCommonSpriteProperties(sprite, pred);
          sprite.BorderColor = RGBA_MAGE_700;
          sprite.FillColor = RGBA_MAGE_400;
          sprite.SizePixel = SPRITE_SIZE_LG;
        });

    info.primaryPoint = info.scene.createSelection<IndexedScalars>()
        .onExit(spriteExit)
        .onBind((sprite: SpriteView, pred: IndexedScalars) => {
          setCommonSpriteProperties(sprite, pred);
          const indexedInput = this.appState.getCurrentInputDataById(pred.id);
          const color = this.colorService.getDatapointColor(indexedInput);
          sprite.BorderColor = RGBA_CYEA_700;
          sprite.FillColor = hexToRGBA(color);
          sprite.SizePixel = SPRITE_SIZE_LG;
        });

    info.hoveredPoint = info.scene.createSelection<IndexedScalars>()
        .onExit(spriteExit)
        .onBind((sprite: SpriteView, pred: IndexedScalars) => {
          setCommonSpriteProperties(sprite, pred);
          sprite.BorderColor = sprite.FillColor = RGBA_MAGE_400;
          sprite.SizePixel = SPRITE_SIZE_LG;
        });
  }

  /** Binds the predictions to the available MegaPlot Selection and Scene. */
  private updatePlots() {
    const margins = this.classificationService.marginSettings[this.model] || {};

    for (const [id, info] of this.plots.entries()) {
      const {hidden, scene} = info;
      if (hidden) continue;

      const selector = `div.scatterplot[data-id="${id}"]`;
      const container =
          this.renderRoot.querySelector<HTMLDivElement>(selector);
      if (!container) continue;

      const {width, height} = container.getBoundingClientRect();
      if (width === 0 || height === 0) continue;

      if (scene == null) this.setupPlot(info, container);

      const {defaultPoints, selectedPoints, pinnedPoint, primaryPoint,
             hoveredPoint} = info;

      if (defaultPoints == null || selectedPoints == null ||
          pinnedPoint == null || primaryPoint == null || hoveredPoint == null) {
        console.error('Uninitialized MegaPlot Selections', info);
        continue;
      }

      defaultPoints.bind(this.preds);

      const selectedPreds = this.preds.filter(pred =>
          this.selectionService.selectedIds.includes(pred.id) &&
          this.selectionService.primarySelectedId !== pred.id &&
          this.pinnedSelectionService.primarySelectedId !== pred.id &&
          this.focusService.focusData?.datapointId !== pred.id);
      selectedPoints.clear().bind(selectedPreds);

      const pinnedPred = this.preds.filter(pred =>
          this.pinnedSelectionService.primarySelectedId === pred.id);
      pinnedPoint.clear().bind(pinnedPred);

      const primaryPred = this.preds.filter(pred =>
          this.selectionService.primarySelectedId === pred.id);
      primaryPoint.clear().bind(primaryPred);

      const hoveredPred = this.preds.filter(pred =>
          this.focusService.focusData?.datapointId === pred.id);
      hoveredPoint.clear().bind(hoveredPred);

      const {key, xScale, yScale} = info;
      if (key == null || !this.classificationKeys.includes(key)) continue;
      if (xScale == null || yScale == null) continue;

      const {output} = this.appState.getModelSpec(this.model);
      const fieldSpec = output[key];
      if (!(fieldSpec instanceof MulticlassPreds) ||
          fieldSpec.vocab.length === 2) continue;
      if (margins[key] == null || margins[key][GLOBAL_FACET] == null) continue;

      const threshold =
          getThresholdFromMargin(margins[key][GLOBAL_FACET].margin);
      const thresholdPosition = xScale(threshold);
      const [yMin, yMax] = yScale.range();

      const lines = d3.select(container).select<SVGGElement>('#lines');
      lines.select('#threshold-line').remove();
      lines.append('line')
            .attr('id', '')
            .attr('x1', thresholdPosition)
            .attr('y1', yMin)
            .attr('x2', thresholdPosition)
            .attr('y2', yMax)
            .style('stroke', DEFAULT_LINE_COLOR);
    }
  }

  // Purposely overridding render() as opposed to renderImpl() as scalars are a
  // special case where the render pass here is just a set up for the containers
  // and the true rendering happens in reactions, due to the nature of this
  // module.
  override render() {
    this.numPlotsRendered = 0;

    const domain = this.colorService.selectedColorOption.scale.domain();
    const sequentialScale = typeof domain[0] === 'number';
    const legendType =
        sequentialScale ? LegendType.SEQUENTIAL: LegendType.CATEGORICAL;

    // clang-format off
    return html`<div class="module-container">
      <div class="module-results-area">
        <div id='container'>
          ${this.classificationKeys.map(key =>
            this.renderClassificationGroup(key))}
          ${this.scalarColumnsToPlot.map(key => this.renderPlot(key, ''))}
        </div>
      </div>
      <div class="module-footer">
        <color-legend legendType=${legendType}
          legendWidth=${this.legendWidth}
          selectedColorName=${this.colorService.selectedColorOption.name}
          .scale=${this.colorService.selectedColorOption.scale}>
        </color-legend>
      </div>
    </div>`;
    // clang-format on
  }

  private renderClassificationGroup(key: string) {
    const spec = this.appState.getModelSpec(this.model);
    const {vocab, null_idx} = spec.output[key] as MulticlassPreds;
    if (vocab == null) return;

    // In the binary classification case, only render one plot that
    // displays the positive class.
    if (vocab?.length === 2 && null_idx != null) {
      return html`${this.renderPlot(key, vocab[1 - null_idx])}`;
    }

    // Otherwise, return one plot per label in the multiclass case.
    // clang-format off
    return html`
        ${null_idx != null ? this.renderMarginSlider(key) : null}
        ${vocab.map(label => this.renderPlot(key, label))}`;
    // clang-format on
  }

  private renderPlot(key: string, label: string) {
    this.numPlotsRendered += 1;
    const id = label ? `${key}:${label}` : key;
    const {primarySelectedId} = this.selectionService;
    const primaryPreds = this.preds.find(pred => pred.id === primarySelectedId);

    let selectedValue;
    if (primaryPreds != null) {
      const modelSpec = this.appState.getModelSpec(this.model);
      const primaryValue =
          this.getValue(primaryPreds, modelSpec, key, label);
      selectedValue = `Value: ${formatForDisplay(primaryValue)}`;
    }

    const plotLabel = `${id}${selectedValue ? ` - ${selectedValue}` : ''}`;

    if (!this.plots.has(id)) {
      const hidden = this.numPlotsRendered > MAX_DEFAULT_PLOTS;
      this.plots.set(id, {key, label, hidden});
    }
    const info = this.plots.get(id)!;

    const rebase = (event: Pick<MouseEvent, 'clientX' | 'clientY'>) => {
      const {clientX, clientY} = event;
      const scene = this.renderRoot.querySelector<HTMLDivElement>(`div.scene`);
      const {top, left} = scene!.getBoundingClientRect();
      return {x: clientX - left, y: clientY - top};
    };

    const select = (event: MouseEvent) => {
      const {x, y} = rebase(event);
      const selected = info.defaultPoints?.hitTest({x, y});
      if (selected?.length) {
        this.selectionService.setPrimarySelection(selected[0].id);
      }
    };

    const hover = (event: MouseEvent) => {
      const {x, y} = rebase(event);
      const hovered = info.defaultPoints?.hitTest({x, y});
      if (hovered?.length) this.focusService.setFocusedDatapoint(hovered[0].id);
    };

    const toggleHidden = () => {
      info.hidden = !info.hidden;
      if (!info.scene) this.requestUpdate();
    };

    // clang-format off
    return html`<div class='plot-holder'>
      <expansion-panel  .label=${plotLabel} ?expanded=${!info.hidden}
                        @expansion-toggle=${toggleHidden}>
        <div  class="scatterplot" data-id=${id} @mousemove=${hover}
              @click=${select}>
          <div class="scene"></div>
          <div class="axes"></div>
        </div>
      </expansion-panel>
    </div>`;
    // clang-format on
  }

  private renderMarginSlider(key: string) {
    const margin = this.classificationService.getMargin(this.model, key);
    const callback = (e: CustomEvent<ThresholdChange>) => {
      this.classificationService.setMargin(this.model, key, e.detail.margin);
    };
    return html`<threshold-slider label=${key} ?isThreshold=${false}
      .margin=${margin} ?showControls=${true} @threshold-changed=${callback}>
    </threshold-slider>`;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'scalar-module': ScalarModule;
  }
}
