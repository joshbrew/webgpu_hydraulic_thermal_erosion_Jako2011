import ewrkr from './erosion.worker.js';

const DEFAULT_MIN_HEIGHT = 0;
const DEFAULT_MAX_HEIGHT = 1;
const DEFAULT_ITERATIONS_PER_FRAME = 5;
const DEFAULT_STEP_ITERATIONS = 128;
const DEFAULT_TIME_STEP = 0.02;
const DEFAULT_RAIN_RATE = 0.001;
const DEFAULT_EVAPORATION_RATE = 0.015;
const DEFAULT_PIPE_AREA = 20.0;
const DEFAULT_GRAVITY = 9.81;
const DEFAULT_CAPACITY_SCALE = 0.82;
const DEFAULT_SUSPENSION_RATE = 0.32;
const DEFAULT_DEPOSITION_RATE = 1.28;
const DEFAULT_SOFTENING_RATE = 2.8;
const DEFAULT_MAX_EROSION_DEPTH = 0.09;
const DEFAULT_THERMAL_RATE = 0.24;
const DEFAULT_TALUS_COEFF = 0.92;
const DEFAULT_TALUS_BIAS = 0.12;
const DEFAULT_RENDER_HEIGHT_SCALE = 0.045;
const DEFAULT_WATER_HEIGHT_SCALE = 0.06;
const DEFAULT_EDGE_WATER_FLOOR = 0.0;
const DEFAULT_WATER_OPACITY = 0.92;
const DEFAULT_SEDIMENT_TINT = 0.35;
const DEFAULT_HARDNESS_BASE = 0.16;
const DEFAULT_HARDNESS_VARIATION = 0.03;
const DEFAULT_SOURCE_X = 35;
const DEFAULT_SOURCE_Y = 18;
const DEFAULT_SOURCE_RADIUS = 4;
const DEFAULT_SOURCE_STRENGTH = 0.06;
const DEFAULT_RAIN_DURATION = 0.0;
const DEFAULT_PULSE2_DURATION = 0.0;
const DEFAULT_SOURCE_LAYOUT_MODE = 0;
const DEFAULT_RANDOM_SPRING_COUNT = 4;
const DEFAULT_SPRING_SEED = 1;
const DEFAULT_METERS_PER_PIXEL = 100;
const DEFAULT_CAMERA_AZIMUTH = 45;
const DEFAULT_CAMERA_ELEVATION = 42;
const DEFAULT_CAMERA_DISTANCE = 2.9;
const DEFAULT_CAMERA_POS_X = 1.5238999619464006;
const DEFAULT_CAMERA_POS_Y = 1.9404787584406888;
const DEFAULT_CAMERA_POS_Z = 1.5238999619464006;
const DEFAULT_CAMERA_MOVE_SPEED = 1.35;
const DEFAULT_CAMERA_LOOK_SENSITIVITY = 0.14;
const DEFAULT_TESSELLATION = 1;
const DEFAULT_HYDRAULIC_8_WAY = true;
const DEFAULT_PRECIP_ENABLED = true;
const DEFAULT_THERMAL_ENABLED = true;
const DEFAULT_HYDRAULIC_EROSION_ENABLED = true;
const DEFAULT_PAINT_RADIUS = 12;
const DEFAULT_PAINT_AMOUNT = 0.03;
const DEFAULT_PAINT_HARDNESS = 0.7;
const DEFAULT_DEM_SOURCE_MODE = 'single';
const LAYER_MATERIAL_PRESETS = {
  sand: { label: 'sand', hardnessMin: 0.42, hardnessMax: 0.70 },
  silt: { label: 'silt', hardnessMin: 0.24, hardnessMax: 0.42 },
  clay: { label: 'clay', hardnessMin: 0.10, hardnessMax: 0.22 },
  soft_rock: { label: 'soft rock', hardnessMin: 0.03, hardnessMax: 0.10 },
  bedrock: { label: 'bedrock', hardnessMin: 0.00, hardnessMax: 0.02 },
  custom: { label: 'custom', hardnessMin: 0.20, hardnessMax: 0.40 },
  default_base: { label: 'base', hardnessMin: DEFAULT_HARDNESS_BASE, hardnessMax: DEFAULT_HARDNESS_BASE + DEFAULT_HARDNESS_VARIATION },
};
const DEFAULT_LAYER_MATERIALS = [
  { enabled: true, preset: 'default_base', heightMin: 0.00, heightMax: 1.00, thermalEnabled: true },
  { enabled: true, preset: 'silt', heightMin: 0.46, heightMax: 0.82, thermalEnabled: true },
  { enabled: true, preset: 'clay', heightMin: 0.20, heightMax: 0.58, thermalEnabled: true },
  { enabled: true, preset: 'bedrock', heightMin: 0.00, heightMax: 0.30, thermalEnabled: true },
];
const DEBUG_LOGGING = false;

function debugLog(...args) {
  if (!DEBUG_LOGGING) return;
  console.log('[WebGPU Erosion UI]', ...args);
}

function makeButton(text) {
  const button = document.createElement('button');
  button.textContent = text;
  button.style.padding = '8px 12px';
  button.style.border = '1px solid #444';
  button.style.borderRadius = '8px';
  button.style.background = '#1f1f1f';
  button.style.color = '#f0f0f0';
  button.style.cursor = 'pointer';
  button.addEventListener('mouseenter', () => { button.style.background = '#2a2a2a'; });
  button.addEventListener('mouseleave', () => { button.style.background = '#1f1f1f'; });
  return button;
}

function makeNumberLabel(text, value, width = '88px', opts = {}) {
  const label = document.createElement('label');
  label.textContent = text;
  label.style.display = 'inline-flex';
  label.style.alignItems = 'center';
  label.style.gap = '8px';

  const input = document.createElement('input');
  input.type = 'number';
  input.value = String(value);
  input.min = String(opts.min ?? -1000000000);
  input.max = String(opts.max ?? 1000000000);
  input.step = String(opts.step ?? 0.01);
  input.style.width = width;
  input.style.padding = '6px 8px';
  input.style.border = '1px solid #444';
  input.style.borderRadius = '6px';
  input.style.background = '#1a1a1a';
  input.style.color = '#e8e8e8';

  label.input = input;
  label.appendChild(input);
  return label;
}

function makeCheckboxLabel(text, checked) {
  const label = document.createElement('label');
  label.style.display = 'inline-flex';
  label.style.alignItems = 'center';
  label.style.gap = '6px';

  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = checked;

  const span = document.createElement('span');
  span.textContent = text;

  label.input = input;
  label.append(input, span);
  return label;
}

function makeFilePickerLabel(text) {
  const label = document.createElement('label');
  label.style.display = 'inline-flex';
  label.style.alignItems = 'center';
  label.style.gap = '8px';
  label.style.flexWrap = 'wrap';

  const span = document.createElement('span');
  span.textContent = text;
  span.style.fontSize = '12px';
  span.style.opacity = '0.9';

  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.png,image/png';
  input.style.padding = '6px 8px';
  input.style.border = '1px solid #444';
  input.style.borderRadius = '6px';
  input.style.background = '#1a1a1a';
  input.style.color = '#e8e8e8';

  label.input = input;
  label.caption = span;
  label.append(span, input);
  return label;
}

function makeSelectLabel(text, values, selected) {
  const label = document.createElement('label');
  label.textContent = text;
  label.style.display = 'inline-flex';
  label.style.alignItems = 'center';
  label.style.gap = '8px';

  const select = document.createElement('select');
  select.style.padding = '6px 8px';
  select.style.border = '1px solid #444';
  select.style.borderRadius = '6px';
  select.style.background = '#1a1a1a';
  select.style.color = '#e8e8e8';

  for (const { value, label: optionLabel } of values) {
    const option = document.createElement('option');
    option.value = String(value);
    option.textContent = optionLabel;
    if (value === selected) option.selected = true;
    select.appendChild(option);
  }

  label.input = select;
  label.appendChild(select);
  return label;
}

function readNumber(label, fallback) {
  const value = Number(label.input.value);
  return Number.isFinite(value) ? value : fallback;
}

function makeSectionCard(title) {
  const section = document.createElement('section');
  section.style.border = '1px solid #2d2d2d';
  section.style.borderRadius = '10px';
  section.style.padding = '10px';
  section.style.background = '#141414';

  const heading = document.createElement('div');
  heading.textContent = title;
  heading.style.fontSize = '13px';
  heading.style.fontWeight = '600';
  heading.style.marginBottom = '8px';
  heading.style.letterSpacing = '0.02em';

  const body = document.createElement('div');
  body.style.display = 'flex';
  body.style.flexWrap = 'wrap';
  body.style.gap = '8px';

  section.append(heading, body);
  section.body = body;
  section.heading = heading;
  return section;
}

function makeSectionNote(text) {
  const note = document.createElement('div');
  note.textContent = text;
  note.style.fontSize = '12px';
  note.style.lineHeight = '1.35';
  note.style.color = 'rgba(255,255,255,0.68)';
  note.style.margin = '-2px 0 8px 0';
  return note;
}

function getSelectedTessellation() {
  return Math.max(1, Number(tessellationLabel.input.value) || DEFAULT_TESSELLATION);
}


function formatRange(range, digits = 4) {
  if (!range || !Number.isFinite(range.min) || !Number.isFinite(range.max)) {
    return 'n/a';
  }
  return `${range.min.toFixed(digits)} to ${range.max.toFixed(digits)}`;
}


function clampValue(value, min, max) {
  return value < min ? min : value > max ? max : value;
}

function degToRad(value) {
  return value * Math.PI / 180;
}

function normalizeVec3(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function subtractVec3(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function crossVec3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function multiplyMat4(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function perspectiveMat4(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy * 0.5);
  const nf = 1 / (near - far);
  const out = new Float32Array(16);
  out[0] = f / Math.max(aspect, 1e-6);
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

function lookAtMat4(eye, target, up) {
  const z = normalizeVec3(subtractVec3(eye, target));
  const x = normalizeVec3(crossVec3(up, z));
  const y = crossVec3(z, x);
  const out = new Float32Array(16);
  out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
  out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
  out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
  out[12] = -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]);
  out[13] = -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]);
  out[14] = -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]);
  out[15] = 1;
  return out;
}

function invertMat4(m) {
  const out = new Float32Array(16);
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  det = 1.0 / det;
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

function transformVec4(m, v) {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
    m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
  ];
}

const state = {
  sourceImageInfo: null,
  sourceFile: null,
  sourceFileName: '',
  sourceUploadedToWorker: false,
  layerSourceFiles: [null, null, null, null],
  worker: null,
  workerReady: false,
  workerWarming: false,
  simReady: false,
  simulationLoading: false,
  simulationRequestId: 0,
  workerCanvasTransferred: false,
  running: false,
  lastFrameMs: 0,
  buildCount: 0,
  readbackPending: false,
  lastReadbackAt: 0,
  gpuStats: null,
  sourcePoints: [],
  isPainting: false,
  lastPaintAt: 0,
  cameraPosX: DEFAULT_CAMERA_POS_X,
  cameraPosY: DEFAULT_CAMERA_POS_Y,
  cameraPosZ: DEFAULT_CAMERA_POS_Z,
  pointerLookActive: false,
  navLoopHandle: 0,
  navLastAt: 0,
  navKeys: { KeyW: false, KeyA: false, KeyS: false, KeyD: false, Space: false, KeyC: false },
  statsFrameHandle: 0,
};

let nextWorkerRequestId = 1;
const pendingWorkerRequests = new Map();

function getCanvasMetrics() {
  const rect = canvasWrap.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width || 960));
  const height = Math.max(1, Math.round(rect.height || Math.round(width * 2 / 3)));
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  return { width, height, dpr };
}

function attachWorkerListeners(worker) {
  worker.addEventListener('message', (event) => {
    const message = event.data || {};
    if (message.type === 'response' && message.requestId) {
      const pending = pendingWorkerRequests.get(message.requestId);
      if (!pending) return;
      pendingWorkerRequests.delete(message.requestId);
      if (message.ok) {
        pending.resolve(message);
      } else {
        pending.reject(new Error(message.error || 'Worker request failed.'));
      }
      return;
    }

    if (message.type === 'status') {
      if (message.stats) state.gpuStats = message.stats;
      if (Array.isArray(message.sourcePoints)) state.sourcePoints = message.sourcePoints;
      if (message.sourceImageInfo) state.sourceImageInfo = message.sourceImageInfo;
      if (Number.isFinite(message.lastFrameMs)) state.lastFrameMs = message.lastFrameMs;
      if (typeof message.running === 'boolean') state.running = message.running;
      if (typeof message.gpuReady === 'boolean') state.workerWarming = !message.gpuReady;
      if (typeof message.simReady === 'boolean') state.simReady = message.simReady;
      updateStatus();
      updateStats();
      return;
    }

    if (message.type === 'workerError') {
      debugLog('workerError', message.message);
      updateStatus(message.message || 'Worker error');
    }
  });

  worker.addEventListener('error', (event) => {
    console.error('[WebGPU Erosion UI] worker error', event);
    updateStatus(event.message || 'Worker crashed');
  });
}

async function ensureWorker() {
  if (state.worker) return state.worker;
  if (!('transferControlToOffscreen' in gpuCanvas)) {
    throw new Error('OffscreenCanvas worker rendering is not available in this browser.');
  }

  const worker = new Worker(ewrkr, { type: 'module' });
  attachWorkerListeners(worker);
  state.worker = worker;

  const offscreen = gpuCanvas.transferControlToOffscreen();
  state.workerCanvasTransferred = true;
  const metrics = getCanvasMetrics();
  const response = await callWorker('init', {
    canvas: offscreen,
    width: metrics.width,
    height: metrics.height,
    dpr: metrics.dpr,
  }, [offscreen]);
  state.workerReady = true;
  state.workerWarming = !!response.warming;
  updateStatus();
  return worker;
}

function postWorker(type, payload = {}, transfer = []) {
  if (!state.worker) return;
  state.worker.postMessage({ type, ...payload }, transfer);
}

function callWorker(type, payload = {}, transfer = []) {
  return new Promise((resolve, reject) => {
    if (!state.worker) {
      reject(new Error('Worker is not initialized.'));
      return;
    }
    const requestId = nextWorkerRequestId++;
    pendingWorkerRequests.set(requestId, { resolve, reject });
    state.worker.postMessage({ type, requestId, ...payload }, transfer);
  });
}


function syncWorkerCanvasSize() {
  if (!state.workerReady) return;
  const metrics = getCanvasMetrics();
  postWorker('resize', metrics);
}

function collectSimulationParams() {
  const primaryLayer = layerMaterialControls[0];
  const primaryErodMin = clampValue(readNumber(primaryLayer.hardnessMinLabel, DEFAULT_HARDNESS_BASE), 0.0, 1.0);
  const primaryErodMax = clampValue(readNumber(primaryLayer.hardnessMaxLabel, DEFAULT_HARDNESS_BASE + DEFAULT_HARDNESS_VARIATION), 0.0, 1.0);
  const normalizedPrimaryErodMin = Math.min(primaryErodMin, primaryErodMax);
  const normalizedPrimaryErodMax = Math.max(primaryErodMin, primaryErodMax);
  return {
    timeStep: readNumber(timeStepLabel, DEFAULT_TIME_STEP),
    rainRate: precipEnabledLabel.input.checked ? readNumber(rainRateLabel, DEFAULT_RAIN_RATE) : 0,
    evaporationRate: readNumber(evaporationRateLabel, DEFAULT_EVAPORATION_RATE),
    pipeArea: readNumber(pipeAreaLabel, DEFAULT_PIPE_AREA),
    gravity: readNumber(gravityLabel, DEFAULT_GRAVITY),
    capacityScale: readNumber(capacityScaleLabel, DEFAULT_CAPACITY_SCALE),
    suspensionRate: readNumber(suspensionRateLabel, DEFAULT_SUSPENSION_RATE),
    depositionRate: readNumber(depositionRateLabel, DEFAULT_DEPOSITION_RATE),
    softeningRate: readNumber(softeningRateLabel, DEFAULT_SOFTENING_RATE),
    maxErosionDepth: readNumber(maxErosionDepthLabel, DEFAULT_MAX_EROSION_DEPTH),
    thermalRate: thermalEnabledLabel.input.checked ? readNumber(thermalRateLabel, DEFAULT_THERMAL_RATE) : 0,
    hydraulicErosionEnabled: !!hydraulicErosionEnabledLabel.input.checked,
    talusSlopeCoeff: readNumber(talusCoeffLabel, DEFAULT_TALUS_COEFF),
    talusSlopeBias: readNumber(talusBiasLabel, DEFAULT_TALUS_BIAS),
    renderHeightScale: readNumber(renderHeightScaleLabel, DEFAULT_RENDER_HEIGHT_SCALE),
    waterHeightScale: readNumber(waterHeightScaleLabel, DEFAULT_WATER_HEIGHT_SCALE),
    edgeWaterFloor: Math.max(0, readNumber(edgeWaterFloorLabel, DEFAULT_EDGE_WATER_FLOOR)),
    waterOpacity: readNumber(waterOpacityLabel, DEFAULT_WATER_OPACITY),
    sedimentTint: readNumber(sedimentTintLabel, DEFAULT_SEDIMENT_TINT),
    hardnessBase: normalizedPrimaryErodMin,
    hardnessVariation: Math.max(0, normalizedPrimaryErodMax - normalizedPrimaryErodMin),
    sourceCenterX: readNumber(sourceXLabel, DEFAULT_SOURCE_X) / 100,
    sourceCenterY: readNumber(sourceYLabel, DEFAULT_SOURCE_Y) / 100,
    sourceRadius: readNumber(sourceRadiusLabel, DEFAULT_SOURCE_RADIUS),
    sourceStrength: readNumber(sourceStrengthLabel, DEFAULT_SOURCE_STRENGTH),
    rainDuration: readNumber(rainDurationLabel, DEFAULT_RAIN_DURATION),
    pulse2Duration: readNumber(pulse2DurationLabel, DEFAULT_PULSE2_DURATION),
    sourceLayoutMode: Number(sourceLayoutLabel.input.value) || 0,
    randomSpringCount: Math.max(1, Math.floor(readNumber(randomSpringCountLabel, DEFAULT_RANDOM_SPRING_COUNT))),
    sourceSeed: Math.floor(readNumber(springSeedLabel, DEFAULT_SPRING_SEED)),
    metersPerPixel: Math.max(1, readNumber(metersPerPixelLabel, DEFAULT_METERS_PER_PIXEL)),
    hydraulic8Way: !!hydraulic8WayLabel.input.checked,
    sourceEnabled: !!sourceEnabledLabel.input.checked,
    renderMode: Number(renderModeLabel.input.value) || 0,
    cameraAzimuthDeg: readNumber(cameraAzimuthLabel, DEFAULT_CAMERA_AZIMUTH),
    cameraElevationDeg: readNumber(cameraElevationLabel, DEFAULT_CAMERA_ELEVATION),
    // cameraDistance: readNumber(cameraDistanceLabel, DEFAULT_CAMERA_DISTANCE),
    cameraPosX: state.cameraPosX,
    cameraPosY: state.cameraPosY,
    cameraPosZ: state.cameraPosZ,
  };
}

function getDemSourceMode() {
  return demSourceModeLabel.input.value || DEFAULT_DEM_SOURCE_MODE;
}

function collectLayerMaterialConfig() {
  return {
    mode: getDemSourceMode(),
    layers: layerMaterialControls.map((layer, index) => ({
      enabled: !!layer.enableLabel.input.checked,
      preset: layer.materialPresetLabel.input.value,
      label: `Layer ${index + 1}`,
      heightMin: readNumber(layer.heightMinLabel, DEFAULT_LAYER_MATERIALS[index].heightMin),
      heightMax: readNumber(layer.heightMaxLabel, DEFAULT_LAYER_MATERIALS[index].heightMax),
      hardnessMin: clampValue(readNumber(layer.hardnessMinLabel, LAYER_MATERIAL_PRESETS.custom.hardnessMin), 0.0, 1.0),
      hardnessMax: clampValue(readNumber(layer.hardnessMaxLabel, LAYER_MATERIAL_PRESETS.custom.hardnessMax), 0.0, 1.0),
      thermalEnabled: !!layer.thermalEnableLabel.input.checked,
    })),
  };
}

function hasAnyDemSourceSelected() {
  return getDemSourceMode() === 'stack4' ? state.layerSourceFiles.some(Boolean) : !!state.sourceFile;
}

function getSourceSummaryName() {
  if (getDemSourceMode() === 'stack4') {
    return state.layerSourceFiles
      .map((file, index) => file ? `L${index + 1}:${file.name}` : null)
      .filter(Boolean)
      .join(' | ');
  }
  return state.sourceFile?.name || '';
}

function applyLayerMaterialPresetSelection(layerControl) {
  const preset = LAYER_MATERIAL_PRESETS[layerControl.materialPresetLabel.input.value];
  if (!preset || layerControl.materialPresetLabel.input.value === 'custom') return;
  layerControl.hardnessMinLabel.input.value = String(preset.hardnessMin);
  layerControl.hardnessMaxLabel.input.value = String(preset.hardnessMax);
}

function syncSingleModeMaterialFields() {
  const primaryLayer = layerMaterialControls[0];
  const heightMin = readNumber(primaryLayer.heightMinLabel, DEFAULT_MIN_HEIGHT);
  const heightMax = readNumber(primaryLayer.heightMaxLabel, DEFAULT_MAX_HEIGHT);
  const erodMin = clampValue(readNumber(primaryLayer.hardnessMinLabel, DEFAULT_HARDNESS_BASE), 0.0, 1.0);
  const erodMax = clampValue(readNumber(primaryLayer.hardnessMaxLabel, DEFAULT_HARDNESS_BASE + DEFAULT_HARDNESS_VARIATION), 0.0, 1.0);
  const normalizedErodMin = Math.min(erodMin, erodMax);
  const normalizedErodMax = Math.max(erodMin, erodMax);
  minHeightLabel.input.value = String(heightMin);
  maxHeightLabel.input.value = String(heightMax);
  hardnessBaseLabel.input.value = String(normalizedErodMin);
  hardnessVariationLabel.input.value = String(Math.max(0, normalizedErodMax - normalizedErodMin));
}

function updateLayerDemUiState() {
  const mode = getDemSourceMode();
  const showStackUploaders = mode === 'stack4';
  const singleMode = mode === 'single';
  fileLabel.style.display = showStackUploaders ? 'none' : 'inline-flex';
  demSourceNote.textContent = singleMode
    ? 'Single grayscale uses only Layer 1. Use that row to set the height range and erodability for the base DEM.'
    : 'Use a packed RGBA DEM where each channel is a layer, or a stack of up to four grayscale PNGs. Height min/max defines each layer top surface, erodability min/max defines how easily each layer cuts, and each layer can opt in or out of thermal collapse.';
  for (const [layerIndex, layer] of layerMaterialControls.entries()) {
    const showRow = !singleMode || layerIndex === 0;
    layer.row.style.display = showRow ? 'grid' : 'none';
    layer.filePickerLabel.style.display = showRow && showStackUploaders ? 'inline-flex' : 'none';
    layer.filePickerLabel.input.disabled = !(showRow && showStackUploaders);
    layer.enableLabel.style.display = singleMode && layerIndex === 0 ? 'none' : 'inline-flex';
    if (singleMode && layerIndex === 0) {
      layer.enableLabel.input.checked = true;
      layer.heading.textContent = 'Layer 1 / base material';
    } else {
      layer.heading.textContent = `Layer ${layerIndex + 1}`;
    }
  }
  syncSingleModeMaterialFields();
}

const root = document.createElement('div');
root.style.boxSizing = 'border-box';
root.style.padding = '16px';
root.style.fontFamily = 'system-ui, sans-serif';
root.style.color = '#e8e8e8';
root.style.background = '#111';
root.style.minHeight = '100vh';

const heading = document.createElement('h2');
heading.textContent = 'WebGPU Fast Hydraulic + Thermal Erosion (Jako)';
heading.style.margin = '0 0 12px 0';
heading.style.fontWeight = '600';

const controls = document.createElement('div');
controls.style.display = 'flex';
controls.style.flexWrap = 'wrap';
controls.style.alignItems = 'center';
controls.style.gap = '8px';
controls.style.marginBottom = '14px';

const fileLabel = document.createElement('label');
fileLabel.textContent = 'DEM PNG';
fileLabel.style.display = 'inline-flex';
fileLabel.style.alignItems = 'center';
fileLabel.style.gap = '8px';

const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.accept = '.png,image/png';
fileInput.style.padding = '6px 8px';
fileInput.style.border = '1px solid #444';
fileInput.style.borderRadius = '6px';
fileInput.style.background = '#1a1a1a';
fileInput.style.color = '#e8e8e8';
fileLabel.appendChild(fileInput);

const demSourceModeLabel = makeSelectLabel('DEM source', [
  { value: 'single', label: 'single grayscale' },
  { value: 'packed_rgba', label: 'packed RGBA layers' },
  { value: 'stack4', label: '4 grayscale stack' },
], DEFAULT_DEM_SOURCE_MODE);

const minHeightLabel = makeNumberLabel('Height floor', DEFAULT_MIN_HEIGHT, '88px');
const maxHeightLabel = makeNumberLabel('Height ceiling', DEFAULT_MAX_HEIGHT, '88px');
const iterationsPerFrameLabel = makeNumberLabel('Sim steps/frame', DEFAULT_ITERATIONS_PER_FRAME, '112px', { min: 1, step: 1 });
const stepIterationsLabel = makeNumberLabel('Single-step size', DEFAULT_STEP_ITERATIONS, '118px', { min: 1, step: 1 });
const timeStepLabel = makeNumberLabel('Time step', DEFAULT_TIME_STEP, '88px', { min: 0.001, step: 0.001 });
const rainRateLabel = makeNumberLabel('Background rain', DEFAULT_RAIN_RATE, '108px', { min: 0, step: 0.0001 });
const evaporationRateLabel = makeNumberLabel('Evaporation', DEFAULT_EVAPORATION_RATE, '96px', { min: 0, step: 0.001 });
const pipeAreaLabel = makeNumberLabel('Water flow width', DEFAULT_PIPE_AREA, '112px', { min: 0.1, step: 0.1 });
const gravityLabel = makeNumberLabel('Gravity', DEFAULT_GRAVITY, '84px', { min: 0.1, step: 0.1 });
const capacityScaleLabel = makeNumberLabel('Sediment carry', DEFAULT_CAPACITY_SCALE, '104px', { min: 0, step: 0.01 });
const suspensionRateLabel = makeNumberLabel('Erode into water', DEFAULT_SUSPENSION_RATE, '112px', { min: 0, step: 0.01 });
const depositionRateLabel = makeNumberLabel('Deposit from water', DEFAULT_DEPOSITION_RATE, '118px', { min: 0, step: 0.01 });
const softeningRateLabel = makeNumberLabel('Bank softening', DEFAULT_SOFTENING_RATE, '108px', { min: 0, step: 0.01 });
const maxErosionDepthLabel = makeNumberLabel('Max cut depth', DEFAULT_MAX_EROSION_DEPTH, '104px', { min: 0.001, step: 0.01 });
const thermalRateLabel = makeNumberLabel('Slope collapse', DEFAULT_THERMAL_RATE, '104px', { min: 0, step: 0.01 });
const talusCoeffLabel = makeNumberLabel('Slope limit scale', DEFAULT_TALUS_COEFF, '118px', { min: 0, step: 0.01 });
const talusBiasLabel = makeNumberLabel('Slope limit bias', DEFAULT_TALUS_BIAS, '110px', { min: 0, step: 0.01 });
const renderHeightScaleLabel = makeNumberLabel('Terrain exaggeration', DEFAULT_RENDER_HEIGHT_SCALE, '130px', { min: 0, step: 0.05 });
const waterHeightScaleLabel = makeNumberLabel('Water height scale', DEFAULT_WATER_HEIGHT_SCALE, '122px', { min: 0, step: 0.01 });
const edgeWaterFloorLabel = makeNumberLabel('Min edge water', DEFAULT_EDGE_WATER_FLOOR, '110px', { min: 0, step: 0.005 });
const waterOpacityLabel = makeNumberLabel('Water visibility', DEFAULT_WATER_OPACITY, '112px', { min: 0, max: 1, step: 0.05 });
const sedimentTintLabel = makeNumberLabel('Sediment boost', DEFAULT_SEDIMENT_TINT, '104px', { min: 0, max: 2, step: 0.05 });
const hardnessBaseLabel = makeNumberLabel('Base erodability', DEFAULT_HARDNESS_BASE, '108px', { min: 0.0, max: 1, step: 0.01 });
const hardnessVariationLabel = makeNumberLabel('Erodability variation', DEFAULT_HARDNESS_VARIATION, '122px', { min: 0, max: 1, step: 0.01 });
const sourceEnabledLabel = makeCheckboxLabel('Enable springs', false);
const sourceXLabel = makeNumberLabel('Spring X %', DEFAULT_SOURCE_X, '92px', { min: 0, max: 100, step: 1 });
const sourceYLabel = makeNumberLabel('Spring Y %', DEFAULT_SOURCE_Y, '92px', { min: 0, max: 100, step: 1 });
const sourceRadiusLabel = makeNumberLabel('Spring radius px', DEFAULT_SOURCE_RADIUS, '108px', { min: 1, step: 1 });
const sourceStrengthLabel = makeNumberLabel('Spring flow rate', DEFAULT_SOURCE_STRENGTH, '104px', { min: 0, step: 0.005 });
const rainDurationLabel = makeNumberLabel('Rain duration s', DEFAULT_RAIN_DURATION, '104px', { min: 0, step: 0.5 });
const pulse2DurationLabel = makeNumberLabel('Pulse 2 sec', DEFAULT_PULSE2_DURATION, '92px', { min: 0, step: 0.5 });
const sourceLayoutLabel = makeSelectLabel('Spring mode', [
  { value: 0, label: 'painted springs' },
  { value: 1, label: 'fixed random springs' },
], DEFAULT_SOURCE_LAYOUT_MODE);
const randomSpringCountLabel = makeNumberLabel('Spring count', DEFAULT_RANDOM_SPRING_COUNT, '96px', { min: 1, max: 16, step: 1 });
const springSeedLabel = makeNumberLabel('Spring seed', DEFAULT_SPRING_SEED, '96px', { min: 0, step: 1 });
const metersPerPixelLabel = makeNumberLabel('Meters / pixel', DEFAULT_METERS_PER_PIXEL, '104px', { min: 1, step: 1 });
const tessellationLabel = makeSelectLabel('Sim tess', [
  { value: 1, label: '1x' },
  { value: 2, label: '2x' },
  { value: 4, label: '4x' },
], DEFAULT_TESSELLATION);
const hydraulic8WayLabel = makeCheckboxLabel('8-way hydraulic pipes', DEFAULT_HYDRAULIC_8_WAY);
const precipEnabledLabel = makeCheckboxLabel('Enable precipitation', DEFAULT_PRECIP_ENABLED);
const thermalEnabledLabel = makeCheckboxLabel('Enable thermal erosion', DEFAULT_THERMAL_ENABLED);
const hydraulicErosionEnabledLabel = makeCheckboxLabel('Enable hydraulic erosion', DEFAULT_HYDRAULIC_EROSION_ENABLED);
const renderModeLabel = makeSelectLabel('View', [
  { value: 0, label: 'shaded' },
  { value: 2, label: 'water' },
  { value: 3, label: 'sediment' },
  { value: 4, label: 'erodability' },
  { value: 5, label: 'thermal' },
  { value: 6, label: 'erosion/deposition history' },
  { value: 7, label: 'natural geomorph' },
], 0);

const presetLabel = makeSelectLabel('Preset', [
  { value: 'paper_balanced', label: 'paper balanced' },
  { value: 'river_cut', label: 'river cut' },
  { value: 'delta_depositor', label: 'delta depositor' },
  { value: 'meander_builder', label: 'meander builder' },
  { value: 'thermal_heavy', label: 'thermal heavy' },
  { value: 'gentle_weathering', label: 'gentle weathering' },
  { value: 'rapid_incision', label: 'rapid incision' },
  { value: 'flash_flood', label: 'flash flood' },
  { value: 'badlands', label: 'badlands' },
  { value: 'canyon_carver', label: 'canyon carver' },
], 'paper_balanced');
const cameraAzimuthLabel = makeNumberLabel('Azimuth', DEFAULT_CAMERA_AZIMUTH, '84px', { min: -180, max: 180, step: 1 });
const cameraElevationLabel = makeNumberLabel('Elevation', DEFAULT_CAMERA_ELEVATION, '84px', { min: 5, max: 89, step: 1 });
const paintModeLabel = makeSelectLabel('Paint', [
  { value: 'none', label: 'off' },
  { value: 'raise', label: 'terrain +' },
  { value: 'lower', label: 'terrain -' },
  { value: 'spring_add', label: 'spring +' },
  { value: 'spring_erase', label: 'spring erase' },
], 'none');
const paintRadiusLabel = makeNumberLabel('Brush radius px', DEFAULT_PAINT_RADIUS, '112px', { min: 1, step: 1 });
const paintAmountLabel = makeNumberLabel('Brush amount', DEFAULT_PAINT_AMOUNT, '104px', { min: 0.001, step: 0.005 });
const paintHardnessLabel = makeNumberLabel('Brush hardness', DEFAULT_PAINT_HARDNESS, '116px', { min: 0.05, max: 1, step: 0.05 });

const materialPresetOptions = Object.entries(LAYER_MATERIAL_PRESETS).map(([value, info]) => ({ value, label: info.label }));
const layerMaterialControls = DEFAULT_LAYER_MATERIALS.map((defaults, index) => {
  const presetInfo = LAYER_MATERIAL_PRESETS[defaults.preset] || LAYER_MATERIAL_PRESETS.custom;
  const row = document.createElement('div');
  row.style.display = 'grid';
  row.style.gridTemplateColumns = 'repeat(2, minmax(0, 1fr))';
  row.style.gap = '8px';
  row.style.padding = '8px';
  row.style.border = '1px solid #2c2c2c';
  row.style.borderRadius = '8px';
  row.style.background = '#121212';

  const heading = document.createElement('div');
  heading.textContent = `Layer ${index + 1}`;
  heading.style.gridColumn = '1 / -1';
  heading.style.fontSize = '12px';
  heading.style.fontWeight = '600';
  heading.style.opacity = '0.95';

  const enableLabel = makeCheckboxLabel('Enable layer', defaults.enabled);
  const materialPresetLabel = makeSelectLabel('Preset', materialPresetOptions, defaults.preset);
  const heightMinLabel = makeNumberLabel('Height min', defaults.heightMin, '86px', { min: -10, max: 10, step: 0.01 });
  const heightMaxLabel = makeNumberLabel('Height max', defaults.heightMax, '86px', { min: -10, max: 10, step: 0.01 });
  const hardnessMinLabel = makeNumberLabel('Erod min', presetInfo.hardnessMin, '86px', { min: 0.0, max: 1, step: 0.01 });
  const hardnessMaxLabel = makeNumberLabel('Erod max', presetInfo.hardnessMax, '86px', { min: 0.0, max: 1, step: 0.01 });
  const thermalEnableLabel = makeCheckboxLabel('Thermal on this layer', defaults.thermalEnabled !== false);
  const filePickerLabel = makeFilePickerLabel(`Layer ${index + 1} PNG`);
  filePickerLabel.style.gridColumn = '1 / -1';

  row.append(
    heading,
    enableLabel,
    materialPresetLabel,
    heightMinLabel,
    heightMaxLabel,
    hardnessMinLabel,
    hardnessMaxLabel,
    thermalEnableLabel,
    filePickerLabel,
  );

  return {
    row,
    heading,
    enableLabel,
    materialPresetLabel,
    heightMinLabel,
    heightMaxLabel,
    hardnessMinLabel,
    hardnessMaxLabel,
    thermalEnableLabel,
    filePickerLabel,
  };
});

const runButton = makeButton('Run');
runButton.disabled = true;
const stepButton = makeButton('Step');
stepButton.disabled = true;
const applyPresetButton = makeButton('Apply preset');
const resetRainTimerButton = makeButton('Reset rain timer');
resetRainTimerButton.disabled = true;
const resetButton = makeButton('Reset DEM');
resetButton.disabled = true;
const clearButton = makeButton('Clear');
const exportDemButton = makeButton('Export DEM PNG');
exportDemButton.disabled = true;
const clearSpringsButton = makeButton('Clear painted springs');
clearSpringsButton.disabled = true;

const statusLine = document.createElement('div');
statusLine.style.marginLeft = '12px';
statusLine.style.opacity = '0.9';
statusLine.style.fontSize = '14px';

controls.append(
  fileLabel,
  presetLabel,
  applyPresetButton,
  runButton,
  stepButton,
  exportDemButton,
  resetButton,
  clearButton,
  statusLine,
);

const layout = document.createElement('div');
layout.style.display = 'grid';
layout.style.gridTemplateColumns = 'minmax(320px, 1fr) minmax(280px, 380px)';
layout.style.gap = '16px';
layout.style.alignItems = 'start';

const viewerPanel = document.createElement('div');
viewerPanel.style.background = '#181818';
viewerPanel.style.border = '1px solid #2d2d2d';
viewerPanel.style.borderRadius = '12px';
viewerPanel.style.padding = '12px';
viewerPanel.style.boxSizing = 'border-box';
viewerPanel.style.minWidth = '0';

const viewerTitle = document.createElement('div');
viewerTitle.textContent = 'Render view:';
viewerTitle.style.fontSize = '15px';
viewerTitle.style.fontWeight = '600';
viewerTitle.style.marginBottom = '8px';

const canvasWrap = document.createElement('div');
canvasWrap.style.width = '100%';
canvasWrap.style.overflow = 'hidden';
canvasWrap.style.border = '1px solid #333';
canvasWrap.style.borderRadius = '10px';
canvasWrap.style.background = '#0d0d0d';
canvasWrap.style.position = 'relative';
canvasWrap.style.aspectRatio = '3 / 1.5';

const previewCanvas = document.createElement('canvas');
previewCanvas.width = 960;
previewCanvas.height = 640;
previewCanvas.style.display = 'block';
previewCanvas.style.width = '100%';
previewCanvas.style.height = '100%';
previewCanvas.style.background = '#0d0d0d';
previewCanvas.style.position = 'absolute';
previewCanvas.style.inset = '0';
previewCanvas.style.zIndex = '0';

const gpuCanvas = document.createElement('canvas');
gpuCanvas.width = 960;
gpuCanvas.height = 640;
gpuCanvas.style.display = 'block';
gpuCanvas.style.width = '100%';
gpuCanvas.style.height = '100%';
gpuCanvas.style.background = '#0d0d0d';
gpuCanvas.style.position = 'absolute';
gpuCanvas.style.inset = '0';
gpuCanvas.style.zIndex = '1';
gpuCanvas.style.visibility = 'hidden';
gpuCanvas.style.touchAction = 'none';
gpuCanvas.tabIndex = 0;

const overlayCanvas = document.createElement('canvas');
overlayCanvas.width = 960;
overlayCanvas.height = 640;
overlayCanvas.style.display = 'block';
overlayCanvas.style.width = '100%';
overlayCanvas.style.height = '100%';
overlayCanvas.style.position = 'absolute';
overlayCanvas.style.inset = '0';
overlayCanvas.style.zIndex = '2';
overlayCanvas.style.pointerEvents = 'none';
overlayCanvas.style.visibility = 'hidden';
canvasWrap.append(previewCanvas, gpuCanvas, overlayCanvas);
viewerPanel.append(viewerTitle, canvasWrap);

const sidePanel = document.createElement('div');
sidePanel.style.background = '#181818';
sidePanel.style.border = '1px solid #2d2d2d';
sidePanel.style.borderRadius = '12px';
sidePanel.style.padding = '12px';
sidePanel.style.boxSizing = 'border-box';
sidePanel.style.maxHeight = '85vh'
sidePanel.style.overflowY = 'scroll';

const statsTitle = document.createElement('div');
statsTitle.textContent = 'Stats';
statsTitle.style.fontSize = '15px';
statsTitle.style.fontWeight = '600';
statsTitle.style.marginBottom = '8px';

const statsBlock = document.createElement('div');
statsBlock.style.whiteSpace = 'pre-line';
statsBlock.style.fontSize = '13px';
statsBlock.style.opacity = '0.95';
statsBlock.style.lineHeight = '1.45';
statsBlock.textContent = 'Load a DEM PNG to begin.';

const settingsStack = document.createElement('div');
settingsStack.style.display = 'grid';
settingsStack.style.gap = '10px';
settingsStack.style.marginBottom = '12px';

const processSection = makeSectionCard('Process toggles');
processSection.append(makeSectionNote('Master on/off switches for precipitation, springs, hydraulic erosion, and thermal erosion.'));
processSection.body.append(
  precipEnabledLabel,
  sourceEnabledLabel,
  thermalEnabledLabel,
  hydraulicErosionEnabledLabel,
  rainDurationLabel,
  resetRainTimerButton,
);

const demSourceSection = makeSectionCard('DEM source and layered materials');
const demSourceNote = makeSectionNote('Use a single grayscale DEM, a packed RGBA DEM where each channel is a layer, or a stack of up to four grayscale PNGs. Height min/max defines each layer top surface, erodability min/max defines how easily each layer cuts, and each layer can opt in or out of thermal collapse.');
demSourceSection.append(demSourceNote);
demSourceSection.body.append(demSourceModeLabel);
for (const layer of layerMaterialControls) {
  demSourceSection.body.append(layer.row);
}

const sharedFlowSection = makeSectionCard('Shared water flow and simulation');
sharedFlowSection.append(makeSectionNote('These affect precipitation, springs, water flow, edge runoff retention, and general simulation pacing. They are not thermal-only.'));
sharedFlowSection.body.append(
  iterationsPerFrameLabel,
  stepIterationsLabel,
  timeStepLabel,
  rainRateLabel,
  evaporationRateLabel,
  pipeAreaLabel,
  gravityLabel,
  tessellationLabel,
  hydraulic8WayLabel,
  metersPerPixelLabel,
  edgeWaterFloorLabel,
);

const hydraulicSection = makeSectionCard('Hydraulic erosion only');
hydraulicSection.append(makeSectionNote('These only change hydraulic erosion and deposition behavior. Water can still animate with hydraulic erosion disabled.'));
hydraulicSection.body.append(
  capacityScaleLabel,
  suspensionRateLabel,
  depositionRateLabel,
  softeningRateLabel,
  maxErosionDepthLabel,
);

const thermalSection = makeSectionCard('Thermal erosion only');
thermalSection.append(makeSectionNote('These only affect slope collapse and talus behavior.'));
thermalSection.body.append(
  thermalRateLabel,
  talusCoeffLabel,
  talusBiasLabel,
);

const springsSection = makeSectionCard('Springs');
springsSection.append(makeSectionNote('Spring mode and spring-source controls. Random-spring controls only apply in fixed random springs mode.'));
springsSection.body.append(
  sourceLayoutLabel,
  sourceRadiusLabel,
  sourceStrengthLabel,
  pulse2DurationLabel,
  randomSpringCountLabel,
  springSeedLabel,
  clearSpringsButton,
);

const viewSection = makeSectionCard('View and shading');
viewSection.append(makeSectionNote('Render-only controls. These do not change erosion behavior. Natural geomorph uses erosion/deposition history plus thermal activity to tint rock, soil, alluvium, and colluvium.'));
viewSection.body.append(
  renderModeLabel,
  renderHeightScaleLabel,
  waterHeightScaleLabel,
  waterOpacityLabel,
  sedimentTintLabel,
);

const cameraSection = makeSectionCard('Camera and navigation');
cameraSection.append(makeSectionNote('Orbit inputs below. Click the canvas with paint off for pointer look, then use WASD to move, Space to go up, C to go down, and Esc to release.'));
cameraSection.body.append(
  cameraAzimuthLabel,
  cameraElevationLabel,
  // cameraDistanceLabel,
);

const paintSection = makeSectionCard('Painting');
paintSection.append(makeSectionNote('Terrain and spring painting tools. Pointer look is disabled while a paint mode is active.'));
paintSection.body.append(
  paintModeLabel,
  paintRadiusLabel,
  paintAmountLabel,
  paintHardnessLabel,
);

settingsStack.append(cameraSection, viewSection, demSourceSection, processSection, sharedFlowSection, hydraulicSection, thermalSection, springsSection, paintSection);

sidePanel.append(settingsStack, statsTitle, statsBlock);
layout.append(viewerPanel, sidePanel);
root.append(heading, controls, layout);

document.body.style.margin = '0';
document.body.style.background = '#111';
document.body.appendChild(root);

window.addEventListener('beforeunload', () => {
  stopLoop();
  if (state.worker) {
    postWorker('destroy');
    state.worker.terminate();
    state.worker = null;
    state.workerReady = false;
    state.workerWarming = false;
    state.simReady = false;
    state.simulationLoading = false;
  }
});

window.addEventListener('resize', () => {
  syncCanvasSizes();
  syncWorkerCanvasSize();
  postWorker('render');
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  void loadSourceImage(file);
});

demSourceModeLabel.input.addEventListener('change', () => {
  state.sourceUploadedToWorker = false;
  updateLayerDemUiState();
  syncSingleModeMaterialFields();
  if (hasAnyDemSourceSelected()) {
    void initializeSimulation();
  } else {
    updateStats();
  }
});

for (const [layerIndex, layer] of layerMaterialControls.entries()) {
  layer.materialPresetLabel.input.addEventListener('change', () => {
    applyLayerMaterialPresetSelection(layer);
    if (layerIndex === 0) syncSingleModeMaterialFields();
    if (hasAnyDemSourceSelected()) {
      void initializeSimulation();
    } else {
      updateStats();
    }
  });
  for (const inputLabel of [layer.enableLabel, layer.heightMinLabel, layer.heightMaxLabel, layer.hardnessMinLabel, layer.hardnessMaxLabel, layer.thermalEnableLabel]) {
    inputLabel.input.addEventListener('input', () => {
      if (layerIndex === 0) syncSingleModeMaterialFields();
      if (hasAnyDemSourceSelected()) {
        void initializeSimulation();
      } else {
        scheduleStatsUpdate();
      }
    });
    inputLabel.input.addEventListener('change', () => {
      if (layerIndex === 0) syncSingleModeMaterialFields();
      if (hasAnyDemSourceSelected()) {
        void initializeSimulation();
      } else {
        scheduleStatsUpdate();
      }
    });
  }
  layer.filePickerLabel.input.addEventListener('change', () => {
    const file = layer.filePickerLabel.input.files?.[0] || null;
    state.layerSourceFiles[layerIndex] = file;
    state.sourceUploadedToWorker = false;
    state.sourceFileName = getSourceSummaryName();
    if (hasAnyDemSourceSelected()) {
      void initializeSimulation();
    } else {
      updateStats();
    }
  });
}

applyPresetButton.addEventListener('click', () => {
  applyPreset(presetLabel.input.value);
});

runButton.addEventListener('click', () => {
  if (state.running) {
    stopLoop();
  } else {
    startLoop();
  }
});

resetRainTimerButton.addEventListener('click', async () => {
  if (!state.workerReady || !state.gpuStats?.ready) return;
  applySimulationParams();
  const response = await callWorker('resetRainTimer');
  if (response.stats) state.gpuStats = response.stats;
  if (Array.isArray(response.sourcePoints)) state.sourcePoints = response.sourcePoints;
  updateStatus('Rain timer reset');
  scheduleStatsUpdate();
});

stepButton.addEventListener('click', async () => {
  if (!state.workerReady || !state.gpuStats?.ready) return;
  applySimulationParams();
  const response = await callWorker('step', {
    iterations: Math.max(1, Math.floor(readNumber(stepIterationsLabel, DEFAULT_STEP_ITERATIONS))),
    render: true,
  });
  if (response.stats) state.gpuStats = response.stats;
  if (Array.isArray(response.sourcePoints)) state.sourcePoints = response.sourcePoints;
  await scheduleReadbackStats(true);
  scheduleStatsUpdate();
});
iterationsPerFrameLabel.input.addEventListener('input', async () => {
  const iterationsPerFrame = Math.max(1, Math.floor(readNumber(iterationsPerFrameLabel, DEFAULT_ITERATIONS_PER_FRAME)));
  scheduleStatsUpdate();
  if (!state.workerReady) return;
  if (state.running) {
    await callWorker('startLoop', { iterationsPerFrame });
  }
});


exportDemButton.addEventListener('click', async () => {
  if (!state.workerReady || !state.gpuStats?.ready) return;
  exportDemButton.disabled = true;
  updateStatus('Exporting DEM PNG…');
  try {
    const response = await callWorker('exportTerrainPng');
    const bytes = response.data instanceof ArrayBuffer ? response.data : response.data?.buffer;
    if (!(bytes instanceof ArrayBuffer)) {
      throw new Error('Terrain export did not return PNG data.');
    }
    const blob = new Blob([bytes], { type: 'image/png' });
    const baseName = (state.sourceFileName || 'terrain').replace(/\.[^.]+$/, '');
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = `${baseName}_dem_export.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    const minTerrain = Number.isFinite(response.minTerrain) ? response.minTerrain.toFixed(4) : 'n/a';
    const maxTerrain = Number.isFinite(response.maxTerrain) ? response.maxTerrain.toFixed(4) : 'n/a';
    updateStatus(`Exported DEM PNG (${response.width}x${response.height}, range ${minTerrain} to ${maxTerrain})`);
  } catch (error) {
    updateStatus(error instanceof Error ? error.message : String(error));
  } finally {
    exportDemButton.disabled = !(state.gpuStats?.ready);
  }
});



clearSpringsButton.addEventListener('click', async () => {
  if (!state.workerReady) return;
  const response = await callWorker('clearPaintedSprings');
  if (response.stats) state.gpuStats = response.stats;
  if (Array.isArray(response.sourcePoints)) state.sourcePoints = response.sourcePoints;
  updateStatus('Cleared painted springs');
  scheduleStatsUpdate();
});

paintModeLabel.input.addEventListener('change', () => {
  updateCanvasInteractionState();
  scheduleStatsUpdate();
});

for (const label of [precipEnabledLabel, thermalEnabledLabel, hydraulicErosionEnabledLabel]) {
  label.input.addEventListener('change', () => {
    refreshProcessControlState();
    applySimulationParams();
    postWorker('render');
    scheduleStatsUpdate();
  });
}

for (const label of [paintRadiusLabel, paintAmountLabel, paintHardnessLabel]) {
  label.input.addEventListener('input', () => {
    scheduleStatsUpdate();
  });
}

resetButton.addEventListener('click', () => {
  void initializeSimulation();
});

clearButton.addEventListener('click', clearAll);

const sourcePointSensitiveInputs = new Set([
  sourceRadiusLabel,
  sourceStrengthLabel,
  randomSpringCountLabel,
  springSeedLabel,
]);

for (const label of [
  timeStepLabel,
  rainRateLabel,
  evaporationRateLabel,
  pipeAreaLabel,
  gravityLabel,
  capacityScaleLabel,
  suspensionRateLabel,
  depositionRateLabel,
  softeningRateLabel,
  maxErosionDepthLabel,
  thermalRateLabel,
  talusCoeffLabel,
  talusBiasLabel,
  renderHeightScaleLabel,
  waterHeightScaleLabel,
  waterOpacityLabel,
  sedimentTintLabel,
  cameraAzimuthLabel,
  cameraElevationLabel,
  // cameraDistanceLabel,
  sourceXLabel,
  sourceYLabel,
  sourceRadiusLabel,
  sourceStrengthLabel,
  rainDurationLabel,
  pulse2DurationLabel,
  randomSpringCountLabel,
  springSeedLabel,
]) {
  label.input.addEventListener('input', () => {
    applySimulationParams();
    if (state.running) {
      postWorker('startLoop', { iterationsPerFrame: Math.max(1, Math.floor(readNumber(iterationsPerFrameLabel, DEFAULT_ITERATIONS_PER_FRAME))) });
    } else {
      postWorker('render');
    }
    if (sourcePointSensitiveInputs.has(label)) {
      void refreshWorkerStatusLight();
    }
    updateStats();
  });
}

sourceEnabledLabel.input.addEventListener('change', () => {
  refreshProcessControlState();
  syncSingleModeMaterialFields();
  applySimulationParams();
  postWorker('render');
  void refreshWorkerStatusLight();
  scheduleStatsUpdate();
});

sourceLayoutLabel.input.addEventListener('change', () => {
  refreshProcessControlState();
  applySimulationParams();
  postWorker('render');
  void refreshWorkerStatusLight();
  scheduleStatsUpdate();
});

hydraulic8WayLabel.input.addEventListener('change', () => {
  applySimulationParams();
  if (state.running) {
    postWorker('startLoop', { iterationsPerFrame: Math.max(1, Math.floor(readNumber(iterationsPerFrameLabel, DEFAULT_ITERATIONS_PER_FRAME))) });
  } else {
    postWorker('render');
  }
  scheduleStatsUpdate();
});

renderModeLabel.input.addEventListener('change', () => {
  applySimulationParams();
  postWorker('render');
  scheduleStatsUpdate();
});

tessellationLabel.input.addEventListener('change', () => {
  updateStats();
  if (hasAnyDemSourceSelected()) {
    void initializeSimulation();
  }
});

applyPreset('paper_balanced');
updateLayerDemUiState();
syncSingleModeMaterialFields();
syncCanvasSizes();
setGPUCanvasVisible(false);
updateStatus();
refreshProcessControlState();
updateCanvasInteractionState();
drawSourceOverlay();

function getSelectedSpringMode() {
  return Number(sourceLayoutLabel.input.value) || 0;
}

function getSelectedSpringModeName() {
  return getSelectedSpringMode() === 1 ? 'fixed random springs' : 'painted springs';
}


function isPointerLookAllowed() {
  return paintModeLabel.input.value === 'none';
}

function updateCanvasInteractionState() {
  const painting = paintModeLabel.input.value !== 'none';
  if (document.pointerLockElement === gpuCanvas && painting) {
    document.exitPointerLock?.();
  }
  gpuCanvas.style.cursor = painting ? 'crosshair' : (document.pointerLockElement === gpuCanvas ? 'none' : 'grab');
}

function getCameraPosition() {
  return [state.cameraPosX, state.cameraPosY, state.cameraPosZ];
}

function getCameraBasis() {
  const azimuth = degToRad(readNumber(cameraAzimuthLabel, DEFAULT_CAMERA_AZIMUTH));
  const elevation = degToRad(readNumber(cameraElevationLabel, DEFAULT_CAMERA_ELEVATION));
  const eye = getCameraPosition();
  const forward = normalizeVec3([
    -Math.cos(elevation) * Math.cos(azimuth),
    -Math.sin(elevation),
    -Math.cos(elevation) * Math.sin(azimuth),
  ]);
  const forwardFlat = normalizeVec3([forward[0], 0, forward[2]]);
  const right = normalizeVec3(crossVec3(forwardFlat, [0, 1, 0]));
  const target = [eye[0] + forward[0], eye[1] + forward[1], eye[2] + forward[2]];
  return { azimuth, elevation, eye, target, forward, forwardFlat, right };
}

function cameraNeedsMovement() {
  return !!(state.navKeys.KeyW || state.navKeys.KeyA || state.navKeys.KeyS || state.navKeys.KeyD || state.navKeys.Space || state.navKeys.KeyC);
}

function applyCameraNavigationStep(now = performance.now()) {
  if (!state.pointerLookActive) {
    state.navLoopHandle = 0;
    state.navLastAt = 0;
    return;
  }
  const dt = state.navLastAt > 0 ? Math.min(0.05, Math.max(0.001, (now - state.navLastAt) / 1000)) : (1 / 60);
  state.navLastAt = now;
  if (cameraNeedsMovement()) {
    const basis = getCameraBasis();
    const moveSpeed = DEFAULT_CAMERA_MOVE_SPEED * dt;
    let dx = 0;
    let dy = 0;
    let dz = 0;
    if (state.navKeys.KeyW) { dx += basis.forwardFlat[0] * moveSpeed; dz += basis.forwardFlat[2] * moveSpeed; }
    if (state.navKeys.KeyS) { dx -= basis.forwardFlat[0] * moveSpeed; dz -= basis.forwardFlat[2] * moveSpeed; }
    if (state.navKeys.KeyD) { dx += basis.right[0] * moveSpeed; dz += basis.right[2] * moveSpeed; }
    if (state.navKeys.KeyA) { dx -= basis.right[0] * moveSpeed; dz -= basis.right[2] * moveSpeed; }
    if (state.navKeys.Space) { dy += moveSpeed; }
    if (state.navKeys.KeyC) { dy -= moveSpeed; }
    state.cameraPosX += dx;
    state.cameraPosY += dy;
    state.cameraPosZ += dz;
    applySimulationParams();
    postWorker('render');
    drawSourceOverlay();
    scheduleStatsUpdate();
  }
  if (state.pointerLookActive) {
    state.navLoopHandle = requestAnimationFrame(applyCameraNavigationStep);
  } else {
    state.navLoopHandle = 0;
  }
}

function ensureCameraNavigationLoop() {
  if (!state.pointerLookActive || state.navLoopHandle) return;
  state.navLastAt = 0;
  state.navLoopHandle = requestAnimationFrame(applyCameraNavigationStep);
}

function handlePointerLockChange() {
  state.pointerLookActive = document.pointerLockElement === gpuCanvas;
  if (!state.pointerLookActive) {
    for (const key of Object.keys(state.navKeys)) state.navKeys[key] = false;
    if (state.navLoopHandle) cancelAnimationFrame(state.navLoopHandle);
    state.navLoopHandle = 0;
    state.navLastAt = 0;
  } else {
    ensureCameraNavigationLoop();
  }
  updateCanvasInteractionState();
}

function handlePointerLookMove(event) {
  if (document.pointerLockElement !== gpuCanvas || !isPointerLookAllowed()) return;
  event.preventDefault();
  const sensitivity = DEFAULT_CAMERA_LOOK_SENSITIVITY;
  const azimuth = readNumber(cameraAzimuthLabel, DEFAULT_CAMERA_AZIMUTH) + event.movementX * sensitivity;
  const elevation = clampValue(readNumber(cameraElevationLabel, DEFAULT_CAMERA_ELEVATION) + event.movementY * sensitivity, 5, 89);
  cameraAzimuthLabel.input.value = String(azimuth);
  cameraElevationLabel.input.value = String(elevation);
  applySimulationParams();
  postWorker('render');
  drawSourceOverlay();
  updateStats();
}

function handleNavigationKey(event, isDown) {
  if (document.pointerLockElement !== gpuCanvas) return;
  if (!(event.code in state.navKeys)) return;
  event.preventDefault();
  state.navKeys[event.code] = isDown;
  if (isDown) {
    ensureCameraNavigationLoop();
  }
}

document.addEventListener('pointerlockchange', handlePointerLockChange);
document.addEventListener('pointermove', handlePointerLookMove, { passive: false });
document.addEventListener('keydown', (event) => handleNavigationKey(event, true), { passive: false });
document.addEventListener('keyup', (event) => handleNavigationKey(event, false), { passive: false });
gpuCanvas.addEventListener('wheel', (event) => {
  if (document.pointerLockElement === gpuCanvas) event.preventDefault();
}, { passive: false });

function refreshProcessControlState() {
  const springsEnabled = !!sourceEnabledLabel.input.checked;
  const precipEnabled = !!precipEnabledLabel.input.checked;
  const thermalEnabled = !!thermalEnabledLabel.input.checked;
  const hydraulicErosionEnabled = !!hydraulicErosionEnabledLabel.input.checked;
  const randomMode = getSelectedSpringMode() === 1;
  rainRateLabel.input.disabled = !precipEnabled;
  capacityScaleLabel.input.disabled = !hydraulicErosionEnabled;
  suspensionRateLabel.input.disabled = !hydraulicErosionEnabled;
  depositionRateLabel.input.disabled = !hydraulicErosionEnabled;
  softeningRateLabel.input.disabled = !hydraulicErosionEnabled;
  maxErosionDepthLabel.input.disabled = !hydraulicErosionEnabled;
  thermalRateLabel.input.disabled = !thermalEnabled;
  talusCoeffLabel.input.disabled = !thermalEnabled;
  talusBiasLabel.input.disabled = !thermalEnabled;
  rainDurationLabel.input.disabled = !precipEnabled;
  sourceLayoutLabel.input.disabled = !springsEnabled;
  sourceRadiusLabel.input.disabled = !springsEnabled || !randomMode;
  sourceStrengthLabel.input.disabled = !springsEnabled || !randomMode;
  randomSpringCountLabel.input.disabled = !springsEnabled || !randomMode;
  springSeedLabel.input.disabled = !springsEnabled || !randomMode;
  resetRainTimerButton.disabled = !state.gpuStats?.ready;
  updateCanvasInteractionState();
}

gpuCanvas.addEventListener('contextmenu', (event) => {
  if (paintModeLabel.input.value !== 'none' || document.pointerLockElement === gpuCanvas) event.preventDefault();
});

gpuCanvas.addEventListener('click', (event) => {
  if (!state.gpuStats?.ready || !isPointerLookAllowed()) return;
  event.preventDefault();
  gpuCanvas.focus();
  if (document.pointerLockElement !== gpuCanvas) {
    gpuCanvas.requestPointerLock?.();
  }
});

gpuCanvas.addEventListener('pointerdown', (event) => {
  if ((paintModeLabel.input.value !== 'none' || document.pointerLockElement === gpuCanvas) && state.gpuStats?.ready) {
    event.preventDefault();
  }
  if (paintModeLabel.input.value === 'none' || !state.gpuStats?.ready) return;
  state.isPainting = true;
  gpuCanvas.setPointerCapture?.(event.pointerId);
  void applyBrushFromEvent(event);
});

gpuCanvas.addEventListener('pointermove', (event) => {
  if (!state.isPainting) return;
  event.preventDefault();
  void applyBrushFromEvent(event);
});

gpuCanvas.addEventListener('pointerup', stopPainting);
gpuCanvas.addEventListener('pointerleave', stopPainting);
gpuCanvas.addEventListener('pointercancel', stopPainting);

function setGPUCanvasVisible(visible) {
  gpuCanvas.style.visibility = visible ? 'visible' : 'hidden';
  overlayCanvas.style.visibility = visible ? 'visible' : 'hidden';
  previewCanvas.style.visibility = visible ? 'hidden' : 'visible';
  drawSourceOverlay();
  updateCanvasInteractionState();
}

function syncCanvasSizes() {
  const rect = canvasWrap.getBoundingClientRect();
  const cssWidth = Math.max(1, Math.round(rect.width || 960));
  const cssHeight = Math.max(1, Math.round(rect.height || Math.round(cssWidth * 2 / 3)));
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  const pixelWidth = Math.max(1, Math.round(cssWidth * dpr));
  const pixelHeight = Math.max(1, Math.round(cssHeight * dpr));
  if (previewCanvas.width !== pixelWidth || previewCanvas.height !== pixelHeight) {
    previewCanvas.width = pixelWidth;
    previewCanvas.height = pixelHeight;
  }
  if (overlayCanvas.width !== pixelWidth || overlayCanvas.height !== pixelHeight) {
    overlayCanvas.width = pixelWidth;
    overlayCanvas.height = pixelHeight;
  }
  drawSourceOverlay();
}

async function loadSourceImage(file) {
  debugLog('loadSourceImage begin', file.name);
  stopLoop();
  setGPUCanvasVisible(false);
  syncCanvasSizes();
  state.sourceFile = file;
  state.sourceFileName = file.name;
  state.sourceImageInfo = null;
  state.sourceUploadedToWorker = false;
  state.gpuStats = null;
  state.sourcePoints = [];
  state.lastFrameMs = 0;
  state.simReady = false;
  state.simulationLoading = false;

  resetButton.disabled = false;
  exportDemButton.disabled = true;
  clearSpringsButton.disabled = true;
  updateStatus('PNG selected, initializing GPU…');
  updateStats();

  try {
    await initializeSimulation();
  } catch (error) {
    clearAll();
void ensureWorker().catch((error) => {
  console.error('[WebGPU Erosion UI] worker prewarm failed', error);
  updateStatus(error instanceof Error ? error.message : String(error));
});
    updateStatus(error instanceof Error ? error.message : String(error));
  }
}

async function initializeSimulation() {
  if (!hasAnyDemSourceSelected()) return;
  const requestId = ++state.simulationRequestId;
  stopLoop();
  exportDemButton.disabled = true;
  state.simulationLoading = true;
  state.simReady = false;
  setGPUCanvasVisible(false);
  updateStatus('Loading DEM…');

  try {
    syncCanvasSizes();
    await ensureWorker();
    if (requestId !== state.simulationRequestId) return;
    syncWorkerCanvasSize();
    syncSingleModeMaterialFields();
    applySimulationParams();
    const demSourceMode = getDemSourceMode();
    const payload = {
      demSourceMode,
      tessellation: getSelectedTessellation(),
      options: {
        minHeight: readNumber(minHeightLabel, DEFAULT_MIN_HEIGHT),
        maxHeight: readNumber(maxHeightLabel, DEFAULT_MAX_HEIGHT),
        demSourceMode,
        layerConfig: collectLayerMaterialConfig(),
      },
    };
    if (!state.sourceUploadedToWorker) {
      if (demSourceMode === 'stack4') {
        payload.layerBlobs = state.layerSourceFiles.slice();
      } else {
        payload.blob = state.sourceFile;
      }
    }
    const response = await callWorker('loadDEMImage', payload);
    if (requestId !== state.simulationRequestId) return;
    state.sourceUploadedToWorker = true;
    state.buildCount++;
    state.sourceFileName = getSourceSummaryName();
    state.sourceImageInfo = response.sourceImageInfo ?? state.sourceImageInfo;
    state.gpuStats = response.stats ?? state.gpuStats;
    state.sourcePoints = response.sourcePoints ?? state.sourcePoints;
    state.simReady = !!(response.stats?.ready || response.ready);
    runButton.disabled = false;
    stepButton.disabled = false;
    exportDemButton.disabled = false;
    clearSpringsButton.disabled = false;
    resetButton.disabled = false;
    refreshProcessControlState();
    setGPUCanvasVisible(state.simReady);
    if (state.simReady) {
      await callWorker('render', { waitForCompletion: true });
      if (requestId !== state.simulationRequestId) return;
      updateStatus('DEM loaded');
      void scheduleReadbackStats(true);
    } else {
      updateStatus('DEM uploaded, waiting for first render…');
    }
    updateStats();
  } catch (error) {
    if (requestId !== state.simulationRequestId) return;
    console.error('[WebGPU Erosion UI] initializeSimulation failed', error);
    state.simReady = false;
    setGPUCanvasVisible(false);
    updateStatus(error instanceof Error ? error.message : String(error));
  } finally {
    if (requestId === state.simulationRequestId) {
      state.simulationLoading = false;
      updateStatus();
    }
  }
}


function scheduleStatsUpdate() {
  if (state.statsFrameHandle) return;
  state.statsFrameHandle = requestAnimationFrame(() => {
    state.statsFrameHandle = 0;
    updateStats();
  });
}

function applySimulationParams() {
  const params = collectSimulationParams();
  if (!state.workerReady) return;
  postWorker('setParams', { params });
}

async function refreshWorkerStatusLight() {
  if (!state.workerReady) return;
  try {
    const response = await callWorker('getStatus');
    if (response.stats) state.gpuStats = response.stats;
    if (Array.isArray(response.sourcePoints)) state.sourcePoints = response.sourcePoints;
    if (typeof response.running === 'boolean') state.running = response.running;
    if (Number.isFinite(response.lastFrameMs)) state.lastFrameMs = response.lastFrameMs;
    drawSourceOverlay();
    scheduleStatsUpdate();
  } catch (error) {
    debugLog('refreshWorkerStatusLight failed', error);
  }
}

function shouldShowRandomSpringMarkers() {
  return paintModeLabel.input.value === 'spring_add' || paintModeLabel.input.value === 'spring_erase';
}

function getBrushSettings() {
  return {
    mode: paintModeLabel.input.value,
    radius: Math.max(1, readNumber(paintRadiusLabel, DEFAULT_PAINT_RADIUS)),
    amount: Math.max(0.001, readNumber(paintAmountLabel, DEFAULT_PAINT_AMOUNT)),
    hardness: clampValue(readNumber(paintHardnessLabel, DEFAULT_PAINT_HARDNESS), 0.05, 1.0),
  };
}

function pickGridFromPointer(event) {
  const stats = state.gpuStats;
  if (!stats?.ready || !stats.width || !stats.height) return null;

  const rect = gpuCanvas.getBoundingClientRect();
  const px = (event.clientX - rect.left) / Math.max(rect.width, 1);
  const py = (event.clientY - rect.top) / Math.max(rect.height, 1);
  if (px < 0 || px > 1 || py < 0 || py > 1) return null;

  const basis = getCameraBasis();
  const worldScale = (stats.width > 1 || stats.height > 1) ? 2.0 / Math.max(stats.width - 1, stats.height - 1, 1) : 1.0;
  const view = lookAtMat4(basis.eye, basis.target, [0, 1, 0]);
  const proj = perspectiveMat4(degToRad(50), Math.max(rect.width / Math.max(rect.height, 1), 1e-6), 0.01, 32.0);
  const invViewProj = invertMat4(multiplyMat4(proj, view));
  if (!invViewProj) return null;

  const ndcX = px * 2 - 1;
  const ndcY = 1 - py * 2;
  const near = transformVec4(invViewProj, [ndcX, ndcY, -1, 1]);
  const far = transformVec4(invViewProj, [ndcX, ndcY, 1, 1]);
  const nearPoint = [near[0] / near[3], near[1] / near[3], near[2] / near[3]];
  const farPoint = [far[0] / far[3], far[1] / far[3], far[2] / far[3]];
  const dir = normalizeVec3(subtractVec3(farPoint, nearPoint));

  let worldX;
  let worldZ;
  const planeY = 0.0;
  if (Math.abs(dir[1]) > 1e-5) {
    const t = (planeY - nearPoint[1]) / dir[1];
    if (t > 0) {
      worldX = nearPoint[0] + dir[0] * t;
      worldZ = nearPoint[2] + dir[2] * t;
    }
  }
  if (!Number.isFinite(worldX) || !Number.isFinite(worldZ)) {
    const gx = px * Math.max(stats.width - 1, 0);
    const gy = py * Math.max(stats.height - 1, 0);
    return { x: clampValue(gx, 0, Math.max(stats.width - 1, 0)), y: clampValue(gy, 0, Math.max(stats.height - 1, 0)) };
  }

  const halfW = (stats.width - 1) * 0.5;
  const halfH = (stats.height - 1) * 0.5;
  const gx = worldX / Math.max(worldScale, 1e-6) + halfW;
  const gy = halfH - worldZ / Math.max(worldScale, 1e-6);
  return {
    x: clampValue(gx, 0, Math.max(stats.width - 1, 0)),
    y: clampValue(gy, 0, Math.max(stats.height - 1, 0)),
  };
}

async function applyBrushFromEvent(event) {
  const brush = getBrushSettings();
  if (brush.mode === 'none' || !state.workerReady || !state.gpuStats?.ready) return;
  const now = performance.now();
  if (now - state.lastPaintAt < 20) return;
  state.lastPaintAt = now;
  const pos = pickGridFromPointer(event);
  if (!pos) return;
  if (state.running) stopLoop();

  if (brush.mode === 'raise' || brush.mode === 'lower') {
    const response = await callWorker('paintTerrainBrush', {
      brush: {
        x: pos.x,
        y: pos.y,
        radius: brush.radius,
        amount: brush.amount,
        hardness: brush.hardness,
        subtract: brush.mode === 'lower',
      },
    });
    if (response.stats) state.gpuStats = response.stats;
    if (Array.isArray(response.sourcePoints)) state.sourcePoints = response.sourcePoints;
    updateStatus(`${brush.mode === 'raise' ? 'Raised' : 'Lowered'} terrain at ${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}`);
  } else {
    const response = await callWorker('paintSpringBrush', {
      brush: {
        x: pos.x,
        y: pos.y,
        radius: brush.radius,
        strength: brush.amount,
        hardness: brush.hardness,
        erase: brush.mode === 'spring_erase',
      },
    });
    if (response.stats) state.gpuStats = response.stats;
    if (Array.isArray(response.sourcePoints)) state.sourcePoints = response.sourcePoints;
    updateStatus(`${brush.mode === 'spring_erase' ? 'Erased' : 'Painted'} spring at ${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}`);
  }
  updateStats();
}

function stopPainting() {
  if (!state.isPainting) return;
  state.isPainting = false;
  void scheduleReadbackStats(true);
}

function applyPreset(name) {
  const presets = {
    paper_balanced: {
      iterationsPerFrame: 5, stepIterations: 128,
      timeStep: 0.02, rainRate: 0.001, evaporationRate: 0.015, pipeArea: 20.0, gravity: 9.81,
      capacityScale: 0.82, suspensionRate: 0.32, depositionRate: 1.22, softeningRate: 2.8,
      maxErosionDepth: 0.09, thermalRate: 0.24, talusCoeff: 0.92, talusBias: 0.12,
      sourceStrength: 0.06, sourceRadius: 4.0, sourceEnabled: false, sourceLayoutMode: 0, randomSpringCount: 1,
      rainDuration: 0.0, pulse2Duration: 0.0, metersPerPixel: 100, waterHeightScale: 0.06, edgeWaterFloor: 0.0, renderMode: 0,
    },
    river_cut: {
      iterationsPerFrame: 5, stepIterations: 160,
      timeStep: 0.02, rainRate: 0.0015, evaporationRate: 0.012, pipeArea: 22.0, gravity: 9.81,
      capacityScale: 0.95, suspensionRate: 0.48, depositionRate: 1.02, softeningRate: 2.8,
      maxErosionDepth: 0.08, thermalRate: 0.22, talusCoeff: 0.88, talusBias: 0.11,
      sourceStrength: 0.09, sourceRadius: 4.0, sourceEnabled: true, sourceLayoutMode: 1, randomSpringCount: 4,
      rainDuration: 0.0, pulse2Duration: 0.0, metersPerPixel: 100, waterHeightScale: 0.06, edgeWaterFloor: 0.0, renderMode: 0,
    },
    delta_depositor: {
      iterationsPerFrame: 5, stepIterations: 192,
      timeStep: 0.02, rainRate: 0.016, evaporationRate: 0.004, pipeArea: 30.0, gravity: 9.81,
      capacityScale: 0.38, suspensionRate: 0.12, depositionRate: 2.2, softeningRate: 1.4,
      maxErosionDepth: 0.04, thermalRate: 0.08, talusCoeff: 0.82, talusBias: 0.10,
      sourceStrength: 0.0, sourceRadius: 5.0, sourceEnabled: false, sourceLayoutMode: 0, randomSpringCount: 1,
      rainDuration: 85.0, pulse2Duration: 0.0, metersPerPixel: 100, waterHeightScale: 0.07, renderMode: 0,
    },
    meander_builder: {
      iterationsPerFrame: 5, stepIterations: 160,
      timeStep: 0.02, rainRate: 0.003, evaporationRate: 0.004, pipeArea: 32.0, gravity: 9.81,
      capacityScale: 0.30, suspensionRate: 0.10, depositionRate: 2.45, softeningRate: 1.1,
      maxErosionDepth: 0.035, thermalRate: 0.18, talusCoeff: 0.86, talusBias: 0.10,
      sourceStrength: 0.11, sourceRadius: 5.0, sourceEnabled: true, sourceLayoutMode: 1, randomSpringCount: 3,
      rainDuration: 0.0, pulse2Duration: 0.0, metersPerPixel: 100, waterHeightScale: 0.07, renderMode: 0,
    },
    thermal_heavy: {
      iterationsPerFrame: 5, stepIterations: 128,
      timeStep: 0.02, rainRate: 0.0, evaporationRate: 0.008, pipeArea: 8.0, gravity: 9.81,
      capacityScale: 0.1, suspensionRate: 0.05, depositionRate: 0.85, softeningRate: 1.0,
      maxErosionDepth: 0.06, thermalRate: 1.10, talusCoeff: 0.55, talusBias: 0.05,
      sourceStrength: 0.0, sourceRadius: 4.0, sourceEnabled: false, sourceLayoutMode: 0, randomSpringCount: 1,
      rainDuration: 0.0, pulse2Duration: 0.0, metersPerPixel: 100, waterHeightScale: 0.06, renderMode: 5,
    },
    gentle_weathering: {
      iterationsPerFrame: 5, stepIterations: 96,
      timeStep: 0.02, rainRate: 0.004, evaporationRate: 0.015, pipeArea: 12.0, gravity: 9.81,
      capacityScale: 0.45, suspensionRate: 0.18, depositionRate: 1.0, softeningRate: 2.0,
      maxErosionDepth: 0.10, thermalRate: 0.12, talusCoeff: 0.8, talusBias: 0.1,
      sourceStrength: 0.0, sourceRadius: 4.0, sourceEnabled: false, sourceLayoutMode: 0, randomSpringCount: 1,
      rainDuration: 0.0, pulse2Duration: 0.0, metersPerPixel: 100, waterHeightScale: 0.06, edgeWaterFloor: 0.0, renderMode: 0,
    },
    rapid_incision: {
      iterationsPerFrame: 5, stepIterations: 192,
      timeStep: 0.022, rainRate: 0.018, evaporationRate: 0.010, pipeArea: 26.0, gravity: 9.81,
      capacityScale: 1.4, suspensionRate: 0.9, depositionRate: 0.75, softeningRate: 6.0,
      maxErosionDepth: 0.10, thermalRate: 0.12, talusCoeff: 0.72, talusBias: 0.08,
      sourceStrength: 0.0, sourceRadius: 4.0, sourceEnabled: false, sourceLayoutMode: 0, randomSpringCount: 1,
      rainDuration: 0.0, pulse2Duration: 0.0, metersPerPixel: 100, waterHeightScale: 0.06, edgeWaterFloor: 0.0, renderMode: 0,
    },
    flash_flood: {
      iterationsPerFrame: 5, stepIterations: 192,
      timeStep: 0.02, rainRate: 0.03, evaporationRate: 0.008, pipeArea: 24.0, gravity: 9.81,
      capacityScale: 1.3, suspensionRate: 0.85, depositionRate: 0.8, softeningRate: 5.0,
      maxErosionDepth: 0.08, thermalRate: 0.10, talusCoeff: 0.75, talusBias: 0.08,
      sourceStrength: 0.0, sourceRadius: 4.0, sourceEnabled: false, sourceLayoutMode: 0, randomSpringCount: 1,
      rainDuration: 50.0, pulse2Duration: 0.0, metersPerPixel: 100, waterHeightScale: 0.06, renderMode: 0,
    },
    badlands: {
      iterationsPerFrame: 5, stepIterations: 160,
      timeStep: 0.02, rainRate: 0.01, evaporationRate: 0.012, pipeArea: 18.0, gravity: 9.81,
      capacityScale: 1.0, suspensionRate: 0.55, depositionRate: 0.95, softeningRate: 4.0,
      maxErosionDepth: 0.09, thermalRate: 0.22, talusCoeff: 0.68, talusBias: 0.08,
      sourceStrength: 0.0, sourceRadius: 4.0, sourceEnabled: false, sourceLayoutMode: 0, randomSpringCount: 1,
      rainDuration: 25.0, pulse2Duration: 0.0, metersPerPixel: 100, waterHeightScale: 0.06, renderMode: 0,
    },
    canyon_carver: {
      iterationsPerFrame: 5, stepIterations: 224,
      timeStep: 0.022, rainRate: 0.014, evaporationRate: 0.010, pipeArea: 28.0, gravity: 9.81,
      capacityScale: 1.55, suspensionRate: 1.0, depositionRate: 0.7, softeningRate: 6.0,
      maxErosionDepth: 0.08, thermalRate: 0.16, talusCoeff: 0.72, talusBias: 0.08,
      sourceStrength: 0.08, sourceRadius: 3.0, sourceEnabled: true, sourceLayoutMode: 1, randomSpringCount: 4,
      rainDuration: 0.0, pulse2Duration: 0.0, metersPerPixel: 100, waterHeightScale: 0.06, edgeWaterFloor: 0.0, renderMode: 0,
    },
  };
  const preset = presets[name] || presets.paper_balanced;
  iterationsPerFrameLabel.input.value = String(preset.iterationsPerFrame ?? DEFAULT_ITERATIONS_PER_FRAME);
  stepIterationsLabel.input.value = String(preset.stepIterations ?? DEFAULT_STEP_ITERATIONS);
  timeStepLabel.input.value = String(preset.timeStep);
  rainRateLabel.input.value = String(preset.rainRate);
  evaporationRateLabel.input.value = String(preset.evaporationRate);
  pipeAreaLabel.input.value = String(preset.pipeArea);
  gravityLabel.input.value = String(preset.gravity);
  capacityScaleLabel.input.value = String(preset.capacityScale);
  suspensionRateLabel.input.value = String(preset.suspensionRate);
  depositionRateLabel.input.value = String(preset.depositionRate);
  softeningRateLabel.input.value = String(preset.softeningRate);
  maxErosionDepthLabel.input.value = String(preset.maxErosionDepth);
  thermalRateLabel.input.value = String(preset.thermalRate);
  talusCoeffLabel.input.value = String(preset.talusCoeff);
  talusBiasLabel.input.value = String(preset.talusBias);
  sourceStrengthLabel.input.value = String(preset.sourceStrength);
  rainDurationLabel.input.value = String(preset.rainDuration ?? DEFAULT_RAIN_DURATION);
  pulse2DurationLabel.input.value = String(preset.pulse2Duration ?? DEFAULT_PULSE2_DURATION);
  sourceRadiusLabel.input.value = String(preset.sourceRadius);
  precipEnabledLabel.input.checked = (preset.rainRate ?? 0) > 0;
  thermalEnabledLabel.input.checked = (preset.thermalRate ?? 0) > 0;
  hydraulicErosionEnabledLabel.input.checked = true;
  sourceEnabledLabel.input.checked = !!preset.sourceEnabled;
  sourceLayoutLabel.input.value = String(preset.sourceLayoutMode ?? DEFAULT_SOURCE_LAYOUT_MODE);
  randomSpringCountLabel.input.value = String(preset.randomSpringCount ?? DEFAULT_RANDOM_SPRING_COUNT);
  metersPerPixelLabel.input.value = String(preset.metersPerPixel ?? DEFAULT_METERS_PER_PIXEL);
  waterHeightScaleLabel.input.value = String(preset.waterHeightScale ?? DEFAULT_WATER_HEIGHT_SCALE);
  edgeWaterFloorLabel.input.value = String(preset.edgeWaterFloor ?? DEFAULT_EDGE_WATER_FLOOR);
  renderModeLabel.input.value = String(preset.renderMode);
  refreshProcessControlState();
  applySimulationParams();
  postWorker('render');
  void refreshWorkerStatusLight();
  updateStats();
}

async function scheduleReadbackStats(force = false) {
  if (!state.workerReady || state.readbackPending) return;
  const now = performance.now();
  if (!force && now - state.lastReadbackAt < 500) return;
  state.readbackPending = true;
  try {
    const response = await callWorker('readbackStats');
    state.gpuStats = response.stats ?? state.gpuStats;
    state.sourcePoints = response.sourcePoints ?? state.sourcePoints;
    state.sourceImageInfo = response.sourceImageInfo ?? state.sourceImageInfo;
    if (Number.isFinite(response.lastFrameMs)) state.lastFrameMs = response.lastFrameMs;
    state.lastReadbackAt = performance.now();
  } catch (error) {
    debugLog('readback failed', error);
  } finally {
    state.readbackPending = false;
  }
}

function startLoop() {
  if (!state.workerReady || state.running) return;
  state.running = true;
  runButton.textContent = 'Pause';
  applySimulationParams();
  postWorker('startLoop', {
    iterationsPerFrame: Math.max(1, Math.floor(readNumber(iterationsPerFrameLabel, DEFAULT_ITERATIONS_PER_FRAME))),
  });
  void scheduleReadbackStats(true);
  updateStatus('Running');
}

function stopLoop() {
  if (!state.running) return;
  state.running = false;
  runButton.textContent = 'Run';
  postWorker('stopLoop');
  updateStatus('Paused');
}

function clearAll() {
  stopLoop();
  document.exitPointerLock?.();
  postWorker('clear');
  state.sourceImageInfo = null;
  state.sourceFile = null;
  state.layerSourceFiles = [null, null, null, null];
  state.sourceUploadedToWorker = false;
  state.buildCount = 0;
  state.lastFrameMs = 0;
  state.readbackPending = false;
  state.lastReadbackAt = 0;
  state.gpuStats = null;
  state.sourcePoints = [];
  state.sourceFileName = '';
  state.simReady = false;
  state.simulationLoading = false;
  state.workerWarming = false;
  state.cameraPosX = DEFAULT_CAMERA_POS_X;
  state.cameraPosY = DEFAULT_CAMERA_POS_Y;
  state.cameraPosZ = DEFAULT_CAMERA_POS_Z;
  fileInput.value = '';
  for (const layer of layerMaterialControls) {
    layer.filePickerLabel.input.value = '';
  }
  runButton.disabled = true;
  stepButton.disabled = true;
  exportDemButton.disabled = true;
  clearSpringsButton.disabled = true;
  resetButton.disabled = true;
  updateStatus();
  updateStats();
  drawSourceOverlay();
  const ctx = previewCanvas.getContext('2d');
  if (ctx) {
    ctx.fillStyle = '#0d0d0d';
    ctx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.font = '16px system-ui, sans-serif';
    ctx.fillText('Load a DEM PNG, packed RGBA DEM, or 4-layer grayscale stack to initialize the worker-owned WebGPU erosion sim.', 24, 36);
  }
}

function drawSourceOverlay() {
  const ctx = overlayCanvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (overlayCanvas.style.visibility === 'hidden') return;
  const stats = state.gpuStats;
  const points = state.sourcePoints || [];
  if (!stats?.ready || !stats.width || !stats.height || points.length === 0) return;

  const basis = getCameraBasis();
  const view = lookAtMat4(basis.eye, basis.target, [0, 1, 0]);
  const proj = perspectiveMat4(degToRad(50), Math.max(overlayCanvas.width / Math.max(overlayCanvas.height, 1), 1e-6), 0.01, 32.0);
  const viewProj = multiplyMat4(proj, view);
  const worldScale = (stats.width > 1 || stats.height > 1) ? 2.0 / Math.max(stats.width - 1, stats.height - 1, 1) : 1.0;
  const halfW = (stats.width - 1) * 0.5;
  const halfH = (stats.height - 1) * 0.5;
  const paintedPixelRadius = Math.max(2.0, Math.min(6.0, Math.min(overlayCanvas.width, overlayCanvas.height) / Math.max(stats.width, stats.height, 1) * 0.85));

  for (const point of points) {
    const worldX = (point.x - halfW) * worldScale;
    const worldZ = (halfH - point.y) * worldScale;
    const worldY = 0.0;
    const clip = transformVec4(viewProj, [worldX, worldY, worldZ, 1]);
    if (!clip[3] || clip[3] <= 0) continue;
    const ndcX = clip[0] / clip[3];
    const ndcY = clip[1] / clip[3];
    if (Math.abs(ndcX) > 1.2 || Math.abs(ndcY) > 1.2) continue;
    const sx = (ndcX * 0.5 + 0.5) * overlayCanvas.width;
    const sy = (1 - (ndcY * 0.5 + 0.5)) * overlayCanvas.height;
    const painted = !!point.painted;
    const active = point.active !== false;

    if (!painted && !shouldShowRandomSpringMarkers()) {
      continue;
    }

    if (painted) {
      const strength = clampValue(Number(point.strength) || 0, 0, 1);
      const radius = paintedPixelRadius * (0.8 + 0.9 * strength);
      ctx.fillStyle = active ? 'rgba(176, 96, 255, 0.98)' : 'rgba(120, 96, 150, 0.85)';
      ctx.fillRect(Math.round(sx - radius), Math.round(sy - radius), Math.max(1, Math.round(radius * 2)), Math.max(1, Math.round(radius * 2)));
      continue;
    }

    const radius = Math.max(5, (point.radius || 2) * Math.min(overlayCanvas.width, overlayCanvas.height) / Math.max(stats.width, stats.height, 1) * 1.2);
    ctx.beginPath();
    ctx.arc(sx, sy, radius, 0, Math.PI * 2);
    ctx.fillStyle = active ? 'rgba(60, 220, 255, 0.30)' : 'rgba(90, 110, 120, 0.18)';
    ctx.fill();
    ctx.lineWidth = 2.0;
    ctx.strokeStyle = active ? 'rgba(138, 240, 255, 0.95)' : 'rgba(140, 165, 175, 0.55)';
    ctx.stroke();
  }
}


function updateStatus(extra = '') {
  const parts = [];
  if (extra) parts.push(extra);
  parts.push(state.sourceFile ? `image: ${state.sourceFileName || 'loaded'}` : 'image: none');
  if (!state.workerReady) {
    parts.push('webgpu: idle');
  } else if (state.workerWarming) {
    parts.push('webgpu: warming');
  } else {
    parts.push('webgpu: worker ready');
  }
  if (state.simulationLoading) {
    parts.push('dem: loading');
  } else if (state.simReady) {
    parts.push('dem: ready');
  } else if (state.sourceFile || state.layerSourceFiles.some(Boolean)) {
    parts.push('dem: not rendered');
  }
  if (state.running) parts.push('sim: running');
  statusLine.textContent = parts.join('  |  ');
}

function updateStats() {
  if (!hasAnyDemSourceSelected() && !state.sourceImageInfo) {
    statsBlock.textContent = 'Load a DEM PNG or layered DEM stack to begin.';
    drawSourceOverlay();
    return;
  }

  const width = state.sourceImageInfo?.width || state.gpuStats?.width || 0;
  const height = state.sourceImageInfo?.height || state.gpuStats?.height || 0;
  const simWidth = state.gpuStats?.width || width;
  const simHeight = state.gpuStats?.height || height;
  const stats = state.gpuStats ?? { width: 0, height: 0, cellCount: 0, ready: false, iterationCount: 0 };
  const fps = state.lastFrameMs > 0 ? (1000 / Math.max(state.lastFrameMs, 1e-6)) : 0;
  const lines = [
    `fps: ${Number.isFinite(fps) && fps > 0 ? fps.toFixed(1) : 'n/a'}`,
    `last frame: ${state.lastFrameMs.toFixed(3)} ms`,
    `erosion cpu: ${Number.isFinite(stats.lastCpuErosionPassMs) ? stats.lastCpuErosionPassMs.toFixed(3) : Number.isFinite(stats.lastErosionPassMs) ? stats.lastErosionPassMs.toFixed(3) : 'n/a'} ms`,
    `erosion gpu: ${Number.isFinite(stats.lastGpuErosionPassMs) ? stats.lastGpuErosionPassMs.toFixed(3) : 'n/a'} ms`,
    `render cpu: ${Number.isFinite(stats.lastCpuRenderPassMs) ? stats.lastCpuRenderPassMs.toFixed(3) : Number.isFinite(stats.lastRenderPassMs) ? stats.lastRenderPassMs.toFixed(3) : 'n/a'} ms`,
    `render gpu: ${Number.isFinite(stats.lastGpuRenderPassMs) ? stats.lastGpuRenderPassMs.toFixed(3) : 'n/a'} ms`,
    `erosion cpu avg: ${Number.isFinite(stats.avgCpuErosionPassMs) ? stats.avgCpuErosionPassMs.toFixed(3) : Number.isFinite(stats.avgErosionPassMs) ? stats.avgErosionPassMs.toFixed(3) : 'n/a'} ms`,
    `erosion gpu avg: ${Number.isFinite(stats.avgGpuErosionPassMs) ? stats.avgGpuErosionPassMs.toFixed(3) : 'n/a'} ms`,
    `render cpu avg: ${Number.isFinite(stats.avgCpuRenderPassMs) ? stats.avgCpuRenderPassMs.toFixed(3) : Number.isFinite(stats.avgRenderPassMs) ? stats.avgRenderPassMs.toFixed(3) : 'n/a'} ms`,
    `render gpu avg: ${Number.isFinite(stats.avgGpuRenderPassMs) ? stats.avgGpuRenderPassMs.toFixed(3) : 'n/a'} ms`,
    '',
    `file: ${state.sourceFileName || '(loaded image)'}`,
    `DEM source: ${getDemSourceMode()}`,
    `layered DEM active: ${stats.layeredDemEnabled ? 'yes' : 'no'}`,
    `active layers: ${Number.isFinite(stats.activeLayerCount) ? stats.activeLayerCount : layerMaterialControls.filter((layer) => layer.enableLabel.input.checked).length}`,
    `image size: ${width} x ${height}`,
    `sim tess: ${getSelectedTessellation()}x`,
    `sim grid: ${stats.width || simWidth} x ${stats.height || simHeight}`,
    `cells: ${stats.cellCount || (simWidth * simHeight)}`,
    `gpu ready: ${stats.ready ? 'yes' : 'no'}`,
    `build count: ${state.buildCount}`,
    `iterations: ${stats.iterationCount || 0}`,
    `sim time: ${Number.isFinite(stats.simTime) ? stats.simTime.toFixed(2) : '0.00'} s`,
    `rain active: ${stats.rainActive ? 'yes' : 'no'}`,
    `precipitation: ${precipEnabledLabel.input.checked ? 'on' : 'off'}`,
    `spring toggle: ${sourceEnabledLabel.input.checked ? 'on' : 'off'}`,
    `thermal erosion: ${thermalEnabledLabel.input.checked ? 'on' : 'off'}`,
    `spring mode: ${getSelectedSpringModeName()}`,
    `painted springs: ${(state.sourcePoints || []).filter((point) => point.painted).length}`,
    `random spring count: ${Math.max(1, Math.floor(readNumber(randomSpringCountLabel, DEFAULT_RANDOM_SPRING_COUNT)))}`,
    `spring seed: ${Math.floor(readNumber(springSeedLabel, DEFAULT_SPRING_SEED))}`,
    `spring centers: ${(state.sourcePoints || []).slice(0, 6).map((point) => `${point.painted ? 'p:' : 'r:'}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(' | ') || 'n/a'}`,
    `meters / pixel: ${readNumber(metersPerPixelLabel, DEFAULT_METERS_PER_PIXEL).toFixed(0)}`,
    `running: ${state.running ? 'yes' : 'no'}`,
    `pointer look: ${state.pointerLookActive ? 'locked' : 'off'}`,
    `camera pos: ${state.cameraPosX.toFixed(2)}, ${state.cameraPosY.toFixed(2)}, ${state.cameraPosZ.toFixed(2)}`,
    `paint mode: ${paintModeLabel.input.options[paintModeLabel.input.selectedIndex]?.textContent || 'off'}`,
    `brush radius: ${readNumber(paintRadiusLabel, DEFAULT_PAINT_RADIUS).toFixed(1)}`,
    `brush amount: ${readNumber(paintAmountLabel, DEFAULT_PAINT_AMOUNT).toFixed(3)}`,
    `brush hardness: ${readNumber(paintHardnessLabel, DEFAULT_PAINT_HARDNESS).toFixed(2)}`,
    `readback pending: ${state.readbackPending ? 'yes' : 'no'}`,
    '',
    `terrain range: ${formatRange(stats.terrainRange)}`,
    `water range: ${formatRange(stats.waterRange)}`,
    `sediment range: ${formatRange(stats.sedimentRange)}`,
    `history range: ${formatRange(stats.historyRange)}`,
    `total water: ${Number.isFinite(stats.totalWater) ? stats.totalWater.toFixed(4) : 'n/a'}`,
    `total sediment: ${Number.isFinite(stats.totalSediment) ? stats.totalSediment.toFixed(4) : 'n/a'}`,
    `avg erodability: ${Number.isFinite(stats.averageHardness) ? stats.averageHardness.toFixed(4) : 'n/a'}`,
    '',
    `Δt: ${readNumber(timeStepLabel, DEFAULT_TIME_STEP).toFixed(3)}`,
    `Kr rain: ${readNumber(rainRateLabel, DEFAULT_RAIN_RATE).toFixed(4)}`,
    `Ke evap: ${readNumber(evaporationRateLabel, DEFAULT_EVAPORATION_RATE).toFixed(3)}`,
    `A pipe: ${readNumber(pipeAreaLabel, DEFAULT_PIPE_AREA).toFixed(2)}`,
    `hydraulic pipes: ${hydraulic8WayLabel.input.checked ? '8-way' : '4-way'}`,
    `g gravity: ${readNumber(gravityLabel, DEFAULT_GRAVITY).toFixed(2)}`,
    `Kc capacity: ${readNumber(capacityScaleLabel, DEFAULT_CAPACITY_SCALE).toFixed(3)}`,
    `Ks suspend: ${readNumber(suspensionRateLabel, DEFAULT_SUSPENSION_RATE).toFixed(3)}`,
    `Kd deposit: ${readNumber(depositionRateLabel, DEFAULT_DEPOSITION_RATE).toFixed(3)}`,
    `Kh soften: ${readNumber(softeningRateLabel, DEFAULT_SOFTENING_RATE).toFixed(3)}`,
    `depth cap: ${readNumber(maxErosionDepthLabel, DEFAULT_MAX_EROSION_DEPTH).toFixed(3)}`,
    `Kt thermal: ${readNumber(thermalRateLabel, DEFAULT_THERMAL_RATE).toFixed(3)}`,
    `Ka talus: ${readNumber(talusCoeffLabel, DEFAULT_TALUS_COEFF).toFixed(3)}`,
    `Ki talus bias: ${readNumber(talusBiasLabel, DEFAULT_TALUS_BIAS).toFixed(3)}`,
    `render h: ${readNumber(renderHeightScaleLabel, DEFAULT_RENDER_HEIGHT_SCALE).toFixed(3)}`,
    `water h: ${readNumber(waterHeightScaleLabel, DEFAULT_WATER_HEIGHT_SCALE).toFixed(3)}`,
    `water α: ${readNumber(waterOpacityLabel, DEFAULT_WATER_OPACITY).toFixed(3)}`,
    `sediment tint: ${readNumber(sedimentTintLabel, DEFAULT_SEDIMENT_TINT).toFixed(3)}`,
    `springs: ${sourceEnabledLabel.input.checked ? 'on' : 'off'}`,
    `rain duration s: ${readNumber(rainDurationLabel, DEFAULT_RAIN_DURATION).toFixed(2)}`,
    `pulse 2 s: ${readNumber(pulse2DurationLabel, DEFAULT_PULSE2_DURATION).toFixed(2)}`,
    `source center: ${readNumber(sourceXLabel, DEFAULT_SOURCE_X).toFixed(0)}%, ${readNumber(sourceYLabel, DEFAULT_SOURCE_Y).toFixed(0)}%`,
    `source radius: ${readNumber(sourceRadiusLabel, DEFAULT_SOURCE_RADIUS).toFixed(1)}`,
    `source strength: ${readNumber(sourceStrengthLabel, DEFAULT_SOURCE_STRENGTH).toFixed(3)}`,
    `view mode: ${renderModeLabel.input.options[renderModeLabel.input.selectedIndex]?.textContent || 'shaded'}`,
    `iters/frame: ${Math.max(1, Math.floor(readNumber(iterationsPerFrameLabel, DEFAULT_ITERATIONS_PER_FRAME)))}`,
    `step iters: ${Math.max(1, Math.floor(readNumber(stepIterationsLabel, DEFAULT_STEP_ITERATIONS)))}`,
  ];

  statsBlock.textContent = lines.join('\n');
  updateStatus();
  drawSourceOverlay();
}

clearAll();
