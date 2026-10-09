import GpuStatistics from './GpuStatistics.js';
import GpuTiming from './GpuTiming.js';

const STATE_FLOATS_PER_CELL = 8;
const VEC4_BYTES_PER_CELL = 16;
const WORKGROUP_SIZE_X = 8;
const WORKGROUP_SIZE_Y = 8;

function clamp(value, min, max) {
  return value < min ? min : value > max ? max : value;
}

function alignTo(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

function sourceHash(seed) {
  let h = seed >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d);
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b);
  return (h ^ (h >>> 16)) >>> 0;
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

export default class WebGPUTerrainErosion {
  constructor(options = {}) {
    this.canvas = options.canvas ?? null;
    this.adapter = null;
    this.device = null;
    this.context = null;
    this.presentationFormat = null;

    this.width = 0;
    this.height = 0;
    this.cellCount = 0;
    this.stateByteLength = 0;
    this.vec4ByteLength = 0;
    this.iterationCount = 0;
    this.simulationTime = 0;
    this.allocatedStateByteLength = 0;
    this.allocatedVec4ByteLength = 0;
    this.dispatchX = 0;
    this.dispatchY = 0;
    this.drawCount = 0;

    this.stateBuffers = [];
    this.fluxBuffer = null;
    this.velocityBuffer = null;
    this.thermalPipeBufferA = null;
    this.thermalPipeBufferB = null;
    this.readbackBuffer = null;
    this.customSpringBuffer = null;
    this.layerTopBuffer = null;
    this.layerSurfaceValues = new Float32Array(0);
    this.layerMaterialValues = new Float32Array(16);
    this.layerMaterialBuffer = null;
    this.paintedSpringMap = new Float32Array(0);
    this.sourceFieldDirty = true;
    this.sourceFieldBuildCount = 0;
    this.allocatedPaintedSpringByteLength = 0;
    this.allocatedReadbackByteLength = 0;
    this.renderUniformValues = new Float32Array(40);
    this.renderUniformBufferTerrain = null;
    this.renderUniformBufferWater = null;
    this.depthTexture = null;
    this.depthTextureView = null;
    this.depthTextureSize = { width: 0, height: 0 };
    this.sceneColorTexture = null;
    this.sceneColorView = null;
    this.sceneColorSize = { width: 0, height: 0 };
    this.sceneSampler = null;
    this.dummySceneTexture = null;
    this.dummySceneView = null;

    this.paramValues = new Float32Array(52);
    this.paramBits = new Uint32Array(this.paramValues.buffer);
    this.paramBuffer = null;
    this.stepParamBuffer = null;
    this.stepParamStride = 256;
    this.stepParamCapacity = 0;
    this.stepParamValuesCpu = null;
    this.stepParamValuesCpuCapacity = 0;
    this.stepParamTemplateDirty = true;
    this.stepParamPreparedIterations = 0;
    this.computeDynamicOffset = new Uint32Array(1);

    this.terrainRenderBundle = null;
    this.finalRenderBundle = null;
    this.terrainRenderBundleList = [];
    this.finalRenderBundleList = [];

    this.computeBindGroupLayout = null;
    this.renderBindGroupLayout = null;
    this.compositeBindGroupLayout = null;
    this.computeBindGroup01 = null;
    this.computeBindGroup10 = null;
    this.renderBindGroupTerrain = null;
    this.renderBindGroupWater = null;
    this.compositeBindGroup = null;

    this.fluxPipeline = null;
    this.sourcePipeline = null;
    this.flowPipeline = null;
    this.erosionPipeline = null;
    this.thermalOutflowPipeline = null;
    this.channelPipeline = null;
    this.channelBendPipeline = null;
    this.renderPipeline = null;
    this.waterRenderPipeline = null;
    this.compositePipeline = null;

    this.initialized = false;
    this.ready = false;
    this.latestReadbackStats = null;
    this.gpuStatistics = null;
    this.lastErosionPassMs = 0;
    this.lastRenderPassMs = 0;
    this.avgErosionPassMs = 0;
    this.avgRenderPassMs = 0;
    this.lastCpuErosionPassMs = 0;
    this.lastCpuRenderPassMs = 0;
    this.avgCpuErosionPassMs = 0;
    this.avgCpuRenderPassMs = 0;
    this.lastGpuErosionPassMs = 0;
    this.lastGpuRenderPassMs = 0;
    this.avgGpuErosionPassMs = 0;
    this.avgGpuRenderPassMs = 0;
    this.layeredDemEnabled = false;
    this.layerConfigMode = 'single';

    this.simulationParams = {
      cellSize: 1,
      timeStep: 0.02,
      rainRate: 0.001,
      evaporationRate: 0.025,
      pipeArea: 16.0,
      gravity: 9.81,
      capacityScale: 1.2,
      suspensionRate: 0.35,
      depositionRate: 1.4,
      softeningRate: 1.2,
      maxErosionDepth: 0.04,
      thermalRate: 0.25,
      talusSlopeCoeff: 0.012,
      talusSlopeBias: 0.002,
      renderHeightScale: 0.08,
      waterOpacity: 0.28,
      sedimentTint: 0.35,
      hardnessBase: 0.24,
      hardnessVariation: 0.12,
      sourceCenterX: 0.5,
      sourceCenterY: 0.5,
      sourceRadius: 4.0,
      sourceStrength: 0.06,
      sourceEnabled: false,
      rainDuration: 0.0,
      pulse2Duration: 0.0,
      sourceLayoutMode: 0,
      randomSpringCount: 4,
      sourceSeed: 1,
      metersPerPixel: 100,
      sourceTimeOffset: 0.0,
      historyDecay: 0.9997,
      edgeDrainStrength: 0.08,
      edgeWaterFloor: 0.0,
      renderMode: 8,
      cameraAzimuthDeg: 45,
      cameraElevationDeg: 42,
      cameraDistance: 2.9,
      cameraPosX: 1.5238999619464006,
      cameraPosY: 1.9404787584406888,
      cameraPosZ: 1.5238999619464006,
      waterHeightScale: 0.12,
      thermalVisualizationScale: 160.0,
      hydraulicErosionEnabled: true,
      closedBoundaries: false,
      renderGridStep: 0,
      flowDrag: 2.0,
      maxOutflowFraction: 0.5,
      bankErosionRate: 0,
      sourceSedimentConcentration: 0,
      bankBendBias: 24,
      bendSedimentRate: 0,
      riverBendLength: 0,
      openBoundaryEdges: 15,
      straightBankErosionFactor: 0.08,
    };

    this.setSimulationParams(options.simulationParams ?? {});
  }

  async initialize(canvas = this.canvas) {
    if (!canvas) {
      throw new Error('A canvas is required to initialize WebGPU erosion.');
    }
    this.canvas = canvas;
    this.#syncCanvasPixelSize();

    if (!navigator.gpu) {
      throw new Error('WebGPU is not available in this browser.');
    }

    this.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!this.adapter) {
      throw new Error('Failed to acquire a WebGPU adapter.');
    }

    this.device = await this.adapter.requestDevice({
      requiredFeatures: this.adapter.features.has('timestamp-query') ? ['timestamp-query'] : [],
    });
    this.stepParamStride = Math.max(256, this.device.limits?.minUniformBufferOffsetAlignment || 256);
    this.context = canvas.getContext('webgpu');
    if (!this.context) {
      throw new Error('Failed to acquire a WebGPU canvas context.');
    }

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: 'opaque',
    });

    this.paramBuffer = this.device.createBuffer({
      size: alignTo(this.paramValues.byteLength, 16),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'paper-erosion-params',
    });

    this.renderUniformBufferTerrain = this.device.createBuffer({
      size: alignTo(this.renderUniformValues.byteLength, 16),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'paper-render-params-terrain',
    });

    this.renderUniformBufferWater = this.device.createBuffer({
      size: alignTo(this.renderUniformValues.byteLength, 16),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'paper-render-params-water',
    });

    this.layerMaterialBuffer = this.device.createBuffer({
      size: alignTo(this.layerMaterialValues.byteLength, 16),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'paper-layer-material-params',
    });

    this.sceneSampler = this.device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    this.#ensureCustomSpringBuffer();
    this.#ensureDummySceneTexture();
    await this.#createPipelines();
    this.#writeParams();
    this.initialized = true;
  }

  destroy() {
    this.gpuStatistics?.destroy();
    this.gpuStatistics = null;
    for (const buffer of this.stateBuffers) {
      buffer?.destroy?.();
    }
    this.stateBuffers = [];
    this.fluxBuffer?.destroy?.();
    this.velocityBuffer?.destroy?.();
    this.thermalPipeBufferA?.destroy?.();
    this.thermalPipeBufferB?.destroy?.();
    this.readbackBuffer?.destroy?.();
    this.customSpringBuffer?.destroy?.();
    this.layerTopBuffer?.destroy?.();
    this.depthTexture?.destroy?.();
    this.sceneColorTexture?.destroy?.();
    this.dummySceneTexture?.destroy?.();
    this.renderUniformBufferTerrain?.destroy?.();
    this.renderUniformBufferWater?.destroy?.();
    this.layerMaterialBuffer?.destroy?.();
    this.paramBuffer?.destroy?.();
    this.stepParamBuffer?.destroy?.();
    this.fluxBuffer = null;
    this.velocityBuffer = null;
    this.thermalPipeBufferA = null;
    this.thermalPipeBufferB = null;
    this.readbackBuffer = null;
    this.customSpringBuffer = null;
    this.layerTopBuffer = null;
    this.layerSurfaceValues = new Float32Array(0);
    this.layerMaterialBuffer = null;
    this.layerMaterialValues = new Float32Array(16);
    this.paintedSpringMap = new Float32Array(0);
    this.allocatedPaintedSpringByteLength = 0;
    this.allocatedReadbackByteLength = 0;
    this.depthTexture = null;
    this.depthTextureView = null;
    this.depthTextureSize = { width: 0, height: 0 };
    this.sceneColorTexture = null;
    this.sceneColorView = null;
    this.sceneColorSize = { width: 0, height: 0 };
    this.dummySceneTexture = null;
    this.dummySceneView = null;
    this.renderUniformBufferTerrain = null;
    this.renderUniformBufferWater = null;
    this.layerMaterialBuffer = null;
    this.paramBuffer = null;
    this.stepParamBuffer = null;
    this.stepParamCapacity = 0;
    this.stepParamValuesCpu = null;
    this.stepParamValuesCpuCapacity = 0;
    this.#invalidateRenderBundles();
    this.computeBindGroup01 = null;
    this.computeBindGroup10 = null;
    this.renderBindGroupTerrain = null;
    this.renderBindGroupWater = null;
    this.compositeBindGroup = null;
    this.ready = false;
    this.width = 0;
    this.height = 0;
    this.cellCount = 0;
    this.iterationCount = 0;
    this.simulationTime = 0;
    this.dispatchX = 0;
    this.dispatchY = 0;
    this.drawCount = 0;
    this.latestReadbackStats = null;
    this.lastErosionPassMs = 0;
    this.lastRenderPassMs = 0;
    this.avgErosionPassMs = 0;
    this.avgRenderPassMs = 0;
    this.lastCpuErosionPassMs = 0;
    this.lastCpuRenderPassMs = 0;
    this.avgCpuErosionPassMs = 0;
    this.avgCpuRenderPassMs = 0;
    this.lastGpuErosionPassMs = 0;
    this.lastGpuRenderPassMs = 0;
    this.avgGpuErosionPassMs = 0;
    this.avgGpuRenderPassMs = 0;
    this.paintedSpringMap = new Float32Array(0);
  }

  setSimulationParams(next = {}) {
    const normalized = { ...next };
    if (Number.isFinite(normalized.erodabilityBase) && !Number.isFinite(normalized.hardnessBase)) normalized.hardnessBase = normalized.erodabilityBase;
    if (Number.isFinite(normalized.erodabilityVariation) && !Number.isFinite(normalized.hardnessVariation)) normalized.hardnessVariation = normalized.erodabilityVariation;
    const sourceKeys = ['sourceRadius', 'sourceStrength', 'sourceSeed', 'randomSpringCount', 'sourceLayoutMode'];
    if (sourceKeys.some(key => key in normalized && normalized[key] !== this.simulationParams[key])) this.sourceFieldDirty = true;
    const layoutChanged = 'sourceLayoutMode' in normalized && normalized.sourceLayoutMode !== this.simulationParams.sourceLayoutMode;
    Object.assign(this.simulationParams, normalized);
    if (layoutChanged && this.simulationParams.sourceLayoutMode < 0.5 && this.device) this.#writeCustomSpringBuffer();
    if (this.layeredDemEnabled && (Number.isFinite(normalized.hardnessBase) || Number.isFinite(normalized.hardnessVariation))) {
      this.layerMaterialValues[0] = this.simulationParams.hardnessBase;
      this.layerMaterialValues[1] = clamp(this.simulationParams.hardnessBase + this.simulationParams.hardnessVariation, 0, 1);
      this.#writeLayerMaterialParams();
    }
    if (this.paramBuffer && this.device) {
      this.#writeParams();
    }
    if ((this.renderUniformBufferTerrain || this.renderUniformBufferWater) && this.device) {
      this.#writeRenderParams();
    }
  }

  resetRainTimer() {
    this.simulationParams.sourceTimeOffset = this.simulationTime;
    if (this.paramBuffer && this.device) {
      this.#writeParams();
    }
  }

  restartSources() {
    this.resetRainTimer();
  }

  async applyTerrainBrush(brush = {}) {
    if (!this.ready || this.width <= 0 || this.height <= 0) {
      throw new Error('Terrain painting is only available after the simulation is initialized.');
    }

    const centerX = clamp(Number(brush.x) || 0, 0, Math.max(this.width - 1, 0));
    const centerY = clamp(Number(brush.y) || 0, 0, Math.max(this.height - 1, 0));
    const radius = Math.max(0.5, Number(brush.radius) || 1);
    const amount = Math.max(0.0001, Number(brush.amount) || 0.01);
    const hardness = clamp(Number(brush.hardness) || 0.5, 0.01, 1.0);
    const subtract = !!brush.subtract;
    const data = await this.#readbackStateData();

    const minX = Math.max(0, Math.floor(centerX - radius - 1));
    const maxX = Math.min(this.width - 1, Math.ceil(centerX + radius + 1));
    const minY = Math.max(0, Math.floor(centerY - radius - 1));
    const maxY = Math.min(this.height - 1, Math.ceil(centerY + radius + 1));
    const falloffPower = 2.65 - hardness * 2.20;
    const direction = subtract ? -1 : 1;

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x - centerX;
        const dy = y - centerY;
        const dist = Math.hypot(dx, dy);
        if (dist > radius) continue;
        const t = 1 - dist / radius;
        const falloff = Math.pow(clamp(t, 0, 1), falloffPower);
        const i = y * this.width + x;
        const base = i * STATE_FLOATS_PER_CELL;
        if (data[base + 4] < 0.5) continue;
        const terrain = clamp(data[base] + direction * amount * falloff, 0.0, 2.0);
        data[base] = terrain;
        const fallbackHardness = Number.isFinite(data[base + 3]) ? data[base + 3] : this.simulationParams.hardnessBase;
        data[base + 3] = this.layerSurfaceValues.length === this.cellCount * 4
          ? this.#resolveLayerHardnessForTerrain(this.layerSurfaceValues, i * 4, terrain, fallbackHardness)
          : fallbackHardness;
      }
    }

    this.device.queue.writeBuffer(this.stateBuffers[0], 0, data);
    this.device.queue.writeBuffer(this.stateBuffers[1], 0, data);
    this.latestReadbackStats = null;
    this.render();
    return this.getStats();
  }

  paintSpringBrush(brush = {}) {
    if (!this.ready || this.width <= 0 || this.height <= 0) {
      throw new Error('Spring painting is only available after the simulation is initialized.');
    }

    const centerX = clamp(Number(brush.x) || 0, 0, Math.max(this.width - 1, 0));
    const centerY = clamp(Number(brush.y) || 0, 0, Math.max(this.height - 1, 0));
    const radius = Math.max(0.5, Number(brush.radius) || 1);
    const amount = Math.max(0.0, Number(brush.strength) || 0.0);
    const hardness = clamp(Number(brush.hardness) || 0.5, 0.01, 1.0);
    const erase = !!brush.erase;

    this.#ensurePaintedSpringMap();

    const minX = Math.max(0, Math.floor(centerX - radius - 1));
    const maxX = Math.min(this.width - 1, Math.ceil(centerX + radius + 1));
    const minY = Math.max(0, Math.floor(centerY - radius - 1));
    const maxY = Math.min(this.height - 1, Math.ceil(centerY + radius + 1));
    const falloffPower = 2.65 - hardness * 2.20;

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dx = x - centerX;
        const dy = y - centerY;
        const dist = Math.hypot(dx, dy);
        if (dist > radius) continue;
        const t = 1 - dist / radius;
        const falloff = Math.pow(clamp(t, 0, 1), falloffPower);
        const i = y * this.width + x;
        if (erase) {
          this.paintedSpringMap[i] = Math.max(0.0, this.paintedSpringMap[i] - Math.max(amount, 1.0) * falloff);
        } else {
          this.paintedSpringMap[i] = clamp(this.paintedSpringMap[i] + amount * falloff, 0.0, 1.0);
        }
      }
    }

    this.#writeCustomSpringBuffer();
    this.#writeParams();
    this.render();
    return this.getSourcePoints();
  }

  clearPaintedSprings() {
    if (this.paintedSpringMap instanceof Float32Array && this.paintedSpringMap.length === this.cellCount) {
      this.paintedSpringMap.fill(0.0);
    }
    this.#writeCustomSpringBuffer();
    this.#writeParams();
    this.render();
    return this.getSourcePoints();
  }

  #normalizeLayerConfigArray(options = {}) {
    const baseHardMin = clamp(this.simulationParams.hardnessBase ?? 0.16, 0.0, 1.0);
    const baseHardMax = clamp(baseHardMin + Math.max(0.0, this.simulationParams.hardnessVariation ?? 0.03), 0.0, 1.0);
    const defaults = [
      { enabled: true, label: 'Base', hardnessMin: baseHardMin, hardnessMax: baseHardMax, thermalEnabled: true },
      { enabled: false, label: 'Silt', hardnessMin: 0.24, hardnessMax: 0.42, thermalEnabled: true },
      { enabled: false, label: 'Clay', hardnessMin: 0.10, hardnessMax: 0.22, thermalEnabled: true },
      { enabled: false, label: 'Bedrock', hardnessMin: 0.00, hardnessMax: 0.02, thermalEnabled: true },
    ];
    const inputLayers = Array.isArray(options.layerConfig?.layers) ? options.layerConfig.layers : [];
    return defaults.map((fallback, index) => {
      const src = inputLayers[index] || {};
      const heightMin = Number.isFinite(src.heightMin) ? src.heightMin : 0.0;
      const heightMax = Number.isFinite(src.heightMax) ? src.heightMax : 1.0;
      const hardMin = Number.isFinite(src.hardnessMin) ? src.hardnessMin : Number.isFinite(src.erodabilityMin) ? src.erodabilityMin : fallback.hardnessMin;
      const hardMax = Number.isFinite(src.hardnessMax) ? src.hardnessMax : Number.isFinite(src.erodabilityMax) ? src.erodabilityMax : fallback.hardnessMax;
      return {
        enabled: !!src.enabled,
        label: src.label || fallback.label,
        preset: src.preset || 'custom',
        heightMin,
        heightMax,
        hardnessMin: clamp(Math.min(hardMin, hardMax), 0.0, 1.0),
        hardnessMax: clamp(Math.max(hardMin, hardMax), 0.0, 1.0),
        thermalEnabled: src.thermalEnabled !== false,
      };
    });
  }

  #resolveLayerHardnessForTerrain(surfaceArray, offset, terrain, fallbackHardness) {
    let chosenIndex = -1;
    let chosenTop = Infinity;
    let highestTop = -Infinity;
    let highestIndex = -1;

    for (let layerIndex = 0; layerIndex < 4; layerIndex++) {
      const matBase = layerIndex * 4;
      if (this.layerMaterialValues[matBase + 2] < 0.5) continue;
      const top = surfaceArray[offset + layerIndex];
      if (!Number.isFinite(top) || top < -1e8) continue;
      if (top >= highestTop) {
        highestTop = top;
        highestIndex = layerIndex;
      }
      if (top >= terrain && top <= chosenTop) {
        chosenTop = top;
        chosenIndex = layerIndex;
      }
    }

    if (chosenIndex < 0) {
      chosenIndex = highestIndex;
      chosenTop = highestTop;
    }
    if (chosenIndex < 0 || !Number.isFinite(chosenTop)) {
      return clamp(fallbackHardness, 0.0, 1.0);
    }

    let lowerTop = -Infinity;
    for (let layerIndex = 0; layerIndex < 4; layerIndex++) {
      const matBase = layerIndex * 4;
      if (this.layerMaterialValues[matBase + 2] < 0.5) continue;
      const top = surfaceArray[offset + layerIndex];
      if (!Number.isFinite(top) || top < -1e8 || top >= chosenTop) continue;
      if (top > lowerTop) lowerTop = top;
    }

    let depthT = 0.0;
    if (Number.isFinite(lowerTop) && lowerTop > -1e8) {
      depthT = clamp((chosenTop - terrain) / Math.max(chosenTop - lowerTop, 1e-6), 0.0, 1.0);
    } else if (terrain < chosenTop) {
      depthT = clamp((chosenTop - terrain) / Math.max(chosenTop, 1e-6), 0.0, 1.0);
    }

    const materialBase = chosenIndex * 4;
    const hardMin = this.layerMaterialValues[materialBase];
    const hardMax = this.layerMaterialValues[materialBase + 1];
    return clamp(hardMin + (hardMax - hardMin) * depthT, 0.0, 1.0);
  }

  #writeLayerMaterialParams() {
    if (this.layerMaterialBuffer && this.device) {
      this.device.queue.writeBuffer(this.layerMaterialBuffer, 0, this.layerMaterialValues);
    }
  }


  getSourcePoints() {
    if (this.width <= 0 || this.height <= 0) {
      return [];
    }

    const layoutMode = this.simulationParams.sourceLayoutMode | 0;
    const springsEnabled = !!this.simulationParams.sourceEnabled;

    if (layoutMode === 0) {
      const values = this.paintedSpringMap;
      if (!(values instanceof Float32Array) || values.length !== this.cellCount) {
        return [];
      }
      const points = [];
      const threshold = 0.0025;
      let activeCount = 0;
      for (let i = 0; i < values.length; i++) {
        if (values[i] > threshold) activeCount++;
      }
      const maxPoints = 8192;
      const stride = Math.max(1, Math.ceil(activeCount / maxPoints));
      let seen = 0;
      for (let y = 0; y < this.height; y++) {
        for (let x = 0; x < this.width; x++) {
          const i = y * this.width + x;
          const strength = values[i];
          if (strength <= threshold) continue;
          if ((seen++ % stride) !== 0) continue;
          points.push({
            x,
            y,
            radius: 0.7,
            strength,
            painted: true,
            active: springsEnabled,
          });
        }
      }
      return points;
    }

    if (!springsEnabled) {
      return [];
    }

    const points = [];
    const count = Math.min(16, Math.max(1, this.simulationParams.randomSpringCount | 0));
    for (let i = 0; i < count; i++) {
      const seed = Math.imul(Math.floor(Math.max(0, this.simulationParams.sourceSeed ?? 1)) >>> 0, 37) + i * 17 + 1;
      const x = (sourceHash(seed) & 0xffffff) / 16777216;
      const y = (sourceHash(seed ^ 0x9e3779b9) & 0xffffff) / 16777216;
      points.push({
        x: x * Math.max(this.width - 1, 0),
        y: y * Math.max(this.height - 1, 0),
        radius: this.simulationParams.sourceRadius,
        strength: this.simulationParams.sourceStrength,
        painted: false,
        active: true,
      });
    }
    return points;
  }


  async setDEM(raster, options = {}) {
    if (!this.initialized) {
      await this.initialize(this.canvas);
    }
    if (!raster || !(raster.values instanceof Float32Array)) {
      throw new Error('setDEM expects a raster with Float32Array values.');
    }

    const width = raster.width | 0;
    const height = raster.height | 0;
    if (width <= 0 || height <= 0) {
      throw new Error('Raster width and height must be positive.');
    }
    if (raster.values.length !== width * height) {
      throw new Error('Raster value count does not match width * height.');
    }

    const minHeight = Number.isFinite(options.minHeight) ? options.minHeight : 0;
    const maxHeight = Number.isFinite(options.maxHeight) ? options.maxHeight : 1;
    const range = Math.max(1e-6, maxHeight - minHeight);
    const mask = raster.mask instanceof Uint8Array && raster.mask.length === width * height
      ? raster.mask
      : null;
    const rasterBands = Array.isArray(raster.bands) ? raster.bands : null;
    const initialWater = raster.water instanceof Float32Array && raster.water.length === width * height ? raster.water : null;
    const layerConfigs = this.#normalizeLayerConfigArray(options);
    const demSourceMode = options.demSourceMode || options.layerConfig?.mode || 'single';
    const hasLayerBands = Array.isArray(rasterBands) && rasterBands.some((band) => band instanceof Float32Array && band.length === width * height);
    const useLayeredDem = hasLayerBands || demSourceMode !== 'single';

    this.width = width;
    this.height = height;
    this.cellCount = width * height;
    this.stateByteLength = this.cellCount * STATE_FLOATS_PER_CELL * 4;
    this.vec4ByteLength = this.cellCount * VEC4_BYTES_PER_CELL;
    this.iterationCount = 0;
    this.simulationTime = 0;
    this.simulationParams.sourceTimeOffset = 0;
    this.sourceFieldDirty = true;
    this.sourceFieldBuildCount = 0;
    this.dispatchX = Math.ceil(this.width / WORKGROUP_SIZE_X);
    this.dispatchY = Math.ceil(this.height / WORKGROUP_SIZE_Y);
    this.drawCount = Math.max(1, (this.width - 1) * (this.height - 1));
    this.#invalidateRenderBundles();

    this.#ensureDepthTexture();
    this.#ensureSceneTargets();
    this.#ensureBuffers();
    this.#writeParams();

    const initialState = new Float32Array(this.cellCount * STATE_FLOATS_PER_CELL);
    const layerSurfaceValues = useLayeredDem ? new Float32Array(this.cellCount * 4) : null;
    if (layerSurfaceValues) layerSurfaceValues.fill(-1e9);
    const hardnessBase = this.simulationParams.hardnessBase;
    const hardnessVariation = this.simulationParams.hardnessVariation;

    this.layerMaterialValues.fill(0.0);
    for (let layerIndex = 0; layerIndex < 4; layerIndex++) {
      const config = layerConfigs[layerIndex];
      const base = layerIndex * 4;
      this.layerMaterialValues[base] = config.hardnessMin;
      this.layerMaterialValues[base + 1] = config.hardnessMax;
      const hasBand = !!(rasterBands && rasterBands[layerIndex] instanceof Float32Array && rasterBands[layerIndex].length === this.cellCount);
      const fallbackSingleLayer = useLayeredDem && !hasLayerBands && layerIndex === 0;
      this.layerMaterialValues[base + 2] = (config.enabled && (hasBand || fallbackSingleLayer)) ? 1.0 : 0.0;
      this.layerMaterialValues[base + 3] = config.thermalEnabled ? 1.0 : 0.0;
    }

    for (let i = 0; i < this.cellCount; i++) {
      const base = i * STATE_FLOATS_PER_CELL;
      const layerBase = i * 4;
      const valid = mask ? (mask[i] ? 1 : 0) : 1;
      const fallbackTerrain = minHeight + raster.values[i] * range;
      const normalizedTerrain = clamp(raster.values[i], 0.0, 1.0);
      const noise = Math.sin(i * 12.9898 + width * 0.031 + height * 0.017) * 43758.5453;
      const frac = noise - Math.floor(noise);
      const noiseTerm = (frac * 2 - 1) * hardnessVariation * 0.25;
      const fallbackHardness = clamp(hardnessBase + (1.0 - normalizedTerrain) * hardnessVariation + noiseTerm, 0.0, 1.0);

      let terrain = fallbackTerrain;
      let hardness = fallbackHardness;
      if (useLayeredDem) {
        let highestLayerTop = -Infinity;
        for (let layerIndex = 0; layerIndex < 4; layerIndex++) {
          const matBase = layerIndex * 4;
          if (this.layerMaterialValues[matBase + 2] < 0.5) continue;
          const band = rasterBands && rasterBands[layerIndex] instanceof Float32Array && rasterBands[layerIndex].length === this.cellCount
            ? rasterBands[layerIndex]
            : (layerIndex === 0 ? raster.values : null);
          if (!band) continue;
          const config = layerConfigs[layerIndex];
          const normalizedBand = clamp(band[i], 0.0, 1.0);
          const topHeight = config.heightMin + normalizedBand * (config.heightMax - config.heightMin);
          layerSurfaceValues[layerBase + layerIndex] = topHeight;
          if (topHeight > highestLayerTop) highestLayerTop = topHeight;
        }
        if (Number.isFinite(highestLayerTop) && highestLayerTop > -1e8) {
          terrain = highestLayerTop;
        } else {
          layerSurfaceValues[layerBase] = terrain;
          this.layerMaterialValues[2] = 1.0;
        }
        hardness = this.#resolveLayerHardnessForTerrain(layerSurfaceValues, layerBase, terrain, fallbackHardness);
      }

      initialState[base] = terrain;
      initialState[base + 1] = valid && Number.isFinite(initialWater?.[i]) ? Math.max(initialWater[i], 0) : 0;
      initialState[base + 2] = 0.0;
      initialState[base + 3] = hardness;
      initialState[base + 4] = valid;
      initialState[base + 5] = 0.0;
      initialState[base + 6] = 0.0;
      initialState[base + 7] = 0.0;
    }

    const clearEncoder = this.device.createCommandEncoder({ label: 'paper-reset-sim-scratch' });
    clearEncoder.clearBuffer(this.fluxBuffer);
    clearEncoder.clearBuffer(this.velocityBuffer);
    clearEncoder.clearBuffer(this.thermalPipeBufferA);
    clearEncoder.clearBuffer(this.thermalPipeBufferB);
    if (!useLayeredDem) {
      clearEncoder.clearBuffer(this.layerTopBuffer);
    }
    this.device.queue.submit([clearEncoder.finish()]);
    this.device.queue.writeBuffer(this.stateBuffers[0], 0, initialState);
    this.device.queue.writeBuffer(this.stateBuffers[1], 0, initialState);
    if (useLayeredDem && layerSurfaceValues) {
      this.device.queue.writeBuffer(this.layerTopBuffer, 0, layerSurfaceValues);
    }
    this.layeredDemEnabled = useLayeredDem;
    this.layerSurfaceValues = this.layeredDemEnabled && layerSurfaceValues ? layerSurfaceValues : new Float32Array(0);
    this.#writeLayerMaterialParams();
    this.layerConfigMode = demSourceMode;
    this.ready = true;
    this.latestReadbackStats = null;
    this.paintedSpringMap = new Float32Array(0);
    this.#writeCustomSpringBuffer();
    this.#writeRenderParams();
    this.#ensureDepthTexture();
    this.#ensureSceneTargets();
    this.render();
  }

  resize() {
    if (!this.context || !this.device || !this.canvas) return;
    this.#syncCanvasPixelSize();
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: 'opaque',
    });
    this.#ensureDepthTexture();
    this.#ensureSceneTargets();
    this.#writeRenderParams();
  }

  #createStepCommandBuffer(iterations = 1, label = 'paper-erosion-step', timing = null) {
    const encoder = this.device.createCommandEncoder({ label });
    this.#encodeStepCommands(encoder, Math.max(1, iterations | 0), timing);
    timing?.resolve(encoder);
    return encoder.finish();
  }

  #createRenderCommandBuffer(label = 'paper-erosion-render', timing = null) {
    this.#ensureDepthTexture();
    this.#ensureSceneTargets();
    this.#writeRenderParams();
    const view = this.context.getCurrentTexture().createView();
    const encoder = this.device.createCommandEncoder({ label });
    this.#encodeRenderCommands(encoder, view, timing);
    timing?.resolve(encoder);
    return encoder.finish();
  }

  async #submitForGpuTiming(commandBuffer, timing) {
    const submitAt = performance.now();
    this.device.queue.submit([commandBuffer]);
    if (timing) return timing.read();
    await this.device.queue.onSubmittedWorkDone();
    return Math.max(0, performance.now() - submitAt);
  }

  async stepBench(iterations = 1) {
    if (!this.ready) return null;
    const timing = this.device.features.has('timestamp-query') ? new GpuTiming(this.device, 1) : null;
    const cpuStartAt = performance.now();
    const commandBuffer = this.#createStepCommandBuffer(Math.max(1, iterations | 0), 'paper-erosion-step-bench', timing);
    const cpuMs = Math.max(0, performance.now() - cpuStartAt);
    const gpuMs = await this.#submitForGpuTiming(commandBuffer, timing);
    this.#recordErosionPassMs(cpuMs);
    this.#recordGpuErosionPassMs(gpuMs);
    return { cpuErosionMs: cpuMs, gpuErosionMs: gpuMs, cpuRenderMs: 0, gpuRenderMs: 0, gpuTimingMode: this.getStats().gpuTimingMode };
  }

  async stepAndRenderBench(iterations = 1) {
    if (!this.ready) return null;
    const canRender = !!(this.context && this.width >= 2 && this.height >= 2);
    const erosionTiming = this.device.features.has('timestamp-query') ? new GpuTiming(this.device, 1) : null;
    const erosionCpuStartAt = performance.now();
    const erosionCommandBuffer = this.#createStepCommandBuffer(Math.max(1, iterations | 0), canRender ? 'paper-erosion-step-bench' : 'paper-erosion-step-render-bench', erosionTiming);
    const cpuErosionMs = Math.max(0, performance.now() - erosionCpuStartAt);
    const gpuErosionMs = await this.#submitForGpuTiming(erosionCommandBuffer, erosionTiming);
    this.#recordErosionPassMs(cpuErosionMs);
    this.#recordGpuErosionPassMs(gpuErosionMs);
    let cpuRenderMs = 0;
    let gpuRenderMs = 0;
    if (canRender) {
      const renderTiming = this.device.features.has('timestamp-query') ? new GpuTiming(this.device, 2) : null;
      const renderCpuStartAt = performance.now();
      const renderCommandBuffer = this.#createRenderCommandBuffer('paper-erosion-render-bench', renderTiming);
      cpuRenderMs = Math.max(0, performance.now() - renderCpuStartAt);
      gpuRenderMs = await this.#submitForGpuTiming(renderCommandBuffer, renderTiming);
      this.#recordRenderPassMs(cpuRenderMs);
      this.#recordGpuRenderPassMs(gpuRenderMs);
    } else {
      this.#recordRenderPassMs(0);
    }
    return { cpuErosionMs, gpuErosionMs, cpuRenderMs, gpuRenderMs, gpuTimingMode: this.getStats().gpuTimingMode };
  }

  step(iterations = 1) {
    if (!this.ready) return;
    const startAt = performance.now();
    const commandBuffer = this.#createStepCommandBuffer(Math.max(1, iterations | 0), 'paper-erosion-step');
    this.device.queue.submit([commandBuffer]);
    this.#recordErosionPassMs(Math.max(0, performance.now() - startAt));
  }

  stepAndRender(iterations = 1) {
    if (!this.ready) return;
    const canRender = !!(this.context && this.width >= 2 && this.height >= 2);
    const encoder = this.device.createCommandEncoder({ label: canRender ? 'paper-erosion-step-render' : 'paper-erosion-step' });
    const erosionStartAt = performance.now();
    this.#encodeStepCommands(encoder, Math.max(1, iterations | 0));
    const erosionEndAt = performance.now();
    let renderStartAt = erosionEndAt;
    let renderEndAt = erosionEndAt;
    if (canRender) {
      renderStartAt = performance.now();
      this.#ensureDepthTexture();
      this.#ensureSceneTargets();
      this.#writeRenderParams();
      const view = this.context.getCurrentTexture().createView();
      this.#encodeRenderCommands(encoder, view);
      renderEndAt = performance.now();
    }
    const submitStartAt = performance.now();
    this.device.queue.submit([encoder.finish()]);
    const submitMs = performance.now() - submitStartAt;
    let erosionMs = Math.max(0, erosionEndAt - erosionStartAt);
    let renderMs = Math.max(0, renderEndAt - renderStartAt);
    if (canRender && (erosionMs + renderMs) > 1e-6) {
      const totalMs = erosionMs + renderMs;
      erosionMs += submitMs * (erosionMs / totalMs);
      renderMs += submitMs * (renderMs / totalMs);
    } else {
      erosionMs += submitMs;
      renderMs = 0;
    }
    this.#recordErosionPassMs(erosionMs);
    this.#recordRenderPassMs(renderMs);
  }

  render() {
    if (!this.ready || !this.context || this.width < 2 || this.height < 2) return;
    const startAt = performance.now();
    const commandBuffer = this.#createRenderCommandBuffer('paper-erosion-render');
    this.device.queue.submit([commandBuffer]);
    this.#recordRenderPassMs(Math.max(0, performance.now() - startAt));
  }

  async readbackStats() {
    if (!this.ready) {
      return this.getStats();
    }

    this.gpuStatistics ??= new GpuStatistics(this.device);
    this.latestReadbackStats = await this.gpuStatistics.read(this.stateBuffers[0], this.cellCount, () => ({
      statisticsIteration: this.iterationCount, statisticsSimTime: this.simulationTime,
    }));

    return this.getStats();
  }

  async exportTerrainPng() {
    if (!this.ready || this.width <= 0 || this.height <= 0) {
      throw new Error('Terrain export is only available after the simulation is initialized.');
    }
    if (typeof OffscreenCanvas !== 'function') {
      throw new Error('OffscreenCanvas is not available for terrain export.');
    }

    const data = await this.#readbackStateData();
    let minTerrain = Infinity;
    let maxTerrain = -Infinity;

    for (let i = 0; i < this.cellCount; i++) {
      const base = i * STATE_FLOATS_PER_CELL;
      if (data[base + 4] < 0.5) continue;
      const terrain = data[base];
      if (terrain < minTerrain) minTerrain = terrain;
      if (terrain > maxTerrain) maxTerrain = terrain;
    }

    if (!Number.isFinite(minTerrain) || !Number.isFinite(maxTerrain)) {
      minTerrain = 0.0;
      maxTerrain = 1.0;
    }

    const range = Math.max(maxTerrain - minTerrain, 1e-6);
    const rgba = new Uint8ClampedArray(this.width * this.height * 4);

    for (let y = 0; y < this.height; y++) {
      const outY = this.height - 1 - y;
      for (let x = 0; x < this.width; x++) {
        const srcIndex = y * this.width + x;
        const base = srcIndex * STATE_FLOATS_PER_CELL;
        const dstIndex = (outY * this.width + x) * 4;
        const mask = data[base + 4] >= 0.5;
        const terrain = data[base];
        const normalized = mask ? clamp((terrain - minTerrain) / range, 0, 1) : 0;
        const value = Math.max(0, Math.min(255, Math.round(normalized * 255)));
        rgba[dstIndex] = value;
        rgba[dstIndex + 1] = value;
        rgba[dstIndex + 2] = value;
        rgba[dstIndex + 3] = mask ? 255 : 0;
      }
    }

    const canvas = new OffscreenCanvas(this.width, this.height);
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to create a terrain export canvas.');
    }
    ctx.putImageData(new ImageData(rgba, this.width, this.height), 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    const arrayBuffer = await blob.arrayBuffer();
    return {
      width: this.width,
      height: this.height,
      minTerrain,
      maxTerrain,
      data: arrayBuffer,
    };
  }


  #blendAverage(prev, next) {
    return prev > 0 ? (prev * 0.88 + next * 0.12) : next;
  }

  #recordErosionPassMs(value) {
    const ms = Math.max(0, Number(value) || 0);
    this.lastErosionPassMs = ms;
    this.lastCpuErosionPassMs = ms;
    this.avgErosionPassMs = this.#blendAverage(this.avgErosionPassMs, ms);
    this.avgCpuErosionPassMs = this.#blendAverage(this.avgCpuErosionPassMs, ms);
  }

  #recordRenderPassMs(value) {
    const ms = Math.max(0, Number(value) || 0);
    this.lastRenderPassMs = ms;
    this.lastCpuRenderPassMs = ms;
    this.avgRenderPassMs = this.#blendAverage(this.avgRenderPassMs, ms);
    this.avgCpuRenderPassMs = this.#blendAverage(this.avgCpuRenderPassMs, ms);
  }

  #recordGpuErosionPassMs(value) {
    const ms = Math.max(0, Number(value) || 0);
    this.lastGpuErosionPassMs = ms;
    this.avgGpuErosionPassMs = this.#blendAverage(this.avgGpuErosionPassMs, ms);
  }

  #recordGpuRenderPassMs(value) {
    const ms = Math.max(0, Number(value) || 0);
    this.lastGpuRenderPassMs = ms;
    this.avgGpuRenderPassMs = this.#blendAverage(this.avgGpuRenderPassMs, ms);
  }

  getStats() {
    const simTime = this.simulationTime;
    const rainDuration = Math.max(0, this.simulationParams.rainDuration ?? 0);
    const sourceAge = Math.max(0, simTime - this.simulationParams.sourceTimeOffset);
    const rainActive = rainDuration <= 0 || sourceAge < rainDuration
      || (this.simulationParams.pulse2Duration > 0 && sourceAge < rainDuration + this.simulationParams.pulse2Duration);
    return {
      width: this.width,
      height: this.height,
      cellCount: this.cellCount,
      ready: this.ready,
      iterationCount: this.iterationCount,
      simTime,
      rainActive,
      rainDuration,
      sourceFieldBuildCount: this.sourceFieldBuildCount,
      layeredDemEnabled: this.layeredDemEnabled,
      layerConfigMode: this.layerConfigMode,
      activeLayerCount: [0, 1, 2, 3].reduce((sum, layerIndex) => sum + (this.layerMaterialValues[layerIndex * 4 + 2] > 0.5 ? 1 : 0), 0),
      lastErosionPassMs: this.lastErosionPassMs,
      lastRenderPassMs: this.lastRenderPassMs,
      avgErosionPassMs: this.avgErosionPassMs,
      avgRenderPassMs: this.avgRenderPassMs,
      lastCpuErosionPassMs: this.lastCpuErosionPassMs,
      lastCpuRenderPassMs: this.lastCpuRenderPassMs,
      avgCpuErosionPassMs: this.avgCpuErosionPassMs,
      avgCpuRenderPassMs: this.avgCpuRenderPassMs,
      gpuTimingMode: this.device?.features.has('timestamp-query') ? 'timestamp-query' : 'completion-time',
      lastGpuErosionPassMs: this.lastGpuErosionPassMs,
      lastGpuRenderPassMs: this.lastGpuRenderPassMs,
      avgGpuErosionPassMs: this.avgGpuErosionPassMs,
      avgGpuRenderPassMs: this.avgGpuRenderPassMs,
      ...(this.latestReadbackStats ?? {}),
    };
  }


  #syncCanvasPixelSize() {
    if (!this.canvas) return;

    if (typeof this.canvas.getBoundingClientRect !== 'function') {
      const pixelWidth = Math.max(1, Math.round(this.canvas.width || 1));
      const pixelHeight = Math.max(1, Math.round(this.canvas.height || 1));
      if (this.canvas.width !== pixelWidth) this.canvas.width = pixelWidth;
      if (this.canvas.height !== pixelHeight) this.canvas.height = pixelHeight;
      return;
    }

    const rect = this.canvas.getBoundingClientRect();
    const cssWidth = Math.max(1, Math.round(rect.width || this.canvas.width || 1));
    const cssHeight = Math.max(1, Math.round(rect.height || this.canvas.height || 1));
    const dpr = Math.max(1, Math.min(globalThis.devicePixelRatio || 1, 2));
    const pixelWidth = Math.max(1, Math.round(cssWidth * dpr));
    const pixelHeight = Math.max(1, Math.round(cssHeight * dpr));
    if (this.canvas.width !== pixelWidth) this.canvas.width = pixelWidth;
    if (this.canvas.height !== pixelHeight) this.canvas.height = pixelHeight;
  }

  #ensureDepthTexture() {
    if (!this.device || !this.canvas) return;
    const width = Math.max(1, this.canvas.width || this.canvas.clientWidth || 1);
    const height = Math.max(1, this.canvas.height || this.canvas.clientHeight || 1);
    if (this.depthTexture && this.depthTextureSize.width === width && this.depthTextureSize.height === height) {
      return;
    }
    this.depthTexture?.destroy?.();
    this.depthTexture = this.device.createTexture({
      size: { width, height },
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      label: 'paper-depth',
    });
    this.depthTextureView = this.depthTexture.createView();
    this.depthTextureSize = { width, height };
  }

  #ensureSceneTargets() {
    if (!this.device || !this.canvas) return;
    const width = Math.max(1, this.canvas.width || this.canvas.clientWidth || 1);
    const height = Math.max(1, this.canvas.height || this.canvas.clientHeight || 1);
    if (this.sceneColorTexture && this.sceneColorSize.width === width && this.sceneColorSize.height === height) {
      return;
    }
    this.sceneColorTexture?.destroy?.();
    this.sceneColorTexture = this.device.createTexture({
      size: { width, height },
      format: this.presentationFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      label: 'paper-scene-color',
    });
    this.sceneColorView = this.sceneColorTexture.createView();
    this.sceneColorSize = { width, height };
    if (this.stateBuffers.length === 2 && this.sceneSampler) {
      this.#rebuildBindGroups();
    }
  }


  async #readbackStateData() {
    this.#ensureReadbackBuffer();
    const encoder = this.device.createCommandEncoder({ label: 'paper-erosion-readback' });
    encoder.copyBufferToBuffer(this.stateBuffers[0], 0, this.readbackBuffer, 0, this.stateByteLength);
    this.device.queue.submit([encoder.finish()]);

    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const mapped = this.readbackBuffer.getMappedRange();
    const copy = mapped.slice(0);
    this.readbackBuffer.unmap();
    return new Float32Array(copy);
  }

  #ensureCustomSpringBuffer() {
    if (!this.device) return;
    const size = Math.max(4, this.cellCount * 4);
    if (this.customSpringBuffer && this.allocatedPaintedSpringByteLength === size) return;
    this.customSpringBuffer?.destroy?.();
    this.customSpringBuffer = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'paper-cached-source-rates',
    });
    this.allocatedPaintedSpringByteLength = size;
    this.#clearCustomSpringBuffer();
  }

  #writeCustomSpringBuffer() {
    this.sourceFieldDirty = true;
    if (!this.device) return;
    this.#ensureCustomSpringBuffer();
    if (!this.customSpringBuffer) return;
    if (!(this.paintedSpringMap instanceof Float32Array) || this.paintedSpringMap.length !== this.cellCount) {
      this.#clearCustomSpringBuffer();
      return;
    }
    this.device.queue.writeBuffer(this.customSpringBuffer, 0, this.paintedSpringMap);
  }


  #ensureReadbackBuffer() {
    if (!this.device) return;
    if (this.readbackBuffer && this.allocatedReadbackByteLength === this.stateByteLength) return;
    this.readbackBuffer?.destroy?.();
    this.readbackBuffer = this.device.createBuffer({
      size: this.stateByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'paper-readback',
    });
    this.allocatedReadbackByteLength = this.stateByteLength;
  }

  #ensurePaintedSpringMap() {
    if (!(this.paintedSpringMap instanceof Float32Array) || this.paintedSpringMap.length !== this.cellCount) {
      this.paintedSpringMap = new Float32Array(this.cellCount);
    }
  }

  #clearCustomSpringBuffer() {
    this.sourceFieldDirty = true;
    if (!this.device) return;
    this.#ensureCustomSpringBuffer();
    if (!this.customSpringBuffer) return;
    const encoder = this.device.createCommandEncoder({ label: 'paper-clear-painted-springs' });
    encoder.clearBuffer(this.customSpringBuffer);
    this.device.queue.submit([encoder.finish()]);
  }


  #ensureDummySceneTexture() {
    if (!this.device || !this.presentationFormat || this.dummySceneTexture) return;
    this.dummySceneTexture = this.device.createTexture({
      size: { width: 1, height: 1 },
      format: this.presentationFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      label: 'paper-dummy-scene-color',
    });
    this.dummySceneView = this.dummySceneTexture.createView();
    this.device.queue.writeTexture(
      { texture: this.dummySceneTexture },
      new Uint8Array([14, 17, 20, 255]),
      { bytesPerRow: 256 },
      { width: 1, height: 1, depthOrArrayLayers: 1 }
    );
  }


  #invalidateRenderBundles() {
    this.terrainRenderBundle = null;
    this.finalRenderBundle = null;
    this.terrainRenderBundleList.length = 0;
    this.finalRenderBundleList.length = 0;
  }

  #ensureRenderBundles() {
    if (!this.device || !this.renderPipeline || !this.compositePipeline || !this.waterRenderPipeline) return;
    if (!this.renderBindGroupTerrain || !this.renderBindGroupWater || !this.compositeBindGroup) return;
    if (this.terrainRenderBundle && this.finalRenderBundle) return;

    const drawCount = this.drawCount || Math.max(1, (this.width - 1) * (this.height - 1));

    const terrainEncoder = this.device.createRenderBundleEncoder({
      colorFormats: [this.presentationFormat],
      depthStencilFormat: 'depth24plus',
    });
    terrainEncoder.setPipeline(this.renderPipeline);
    terrainEncoder.setBindGroup(0, this.renderBindGroupTerrain);
    terrainEncoder.draw(6, drawCount, 0, 0);
    this.terrainRenderBundle = terrainEncoder.finish({ label: 'paper-render-terrain-bundle' });
    this.terrainRenderBundleList[0] = this.terrainRenderBundle;

    const finalEncoder = this.device.createRenderBundleEncoder({
      colorFormats: [this.presentationFormat],
      depthStencilFormat: 'depth24plus',
    });
    finalEncoder.setPipeline(this.compositePipeline);
    finalEncoder.setBindGroup(0, this.compositeBindGroup);
    finalEncoder.draw(3, 1, 0, 0);
    finalEncoder.setPipeline(this.waterRenderPipeline);
    finalEncoder.setBindGroup(0, this.renderBindGroupWater);
    finalEncoder.draw(6, drawCount, 0, 0);
    this.finalRenderBundle = finalEncoder.finish({ label: 'paper-render-final-bundle' });
    this.finalRenderBundleList[0] = this.finalRenderBundle;
  }

  #encodeStepCommands(encoder, iterCount, timing = null) {
    if (!this.ready) return;
    const dispatchX = this.dispatchX || Math.ceil(this.width / WORKGROUP_SIZE_X);
    const dispatchY = this.dispatchY || Math.ceil(this.height / WORKGROUP_SIZE_Y);
    this.#ensureStepParamTemplate(iterCount);

    const strideFloats = this.stepParamStride >>> 2;
    const stepParamValues = this.stepParamValuesCpu;
    const dt = this.paramValues[3];
    const baseTime = this.simulationTime;
    const integration = this.#sourceIntegrationCoefficients(dt);
    for (let i = 0; i < iterCount; i++) {
      const offset = i * strideFloats;
      stepParamValues[offset + 25] = baseTime + i * dt;
      this.#writeSourceIntegration(stepParamValues, offset, baseTime + i * dt, integration);
    }
    this.device.queue.writeBuffer(this.stepParamBuffer, 0, stepParamValues.buffer, 0, this.stepParamStride * iterCount);

    const fluxPipeline = this.fluxPipeline;
    const flowPipeline = this.flowPipeline;
    const thermalOutflowPipeline = this.thermalOutflowPipeline;
    const erosionPipeline = this.erosionPipeline;
    const computeBindGroup01 = this.computeBindGroup01;
    const computeBindGroup10 = this.computeBindGroup10;
    const dynamicOffset = this.computeDynamicOffset;

    const pass = encoder.beginComputePass({ label: 'paper-erosion-compute', timestampWrites: timing?.writes() });
    if (this.sourceFieldDirty && this.simulationParams.sourceEnabled && this.simulationParams.sourceLayoutMode >= 0.5) {
      dynamicOffset[0] = 0;
      pass.setBindGroup(0, computeBindGroup01, dynamicOffset);
      pass.setPipeline(this.sourcePipeline);
      pass.dispatchWorkgroups(dispatchX, dispatchY);
      this.sourceFieldDirty = false;
      this.sourceFieldBuildCount++;
    }
    for (let i = 0; i < iterCount; i++) {
      dynamicOffset[0] = i * this.stepParamStride;

      pass.setBindGroup(0, computeBindGroup01, dynamicOffset);
      pass.setPipeline(fluxPipeline);
      pass.dispatchWorkgroups(dispatchX, dispatchY);

      pass.setPipeline(flowPipeline);
      pass.dispatchWorkgroups(dispatchX, dispatchY);

      pass.setBindGroup(0, computeBindGroup10, dynamicOffset);
      if (this.simulationParams.bendSedimentRate > 0 && this.simulationParams.riverBendLength > 0 && this.simulationParams.hydraulicErosionEnabled) {
        pass.setPipeline(this.channelPipeline);
        pass.dispatchWorkgroups(this.height);
        pass.setPipeline(this.channelBendPipeline);
        pass.dispatchWorkgroups(Math.ceil(this.height / 64));
      }
      if (this.simulationParams.thermalRate > 0 || ((this.simulationParams.bankErosionRate > 0 || this.simulationParams.bendSedimentRate > 0) && this.simulationParams.hydraulicErosionEnabled)) {
        pass.setPipeline(thermalOutflowPipeline);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
      }

      pass.setPipeline(erosionPipeline);
      pass.dispatchWorkgroups(dispatchX, dispatchY);

    }
    pass.end();

    this.iterationCount += iterCount;
    this.simulationTime += iterCount * dt;
    this.#fillParamValues(this.iterationCount, this.paramValues);
  }

  #encodeRenderCommands(encoder, view, timing = null) {
    this.#ensureRenderBundles();

    {
      const pass = encoder.beginRenderPass({
        timestampWrites: timing?.writes(0),
        colorAttachments: [{
          view: this.sceneColorView,
          clearValue: { r: 0.055, g: 0.065, b: 0.08, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
        depthStencilAttachment: this.depthTextureView ? {
          view: this.depthTextureView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        } : undefined,
      });
      if (this.terrainRenderBundle) {
        pass.executeBundles(this.terrainRenderBundleList);
      } else {
        const drawCount = this.drawCount || Math.max(1, (this.width - 1) * (this.height - 1));
        pass.setPipeline(this.renderPipeline);
        pass.setBindGroup(0, this.renderBindGroupTerrain);
        pass.draw(6, drawCount, 0, 0);
      }
      pass.end();
    }

    {
      const pass = encoder.beginRenderPass({
        timestampWrites: timing?.writes(1),
        colorAttachments: [{
          view,
          clearValue: { r: 0.055, g: 0.065, b: 0.08, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
        depthStencilAttachment: this.depthTextureView ? {
          view: this.depthTextureView,
          depthLoadOp: 'load',
          depthStoreOp: 'discard',
        } : undefined,
      });
      if (this.finalRenderBundle) {
        pass.executeBundles(this.finalRenderBundleList);
      } else {
        const drawCount = this.drawCount || Math.max(1, (this.width - 1) * (this.height - 1));
        pass.setPipeline(this.compositePipeline);
        pass.setBindGroup(0, this.compositeBindGroup);
        pass.draw(3, 1, 0, 0);
        pass.setPipeline(this.waterRenderPipeline);
        pass.setBindGroup(0, this.renderBindGroupWater);
        pass.draw(6, drawCount, 0, 0);
      }
      pass.end();
    }
  }

  #writeRenderParams() {
    if ((!this.renderUniformBufferTerrain && !this.renderUniformBufferWater) || !this.device) return;
    const aspect = this.canvas ? Math.max((this.canvas.width || 1) / Math.max(this.canvas.height || 1, 1), 1e-6) : 1;
    const azimuth = degToRad(this.simulationParams.cameraAzimuthDeg);
    const elevation = degToRad(this.simulationParams.cameraElevationDeg);
    const worldScale = this.width > 1 || this.height > 1 ? 2.0 / Math.max(this.width - 1, this.height - 1, 1) : 1.0;
    const eye = [
      this.simulationParams.cameraPosX ?? 1.5238999619464006,
      this.simulationParams.cameraPosY ?? 1.9404787584406888,
      this.simulationParams.cameraPosZ ?? 1.5238999619464006,
    ];
    const forward = normalizeVec3([
      -Math.cos(elevation) * Math.cos(azimuth),
      -Math.sin(elevation),
      -Math.cos(elevation) * Math.sin(azimuth),
    ]);
    const target = [eye[0] + forward[0], eye[1] + forward[1], eye[2] + forward[2]];
    const view = lookAtMat4(eye, target, [0, 1, 0]);
    const proj = perspectiveMat4(degToRad(50), aspect, 0.01, 32.0);
    const viewProj = multiplyMat4(proj, view);
    const effectiveCellSize = Math.max(0.25, (this.simulationParams.metersPerPixel ?? 100) / 100);
    this.renderUniformValues.set(viewProj, 0);
    this.renderUniformValues[16] = this.width;
    this.renderUniformValues[17] = this.height;
    this.renderUniformValues[18] = effectiveCellSize;
    this.renderUniformValues[19] = this.simulationParams.renderHeightScale;
    this.renderUniformValues[20] = this.simulationParams.waterOpacity;
    this.renderUniformValues[21] = this.simulationParams.sedimentTint;
    this.renderUniformValues[22] = this.simulationParams.renderMode;
    this.renderUniformValues[23] = this.simulationParams.thermalVisualizationScale;
    this.renderUniformValues[24] = worldScale;
    this.renderUniformValues[25] = this.simulationParams.waterHeightScale;
    this.renderUniformValues[27] = 0.0005;
    const light = normalizeVec3([0.45, 0.82, 0.36]);
    this.renderUniformValues[28] = light[0];
    this.renderUniformValues[29] = light[1];
    this.renderUniformValues[30] = light[2];
    const meshStep = Math.max(1, Math.floor(this.simulationParams.renderGridStep)
      || Math.ceil(Math.max(this.width - 1, this.height - 1, 1) / 512));
    const drawCount = Math.max(1, Math.ceil((this.width - 1) / meshStep) * Math.ceil((this.height - 1) / meshStep));
    if (this.drawCount !== drawCount) {
      this.drawCount = drawCount;
      this.#invalidateRenderBundles();
    }
    this.renderUniformValues[31] = meshStep;
    this.renderUniformValues[32] = eye[0];
    this.renderUniformValues[33] = eye[1];
    this.renderUniformValues[34] = eye[2];
    this.renderUniformValues[35] = this.layeredDemEnabled ? 1.0 : 0.0;
    this.renderUniformValues[36] = this.simulationTime;
    this.renderUniformValues[37] = this.iterationCount;
    this.renderUniformValues[38] = this.canvas?.width || 1;
    this.renderUniformValues[39] = this.canvas?.height || 1;

    this.renderUniformValues[26] = 0.0;
    this.device.queue.writeBuffer(this.renderUniformBufferTerrain, 0, this.renderUniformValues);
    this.renderUniformValues[26] = 1.0;
    this.device.queue.writeBuffer(this.renderUniformBufferWater, 0, this.renderUniformValues);
  }

  #ensureBuffers() {
    const needsRebuild =
      this.stateBuffers.length !== 2 ||
      !this.fluxBuffer ||
      !this.velocityBuffer ||
      !this.thermalPipeBufferA ||
      !this.thermalPipeBufferB ||
      !this.layerTopBuffer ||
      this.allocatedStateByteLength !== this.stateByteLength ||
      this.allocatedVec4ByteLength !== this.vec4ByteLength ||
      this.velocityBuffer?.size !== this.vec4ByteLength + this.height * 32;

    if (needsRebuild) {
      for (const buffer of this.stateBuffers) buffer?.destroy?.();
      this.fluxBuffer?.destroy?.();
      this.velocityBuffer?.destroy?.();
      this.thermalPipeBufferA?.destroy?.();
      this.thermalPipeBufferB?.destroy?.();
      this.readbackBuffer?.destroy?.();
      this.customSpringBuffer?.destroy?.();
      this.layerTopBuffer?.destroy?.();
      this.customSpringBuffer = null;
      this.layerTopBuffer = null;

      this.stateBuffers = [0, 1].map((index) => this.device.createBuffer({
        size: this.stateByteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: `paper-state-${index}`,
      }));

      this.fluxBuffer = this.device.createBuffer({
        size: this.vec4ByteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: 'paper-flux',
      });

      this.velocityBuffer = this.device.createBuffer({
        size: this.vec4ByteLength + this.height * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: 'paper-velocity',
      });

      this.thermalPipeBufferA = this.device.createBuffer({
        // Second half contains conservative cardinal bank-to-water transfers.
        size: this.vec4ByteLength * 2,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: 'paper-thermal-pipes-a',
      });

      this.thermalPipeBufferB = this.device.createBuffer({
        size: this.vec4ByteLength * 2,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'paper-thermal-pipes-b',
      });

      this.layerTopBuffer = this.device.createBuffer({
        size: this.vec4ByteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'paper-layer-top-heights',
      });

      this.readbackBuffer = null;
      this.allocatedReadbackByteLength = 0;
      this.allocatedStateByteLength = this.stateByteLength;
      this.allocatedVec4ByteLength = this.vec4ByteLength;
      this.#rebuildBindGroups();
    }
  }

  #sourceIntegrationCoefficients(dt) {
    const half = Math.max(dt, 0) * 0.5;
    const rate = Math.max(this.simulationParams.evaporationRate, 0);
    const duration = Math.max(this.simulationParams.rainDuration ?? 0, 0);
    return {
      half, rate, decay: Math.exp(-rate * half),
      fullWeight: rate > 0 ? -Math.expm1(-rate * half) / rate : half,
      end: duration > 0 ? this.simulationParams.sourceTimeOffset + duration + Math.max(this.simulationParams.pulse2Duration ?? 0, 0) : Infinity,
    };
  }

  #writeSourceIntegration(target, offset, at, integration) {
    const { half, rate, decay, fullWeight, end } = integration;
    const active = clamp(end - at, 0, half * 2);
    const first = Math.min(active, half);
    const second = Math.max(active - half, 0);
    target[offset + 40] = first === half ? fullWeight : rate > 0 ? -Math.expm1(-rate * first) / rate * Math.exp(-rate * (half - first)) : first;
    target[offset + 41] = decay;
    target[offset + 42] = second === half ? fullWeight : rate > 0 ? -Math.expm1(-rate * second) / rate * Math.exp(-rate * (half - second)) : second;
    // Solid input uses active duration, never the evaporating water weights.
    target[offset + 43] = first;
    target[offset + 47] = second;
  }

  #fillParamValues(iterationCount, target = this.paramValues) {
    const effectiveCellSize = Math.max(0.25, (this.simulationParams.metersPerPixel ?? 100) / 100);

    target[0] = this.width;
    target[1] = this.height;
    target[2] = effectiveCellSize;
    target[3] = this.simulationParams.timeStep;

    target[4] = this.simulationParams.rainRate;
    target[5] = this.simulationParams.evaporationRate;
    target[6] = this.simulationParams.pipeArea;
    target[7] = this.simulationParams.gravity;

    target[8] = this.simulationParams.capacityScale;
    target[9] = this.simulationParams.suspensionRate;
    target[10] = this.simulationParams.depositionRate;
    target[11] = this.simulationParams.softeningRate;

    target[12] = this.simulationParams.maxErosionDepth;
    target[13] = this.simulationParams.thermalRate;
    target[14] = this.simulationParams.talusSlopeCoeff;
    target[15] = this.simulationParams.talusSlopeBias;

    target[16] = this.simulationParams.sourceLayoutMode;
    target[17] = this.simulationParams.randomSpringCount;
    target[18] = this.simulationParams.pulse2Duration;
    target[19] = this.simulationParams.edgeDrainStrength;

    target[20] = this.simulationParams.sourceCenterX;
    target[21] = this.simulationParams.sourceCenterY;
    target[22] = this.simulationParams.sourceRadius;
    target[23] = this.simulationParams.sourceStrength;

    target[24] = this.simulationParams.sourceEnabled ? 1 : 0;
    target[25] = this.simulationTime;
    target[26] = Math.max(0.0, this.simulationParams.rainDuration ?? 0.0);
    target[27] = this.simulationParams.sourceTimeOffset;

    target[28] = this.simulationParams.historyDecay;
    target[29] = this.simulationParams.hardnessBase;
    target[30] = this.simulationParams.closedBoundaries ? 1.0 : 0.0;
    target[31] = 4.0;
    const bits = target === this.paramValues ? this.paramBits : new Uint32Array(target.buffer, target.byteOffset, target.length);
    bits[32] = Math.floor(Math.max(0, this.simulationParams.sourceSeed ?? 1)) >>> 0;
    target[33] = this.simulationParams.metersPerPixel ?? 100;
    target[34] = this.simulationParams.hydraulicErosionEnabled ? 1.0 : 0.0;
    target[35] = Math.max(0.0, this.simulationParams.edgeWaterFloor ?? 0.0);
    target[36] = this.simulationParams.hardnessBase;
    target[37] = this.simulationParams.hardnessVariation;
    target[38] = Math.max(0, this.simulationParams.flowDrag ?? 2);
    target[39] = clamp(this.simulationParams.maxOutflowFraction ?? 0.5, 0.05, 1);
    target[44] = Math.max(0, this.simulationParams.bankErosionRate ?? 0);
    target[45] = Math.max(0, this.simulationParams.bankBendBias ?? 24);
    target[46] = Math.max(0, this.simulationParams.sourceSedimentConcentration ?? 0);
    target[47] = 0;
    target[48] = Math.max(0, this.simulationParams.bendSedimentRate ?? 0);
    target[49] = Math.max(0, this.simulationParams.straightBankErosionFactor ?? 0.08);
    target[50] = Math.max(0, this.simulationParams.riverBendLength ?? 0);
    target[51] = Math.max(0, Math.min(15, Math.floor(this.simulationParams.openBoundaryEdges ?? 15)));
    this.#writeSourceIntegration(target, 0, this.simulationTime, this.#sourceIntegrationCoefficients(target[3]));
    return target;
  }

  #writeParams() {
    this.#fillParamValues(this.iterationCount, this.paramValues);
    this.stepParamTemplateDirty = true;
    this.stepParamPreparedIterations = 0;
    if (this.paramBuffer) {
      this.device.queue.writeBuffer(this.paramBuffer, 0, this.paramValues);
    }
  }

  #ensureStepParamTemplate(requiredIterations = 1) {
    this.#ensureStepParamBuffer(requiredIterations);
    if (!this.stepParamValuesCpu) return;
    if (!this.stepParamTemplateDirty && this.stepParamPreparedIterations >= requiredIterations) {
      return;
    }
    const strideFloats = this.stepParamStride >>> 2;
    const startIteration = this.stepParamTemplateDirty ? 0 : this.stepParamPreparedIterations;
    for (let i = startIteration; i < requiredIterations; i++) {
      this.stepParamValuesCpu.set(this.paramValues, i * strideFloats);
    }
    this.stepParamTemplateDirty = false;
    this.stepParamPreparedIterations = Math.max(this.stepParamPreparedIterations, requiredIterations);
  }

  #ensureStepParamBuffer(requiredIterations = 1) {
    const neededCapacity = Math.max(1, requiredIterations | 0);
    if (this.stepParamBuffer && this.stepParamCapacity >= neededCapacity && this.stepParamValuesCpu && this.stepParamValuesCpuCapacity >= neededCapacity) {
      return;
    }

    const nextCapacity = Math.max(8, neededCapacity, this.stepParamCapacity > 0 ? this.stepParamCapacity * 2 : 0);
    const strideFloats = this.stepParamStride >>> 2;

    if (!this.stepParamValuesCpu || this.stepParamValuesCpuCapacity < neededCapacity) {
      this.stepParamValuesCpu = new Float32Array(strideFloats * nextCapacity);
      this.stepParamValuesCpuCapacity = nextCapacity;
      this.stepParamTemplateDirty = true;
      this.stepParamPreparedIterations = 0;
    }

    if (!this.stepParamBuffer || this.stepParamCapacity < neededCapacity) {
      this.stepParamBuffer?.destroy?.();
      this.stepParamBuffer = this.device.createBuffer({
        size: this.stepParamStride * nextCapacity,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: 'paper-step-params',
      });
      this.stepParamCapacity = nextCapacity;
      this.stepParamTemplateDirty = true;
      this.stepParamPreparedIterations = 0;
      if (this.stateBuffers.length === 2) {
        this.#rebuildBindGroups();
      }
    }
  }

  async #createPipelines() {
    const computeModule = this.device.createShaderModule({
      code: COMPUTE_WGSL_SOURCE,
      label: 'paper-erosion-compute-wgsl',
    });

    const renderModule = this.device.createShaderModule({
      code: RENDER_WGSL_SOURCE,
      label: 'paper-erosion-render-wgsl',
    });

    const compositeModule = this.device.createShaderModule({
      code: COMPOSITE_WGSL_SOURCE,
      label: 'paper-erosion-composite-wgsl',
    });

    const createComputePipeline = typeof this.device.createComputePipelineAsync === 'function'
      ? (descriptor) => this.device.createComputePipelineAsync(descriptor)
      : async (descriptor) => this.device.createComputePipeline(descriptor);
    const createRenderPipeline = typeof this.device.createRenderPipelineAsync === 'function'
      ? (descriptor) => this.device.createRenderPipelineAsync(descriptor)
      : async (descriptor) => this.device.createRenderPipeline(descriptor);

    this.computeBindGroupLayout = this.device.createBindGroupLayout({
      label: 'paper-compute-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    this.renderBindGroupLayout = this.device.createBindGroupLayout({
      label: 'paper-render-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.compositeBindGroupLayout = this.device.createBindGroupLayout({
      label: 'paper-composite-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });

    const computeLayout = this.device.createPipelineLayout({
      label: 'paper-compute-layout',
      bindGroupLayouts: [this.computeBindGroupLayout],
    });

    const renderLayout = this.device.createPipelineLayout({
      label: 'paper-render-layout',
      bindGroupLayouts: [this.renderBindGroupLayout],
    });

    const compositeLayout = this.device.createPipelineLayout({
      label: 'paper-composite-layout',
      bindGroupLayouts: [this.compositeBindGroupLayout],
    });

    [
      this.fluxPipeline,
      this.sourcePipeline,
      this.flowPipeline,
      this.erosionPipeline,
      this.thermalOutflowPipeline,
      this.channelPipeline,
      this.channelBendPipeline,
      this.renderPipeline,
      this.waterRenderPipeline,
      this.compositePipeline,
    ] = await Promise.all([
      createComputePipeline({
        label: 'paper-flux-pipeline',
        layout: computeLayout,
        compute: { module: computeModule, entryPoint: 'fluxMain' },
      }),
      createComputePipeline({
        label: 'paper-source-pipeline',
        layout: computeLayout,
        compute: { module: computeModule, entryPoint: 'sourceMain' },
      }),
      createComputePipeline({
        label: 'paper-flow-pipeline',
        layout: computeLayout,
        compute: { module: computeModule, entryPoint: 'flowMain' },
      }),
      createComputePipeline({
        label: 'paper-erosion-pipeline',
        layout: computeLayout,
        compute: { module: computeModule, entryPoint: 'erosionMain' },
      }),
      createComputePipeline({
        label: 'paper-thermal-outflow-pipeline',
        layout: computeLayout,
        compute: { module: computeModule, entryPoint: 'thermalOutflowMain' },
      }),
      createComputePipeline({
        label: 'river-channel-guide-pipeline',
        layout: computeLayout,
        compute: { module: computeModule, entryPoint: 'channelMain' },
      }),
      createComputePipeline({
        label: 'paper-channel-bend-pipeline',
        layout: computeLayout,
        compute: { module: computeModule, entryPoint: 'channelBendMain' },
      }),
      createRenderPipeline({
        label: 'paper-render-pipeline',
        layout: renderLayout,
        vertex: { module: renderModule, entryPoint: 'vsMesh' },
        fragment: {
          module: renderModule,
          entryPoint: 'fsMesh',
          targets: [{ format: this.presentationFormat }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: true,
          depthCompare: 'less',
        },
      }),
      createRenderPipeline({
        label: 'paper-water-render-pipeline',
        layout: renderLayout,
        vertex: { module: renderModule, entryPoint: 'vsMesh' },
        fragment: {
          module: renderModule,
          entryPoint: 'fsMesh',
          targets: [{
            format: this.presentationFormat,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: false,
          depthCompare: 'less-equal',
        },
      }),
      createRenderPipeline({
        label: 'paper-composite-pipeline',
        layout: compositeLayout,
        vertex: { module: compositeModule, entryPoint: 'vsComposite' },
        fragment: {
          module: compositeModule,
          entryPoint: 'fsComposite',
          targets: [{ format: this.presentationFormat }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: false,
          depthCompare: 'always',
        },
      }),
    ]);
  }

  #rebuildBindGroups() {
    if (!this.sceneColorView || !this.sceneSampler) {
      return;
    }
    this.#invalidateRenderBundles();
    this.#ensureDummySceneTexture();
    this.#ensureCustomSpringBuffer();
    this.#ensureStepParamBuffer(1);
    if (!this.dummySceneView || !this.stepParamBuffer || !this.customSpringBuffer || !this.layerTopBuffer || !this.layerMaterialBuffer) {
      return;
    }

    this.computeBindGroup01 = this.device.createBindGroup({
      label: 'paper-compute-0to1',
      layout: this.computeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.stepParamBuffer, size: this.paramValues.byteLength } },
        { binding: 1, resource: { buffer: this.stateBuffers[0] } },
        { binding: 2, resource: { buffer: this.stateBuffers[1] } },
        { binding: 3, resource: { buffer: this.fluxBuffer } },
        { binding: 4, resource: { buffer: this.velocityBuffer } },
        { binding: 5, resource: { buffer: this.thermalPipeBufferA } },
        { binding: 6, resource: { buffer: this.thermalPipeBufferB } },
        { binding: 7, resource: { buffer: this.customSpringBuffer } },
        { binding: 8, resource: { buffer: this.layerTopBuffer } },
        { binding: 9, resource: { buffer: this.layerMaterialBuffer } },
      ],
    });

    this.computeBindGroup10 = this.device.createBindGroup({
      label: 'paper-compute-1to0',
      layout: this.computeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.stepParamBuffer, size: this.paramValues.byteLength } },
        { binding: 1, resource: { buffer: this.stateBuffers[1] } },
        { binding: 2, resource: { buffer: this.stateBuffers[0] } },
        { binding: 3, resource: { buffer: this.fluxBuffer } },
        { binding: 4, resource: { buffer: this.velocityBuffer } },
        { binding: 5, resource: { buffer: this.thermalPipeBufferA } },
        { binding: 6, resource: { buffer: this.thermalPipeBufferB } },
        { binding: 7, resource: { buffer: this.customSpringBuffer } },
        { binding: 8, resource: { buffer: this.layerTopBuffer } },
        { binding: 9, resource: { buffer: this.layerMaterialBuffer } },
      ],
    });

    this.renderBindGroupTerrain = this.device.createBindGroup({
      label: 'paper-render-terrain',
      layout: this.renderBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.renderUniformBufferTerrain } },
        { binding: 1, resource: { buffer: this.stateBuffers[0] } },
        { binding: 2, resource: this.sceneSampler },
        { binding: 3, resource: this.dummySceneView },
        { binding: 4, resource: { buffer: this.layerTopBuffer } },
        { binding: 5, resource: { buffer: this.layerMaterialBuffer } },
        { binding: 6, resource: { buffer: this.velocityBuffer } },
      ],
    });

    this.renderBindGroupWater = this.device.createBindGroup({
      label: 'paper-render-water',
      layout: this.renderBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.renderUniformBufferWater } },
        { binding: 1, resource: { buffer: this.stateBuffers[0] } },
        { binding: 2, resource: this.sceneSampler },
        { binding: 3, resource: this.sceneColorView },
        { binding: 4, resource: { buffer: this.layerTopBuffer } },
        { binding: 5, resource: { buffer: this.layerMaterialBuffer } },
        { binding: 6, resource: { buffer: this.velocityBuffer } },
      ],
    });

    this.compositeBindGroup = this.device.createBindGroup({
      label: 'paper-composite',
      layout: this.compositeBindGroupLayout,
      entries: [
        { binding: 0, resource: this.sceneSampler },
        { binding: 1, resource: this.sceneColorView },
      ],
    });
  }
}

const COMPUTE_WGSL_SOURCE = /* wgsl */ `
struct SimParams {
  dims: vec4<f32>,
  hydro0: vec4<f32>,
  hydro1: vec4<f32>,
  thermal0: vec4<f32>,
  render0: vec4<f32>,
  source0: vec4<f32>,
  source1: vec4<f32>,
  misc0: vec4<f32>,
  sourceSeed: u32,
  sourceMetersPerPixel: f32,
  hydraulicEnabled: f32,
  edgeWaterFloor: f32,
  material0: vec4<f32>,
  integration: vec4<f32>,
  banks: vec4<f32>,
  meander: vec4<f32>,
}

struct CellState {
  terrain: f32,
  water: f32,
  sediment: f32,
  hardness: f32,
  mask: f32,
  aux0: f32,
  aux1: f32,
  aux2: f32,
}

struct StateBuffer {
  cells: array<CellState>,
}

struct Vec4Buffer {
  values: array<vec4<f32>>,
}

struct FloatBuffer {
  values: array<f32>,
}

struct LayerMaterialParams {
  layers: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> srcState: StateBuffer;
@group(0) @binding(2) var<storage, read_write> dstState: StateBuffer;
@group(0) @binding(3) var<storage, read_write> fluxState: Vec4Buffer;
@group(0) @binding(4) var<storage, read_write> velocityState: Vec4Buffer;
@group(0) @binding(5) var<storage, read_write> thermalPipeA: Vec4Buffer;
@group(0) @binding(6) var<storage, read_write> thermalPipeB: Vec4Buffer;
@group(0) @binding(7) var<storage, read_write> paintedSourceState: FloatBuffer;
@group(0) @binding(8) var<storage, read> layerTopState: Vec4Buffer;
@group(0) @binding(9) var<uniform> layerMaterials: LayerMaterialParams;

fn gridWidth() -> u32 { return u32(params.dims.x); }
fn gridHeight() -> u32 { return u32(params.dims.y); }
fn timeStep() -> f32 { return params.dims.w; }
fn cellSize() -> f32 { return max(params.dims.z, 1e-6); }
fn cellArea() -> f32 { let c = cellSize(); return c * c; }
fn idx(x: u32, y: u32) -> u32 { return y * gridWidth() + x; }
fn hydraulicErosionEnabled() -> bool { return params.hydraulicEnabled > 0.5; }

fn clampCoord(v: i32, dim: u32) -> u32 {
  return u32(clamp(v, 0, max(i32(dim) - 1, 0)));
}

fn inBounds(x: i32, y: i32) -> bool {
  return x >= 0 && y >= 0 && x < i32(gridWidth()) && y < i32(gridHeight());
}

fn readStateClamped(x: i32, y: i32) -> CellState {
  return srcState.cells[idx(clampCoord(x, gridWidth()), clampCoord(y, gridHeight()))];
}

fn readFluxOrZero(x: i32, y: i32) -> vec4<f32> {
  if (!inBounds(x, y)) { return vec4<f32>(0.0); }
  return fluxState.values[idx(u32(x), u32(y))];
}

fn readVelocityClamped(x: i32, y: i32) -> vec4<f32> {
  return velocityState.values[idx(clampCoord(x, gridWidth()), clampCoord(y, gridHeight()))];
}

fn readThermalAOrZero(x: i32, y: i32) -> vec4<f32> {
  if (!inBounds(x, y)) { return vec4<f32>(0.0); }
  return thermalPipeA.values[idx(u32(x), u32(y))];
}

fn readThermalBOrZero(x: i32, y: i32) -> vec4<f32> {
  if (!inBounds(x, y)) { return vec4<f32>(0.0); }
  return thermalPipeB.values[idx(u32(x), u32(y))];
}

fn totalHeight(cell: CellState) -> f32 { return cell.terrain + cell.water; }

fn finiteOr(value: f32, fallback: f32) -> f32 {
  if (value == value && abs(value) < 1e30) { return value; }
  return fallback;
}

fn resolveMaterialInfo(i: u32, terrain: f32) -> vec4<f32> {
  let tops = layerTopState.values[i];
  var chosenIndex: u32 = 0u;
  var foundChoice = false;
  var bestAbove = 1e30;
  var highestTop = -1e30;
  var highestIndex: u32 = 0u;

  for (var layerIndex: u32 = 0u; layerIndex < 4u; layerIndex = layerIndex + 1u) {
    let material = layerMaterials.layers[layerIndex];
    if (material.z < 0.5) { continue; }
    let top = tops[layerIndex];
    if (!(top > -1e8)) { continue; }
    if (top >= highestTop) {
      highestTop = top;
      highestIndex = layerIndex;
    }
    // Later layers are deeper: coincident tops mean no upper-layer thickness.
    if (top >= terrain && top <= bestAbove) {
      bestAbove = top;
      chosenIndex = layerIndex;
      foundChoice = true;
    }
  }

  if (!foundChoice) {
    if (!(highestTop > -1e8)) {
      return vec4<f32>(-1.0, 1.0, -1.0, 0.0);
    }
    chosenIndex = highestIndex;
  }

  let chosenTop = tops[chosenIndex];
  var lowerTop = -1e30;
  for (var layerIndex: u32 = 0u; layerIndex < 4u; layerIndex = layerIndex + 1u) {
    let material = layerMaterials.layers[layerIndex];
    if (material.z < 0.5) { continue; }
    let top = tops[layerIndex];
    if (!(top > -1e8) || top >= chosenTop) { continue; }
    if (top > lowerTop) { lowerTop = top; }
  }

  var depthT = 0.0;
  if (lowerTop > -1e8) {
    depthT = clamp((chosenTop - terrain) / max(chosenTop - lowerTop, 1e-6), 0.0, 1.0);
  } else if (terrain < chosenTop) {
    depthT = clamp((chosenTop - terrain) / max(chosenTop, 1e-6), 0.0, 1.0);
  }

  let selectedMaterial = layerMaterials.layers[chosenIndex];
  return vec4<f32>(clamp(mix(selectedMaterial.x, selectedMaterial.y, depthT), 0.0, 1.0), selectedMaterial.w, f32(chosenIndex), depthT);
}

fn resolveMaterialHardness(i: u32, terrain: f32, fallbackHardness: f32) -> f32 {
  let info = resolveMaterialInfo(i, terrain);
  if (info.z < 0.0) {
    return clamp(params.material0.x + (1.0 - clamp(terrain, 0.0, 1.0)) * params.material0.y, 0.0, 1.0);
  }
  return info.x;
}

fn resolveMaterialThermalEnabled(i: u32, terrain: f32) -> bool {
  let info = resolveMaterialInfo(i, terrain);
  if (info.z < 0.0) {
    return true;
  }
  return info.y >= 0.5;
}

// Stop at the next material interface so a large cut cannot skip a resistant
// stratum. The following step resolves that newly exposed material's rates.
fn materialCutLimit(i: u32, terrain: f32) -> f32 {
  let tops = layerTopState.values[i];
  var floorHeight = 0.0;
  for (var layerIndex = 0u; layerIndex < 4u; layerIndex++) {
    if (layerMaterials.layers[layerIndex].z < 0.5) { continue; }
    let top = tops[layerIndex];
    if (top < terrain && top > floorHeight) { floorHeight = top; }
  }
  return max(terrain - floorHeight, 0.0);
}

fn bankTransfersEnabled() -> bool { return params.banks.x > 0.0 && hydraulicErosionEnabled(); }
fn readBankOrZero(x: i32, y: i32) -> vec4<f32> {
  if (!inBounds(x, y)) { return vec4<f32>(0.0); }
  return thermalPipeA.values[gridWidth() * gridHeight() + idx(u32(x), u32(y))];
}
fn directionAt(x: i32, y: i32, fallback: vec2<f32>) -> vec2<f32> {
  if (!inBounds(x, y)) { return fallback; }
  let cell = srcState.cells[idx(u32(x), u32(y))];
  let v = velocityState.values[idx(u32(x), u32(y))];
  if (cell.mask < 0.5 || cell.water < 0.0005 || v.z < 0.01) { return fallback; }
  return v.xy / max(length(v.xy), 1e-6);
}
// Empirical bank undercutting. Flow-direction change along the current gives
// the bend's outward side; straight banks receive only a small baseline rate.
// Every donor transfer is later received as suspended solid, never deleted.
fn averagedFlowDirection(x: i32, y: i32, fallback: vec2<f32>) -> vec2<f32> {
  var momentum = vec2<f32>(0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (!inBounds(x + dx, y + dy)) { continue; }
      let i = idx(u32(x + dx), u32(y + dy));
      let cell = srcState.cells[i];
      let v = velocityState.values[i];
      if (cell.mask > 0.5 && cell.water > 0.0005 && v.w > 1e-5) { momentum += v.xy * cell.water; }
    }
  }
  if (length(momentum) < 1e-6) { return fallback; }
  return normalize(momentum);
}
fn flowTurn(nx: i32, ny: i32, direction: vec2<f32>) -> vec2<f32> {
  if (params.meander.x > 0.0 && params.meander.z > 0.0) {
    let row = u32(clamp(ny, 0, i32(gridHeight()) - 1));
    return velocityState.values[gridWidth() * gridHeight() + gridHeight() + row].xy;
  }
  if (params.meander.x > 0.0) {
    // Sample several cells along the current, suppressing cell-scale bends.
    let step = vec2<i32>(round(direction * 3.0));
    return (averagedFlowDirection(nx + step.x, ny + step.y, direction)
      - averagedFlowDirection(nx - step.x, ny - step.y, direction)) / 6.0;
  }
  let dx = (directionAt(nx + 1, ny, direction) - directionAt(nx - 1, ny, direction)) * 0.5;
  let dy = (directionAt(nx, ny + 1, direction) - directionAt(nx, ny - 1, direction)) * 0.5;
  return direction.x * dx + direction.y * dy;
}
fn channelCenter(y: i32) -> f32 {
  var moment = 0.0;
  var weight = 0.0;
  for (var d = -3; d <= 3; d++) {
    let row = u32(clamp(y + d, 0, i32(gridHeight()) - 1));
    let data = velocityState.values[gridWidth() * gridHeight() + row];
    let w = data.y * f32(4 - abs(d));
    moment += data.x * w; weight += w;
  }
  return select(f32(gridWidth()) * 0.5, moment / max(weight, 1e-10), weight > 1e-8);
}
// Optional streamwise guide for north-to-south lowland rivers. It comes from
// current discharge, not an imposed curve or an elapsed-time animation.
var<workgroup> channelMoments: array<f32, 64>;
var<workgroup> channelWeights: array<f32, 64>;
@compute @workgroup_size(64)
fn channelMain(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let row = group.x;
  var moment = 0.0;
  var weight = 0.0;
  for (var x = local; x < gridWidth(); x += 64u) {
    let i = idx(x, row);
    let cell = srcState.cells[i];
    if (cell.mask < 0.5 || cell.water < 0.0005) { continue; }
    let f = fluxState.values[i];
    let discharge = max(f.w - f.z, 0.0);
    // Prefer the active channel over broad, slowly moving floodplain films.
    let q = discharge * discharge;
    moment += f32(x) * q; weight += q;
  }
  channelMoments[local] = moment; channelWeights[local] = weight;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local < stride) { channelMoments[local] += channelMoments[local + stride]; channelWeights[local] += channelWeights[local + stride]; }
    workgroupBarrier();
  }
  if (local == 0u) {
    velocityState.values[gridWidth() * gridHeight() + row] = vec4<f32>(channelMoments[0] / max(channelWeights[0], 1e-10), channelWeights[0], 0.0, 0.0);
  }
}
// Cache the smoothed bend once per row instead of repeating its 21 guide
// reads for every wet receiver and adjacent bank in the erosion pass.
@compute @workgroup_size(64)
fn channelBendMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridHeight()) { return; }
  let y = i32(gid.x);
  let distance = max(4, i32(round(f32(gridHeight()) * params.meander.z)));
  let center = channelCenter(y);
  let above = channelCenter(y - distance);
  let below = channelCenter(y + distance);
  let slope = (below - above) / f32(distance * 2);
  let curve = (below - 2.0 * center + above) / f32(distance * distance);
  let turn = vec2<f32>(1.0, -slope) * curve / pow(1.0 + slope * slope, 2.0);
  velocityState.values[gridWidth() * gridHeight() + gridHeight() + gid.x] = vec4<f32>(turn, 0.0, 0.0);
}
fn bankWeight(bank: CellState, nx: i32, ny: i32, toBank: vec2<f32>) -> f32 {
  if (!inBounds(nx, ny)) { return 0.0; }
  let receiver = srcState.cells[idx(u32(nx), u32(ny))];
  let v = velocityState.values[idx(u32(nx), u32(ny))];
  if (receiver.mask < 0.5 || receiver.water < 0.001 || v.z < 0.02 || v.w < 1e-5) { return 0.0; }
  let direction = v.xy / max(length(v.xy), 1e-6);
  let turn = flowTurn(nx, ny, direction);
  let side = pow(max(1.0 - abs(dot(direction, toBank)), 0.0), 2.0);
  let bend = clamp(-dot(turn, toBank) * params.banks.y, 0.0, 3.0);
  let relief = max(bank.terrain - receiver.terrain - 0.0005, 0.0);
  return relief * side * (params.meander.y + bend) * min(v.z, 2.0) * smoothstep(0.001, 0.008, receiver.water);
}

fn bendSedimentEnabled() -> bool { return params.meander.x > 0.0 && hydraulicErosionEnabled(); }
fn readBendSediment(x: i32, y: i32) -> vec4<f32> {
  if (!inBounds(x, y)) { return vec4<f32>(0.0); }
  return thermalPipeB.values[gridWidth() * gridHeight() + idx(u32(x), u32(y))];
}
fn sedimentReceiver(nx: i32, ny: i32, cell: CellState) -> f32 {
  if (!inBounds(nx, ny)) { return 0.0; }
  let receiver = srcState.cells[idx(u32(nx), u32(ny))];
  return select(0.0, 1.0, receiver.mask > 0.5 && receiver.terrain < cell.terrain + cell.water);
}
// Empirical transverse bed-load drift toward a bend's inner side. These
// donor-limited internal transfers help point bars deflect the current.
fn writeBendSediment(i: u32, x: i32, y: i32, cell: CellState) {
  if (!bendSedimentEnabled()) { return; }
  let offset = gridWidth() * gridHeight() + i;
  thermalPipeB.values[offset] = vec4<f32>(0.0);
  let v = velocityState.values[i];
  if (cell.mask < 0.5 || cell.water < 0.001 || v.z < 0.02 || v.w < 1e-5 || cell.sediment <= 0.0) { return; }
  let direction = v.xy / max(length(v.xy), 1e-6);
  let turn = flowTurn(x, y, direction);
  let drift = turn - direction * dot(turn, direction);
  let weights = max(vec4<f32>(-drift.x, drift.x, -drift.y, drift.y), vec4<f32>(0.0))
    * vec4<f32>(sedimentReceiver(x - 1, y, cell), sedimentReceiver(x + 1, y, cell), sedimentReceiver(x, y - 1, cell), sedimentReceiver(x, y + 1, cell));
  let sum = dot(weights, vec4<f32>(1.0));
  if (sum < 1e-8) { return; }
  let fraction = min(0.5, 1.0 - exp(-params.meander.x * min(sum * params.banks.y, 3.0) * timeStep()));
  thermalPipeB.values[offset] = weights * (cell.sediment * fraction / sum);
}

fn writeBankTransfers(i: u32, x: i32, y: i32, cell: CellState) -> f32 {
  if (!bankTransfersEnabled()) { return 0.0; }
  let offset = gridWidth() * gridHeight() + i;
  thermalPipeA.values[offset] = vec4<f32>(0.0);
  if (cell.mask < 0.5) { return 0.0; }
  let weights = vec4<f32>(bankWeight(cell, x - 1, y, vec2<f32>(1.0, 0.0)), bankWeight(cell, x + 1, y, vec2<f32>(-1.0, 0.0)),
    bankWeight(cell, x, y - 1, vec2<f32>(0.0, 1.0)), bankWeight(cell, x, y + 1, vec2<f32>(0.0, -1.0)));
  let sum = dot(weights, vec4<f32>(1.0));
  if (sum <= 1e-10) { return 0.0; }
  let erodability = resolveMaterialHardness(i, cell.terrain, cell.hardness);
  let response = 1.0 - exp(-params.banks.x * erodability * timeStep());
  let amount = min(materialCutLimit(i, cell.terrain), min(max(params.thermal0.x, 0.0), sum * response));
  thermalPipeA.values[offset] = weights * (amount / sum);
  return amount;
}

fn neighborTerrain(x: i32, y: i32, center: f32) -> f32 {
  let cell = readStateClamped(x, y);
  return select(center, finiteOr(cell.terrain, center), cell.mask > 0.5);
}

fn terrainNormal(x: i32, y: i32) -> vec3<f32> {
  let center = finiteOr(readStateClamped(x, y).terrain, 0.0);
  let tl = neighborTerrain(x - 1, y - 1, center);
  let tc = neighborTerrain(x, y - 1, center);
  let tr = neighborTerrain(x + 1, y - 1, center);
  let ml = neighborTerrain(x - 1, y, center);
  let mr = neighborTerrain(x + 1, y, center);
  let bl = neighborTerrain(x - 1, y + 1, center);
  let bc = neighborTerrain(x, y + 1, center);
  let br = neighborTerrain(x + 1, y + 1, center);
  let inv = 1.0 / max(32.0 * cellSize(), 1e-6);
  let dzdx = ((3.0 * tr + 10.0 * mr + 3.0 * br) - (3.0 * tl + 10.0 * ml + 3.0 * bl)) * inv;
  let dzdy = ((3.0 * bl + 10.0 * bc + 3.0 * br) - (3.0 * tl + 10.0 * tc + 3.0 * tr)) * inv;
  return normalize(vec3<f32>(-dzdx, 1.0, -dzdy));
}

fn terrainSinAlpha(x: i32, y: i32) -> f32 {
  let n = terrainNormal(x, y);
  return length(n.xz);
}

fn depthLimiter(water: f32) -> f32 {
  let limit = max(params.thermal0.x, 1e-6);
  if (water <= 0.0) { return 0.0; }
  if (water >= limit) { return 1.0; }
  return 1.0 - (limit - water) / limit;
}

fn sourceHash(seed: u32) -> u32 {
  var h = seed;
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  return h ^ (h >> 16u);
}
fn springHash(seed: u32) -> vec2<f32> {
  return vec2<f32>(f32(sourceHash(seed) & 0xffffffu), f32(sourceHash(seed ^ 0x9e3779b9u) & 0xffffffu)) / 16777216.0;
}

fn rainAmountAt(x: u32, y: u32) -> f32 {
  var rate = max(params.hydro0.x, 0.0);
  if (params.source1.x > 0.5) { rate += max(paintedSourceState.values[idx(x, y)], 0.0); }
  return rate;
}

fn availableWater(x: u32, y: u32) -> f32 {
  // Integrate sources and evaporation for half a step before transport.
  return max(srcState.cells[idx(x, y)].water * params.integration.y + params.integration.x * rainAmountAt(x, y), 0.0);
}

// Rebuild only when spring settings or the grid change. Painted sources reuse
// this buffer directly; CPU paint data is retained when switching layouts.
@compute @workgroup_size(8, 8)
fn sourceMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }
  let pos = vec2<f32>(gid.xy);
  let radius = max(params.source0.z, 1.0);
  let count = clamp(u32(max(params.render0.y, 1.0)), 1u, 16u);
  let perSpring = max(params.source0.w, 0.0) / f32(count);
  var rate = 0.0;
  for (var j = 0u; j < count; j++) {
    let uv = springHash(params.sourceSeed * 37u + j * 17u + 1u);
    let center = uv * vec2<f32>(f32(gridWidth() - 1u), f32(gridHeight() - 1u));
    let falloff = max(1.0 - distance(pos, center) / radius, 0.0);
    rate += perSpring * falloff * falloff;
  }
  paintedSourceState.values[idx(gid.x, gid.y)] = rate;
}

// Sediment is a solid thickness per cell. Each face exports the same
// fraction of the donor's sediment as its available water volume.
fn springSedimentRate(x: u32, y: u32) -> f32 {
  return select(0.0, max(paintedSourceState.values[idx(x, y)], 0.0) * params.banks.z, params.source1.x > 0.5);
}
fn availableSediment(x: u32, y: u32, sediment: f32) -> f32 {
  return max(sediment, 0.0) + params.integration.w * springSedimentRate(x, y);
}
fn sedimentFlux(x: i32, y: i32) -> vec4<f32> {
  if (!inBounds(x, y)) { return vec4<f32>(0.0); }
  let i = idx(u32(x), u32(y));
  let cell = srcState.cells[i];
  if (cell.mask < 0.5) { return vec4<f32>(0.0); }
  let wetDepth = availableWater(u32(x), u32(y));
  if (wetDepth <= 1e-8) { return vec4<f32>(0.0); }
  return fluxState.values[i] * (timeStep() / cellArea()) * (availableSediment(u32(x), u32(y), cell.sediment) / wetDepth);
}

fn thermalDistanceScale(dx: i32, dy: i32) -> f32 {
  if (dx == 0 || dy == 0) {
    return 1.0;
  }
  return 1.41421356;
}

@compute @workgroup_size(8, 8)
fn fluxMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }

  let i = idx(gid.x, gid.y);
  let cell = srcState.cells[i];
  if (cell.mask < 0.5) {
    fluxState.values[i] = vec4<f32>(0.0);
    velocityState.values[i] = vec4<f32>(0.0);
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let dt = timeStep();
  let rainWater = availableWater(gid.x, gid.y);
  let centerTotal = cell.terrain + rainWater;

  let leftCell = readStateClamped(x - 1, y);
  let rightCell = readStateClamped(x + 1, y);
  let topCell = readStateClamped(x, y - 1);
  let bottomCell = readStateClamped(x, y + 1);

  let edgeDrain = max(params.render0.w, 0.0);
  let retainedEdgeWater = max(params.edgeWaterFloor, 0.0);
  let outsideTotal = select(centerTotal, cell.terrain + retainedEdgeWater - edgeDrain, rainWater > retainedEdgeWater + 1e-6);
  let leftTotal = select(outsideTotal, leftCell.terrain + availableWater(u32(max(x - 1, 0)), gid.y), inBounds(x - 1, y));
  let rightTotal = select(outsideTotal, rightCell.terrain + availableWater(u32(min(x + 1, i32(gridWidth()) - 1)), gid.y), inBounds(x + 1, y));
  let topTotal = select(outsideTotal, topCell.terrain + availableWater(gid.x, u32(max(y - 1, 0))), inBounds(x, y - 1));
  let bottomTotal = select(outsideTotal, bottomCell.terrain + availableWater(gid.x, u32(min(y + 1, i32(gridHeight()) - 1))), inBounds(x, y + 1));

  let oldFlux = fluxState.values[i] * exp(-max(params.material0.z, 0.0) * dt);
  let flowScale = dt * params.hydro0.z * params.hydro0.w / cellSize();
  var nextFlux = vec4<f32>(
    max(0.0, oldFlux.x + flowScale * (centerTotal - leftTotal)),
    max(0.0, oldFlux.y + flowScale * (centerTotal - rightTotal)),
    max(0.0, oldFlux.z + flowScale * (centerTotal - topTotal)),
    max(0.0, oldFlux.w + flowScale * (centerTotal - bottomTotal))
  );

  if (inBounds(x - 1, y) && leftCell.mask < 0.5) { nextFlux.x = 0.0; }
  if (inBounds(x + 1, y) && rightCell.mask < 0.5) { nextFlux.y = 0.0; }
  if (inBounds(x, y - 1) && topCell.mask < 0.5) { nextFlux.z = 0.0; }
  if (inBounds(x, y + 1) && bottomCell.mask < 0.5) { nextFlux.w = 0.0; }
  if (params.misc0.z > 0.5) {
    if (x == 0) { nextFlux.x = 0.0; }
    if (x == i32(gridWidth()) - 1) { nextFlux.y = 0.0; }
    if (y == 0) { nextFlux.z = 0.0; }
    if (y == i32(gridHeight()) - 1) { nextFlux.w = 0.0; }
  }
  let edges = u32(params.meander.w);
  if (x == 0 && (edges & 1u) == 0u) { nextFlux.x = 0.0; }
  if (x == i32(gridWidth()) - 1 && (edges & 2u) == 0u) { nextFlux.y = 0.0; }
  if (y == 0 && (edges & 4u) == 0u) { nextFlux.z = 0.0; }
  if (y == i32(gridHeight()) - 1 && (edges & 8u) == 0u) { nextFlux.w = 0.0; }

  nextFlux = vec4<f32>(
    finiteOr(max(0.0, nextFlux.x), 0.0),
    finiteOr(max(0.0, nextFlux.y), 0.0),
    finiteOr(max(0.0, nextFlux.z), 0.0),
    finiteOr(max(0.0, nextFlux.w), 0.0)
  );
  let sumOut = finiteOr(nextFlux.x + nextFlux.y + nextFlux.z + nextFlux.w, 0.0);
  let maxOut = finiteOr(rainWater * cellArea() / max(dt, 1e-6), 0.0) * clamp(params.material0.w, 0.05, 1.0);
  if (sumOut > maxOut && sumOut > 0.0) {
    nextFlux *= maxOut / sumOut;
  }

  let maxOutsideOut = max(rainWater - retainedEdgeWater, 0.0) * cellArea() / max(dt, 1e-6);
  let outsideFluxX = select(0.0, nextFlux.x, !inBounds(x - 1, y));
  let outsideFluxY = select(0.0, nextFlux.y, !inBounds(x + 1, y));
  let outsideFluxZ = select(0.0, nextFlux.z, !inBounds(x, y - 1));
  let outsideFluxW = select(0.0, nextFlux.w, !inBounds(x, y + 1));
  let sumOutsideOut = outsideFluxX + outsideFluxY + outsideFluxZ + outsideFluxW;
  if (sumOutsideOut > maxOutsideOut && sumOutsideOut > 0.0) {
    let outsideScale = maxOutsideOut / sumOutsideOut;
    if (!inBounds(x - 1, y)) { nextFlux.x *= outsideScale; }
    if (!inBounds(x + 1, y)) { nextFlux.y *= outsideScale; }
    if (!inBounds(x, y - 1)) { nextFlux.z *= outsideScale; }
    if (!inBounds(x, y + 1)) { nextFlux.w *= outsideScale; }
  }
  fluxState.values[i] = vec4<f32>(
    finiteOr(nextFlux.x, 0.0),
    finiteOr(nextFlux.y, 0.0),
    finiteOr(nextFlux.z, 0.0),
    finiteOr(nextFlux.w, 0.0)
  );
}

@compute @workgroup_size(8, 8)
fn flowMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }

  let i = idx(gid.x, gid.y);
  let cell = srcState.cells[i];
  if (cell.mask < 0.5) {
    dstState.cells[i] = CellState(0.0, 0.0, 0.0, cell.hardness, 0.0, 0.0, 0.0, 0.0);
    velocityState.values[i] = vec4<f32>(0.0);
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let dt = timeStep();
  let rainWater = availableWater(gid.x, gid.y);

  let localFlux = fluxState.values[i];
  let inflow =
    readFluxOrZero(x - 1, y).y +
    readFluxOrZero(x + 1, y).x +
    readFluxOrZero(x, y - 1).w +
    readFluxOrZero(x, y + 1).z;

  let outflow = localFlux.x + localFlux.y + localFlux.z + localFlux.w;
  let deltaVolume = dt * (inflow - outflow);
  let water = max(0.0, rainWater + deltaVolume / cellArea());

  let deltaWx = 0.5 * (
    readFluxOrZero(x - 1, y).y - localFlux.x +
    localFlux.y - readFluxOrZero(x + 1, y).x
  );
  let deltaWy = 0.5 * (
    readFluxOrZero(x, y - 1).w - localFlux.z +
    localFlux.w - readFluxOrZero(x, y + 1).z
  );
  let avgWater = max(0.5 * (rainWater + water), 1e-4);
  var vel = vec2<f32>(deltaWx, deltaWy) / max(cellSize() * avgWater, 1e-4);
  vel = vec2<f32>(finiteOr(vel.x, 0.0), finiteOr(vel.y, 0.0));
  var speed = finiteOr(length(vel), 0.0);
  let maxVelocity = max(params.misc0.w, 0.05);
  if (speed > maxVelocity && speed > 1e-6) {
    vel *= maxVelocity / speed;
    speed = maxVelocity;
  }
  velocityState.values[i] = vec4<f32>(finiteOr(vel.x, 0.0), finiteOr(vel.y, 0.0), finiteOr(speed, 0.0), clamp(dt * outflow / max(cellArea() * rainWater, 1e-8), 0.0, 1.0));

  let selfSedimentFlux = sedimentFlux(x, y);
  let sedimentOut = dot(selfSedimentFlux, vec4<f32>(1.0));
  let sedimentIn = sedimentFlux(x - 1, y).y + sedimentFlux(x + 1, y).x
    + sedimentFlux(x, y - 1).w + sedimentFlux(x, y + 1).z;
  let sediment = max(0.0, availableSediment(gid.x, gid.y, cell.sediment) - sedimentOut + sedimentIn);
  let activeHardness = resolveMaterialHardness(i, finiteOr(cell.terrain, 0.0), finiteOr(cell.hardness, 0.1));
  dstState.cells[i] = CellState(finiteOr(cell.terrain, 0.0), finiteOr(water, 0.0), finiteOr(sediment, 0.0), activeHardness, cell.mask, 0.0, 0.0, finiteOr(cell.aux2, 0.0));
}

// Exchange, evaporation, settling and thermal application share one state write.
@compute @workgroup_size(8, 8)
fn erosionMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }
  let i = idx(gid.x, gid.y);
  let cell = srcState.cells[i];
  if (cell.mask < 0.5) { dstState.cells[i] = cell; return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let dt = timeStep();
  let velocity = velocityState.values[i];
  let speed = max(finiteOr(velocity.z, 0.0), 0.0);
  var terrain = finiteOr(cell.terrain, 0.0);
  var sediment = max(finiteOr(cell.sediment, 0.0), 0.0) + params.banks.w * springSedimentRate(gid.x, gid.y);
  // Finish the source/evaporation split after transport. Sediment is unaffected.
  var water = max(finiteOr(cell.water, 0.0), 0.0) * params.integration.y
    + params.integration.z * rainAmountAt(gid.x, gid.y);
  var erodability = resolveMaterialHardness(i, terrain, cell.hardness);
  var history = finiteOr(cell.aux2, 0.0) * pow(clamp(params.misc0.x, 0.0, 1.0), dt / 0.02);
  var selfOut = 0.0;
  var incoming = 0.0;
  var bankOut = 0.0;
  if (bendSedimentEnabled()) {
    sediment = max(sediment - dot(readBendSediment(x, y), vec4<f32>(1.0)), 0.0)
      + readBendSediment(x - 1, y).y + readBendSediment(x + 1, y).x
      + readBendSediment(x, y - 1).w + readBendSediment(x, y + 1).z;
  }
  if (bankTransfersEnabled()) {
    bankOut = dot(readBankOrZero(x, y), vec4<f32>(1.0));
    sediment += readBankOrZero(x - 1, y).y + readBankOrZero(x + 1, y).x
      + readBankOrZero(x, y - 1).w + readBankOrZero(x, y + 1).z;
    history -= bankOut * 240.0;
  }
  if (params.thermal0.y > 0.0) {
    selfOut = dot(thermalPipeA.values[i] + thermalPipeB.values[i], vec4<f32>(1.0));
    incoming = readThermalAOrZero(x - 1, y).y + readThermalAOrZero(x + 1, y).x
      + readThermalAOrZero(x, y - 1).w + readThermalAOrZero(x, y + 1).z
      + readThermalBOrZero(x - 1, y - 1).w + readThermalBOrZero(x + 1, y - 1).z
      + readThermalBOrZero(x - 1, y + 1).y + readThermalBOrZero(x + 1, y + 1).x;
  }
  var capacity = 0.0;
  if (hydraulicErosionEnabled()) {
    // A tangent/normal collision dot product is identically zero. Use the
    // actual terrain grade and require outgoing water before entrainment.
    let slope = terrainSinAlpha(x, y);
    let movingWater = select(0.0, 1.0, velocity.w > 1e-6 && water > 1e-6);
    capacity = max(params.hydro1.x, 0.0) * slope * speed * depthLimiter(water) * movingWater;
    if (capacity > sediment) {
      // Recent deposition is a proxy for loose soil above the active material.
      let looseSoil = clamp(erodability * (1.0 + max(cell.aux2, 0.0) * max(params.hydro1.w, 0.0)), 0.0, 1.0);
      let response = 1.0 - exp(-max(params.hydro1.y, 0.0) * looseSoil * dt);
      var downhillFloor = terrain;
      let left = readStateClamped(x - 1, y);
      let right = readStateClamped(x + 1, y);
      let top = readStateClamped(x, y - 1);
      let bottom = readStateClamped(x, y + 1);
      if (left.mask > 0.5) { downhillFloor = min(downhillFloor, left.terrain); }
      if (right.mask > 0.5) { downhillFloor = min(downhillFloor, right.terrain); }
      if (top.mask > 0.5) { downhillFloor = min(downhillFloor, top.terrain); }
      if (bottom.mask > 0.5) { downhillFloor = min(downhillFloor, bottom.terrain); }
      var cutLimit = max(terrain - downhillFloor, 0.0) * 0.5;
      // An open edge has an outlet beyond the sampled height field.
      if (params.misc0.z < 0.5 && (x == 0 || y == 0 || x == i32(gridWidth()) - 1 || y == i32(gridHeight()) - 1)) {
        cutLimit = max(params.thermal0.x, 0.0);
      }
      // Reserve the soil already committed to talus transfer and limit each
      // cut by water depth and the user-selected maximum cut depth.
      let eroded = min((capacity - sediment) * response,
        min(max(materialCutLimit(i, terrain) - selfOut - bankOut, 0.0), min(cutLimit, min(water, max(params.thermal0.x, 0.0)))));
      terrain -= eroded;
      sediment += eroded;
      history -= eroded * 240.0;
    } else {
      let deposited = (sediment - capacity) * (1.0 - exp(-max(params.hydro1.z, 0.0) * dt));
      terrain += deposited;
      sediment -= deposited;
      history += deposited * 180.0;
    }
  }
  // A dry cell cannot carry suspended solid, even with hydraulic erosion off.
  if (water <= 1e-6) {
    terrain += sediment;
    history += sediment * 180.0;
    sediment = 0.0;
  }
  terrain = terrain - selfOut - bankOut + incoming;
  erodability = resolveMaterialHardness(i, terrain, erodability);
  // No upper clamps: clipping terrain, water or sediment destroys mass.
  dstState.cells[i] = CellState(finiteOr(terrain, cell.terrain), finiteOr(water, 0.0),
    finiteOr(sediment, 0.0), erodability, cell.mask, selfOut, incoming, clamp(history, -1.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn thermalOutflowMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }

  let i = idx(gid.x, gid.y);
  let cell = srcState.cells[i];
  let bankOut = writeBankTransfers(i, i32(gid.x), i32(gid.y), cell);
  writeBendSediment(i, i32(gid.x), i32(gid.y), cell);
  if (cell.mask < 0.5) {
    thermalPipeA.values[i] = vec4<f32>(0.0);
    thermalPipeB.values[i] = vec4<f32>(0.0);
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let activeHardness = resolveMaterialHardness(i, finiteOr(cell.terrain, 0.0), finiteOr(cell.hardness, params.misc0.y));
  let thermalAllowed = resolveMaterialThermalEnabled(i, finiteOr(cell.terrain, 0.0));
  if (!thermalAllowed || params.thermal0.y <= 1e-6) {
    thermalPipeA.values[i] = vec4<f32>(0.0);
    thermalPipeB.values[i] = vec4<f32>(0.0);
    return;
  }
  let softness = clamp(activeHardness, 0.0, 1.0);
  let thresholdBase = cellSize() * max(0.0, params.thermal0.w + params.thermal0.z * (1.0 - softness));

  var rawCard = vec4<f32>(0.0);
  var rawDiag = vec4<f32>(0.0);
  var weightSum = 0.0;
  var maxExcess = 0.0;

  let left = readStateClamped(x - 1, y);
  let dropL = cell.terrain - left.terrain;
  let thrL = thresholdBase * thermalDistanceScale(-1, 0);
  let excessL = max(0.0, dropL - thrL);
  if (inBounds(x - 1, y) && left.mask > 0.5 && excessL > 0.0) { rawCard.x = excessL; weightSum += excessL; maxExcess = max(maxExcess, excessL); }

  let right = readStateClamped(x + 1, y);
  let dropR = cell.terrain - right.terrain;
  let thrR = thresholdBase * thermalDistanceScale(1, 0);
  let excessR = max(0.0, dropR - thrR);
  if (inBounds(x + 1, y) && right.mask > 0.5 && excessR > 0.0) { rawCard.y = excessR; weightSum += excessR; maxExcess = max(maxExcess, excessR); }

  let top = readStateClamped(x, y - 1);
  let dropT = cell.terrain - top.terrain;
  let thrT = thresholdBase * thermalDistanceScale(0, -1);
  let excessT = max(0.0, dropT - thrT);
  if (inBounds(x, y - 1) && top.mask > 0.5 && excessT > 0.0) { rawCard.z = excessT; weightSum += excessT; maxExcess = max(maxExcess, excessT); }

  let bottom = readStateClamped(x, y + 1);
  let dropB = cell.terrain - bottom.terrain;
  let thrB = thresholdBase * thermalDistanceScale(0, 1);
  let excessB = max(0.0, dropB - thrB);
  if (inBounds(x, y + 1) && bottom.mask > 0.5 && excessB > 0.0) { rawCard.w = excessB; weightSum += excessB; maxExcess = max(maxExcess, excessB); }

  let tl = readStateClamped(x - 1, y - 1);
  let dropTL = cell.terrain - tl.terrain;
  let thrTL = thresholdBase * thermalDistanceScale(-1, -1);
  let excessTL = max(0.0, dropTL - thrTL);
  if (inBounds(x - 1, y - 1) && tl.mask > 0.5 && excessTL > 0.0) { rawDiag.x = excessTL; weightSum += excessTL; maxExcess = max(maxExcess, excessTL); }

  let tr = readStateClamped(x + 1, y - 1);
  let dropTR = cell.terrain - tr.terrain;
  let thrTR = thresholdBase * thermalDistanceScale(1, -1);
  let excessTR = max(0.0, dropTR - thrTR);
  if (inBounds(x + 1, y - 1) && tr.mask > 0.5 && excessTR > 0.0) { rawDiag.y = excessTR; weightSum += excessTR; maxExcess = max(maxExcess, excessTR); }

  let bl = readStateClamped(x - 1, y + 1);
  let dropBL = cell.terrain - bl.terrain;
  let thrBL = thresholdBase * thermalDistanceScale(-1, 1);
  let excessBL = max(0.0, dropBL - thrBL);
  if (inBounds(x - 1, y + 1) && bl.mask > 0.5 && excessBL > 0.0) { rawDiag.z = excessBL; weightSum += excessBL; maxExcess = max(maxExcess, excessBL); }

  let br = readStateClamped(x + 1, y + 1);
  let dropBR = cell.terrain - br.terrain;
  let thrBR = thresholdBase * thermalDistanceScale(1, 1);
  let excessBR = max(0.0, dropBR - thrBR);
  if (inBounds(x + 1, y + 1) && br.mask > 0.5 && excessBR > 0.0) { rawDiag.w = excessBR; weightSum += excessBR; maxExcess = max(maxExcess, excessBR); }

  if (weightSum <= 1e-6 || maxExcess <= 1e-6) {
    thermalPipeA.values[i] = vec4<f32>(0.0);
    thermalPipeB.values[i] = vec4<f32>(0.0);
    return;
  }

  let response = 1.0 - exp(-max(params.thermal0.y, 0.0) * softness * timeStep());
  let totalOut = min(max(materialCutLimit(i, cell.terrain) - bankOut, 0.0), 0.5 * maxExcess * response);
  thermalPipeA.values[i] = rawCard * (totalOut / weightSum);
  thermalPipeB.values[i] = rawDiag * (totalOut / weightSum);
}
`;

const COMPOSITE_WGSL_SOURCE = /* wgsl */ `
@group(0) @binding(0) var sceneSampler: sampler;
@group(0) @binding(1) var sceneTexture: texture_2d<f32>;

struct CompositeOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vsComposite(@builtin(vertex_index) vertexIndex: u32) -> CompositeOut {
  var out: CompositeOut;
  let pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(3.0, 1.0)
  );
  let uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 2.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(2.0, 0.0)
  );
  out.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
  out.uv = uv[vertexIndex];
  return out;
}

@fragment
fn fsComposite(in: CompositeOut) -> @location(0) vec4<f32> {
  return textureSampleLevel(sceneTexture, sceneSampler, clamp(in.uv, vec2<f32>(0.001), vec2<f32>(0.999)), 0.0);
}
`;

// Optional post-processing renderers share the same terrain/material shading.
export function getTerrainRenderShader() { return RENDER_WGSL_SOURCE; }

const RENDER_WGSL_SOURCE = /* wgsl */ `
struct RenderParams {
  viewProj: mat4x4<f32>,
  dims: vec4<f32>,
  shading: vec4<f32>,
  misc: vec4<f32>,
  lightDir: vec4<f32>,
  cameraPos: vec4<f32>,
  timeData: vec4<f32>,
}

struct CellState {
  terrain: f32,
  water: f32,
  sediment: f32,
  hardness: f32,
  mask: f32,
  aux0: f32,
  aux1: f32,
  aux2: f32,
}

struct StateBuffer {
  cells: array<CellState>,
}

fn finiteOr(value: f32, fallback: f32) -> f32 {
  if (abs(value) < 1e30) { return value; }
  return fallback;
}

struct RenderVertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) terrain: f32,
  @location(2) water: f32,
  @location(3) sediment: f32,
  @location(4) hardness: f32,
  @location(5) thermal: f32,
  @location(6) history: f32,
  @location(7) mask: f32,
  @location(8) worldPos: vec3<f32>,
  @location(9) gridPos: vec2<f32>,
  @location(10) materialColor: vec3<f32>,
  @location(11) cavity: f32,
  @location(12) flow: vec3<f32>,
}

@group(0) @binding(0) var<uniform> renderParams: RenderParams;
@group(0) @binding(1) var<storage, read> renderState: StateBuffer;
@group(0) @binding(2) var sceneSampler: sampler;
@group(0) @binding(3) var sceneTexture: texture_2d<f32>;
struct LayerTops { values: array<vec4<f32>> }
struct LayerMaterials { layers: array<vec4<f32>, 4> }
@group(0) @binding(4) var<storage, read> renderLayerTops: LayerTops;
@group(0) @binding(5) var<uniform> renderLayerMaterials: LayerMaterials;
struct Velocities { values: array<vec4<f32>> }
@group(0) @binding(6) var<storage, read> renderVelocity: Velocities;

fn exposedMaterialColor(i: u32, terrain: f32) -> vec3<f32> {
  if (renderParams.cameraPos.w < 0.5) { return vec3<f32>(0.51, 0.45, 0.34); }
  let tops = renderLayerTops.values[i];
  var chosen = 0u;
  var bestAbove = 1e30;
  var highest = -1e30;
  var highestIndex = 0u;
  var found = false;
  for (var layer = 0u; layer < 4u; layer++) {
    if (renderLayerMaterials.layers[layer].z < 0.5 || tops[layer] < -1e8) { continue; }
    if (tops[layer] >= highest) { highest = tops[layer]; highestIndex = layer; }
    if (tops[layer] >= terrain && tops[layer] <= bestAbove) {
      bestAbove = tops[layer]; chosen = layer; found = true;
    }
  }
  if (!found) { chosen = highestIndex; }
  switch chosen {
    case 0u: { return vec3<f32>(0.61, 0.54, 0.40); }
    case 1u: { return vec3<f32>(0.52, 0.40, 0.28); }
    case 2u: { return vec3<f32>(0.43, 0.44, 0.40); }
    default: { return vec3<f32>(0.29, 0.31, 0.30); }
  }
}

fn idx(x: u32, y: u32) -> u32 {
  return y * u32(renderParams.dims.x) + x;
}

fn renderStateAt(x: i32, y: i32) -> CellState {
  let maxX = max(i32(renderParams.dims.x) - 1, 0);
  let maxY = max(i32(renderParams.dims.y) - 1, 0);
  let cx = clamp(x, 0, maxX);
  let cy = clamp(y, 0, maxY);
  return renderState.cells[idx(u32(cx), u32(cy))];
}

fn renderTerrainHeightAt(x: i32, y: i32) -> f32 {
  return max(finiteOr(renderStateAt(x, y).terrain, 0.0), 0.0) * renderParams.dims.w;
}

fn renderWaterDepthAt(x: i32, y: i32) -> f32 {
  return max(finiteOr(renderStateAt(x, y).water, 0.0), 0.0) * renderParams.misc.y;
}

fn renderRelief(x: i32, y: i32, centerHeight: f32) -> vec4<f32> {
  let worldScale = max(renderParams.misc.x, 1e-5);
  let tl = renderTerrainHeightAt(x - 1, y - 1);
  let tc = renderTerrainHeightAt(x, y - 1);
  let tr = renderTerrainHeightAt(x + 1, y - 1);
  let ml = renderTerrainHeightAt(x - 1, y);
  let mr = renderTerrainHeightAt(x + 1, y);
  let bl = renderTerrainHeightAt(x - 1, y + 1);
  let bc = renderTerrainHeightAt(x, y + 1);
  let br = renderTerrainHeightAt(x + 1, y + 1);
  let inv = 1.0 / max(32.0 * worldScale, 1e-5);
  let dzdx = ((3.0 * tr + 10.0 * mr + 3.0 * br) - (3.0 * tl + 10.0 * ml + 3.0 * bl)) * inv;
  let dzdz = ((3.0 * tl + 10.0 * tc + 3.0 * tr) - (3.0 * bl + 10.0 * bc + 3.0 * br)) * inv;
  let cavity = clamp((ml + mr + tc + bc - 4.0 * centerHeight) / worldScale * 1.8, -1.0, 1.0);
  return vec4<f32>(normalize(vec3<f32>(-dzdx, 1.0, -dzdz)), cavity);
}

// Display the solver's water surface. Smoothing depth across unequal terrain
// creates artificial waves in a resting lake and spills water onto dry cells.
fn renderWaterSurfaceAt(x: i32, y: i32) -> f32 {
  return renderTerrainHeightAt(x, y) + renderWaterDepthAt(x, y);
}

fn cornerCoord(localVertex: u32) -> vec2<u32> {
  switch localVertex {
    case 0u: { return vec2<u32>(0u, 0u); }
    case 1u: { return vec2<u32>(1u, 0u); }
    case 2u: { return vec2<u32>(0u, 1u); }
    case 3u: { return vec2<u32>(0u, 1u); }
    case 4u: { return vec2<u32>(1u, 0u); }
    default: { return vec2<u32>(1u, 1u); }
  }
}

@vertex
fn vsMesh(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> RenderVertexOut {
  var out: RenderVertexOut;
  let width = u32(renderParams.dims.x);
  let height = u32(renderParams.dims.y);
  let meshStep = max(u32(renderParams.lightDir.w), 1u);
  let cellsWide = max((width - 1u + meshStep - 1u) / meshStep, 1u);
  let cellX = (instanceIndex % cellsWide) * meshStep;
  let cellY = (instanceIndex / cellsWide) * meshStep;
  if (cellY >= max(height - 1u, 1u)) {
    out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
    out.normal = vec3<f32>(0.0, 1.0, 0.0);
    out.terrain = 0.0;
    out.water = 0.0;
    out.sediment = 0.0;
    out.hardness = 0.0;
    out.thermal = 0.0;
    out.history = 0.0;
    out.mask = 0.0;
    out.worldPos = vec3<f32>(0.0);
    out.gridPos = vec2<f32>(0.0);
    out.materialColor = vec3<f32>(0.0);
    out.cavity = 0.0;
    return out;
  }

  let corner = cornerCoord(vertexIndex % 6u);
  let gx = min(cellX + corner.x * meshStep, width - 1u);
  let gy = min(cellY + corner.y * meshStep, height - 1u);
  let cell = renderState.cells[idx(gx, gy)];
  let halfW = (renderParams.dims.x - 1.0) * 0.5;
  let halfH = (renderParams.dims.y - 1.0) * 0.5;
  let worldScale = renderParams.misc.x;
  let worldX = (f32(gx) - halfW) * worldScale;
  let worldZ = (halfH - f32(gy)) * worldScale;
  let terrainY = renderTerrainHeightAt(i32(gx), i32(gy));
  let waterSurfaceY = renderWaterSurfaceAt(i32(gx), i32(gy));
  let waterY = max(waterSurfaceY - terrainY, 0.0);
  let waterPass = renderParams.misc.z > 0.5;
  let worldY = select(terrainY, waterSurfaceY, waterPass);

  out.position = renderParams.viewProj * vec4<f32>(worldX, worldY, worldZ, 1.0);
  out.normal = vec3<f32>(0.0, 1.0, 0.0);
  out.cavity = 0.0;
  out.materialColor = vec3<f32>(0.51, 0.45, 0.34);
  if (!waterPass) {
    let relief = renderRelief(i32(gx), i32(gy), terrainY);
    out.normal = relief.xyz;
    out.cavity = relief.w;
    out.materialColor = exposedMaterialColor(idx(gx, gy), cell.terrain);
  }
  out.terrain = finiteOr(cell.terrain, 0.0);
  out.water = finiteOr(waterY / max(renderParams.misc.y, 1e-6), 0.0);
  out.sediment = finiteOr(cell.sediment, 0.0);
  out.hardness = finiteOr(cell.hardness, 0.1);
  out.thermal = clamp(pow(max(finiteOr((cell.aux0 + cell.aux1) * renderParams.shading.w, 0.0), 0.0), 0.45), 0.0, 1.0);
  out.history = clamp(finiteOr(cell.aux2, 0.0), -1.0, 1.0);
  out.mask = cell.mask;
  out.worldPos = vec3<f32>(worldX, worldY, worldZ);
  out.gridPos = vec2<f32>(f32(gx), f32(gy));
  out.flow = vec3<f32>(0.0);
  if (waterPass && renderParams.shading.z == 9.0) {
    let v = renderVelocity.values[idx(gx, gy)];
    out.flow = vec3<f32>(v.xy, v.w);
  }
  return out;
}

@fragment
fn fsMesh(in: RenderVertexOut) -> @location(0) vec4<f32> {
  // Derivatives of the displayed surface avoid repeated per-pixel storage
  // reads and keep reflections consistent with the actual water geometry.
  let faceNormal = cross(dpdx(in.worldPos), dpdy(in.worldPos));
  let faceLength = max(length(faceNormal), 1e-12);
  let waterNormal = faceNormal / faceLength * select(-1.0, 1.0, faceNormal.y >= 0.0);
  if (in.mask < 0.5) {
    return vec4<f32>(0.03, 0.04, 0.07, 1.0);
  }

  let waterPass = renderParams.misc.z > 0.5;
  let terrainNormal = normalize(mix(in.normal, waterNormal, 0.2));
  let lambert = clamp(dot(terrainNormal, normalize(renderParams.lightDir.xyz)), 0.0, 1.0);
  let hemi = 0.80 + 0.20 * clamp(terrainNormal.y * 0.5 + 0.5, 0.0, 1.0);

  let h = clamp(in.terrain, 0.0, 1.0);
  var topo = mix(vec3<f32>(0.05, 0.18, 0.50), vec3<f32>(0.10, 0.68, 0.82), smoothstep(0.00, 0.20, h));
  topo = mix(topo, vec3<f32>(0.18, 0.74, 0.34), smoothstep(0.16, 0.36, h));
  topo = mix(topo, vec3<f32>(0.88, 0.84, 0.20), smoothstep(0.34, 0.58, h));
  topo = mix(topo, vec3<f32>(0.90, 0.62, 0.18), smoothstep(0.56, 0.76, h));
  topo = mix(topo, vec3<f32>(0.95, 0.94, 0.90), smoothstep(0.78, 1.00, h));
  let litTopo = topo * (0.72 + 0.20 * lambert + 0.16 * hemi);
  let contour = 0.92 + 0.08 * abs(sin(h * 44.0));
  let terrainColor = clamp(litTopo * contour, vec3<f32>(0.0), vec3<f32>(1.0));

  let shorelineFade = smoothstep(0.0002, 0.004, in.water);
  // Depth-dependent absorption keeps shallow rivers visible while leaving
  // thin wet films and dry shore cells transparent.
  let waterAlpha = clamp((1.0 - exp(-in.water * max(renderParams.shading.x, 0.0) * 96.0)) * shorelineFade, 0.0, 0.97);
  let mode = i32(round(renderParams.shading.z));

  if (waterPass) {
    if (mode == 9) {
      let wet = smoothstep(0.0002, 0.0015, in.water);
      let speed = length(in.flow.xy);
      let direction = in.flow.xy / max(speed, 1e-6);
      let tile = fract(in.gridPos / 6.0) - 0.5;
      let along = dot(tile, direction);
      let across = abs(dot(tile, vec2<f32>(-direction.y, direction.x)));
      let stem = select(0.0, 1.0, along > -0.28 && along < 0.13 && across < 0.028);
      let head = select(0.0, 1.0, along > 0.02 && along < 0.29 && across < (0.29 - along) * 0.7);
      let arrow = max(stem, head) * smoothstep(0.02, 0.15, speed) * smoothstep(1e-5, 0.001, in.flow.z);
      let flowColor = mix(vec3<f32>(0.025, 0.22, 0.34), vec3<f32>(0.24, 0.60, 0.72), clamp(speed * 0.25, 0.0, 1.0));
      return vec4<f32>(mix(flowColor, vec3<f32>(0.88, 0.99, 1.0), arrow), wet * 0.88);
    }
    if (waterAlpha <= max(renderParams.misc.w, 0.01)) {
      return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let w = clamp(in.water * 2.2, 0.0, 1.0);
    let s = clamp(in.sediment * renderParams.shading.y * 2.7, 0.0, 1.0);
    let viewDir = normalize(renderParams.cameraPos.xyz - in.worldPos);
    let lightDir = normalize(renderParams.lightDir.xyz);
    let phase = renderParams.timeData.x;
    let ripple = vec2<f32>(sin(in.worldPos.x * 95.0 + phase * 1.7), cos(in.worldPos.z * 87.0 - phase * 1.3)) * 0.008;
    let combinedNormal = normalize(waterNormal + vec3<f32>(ripple.x, 0.0, ripple.y));
    let ndv = clamp(dot(combinedNormal, viewDir), 0.0, 1.0);
    let fresnel = pow(1.0 - ndv, 3.8);
    let sunFacing = clamp(dot(combinedNormal, lightDir), 0.0, 1.0);
    let halfDir = normalize(lightDir + viewDir);
    let sunSpecTight = pow(clamp(dot(combinedNormal, halfDir), 0.0, 1.0), 54.0);
    let sunSpecBroad = pow(clamp(dot(combinedNormal, halfDir), 0.0, 1.0), 16.0);
    let rippleGlow = pow(clamp(1.0 - combinedNormal.y, 0.0, 1.0), 0.75);

    if (mode == 3) {
      let deepTint = mix(vec3<f32>(0.002, 0.008, 0.016), vec3<f32>(0.006, 0.024, 0.040), w);
      let shallowTint = mix(vec3<f32>(0.008, 0.030, 0.050), vec3<f32>(0.018, 0.060, 0.080), w);
      let mutedWater = mix(deepTint, shallowTint, clamp(0.22 + 0.48 * w, 0.0, 1.0));
      let suspendedSedimentGlow = mix(vec3<f32>(0.16, 0.12, 0.05), vec3<f32>(0.52, 0.36, 0.10), s);
      var waterColorOnly = mutedWater * (0.82 + 0.06 * sunFacing);
      waterColorOnly += suspendedSedimentGlow * (0.05 + 0.12 * s);
      waterColorOnly += vec3<f32>(0.000, 0.010, 0.018) * fresnel;
      let finalAlpha = clamp(waterAlpha * (0.18 + 0.14 * w), 0.0, 0.32);
      return vec4<f32>(clamp(waterColorOnly, vec3<f32>(0.0), vec3<f32>(1.0)), finalAlpha);
    }

    let deepTint = mix(vec3<f32>(0.003, 0.018, 0.060), vec3<f32>(0.008, 0.050, 0.135), w);
    let shallowTint = mix(vec3<f32>(0.015, 0.120, 0.220), vec3<f32>(0.045, 0.280, 0.380), w);
    let waterBase = mix(deepTint, shallowTint, clamp(0.20 + 0.58 * w, 0.0, 1.0));
    var waterColorOnly = waterBase * (0.78 + 0.12 * sunFacing);
    waterColorOnly += vec3<f32>(0.010, 0.095, 0.145) * (0.24 + 0.76 * w);
    waterColorOnly += vec3<f32>(0.020, 0.220, 0.280) * rippleGlow * (0.18 + 0.42 * w);
    waterColorOnly += vec3<f32>(1.30, 1.18, 0.96) * sunSpecTight * (0.10 + 0.32 * w);
    waterColorOnly += vec3<f32>(0.18, 0.16, 0.13) * sunSpecBroad * (0.06 + 0.12 * w);
    waterColorOnly += vec3<f32>(0.000, 0.028, 0.050) * fresnel;
    let finalAlpha = clamp(waterAlpha * (0.95 + 0.05 * fresnel), 0.0, 0.97);
    return vec4<f32>(clamp(waterColorOnly, vec3<f32>(0.0), vec3<f32>(1.0)), finalAlpha);
  }

  if (mode == 0) {
    return vec4<f32>(terrainColor, 1.0);
  }
  if (mode == 2) {
    let w = clamp(in.water * 1.8, 0.0, 1.0);
    let abyss = mix(vec3<f32>(0.004, 0.012, 0.026), vec3<f32>(0.010, 0.030, 0.052), h);
    let shelf = mix(vec3<f32>(0.018, 0.040, 0.060), vec3<f32>(0.050, 0.100, 0.110), smoothstep(0.02, 0.32, h));
    let seabed = mix(abyss, shelf, smoothstep(0.0, 0.65, w));
    let wetSediment = mix(vec3<f32>(0.08, 0.22, 0.18), vec3<f32>(0.24, 0.52, 0.30), clamp(in.sediment * 2.2, 0.0, 1.0));
    let seabedLit = mix(seabed, wetSediment, clamp(in.sediment * 0.22 + (1.0 - w) * 0.08, 0.0, 0.32));
    return vec4<f32>(clamp(seabedLit * (0.78 + 0.10 * lambert + 0.12 * hemi), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
  }
  if (mode == 3) {
    let s = clamp(in.sediment * renderParams.shading.y * 2.7, 0.0, 1.0);
    let matteLambert = 0.80 + 0.18 * lambert + 0.14 * hemi;
    let sedimentBase = mix(vec3<f32>(0.18, 0.14, 0.10), vec3<f32>(1.00, 0.86, 0.46), s);
    var sedimentColor = sedimentBase * matteLambert;
    sedimentColor = mix(sedimentColor, vec3<f32>(1.00, 0.94, 0.70), s * 0.42);
    sedimentColor += vec3<f32>(0.04, 0.06, 0.03) * pow(s, 0.72);
    return vec4<f32>(clamp(sedimentColor, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
  }
  if (mode == 4) {
    return vec4<f32>(mix(vec3<f32>(0.25, 0.00, 0.45), vec3<f32>(0.95, 0.95, 1.00), clamp(in.hardness, 0.0, 1.0)), 1.0);
  }
  if (mode == 5) {
    return vec4<f32>(mix(vec3<f32>(0.08, 0.05, 0.18), vec3<f32>(1.00, 0.88, 0.05), in.thermal), 1.0);
  }
  if (mode == 6) {
    let signedHistory = clamp(in.history, -1.0, 1.0);
    let mag = pow(abs(signedHistory), 0.55);
    let erosionColor = vec3<f32>(1.00, 0.08, 0.05);
    let depositionColor = vec3<f32>(0.08, 0.95, 0.20);
    let diagColor = select(mix(terrainColor * 0.18, depositionColor, mag), mix(terrainColor * 0.18, erosionColor, mag), signedHistory < 0.0);
    return vec4<f32>(diagColor, 1.0);
  }
  let slope = length(terrainNormal.xz);
  let steepness = smoothstep(0.1, 0.88, slope);
  let reliefLight = (0.24 + 0.64 * lambert + 0.12 * hemi) * (1.0 - 0.2 * max(in.cavity, 0.0) + 0.05 * max(-in.cavity, 0.0));
  if (mode == 8 || mode == 9) {
    let slopeColor = mix(mix(vec3<f32>(0.38, 0.47, 0.29), vec3<f32>(0.72, 0.57, 0.34), smoothstep(0.05, 0.55, slope)), vec3<f32>(0.45, 0.40, 0.37), smoothstep(0.55, 0.93, slope));
    return vec4<f32>(slopeColor * reliefLight, 1.0);
  }
  if (mode == 7) {
    let erosion = smoothstep(0.03, 0.85, max(-in.history, 0.0));
    let deposition = smoothstep(0.03, 0.85, max(in.history, 0.0));
    let moisture = smoothstep(0.0005, 0.035, in.water);
    let lowland = 1.0 - smoothstep(0.18, 0.65, h);
    let vegetation = moisture * lowland * (1.0 - steepness) * 0.38;
    let soil = mix(vec3<f32>(0.40, 0.35, 0.26), vec3<f32>(0.55, 0.49, 0.39), smoothstep(0.18, 0.85, h));
    let rock = mix(vec3<f32>(0.36, 0.34, 0.31), vec3<f32>(0.61, 0.59, 0.54), h);
    let alluvium = vec3<f32>(0.55, 0.45, 0.31);
    let stratum = mix(soil, in.materialColor, select(0.0, 0.7, renderParams.cameraPos.w > 0.5));
    let slopeTint = mix(vec3<f32>(0.64, 0.50, 0.33), rock * 0.86, smoothstep(0.45, 0.9, slope));
    var natural = mix(stratum, slopeTint, steepness * 0.48 + erosion * 0.12);
    natural = mix(natural, alluvium, deposition * (1.0 - steepness) * 0.55);
    natural = mix(natural, vec3<f32>(0.25, 0.33, 0.21), vegetation);
    natural *= 1.0 - moisture * 0.18;
    // Subtle spatial grain, without height bands or diagnostic history stripes.
    let grain = 0.98 + 0.04 * fract(sin(dot(in.worldPos.xz, vec2<f32>(127.1, 311.7))) * 43758.5453);
    return vec4<f32>(clamp(natural * reliefLight * grain, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
  }

  return vec4<f32>(terrainColor, 1.0);
}
`;
