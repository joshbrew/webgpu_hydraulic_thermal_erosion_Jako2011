import WebGPUTerrainErosion from './WebGPUTerrainErosion.js';

let gpu = null;
let offscreenCanvas = null;
let canvasMetrics = { width: 1, height: 1, dpr: 1 };
let running = false;
let loopHandle = 0;
let lastFrameAt = 0;
let lastFrameMs = 0;
let lastStatusPostAt = 0;
let lastRenderAt = 0;
let loopIterationsPerFrame = 5;
let simulationParams = {};
let sourceRaster = null;
let sourceImageInfo = null;

const hasWorkerRAF = typeof self.requestAnimationFrame === 'function';
const workerRAF = hasWorkerRAF
  ? (cb) => self.requestAnimationFrame(cb)
  : (cb) => self.setTimeout(() => cb(performance.now()), 0);
const workerCAF = hasWorkerRAF
  ? (id) => self.cancelAnimationFrame(id)
  : (id) => self.clearTimeout(id);

function cloneStats(stats) {
  if (!stats) return null;
  return {
    ...stats,
    terrainRange: stats.terrainRange ? { ...stats.terrainRange } : null,
    waterRange: stats.waterRange ? { ...stats.waterRange } : null,
    sedimentRange: stats.sedimentRange ? { ...stats.sedimentRange } : null,
    historyRange: stats.historyRange ? { ...stats.historyRange } : null,
  };
}

function getSourcePoints() {
  return gpu?.getSourcePoints?.() || [];
}

function postStatus(force = false) {
  const stats = gpu ? cloneStats(gpu.getStats()) : null;
  self.postMessage({
    type: 'status',
    stats,
    sourcePoints: force ? getSourcePoints() : undefined,
    lastFrameMs,
    running,
    sourceImageInfo: force ? sourceImageInfo : undefined,
  });
}

function applyCanvasMetrics() {
  if (!offscreenCanvas) return;
  const width = Math.max(1, Math.round((canvasMetrics.width || 1) * (canvasMetrics.dpr || 1)));
  const height = Math.max(1, Math.round((canvasMetrics.height || 1) * (canvasMetrics.dpr || 1)));
  offscreenCanvas.width = width;
  offscreenCanvas.height = height;
  if (gpu) {
    gpu.resize();
  }
}

async function ensureGpu() {
  if (!offscreenCanvas) {
    throw new Error('Worker canvas has not been initialized.');
  }
  if (!gpu) {
    gpu = new WebGPUTerrainErosion({ canvas: offscreenCanvas });
    await gpu.initialize();
    if (Object.keys(simulationParams).length > 0) {
      gpu.setSimulationParams(simulationParams);
    }
    applyCanvasMetrics();
  }
  return gpu;
}

function stopLoopInternal() {
  if (!running) return;
  running = false;
  if (loopHandle) {
    workerCAF(loopHandle);
    loopHandle = 0;
  }
}

function scheduleNextFrame() {
  loopHandle = workerRAF((now) => {
    void frame(now);
  });
}

function getLoopRenderIntervalMs() {
  const iterations = Math.max(1, loopIterationsPerFrame | 0);
  if (iterations <= 4) return 0;
  if (iterations <= 8) return 16;
  if (iterations <= 16) return 33;
  return 50;
}

async function frame(now) {
  if (!running) return;
  const currentGpu = gpu;
  if (!currentGpu?.ready) {
    scheduleNextFrame();
    return;
  }

  lastFrameMs = lastFrameAt > 0 ? Math.max(0, now - lastFrameAt) : 0;
  lastFrameAt = now;

  try {
    const iterations = Math.max(1, loopIterationsPerFrame | 0);
    const renderIntervalMs = getLoopRenderIntervalMs();
    const shouldRender = renderIntervalMs <= 0 || lastRenderAt <= 0 || (now - lastRenderAt) >= renderIntervalMs;
    if (shouldRender) {
      currentGpu.stepAndRender(iterations);
      lastRenderAt = now;
    } else {
      currentGpu.step(iterations);
    }
    if (now - lastStatusPostAt >= 1000) {
      lastStatusPostAt = now;
      postStatus();
    }
  } catch (error) {
    stopLoopInternal();
    self.postMessage({
      type: 'workerError',
      message: error instanceof Error ? error.message : String(error),
    });
    return;
  }

  scheduleNextFrame();
}

function reply(requestId, payload = {}, transfer = []) {
  if (!requestId) return;
  self.postMessage({ type: 'response', requestId, ok: true, ...payload }, transfer);
}

function replyError(requestId, error) {
  if (!requestId) return;
  self.postMessage({
    type: 'response',
    requestId,
    ok: false,
    error: error instanceof Error ? error.message : String(error),
  });
}

async function buildNormalizedRasterFromBlob(blob) {
  if (!(blob instanceof Blob)) {
    throw new Error('DEM image payload must be a Blob or File.');
  }
  if (typeof createImageBitmap !== 'function') {
    throw new Error('createImageBitmap is not available in this worker.');
  }
  const bitmap = await createImageBitmap(blob, { imageOrientation: 'none' });
  try {
    return await buildNormalizedRasterFromBitmap(bitmap);
  } finally {
    bitmap.close?.();
  }
}

async function buildNormalizedRasterFromBitmap(bitmap) {
  const width = bitmap.width | 0;
  const height = bitmap.height | 0;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('Failed to acquire worker raster canvas context.');
  }
  ctx.setTransform(1, 0, 0, -1, 0, height);
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(bitmap, 0, 0, width, height);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const imageData = ctx.getImageData(0, 0, width, height);
  const rgba = imageData.data;
  const values = new Float32Array(width * height);
  const mask = new Uint8Array(width * height);

  for (let i = 0, j = 0; i < values.length; i++, j += 4) {
    values[i] = (rgba[j] + rgba[j + 1] + rgba[j + 2]) / (3 * 255);
    mask[i] = rgba[j + 3] === 0 ? 0 : 1;
  }

  return { width, height, values, mask };
}

function resampleRasterBilinear(raster, scale) {
  const safeScale = Math.max(1, Math.floor(Number(scale) || 1));
  if (!raster || safeScale === 1) return raster;

  const srcWidth = raster.width | 0;
  const srcHeight = raster.height | 0;
  const dstWidth = Math.max(1, ((srcWidth - 1) * safeScale) + 1);
  const dstHeight = Math.max(1, ((srcHeight - 1) * safeScale) + 1);
  const dstValues = new Float32Array(dstWidth * dstHeight);
  const dstMask = new Uint8Array(dstWidth * dstHeight);
  const srcValues = raster.values;
  const srcMask = raster.mask instanceof Uint8Array ? raster.mask : null;

  function sampleValue(px, py) {
    const x0 = Math.floor(px);
    const y0 = Math.floor(py);
    const x1 = Math.min(x0 + 1, srcWidth - 1);
    const y1 = Math.min(y0 + 1, srcHeight - 1);
    const tx = px - x0;
    const ty = py - y0;
    const i00 = y0 * srcWidth + x0;
    const i10 = y0 * srcWidth + x1;
    const i01 = y1 * srcWidth + x0;
    const i11 = y1 * srcWidth + x1;
    const v00 = srcValues[i00];
    const v10 = srcValues[i10];
    const v01 = srcValues[i01];
    const v11 = srcValues[i11];
    const a = v00 + (v10 - v00) * tx;
    const b = v01 + (v11 - v01) * tx;
    return a + (b - a) * ty;
  }

  function sampleMask(px, py) {
    if (!srcMask) return 1;
    const x0 = Math.floor(px);
    const y0 = Math.floor(py);
    const x1 = Math.min(x0 + 1, srcWidth - 1);
    const y1 = Math.min(y0 + 1, srcHeight - 1);
    const tx = px - x0;
    const ty = py - y0;
    const i00 = y0 * srcWidth + x0;
    const i10 = y0 * srcWidth + x1;
    const i01 = y1 * srcWidth + x0;
    const i11 = y1 * srcWidth + x1;
    const m00 = srcMask[i00];
    const m10 = srcMask[i10];
    const m01 = srcMask[i01];
    const m11 = srcMask[i11];
    const a = m00 + (m10 - m00) * tx;
    const b = m01 + (m11 - m01) * tx;
    return (a + (b - a) * ty) >= 0.5 ? 1 : 0;
  }

  for (let y = 0; y < dstHeight; y++) {
    const srcY = y / safeScale;
    for (let x = 0; x < dstWidth; x++) {
      const srcX = x / safeScale;
      const dstIndex = y * dstWidth + x;
      dstValues[dstIndex] = sampleValue(srcX, srcY);
      dstMask[dstIndex] = sampleMask(srcX, srcY);
    }
  }

  return { width: dstWidth, height: dstHeight, values: dstValues, mask: dstMask };
}

async function loadRasterIntoGpu(message) {
  const currentGpu = await ensureGpu();
  if (Object.keys(simulationParams).length > 0) {
    currentGpu.setSimulationParams(simulationParams);
  }
  if (message.blob) {
    sourceRaster = await buildNormalizedRasterFromBlob(message.blob);
    sourceImageInfo = { width: sourceRaster.width, height: sourceRaster.height };
  }
  if (!sourceRaster) {
    throw new Error('No DEM image has been loaded into the worker yet.');
  }
  const tessellation = Math.max(1, Math.floor(Number(message.tessellation) || 1));
  const raster = resampleRasterBilinear(sourceRaster, tessellation);
  await currentGpu.setDEM(raster, message.options || {});
  currentGpu.render();
  const stats = cloneStats(await currentGpu.readbackStats());
  return {
    stats,
    sourcePoints: getSourcePoints(),
    ready: true,
    sourceImageInfo,
    simSize: { width: raster.width, height: raster.height },
  };
}

self.onmessage = async (event) => {
  const message = event.data || {};
  const { type, requestId } = message;

  try {
    switch (type) {
      case 'init': {
        offscreenCanvas = message.canvas;
        canvasMetrics = {
          width: Math.max(1, Number(message.width) || 1),
          height: Math.max(1, Number(message.height) || 1),
          dpr: Math.max(1, Number(message.dpr) || 1),
        };
        await ensureGpu();
        reply(requestId, { initialized: true });
        postStatus(true);
        break;
      }

      case 'resize': {
        canvasMetrics = {
          width: Math.max(1, Number(message.width) || canvasMetrics.width || 1),
          height: Math.max(1, Number(message.height) || canvasMetrics.height || 1),
          dpr: Math.max(1, Number(message.dpr) || canvasMetrics.dpr || 1),
        };
        applyCanvasMetrics();
        gpu?.render?.();
        reply(requestId, { resized: true });
        break;
      }

      case 'setParams': {
        simulationParams = { ...simulationParams, ...(message.params || {}) };
        if (gpu) {
          gpu.setSimulationParams(simulationParams);
        }
        reply(requestId, { applied: true });
        break;
      }

      case 'loadDEMImage': {
        const payload = await loadRasterIntoGpu(message);
        reply(requestId, payload);
        postStatus(true);
        break;
      }

      case 'setDEM': {
        const currentGpu = await ensureGpu();
        if (Object.keys(simulationParams).length > 0) {
          currentGpu.setSimulationParams(simulationParams);
        }
        const raster = message.raster || {};
        await currentGpu.setDEM({
          width: raster.width | 0,
          height: raster.height | 0,
          values: raster.values instanceof Float32Array ? raster.values : new Float32Array(raster.values || []),
          mask: raster.mask ? (raster.mask instanceof Uint8Array ? raster.mask : new Uint8Array(raster.mask)) : null,
        }, message.options || {});
        currentGpu.render();
        const stats = cloneStats(await currentGpu.readbackStats());
        reply(requestId, { stats, sourcePoints: getSourcePoints(), ready: true, sourceImageInfo });
        postStatus(true);
        break;
      }

      case 'render': {
        gpu?.render?.();
        reply(requestId, { rendered: true });
        break;
      }

      case 'step': {
        if (gpu?.ready) {
          if (message.render !== false) {
            gpu.stepAndRender(Math.max(1, Number(message.iterations) || 1));
          } else {
            gpu.step(Math.max(1, Number(message.iterations) || 1));
          }
        }
        reply(requestId, { stepped: true, stats: cloneStats(gpu?.getStats?.() || null), sourcePoints: getSourcePoints() });
        break;
      }

      case 'readbackStats': {
        const stats = gpu?.ready ? cloneStats(await gpu.readbackStats()) : cloneStats(gpu?.getStats?.() || null);
        reply(requestId, { stats, sourcePoints: getSourcePoints(), lastFrameMs, running, sourceImageInfo });
        break;
      }

      case 'exportTerrainPng': {
        if (!gpu?.ready) {
          throw new Error('Terrain export is only available after the GPU sim is ready.');
        }
        const exportResult = await gpu.exportTerrainPng();
        reply(requestId, {
          width: exportResult.width,
          height: exportResult.height,
          minTerrain: exportResult.minTerrain,
          maxTerrain: exportResult.maxTerrain,
          data: exportResult.data,
        }, [exportResult.data]);
        break;
      }

      case 'paintTerrainBrush': {
        if (!gpu?.ready) {
          throw new Error('Terrain painting is only available after the GPU sim is ready.');
        }
        const stats = cloneStats(await gpu.applyTerrainBrush(message.brush || {}));
        gpu.render();
        reply(requestId, { stats, sourcePoints: getSourcePoints() });
        postStatus(true);
        break;
      }

      case 'paintSpringBrush': {
        if (!gpu?.ready) {
          throw new Error('Spring painting is only available after the GPU sim is ready.');
        }
        gpu.paintSpringBrush(message.brush || {});
        const stats = cloneStats(gpu.getStats());
        reply(requestId, { stats, sourcePoints: getSourcePoints() });
        postStatus(true);
        break;
      }

      case 'clearPaintedSprings': {
        if (gpu?.ready) {
          gpu.clearPaintedSprings();
        }
        const stats = cloneStats(gpu?.getStats?.() || null);
        reply(requestId, { stats, sourcePoints: getSourcePoints() });
        postStatus(true);
        break;
      }

      case 'resetRainTimer': {
        if (gpu?.ready) {
          gpu.resetRainTimer();
          gpu.render();
        }
        const stats = gpu?.ready ? cloneStats(await gpu.readbackStats()) : cloneStats(gpu?.getStats?.() || null);
        reply(requestId, { stats, sourcePoints: getSourcePoints() });
        break;
      }

      case 'restartSources': {
        if (gpu?.ready) {
          gpu.resetRainTimer();
          gpu.render();
        }
        const stats = gpu?.ready ? cloneStats(await gpu.readbackStats()) : cloneStats(gpu?.getStats?.() || null);
        reply(requestId, { stats, sourcePoints: getSourcePoints() });
        break;
      }

      case 'startLoop': {
        if (Number.isFinite(message.iterationsPerFrame)) {
          loopIterationsPerFrame = Math.max(1, Number(message.iterationsPerFrame) | 0);
        }
        if (!running) {
          running = true;
          lastFrameAt = 0;
          lastStatusPostAt = 0;
          lastRenderAt = 0;
          scheduleNextFrame();
        }
        reply(requestId, { running: true, iterationsPerFrame: loopIterationsPerFrame });
        break;
      }

      case 'stopLoop': {
        stopLoopInternal();
        if (gpu?.ready) {
          gpu.render();
        }
        reply(requestId, { running: false });
        postStatus(true);
        break;
      }

      case 'clear': {
        stopLoopInternal();
        if (gpu) {
          gpu.destroy();
          gpu = null;
        }
        sourceRaster = null;
        sourceImageInfo = null;
        reply(requestId, { cleared: true });
        postStatus(true);
        break;
      }

      case 'getStatus': {
        reply(requestId, {
          stats: cloneStats(gpu?.getStats?.() || null),
          sourcePoints: getSourcePoints(),
          lastFrameMs,
          running,
          sourceImageInfo,
        });
        break;
      }

      case 'destroy': {
        stopLoopInternal();
        if (gpu) {
          gpu.destroy();
          gpu = null;
        }
        sourceRaster = null;
        sourceImageInfo = null;
        reply(requestId, { destroyed: true });
        self.close();
        break;
      }

      default:
        throw new Error(`Unknown worker message type: ${String(type)}`);
    }
  } catch (error) {
    replyError(requestId, error);
  }
};
