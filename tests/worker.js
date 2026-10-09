import { EROSION_PRESETS } from '../erosionPresets.js';
import { TERRAIN_EXAMPLES, terrainExampleOptions } from '../terrainExamples.js';
const assert = (value, message) => { if (!value) throw Error(message); };
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

export async function checkWorkerPacing() {
  const worker = new Worker('/erosion.worker.js', { type: 'module' });
  const pending = new Map();
  const errors = [];
  let nextId = 1;
  worker.onmessage = ({ data }) => {
    if (data.type === 'workerError') {
      errors.push(data.message);
      for (const { reject } of pending.values()) reject(Error(data.message));
      pending.clear();
    }
    if (data.type === 'response') {
      const request = pending.get(data.requestId);
      if (!request) return;
      pending.delete(data.requestId);
      data.ok ? request.resolve(data) : request.reject(Error(data.error));
    }
  };
  worker.onerror = error => errors.push(error.message);
  const request = (type, payload = {}, transfer = []) => new Promise((resolve, reject) => {
    const requestId = nextId++;
    const timeout = setTimeout(() => { pending.delete(requestId); reject(Error(`Worker ${type} timed out`)); }, 30000);
    pending.set(requestId, { resolve: value => { clearTimeout(timeout); resolve(value); }, reject: error => { clearTimeout(timeout); reject(error); } });
    worker.postMessage({ type, requestId, ...payload }, transfer);
  });
  try {
    const canvas = new OffscreenCanvas(320, 240);
    await request('init', { canvas, width: 320, height: 240, dpr: 1 }, [canvas]);
    await request('setParams', { params: { rainRate: 0.001, evaporationRate: 0.1, closedBoundaries: true,
      timeStep: 0.02, sourceEnabled: false, thermalRate: 0, hydraulicErosionEnabled: false } });
    await request('setDEM', { raster: { width: 32, height: 32, values: new Float32Array(1024).fill(0.5) } });
    await request('startLoop', { iterationsPerFrame: 4, autoBatch: true });
    await delay(500);
    const active = await request('getStatus');
    assert(active.stats.autoBatch && active.stats.iterationCount > 0, 'adaptive loop advances');
    assert(active.stats.actualIterationsPerFrame >= 1 && active.stats.actualIterationsPerFrame <= 64, 'adaptive batch bound');
    assert(active.stats.inFlightFrames <= 2, 'bounded queue');
    await request('stopLoop');
    const paused = await request('getStatus'); await delay(100);
    assert((await request('getStatus')).stats.iterationCount === paused.stats.iterationCount, 'pause stops submissions');
    const measured = (await request('readbackStats', { includeMass: true })).stats;
    const expected = 0.001 / 0.1 * (1 - Math.exp(-0.1 * measured.simTime));
    assert(Math.abs(measured.totalWater / 1024 - expected) < 2e-7, 'adaptive batches retain analytic water balance');
    await request('startLoop', { iterationsPerFrame: 2, autoBatch: false });
    await delay(120);
    const fixed = await request('getStatus');
    assert(!fixed.stats.autoBatch && fixed.stats.actualIterationsPerFrame === 2, 'fixed mode restores requested batch');
    assert(fixed.stats.iterationCount > paused.stats.iterationCount, 'fixed restart advances');
    await request('stopLoop');
    await request('startLoop', { iterationsPerFrame: 4, autoBatch: true });
    await delay(120); await request('stopLoop');
    assert((await request('getStatus')).stats.autoBatch, 'adaptive restart remains available');
    for (const name of ['river_delta', 'mountain_canyon', 'volcano', 'crater_lake', 'lowland_river']) {
      const example = TERRAIN_EXAMPLES[name], preset = EROSION_PRESETS[example.preset];
      await request('setParams', { params: { ...preset, closedBoundaries: !!example.closed,
        talusSlopeCoeff: preset.talusCoeff, talusSlopeBias: preset.talusBias,
        hardnessBase: preset.erodabilityMin, hardnessVariation: preset.erodabilityMax - preset.erodabilityMin } });
      const options = terrainExampleOptions(name, preset);
      const generated = await request('generateExample', { name, size: 64, seed: 7, options });
      const initial = (await request('readbackStats', { includeMass: true })).stats;
      assert(name === 'volcano' ? initial.totalWater === 0 : initial.totalWater > 0, `${name} initial water`);
      if (preset.sourceEnabled) assert(generated.sourcePoints.length > 0, `${name} has a headwater spring`);
      await request('step', { iterations: 100, render: false });
      const reset = await request('loadDEMImage', { options });
      const restored = (await request('readbackStats', { includeMass: true })).stats;
      assert(restored.iterationCount === 0, `${name} reset restarts time`);
      assert(Math.abs(restored.totalWater - initial.totalWater) < 1e-6 && Math.abs(restored.totalSolid - initial.totalSolid) < 1e-6, `${name} reset restores terrain and water`);
      assert(JSON.stringify(reset.sourcePoints) === JSON.stringify(generated.sourcePoints), `${name} reset restores sources`);
    }
    assert(errors.length === 0, errors.join('\n'));
    await request('destroy');
    return { passed: true, adaptiveSteps: active.stats.actualIterationsPerFrame, pausedIterations: paused.stats.iterationCount,
      sceneResets: ['river_delta', 'mountain_canyon', 'volcano', 'crater_lake', 'lowland_river'] };
  } finally { worker.terminate(); }
}
