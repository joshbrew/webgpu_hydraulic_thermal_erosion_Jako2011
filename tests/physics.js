import Erosion from '../WebGPUTerrainErosion.js';
import { EXAMPLE_MATERIALS, generateTerrainExample, seedTerrainExampleSources, terrainExampleOptions, TERRAIN_EXAMPLES } from '../terrainExamples.js';
import { EROSION_PRESETS } from '../erosionPresets.js';

const results = [];
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const near = (actual, expected, epsilon, label) => assert(Math.abs(actual - expected) <= epsilon, `${label}: ${actual} vs ${expected}`);
const sim = new Erosion({ canvas: document.querySelector('canvas') });
const errors = [];
const base = {
  rainRate: 0, sourceEnabled: false, timeStep: 0.02, evaporationRate: 0,
  thermalRate: 0, closedBoundaries: true, hydraulicErosionEnabled: true,
  bankErosionRate: 0, sourceSedimentConcentration: 0,
  bendSedimentRate: 0, riverBendLength: 0, straightBankErosionFactor: 0.08, openBoundaryEdges: 15,
  hardnessBase: 0.5, hardnessVariation: 0, metersPerPixel: 100,
  pipeArea: 16, gravity: 9.81, flowDrag: 2, maxOutflowFraction: 0.5,
  capacityScale: 1.2, suspensionRate: 0.35, depositionRate: 1.4, softeningRate: 1.2,
  maxErosionDepth: 0.04, talusSlopeCoeff: 0.012, talusSlopeBias: 0.002,
  rainDuration: 0, pulse2Duration: 0, sourceTimeOffset: 0, edgeDrainStrength: 0.08, edgeWaterFloor: 0,
};
async function read(buffer, bytes) {
  const target = sim.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = sim.device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, target, 0, bytes);
  sim.device.queue.submit([encoder.finish()]);
  await target.mapAsync(GPUMapMode.READ);
  const values = new Float32Array(target.getMappedRange()).slice();
  target.unmap(); target.destroy();
  return values;
}
const state = () => read(sim.stateBuffers[0], sim.stateByteLength);
function totals(values) {
  let solid = 0, water = 0, sediment = 0;
  for (let i = 0; i < values.length; i += 8) {
    if (values[i + 4] < 0.5) continue;
    assert(values.slice(i, i + 8).every(Number.isFinite), 'non-finite cell');
    assert(values[i] >= -1e-7 && values[i + 1] >= 0 && values[i + 2] >= 0, 'negative physical state');
    solid += values[i] + values[i + 2]; water += values[i + 1]; sediment += values[i + 2];
  }
  return { solid, water, sediment };
}
async function initialize({ size = 16, terrain = () => 0.5, water = () => 0.1, sediment = () => 0, mask = null, params = {} } = {}) {
  sim.setSimulationParams({ ...base, ...params });
  await sim.setDEM({ width: size, height: size, values: Float32Array.from({ length: size * size }, (_, i) => terrain(i % size, Math.floor(i / size))), mask });
  const values = await state();
  for (let i = 0; i < size * size; i++) {
    values[i * 8 + 1] = water(i % size, Math.floor(i / size));
    values[i * 8 + 2] = sediment(i % size, Math.floor(i / size));
  }
  sim.device.queue.writeBuffer(sim.stateBuffers[0], 0, values);
  sim.device.queue.writeBuffer(sim.stateBuffers[1], 0, values);
  return values;
}
async function test(name, fn) {
  sim.device.pushErrorScope('validation');
  await fn();
  const error = await sim.device.popErrorScope();
  assert(!error, error?.message);
  results.push({ name, passed: true });
  document.querySelector('pre').textContent = JSON.stringify(results, null, 2);
}
async function loadScene(name, overrides = {}, size = 64) {
  const preset = EROSION_PRESETS[TERRAIN_EXAMPLES[name].preset];
  sim.setSimulationParams({ ...preset, closedBoundaries: !!TERRAIN_EXAMPLES[name].closed,
    talusSlopeCoeff: preset.talusCoeff, talusSlopeBias: preset.talusBias,
    hardnessBase: preset.erodabilityMin, hardnessVariation: preset.erodabilityMax - preset.erodabilityMin, ...overrides });
  const raster = await generateTerrainExample(sim.device, name, size, 7);
  await sim.setDEM(raster, terrainExampleOptions(name, preset));
  seedTerrainExampleSources(sim, raster, name);
  return { raster, initial: await state() };
}
async function advance(seconds) {
  const steps = Math.round(seconds / sim.simulationParams.timeStep);
  for (let i = 0; i < steps; i += 100) {
    sim.step(Math.min(100, steps - i));
    await sim.device.queue.onSubmittedWorkDone();
  }
  return state();
}
try {
  await sim.initialize();
  sim.device.addEventListener('uncapturederror', e => errors.push(e.error.message));
  await test('Stagnant water cannot entrain soil', async () => {
    const initial = await initialize();
    sim.step(200);
    const actual = await state();
    for (let i = 0; i < initial.length; i += 8) near(actual[i], initial[i], 1e-7, 'stagnant terrain');
    near(totals(actual).water, totals(initial).water, 1e-5, 'stagnant water');
  });
  await test('Evaporation deposits solids instead of deleting them', async () => {
    const initial = await initialize({ sediment: () => 0.02, params: { evaporationRate: 200, depositionRate: 0 } });
    sim.step(8);
    const actual = totals(await state());
    near(actual.solid, totals(initial).solid, 2e-5, 'dry solid');
    near(actual.sediment, 0, 1e-7, 'dry suspension');
    near(actual.water, 0, 1e-7, 'dry water');
  });
  await test('Deposition exchanges soil and sediment without changing water', async () => {
    const initial = await initialize({ sediment: () => 0.04, params: { depositionRate: 8 } });
    sim.step(50);
    const actual = totals(await state());
    near(actual.solid, totals(initial).solid, 1e-4, 'deposition solid');
    near(actual.water, totals(initial).water, 1e-5, 'deposition water');
  });
  await test('Closed basin conserves water and solids during flow and incision', async () => {
    const initial = await initialize({ terrain: (x, y) => 0.1 + 0.015 * Math.hypot(x - 8, y - 8), sediment: x => x < 4 ? 0.05 : 0 });
    sim.step(400);
    const actual = totals(await state());
    near(actual.solid, totals(initial).solid, 0.0005, 'basin solid');
    near(actual.water, totals(initial).water, 0.0002, 'basin water');
  });
  await test('Masked cells block flow and preserve surrounding mass', async () => {
    const mask = Uint8Array.from({ length: 256 }, (_, i) => i % 16 === 8 ? 0 : 1);
    const initial = await initialize({ mask, terrain: x => 0.3 + x * 0.01, sediment: () => 0.02 });
    sim.step(200);
    const actual = totals(await state());
    near(actual.solid, totals(initial).solid, 0.0004, 'masked solid');
    near(actual.water, totals(initial).water, 0.0002, 'masked water');
  });
  await test('Open boundaries export exactly the donor sediment flux', async () => {
    const initial = await initialize({ sediment: () => 0.025, params: { closedBoundaries: false, suspensionRate: 0, depositionRate: 0 } });
    sim.step(1);
    const flux = await read(sim.fluxBuffer, sim.vec4ByteLength);
    let sedimentExport = 0, waterExport = 0;
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
      const f = (y * 16 + x) * 4;
      const outgoing = (x === 0 ? flux[f] : 0) + (x === 15 ? flux[f + 1] : 0) + (y === 0 ? flux[f + 2] : 0) + (y === 15 ? flux[f + 3] : 0);
      waterExport += 0.02 * outgoing;
      sedimentExport += 0.02 * outgoing * 0.25;
    }
    const actual = totals(await state());
    near(actual.solid, totals(initial).solid - sedimentExport, 3e-5, 'exported solid');
    near(actual.water, totals(initial).water - waterExport, 3e-5, 'exported water');
  });
  await test('Thermal collapse conserves material and zero erodability resists collapse', async () => {
    const params = { thermalRate: 5, talusSlopeCoeff: 0.01, talusSlopeBias: 0.001 };
    const initial = await initialize({ terrain: x => x < 8 ? 0.9 : 0.1, water: () => 0, params });
    sim.step(200);
    const actual = await state();
    near(totals(actual).solid, totals(initial).solid, 0.001, 'talus solid');
    assert(actual[7 * 8] < initial[7 * 8], 'cliff must relax');
    const rock = await initialize({ terrain: x => x < 8 ? 0.9 : 0.1, water: () => 0, params: { ...params, hardnessBase: 0 } });
    sim.step(200);
    const unchanged = await state();
    for (let i = 0; i < rock.length; i += 8) near(unchanged[i], rock[i], 1e-7, 'bedrock');
  });
  await test('Evaporation rate is independent of timestep subdivision', async () => {
    await initialize({ params: { timeStep: 0.02, evaporationRate: 0.3 } });
    sim.step(100); const a = totals(await state());
    await initialize({ params: { timeStep: 0.01, evaporationRate: 0.3 } });
    sim.step(200); const b = totals(await state());
    near(a.water, b.water, 0.00015, 'evaporation subdivision');
  });
  await test('Changing timestep preserves elapsed time and rain pulse timing', async () => {
    await initialize({ params: { timeStep: 0.02, rainDuration: 3 } });
    sim.step(100);
    near(sim.getStats().simTime, 2, 1e-6, 'elapsed time');
    sim.setSimulationParams({ timeStep: 0.01 });
    near(sim.getStats().simTime, 2, 1e-6, 'elapsed time after edit');
    sim.step(100);
    near(sim.getStats().simTime, 3, 1e-6, 'elapsed time after steps');
    sim.resetRainTimer();
    assert(sim.getStats().rainActive, 'reset must restart pulse');
  });
  await test('Large solid inventories remain conserved above the old clipping limit', async () => {
    const initial = await initialize({ terrain: () => 2.5, sediment: () => 0.5, params: { depositionRate: 8 } });
    sim.step(200);
    near(totals(await state()).solid, totals(initial).solid, 0.003, 'unclipped solid');
  });
  await test('Preset erodability changes the active material without resetting terrain', async () => {
    await initialize();
    sim.setSimulationParams({ hardnessBase: 0.05, hardnessVariation: 0 });
    sim.step(1); const rock = await state();
    near(rock[3], 0.05, 1e-7, 'active erodability');
    near(rock[0], 0.5, 1e-7, 'preserved height');
  });
  await test('Every seeded example and preset compiles and remains finite', async () => {
    for (const [name, example] of Object.entries(TERRAIN_EXAMPLES)) {
      const raster = await generateTerrainExample(sim.device, name, 64, 7);
      assert(raster.values.every(h => Number.isFinite(h) && h > 0 && h < 1), name + ' heights');
      const again = await generateTerrainExample(sim.device, name, 64, 7);
      assert(raster.values.every((h, i) => h === again.values[i]), name + ' deterministic seed');
      assert(raster.bands.length === EXAMPLE_MATERIALS[name].length && raster.bands.length >= 3, name + ' material stack');
      for (let band = 0; band < raster.bands.length; band++) {
        assert(raster.bands[band].every((h, i) => Number.isFinite(h) && h >= 0 && h === again.bands[band][i]), name + ' seeded material surfaces');
        if (band) assert(raster.bands[band].every((h, i) => h <= raster.bands[band - 1][i]), name + ' ordered layers');
      }
      const preset = EROSION_PRESETS[example.preset];
      sim.setSimulationParams({ ...preset, talusSlopeCoeff: preset.talusCoeff, talusSlopeBias: preset.talusBias,
        closedBoundaries: !!example.closed, hardnessBase: preset.erodabilityMin, hardnessVariation: preset.erodabilityMax - preset.erodabilityMin });
      await sim.setDEM(raster, terrainExampleOptions(name, preset));
      seedTerrainExampleSources(sim, raster, name);
      sim.step(240); sim.render(); totals(await state());
    }
  });
  await test('CPU spring markers match the GPU source field', async () => {
    await initialize({ water: () => 0, params: { sourceEnabled: true, sourceLayoutMode: 1, sourceSeed: 7,
      sourceStrength: 0.12, sourceRadius: 4, randomSpringCount: 4, rainDuration: 0 } });
    const points = sim.getSourcePoints();
    let sourceWater = 0;
    const expected = new Float32Array(256);
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) for (const point of points) {
      const falloff = Math.max(1 - Math.hypot(x - point.x, y - point.y) / 4, 0);
      sourceWater += 0.02 * (0.12 / 4) * falloff * falloff;
      expected[y * 16 + x] += 0.02 * (0.12 / 4) * falloff * falloff;
    }
    sim.step(1);
    const sourceField = await read(sim.customSpringBuffer, sim.cellCount * 4);
    for (let i = 0; i < 256; i++) near(sourceField[i] * 0.02, expected[i], 1e-8, 'source cell');
    near(totals(await state()).water, sourceWater, 1e-7, 'source footprint water');
  });
  await test('Coarse preview mesh handles odd rectangular grids and full-detail switching', async () => {
    sim.setSimulationParams({ ...base, renderGridStep: 4 });
    await sim.setDEM({ width: 67, height: 35, values: new Float32Array(67 * 35).fill(0.5) });
    sim.render(); await sim.device.queue.onSubmittedWorkDone();
    assert(sim.drawCount === Math.ceil(66 / 4) * Math.ceil(34 / 4), 'coarse mesh count');
    sim.setSimulationParams({ renderGridStep: 1 });
    sim.render(); await sim.device.queue.onSubmittedWorkDone();
    assert(sim.drawCount === 66 * 34, 'full mesh count');
  });
  await test('Thin stationary films retain water while sediment settles', async () => {
    const initial = await initialize({ water: () => 5e-7, sediment: () => 0.001 });
    sim.step(20);
    const actual = totals(await state());
    near(actual.water, totals(initial).water, 1e-10, 'thin-film water');
    near(actual.solid, totals(initial).solid, 1e-5, 'thin-film solids');
    near(actual.sediment, 0, 1e-8, 'thin-film settling');
  });
  await test('Drainage limits conserve very thin flowing films', async () => {
    const initial = await initialize({ terrain: x => 0.4 + x * 0.01, water: () => 1e-9,
      params: { maxOutflowFraction: 0.25, hydraulicErosionEnabled: false } });
    sim.step(1);
    const flux = await read(sim.fluxBuffer, sim.vec4ByteLength);
    for (let i = 0; i < flux.length; i += 4) {
      const outgoing = (flux[i] + flux[i + 1] + flux[i + 2] + flux[i + 3]) * 0.02;
      assert(outgoing <= 0.25e-9 + 1e-16, 'configured drainage limit');
    }
    sim.step(99);
    near(totals(await state()).water, totals(initial).water, 1e-12, 'flowing thin-film water');
  });
  await test('GPU statistics match full-state totals across multiple reduction levels', async () => {
    const mask = Uint8Array.from({ length: 273 * 273 }, (_, i) => i % 7 ? 1 : 0);
    await initialize({ size: 273, mask, terrain: (x, y) => 0.1 + (x + y) / 800,
      water: (x, y) => (x + y + 1) / 10000, sediment: x => x / 9000 });
    const values = await state();
    const expected = totals(values);
    const [actual, second] = await Promise.all([sim.readbackStats(), sim.readbackStats()]);
    near(actual.totalSolid, expected.solid, expected.solid * 2e-7, 'GPU solid sum');
    near(actual.totalWater, expected.water, expected.water * 2e-7, 'GPU water sum');
    assert(actual.validCells === mask.reduce((sum, valid) => sum + valid, 0), 'GPU mask count');
    assert(actual.statisticsReadbackBytes === 64 && second.statisticsReadbackBytes === 64, 'small readback');
    assert(actual.invalidCells === 0, 'valid inventory');
    const pending = sim.readbackStats();
    sim.step(4);
    const snapshot = await pending;
    assert(snapshot.statisticsIteration === 4, 'snapshot metadata matches queued state');
    near(snapshot.statisticsSimTime, sim.getStats().simTime, 1e-8, 'snapshot time');
  });
  await test('Explicit benchmarks measure compute and both render passes', async () => {
    await initialize({ size: 64 });
    const timing = await sim.stepAndRenderBench(20);
    assert(['timestamp-query', 'completion-time'].includes(timing.gpuTimingMode), 'timing mode');
    for (const field of ['cpuErosionMs', 'gpuErosionMs', 'cpuRenderMs', 'gpuRenderMs']) {
      assert(Number.isFinite(timing[field]) && timing[field] >= 0, field);
    }
    assert(sim.iterationCount === 20, 'benchmark advances exactly once');
    const compute = await sim.stepBench(10);
    assert(Number.isFinite(compute.gpuErosionMs) && compute.gpuErosionMs >= 0, 'compute timestamp');
    assert(sim.iterationCount === 30, 'compute benchmark advances exactly once');
  });
  await test('Strong incision and talus cannot skip a resistant material interface', async () => {
    sim.setSimulationParams({ ...base, thermalRate: 8, capacityScale: 20, suspensionRate: 20, maxErosionDepth: 0.1 });
    const values = Float32Array.from({ length: 256 }, (_, i) => 0.501 + (i % 16) * 0.012);
    await sim.setDEM({ width: 16, height: 16, values, bands: [values, new Float32Array(256).fill(0.5)] }, {
      layerConfig: { layers: [
        { enabled: true, heightMin: 0, heightMax: 1, hardnessMin: 1, hardnessMax: 1, thermalEnabled: true },
        { enabled: true, heightMin: 0, heightMax: 1, hardnessMin: 0, hardnessMax: 0.02, thermalEnabled: false },
      ] },
    });
    const initial = await state();
    for (let i = 0; i < initial.length; i += 8) initial[i + 1] = 0.1;
    sim.device.queue.writeBuffer(sim.stateBuffers[0], 0, initial);
    sim.step(500);
    const actual = await state();
    for (let i = 0; i < actual.length; i += 8) assert(actual[i] >= 0.5 - 1e-6, 'resistant layer was skipped');
    near(totals(actual).solid, totals(initial).solid, 0.001, 'layered solids');
    assert(actual.some((h, i) => i % 8 === 0 && h < initial[i] - 0.0001), 'upper soil should erode');
  });
  await test('Coincident layer surfaces expose the deeper material consistently', async () => {
    sim.setSimulationParams(base);
    const values = new Float32Array(256).fill(0.5);
    await sim.setDEM({ width: 16, height: 16, values, bands: [values, values] }, {
      layerConfig: { layers: [
        { enabled: true, heightMin: 0, heightMax: 1, hardnessMin: 1, hardnessMax: 1, thermalEnabled: true },
        { enabled: true, heightMin: 0, heightMax: 1, hardnessMin: 0, hardnessMax: 0.02, thermalEnabled: false },
      ] },
    });
    const initial = await state();
    near(initial[3], 0, 1e-8, 'CPU material');
    sim.step(10);
    const actual = await state();
    near(actual[3], 0, 1e-8, 'GPU material');
    near(actual[0], 0.5, 1e-8, 'exposed rock');
  });
  await test('Timed storms deliver the same rain across timestep subdivisions', async () => {
    for (const dt of [0.02, 0.03, 0.007]) {
      await initialize({ water: () => 0, params: { timeStep: dt, rainRate: 0.02, rainDuration: 0.071, pulse2Duration: 0.013 } });
      sim.step(Math.ceil(0.12 / dt));
      near(totals(await state()).water, 256 * 0.02 * 0.084, 3e-7, 'integrated storm');
      assert(!sim.getStats().rainActive, 'storm ends');
    }
  });
  await test('Spring fields are cached and invalidated by source changes and layout switches', async () => {
    const params = { sourceEnabled: true, sourceLayoutMode: 1, sourceSeed: 7, sourceRadius: 4,
      sourceStrength: 0.12, randomSpringCount: 4, rainDuration: 0, pulse2Duration: 0 };
    await initialize({ water: () => 0, params });
    sim.step(40); const original = await read(sim.customSpringBuffer, sim.cellCount * 4);
    assert(sim.getStats().sourceFieldBuildCount === 1, 'build once per batch');
    sim.step(100);
    assert(sim.getStats().sourceFieldBuildCount === 1, 'reuse across batches');
    sim.setSimulationParams({ ...params, sourceSeed: 11 }); sim.step(1);
    const changed = await read(sim.customSpringBuffer, sim.cellCount * 4);
    assert(changed.some((rate, i) => Math.abs(rate - original[i]) > 1e-6), 'seed changes field');
    assert(sim.getStats().sourceFieldBuildCount === 2, 'rebuild changed sources');
    sim.setSimulationParams({ sourceSeed: 11 }); sim.step(1);
    assert(sim.getStats().sourceFieldBuildCount === 2, 'identical values preserve cache');
    sim.setSimulationParams({ sourceEnabled: false }); sim.step(1);
    sim.setSimulationParams({ sourceEnabled: true }); sim.step(1);
    assert(sim.getStats().sourceFieldBuildCount === 2, 'toggle preserves cache');
    sim.setSimulationParams({ sourceLayoutMode: 0 }); sim.step(1);
    const painted = await read(sim.customSpringBuffer, sim.cellCount * 4);
    assert(painted.every(rate => rate === 0), 'painted layout restores retained empty map');
    sim.paintSpringBrush({ x: 8, y: 8, radius: 3, strength: 0.1 });
    const paint = await read(sim.customSpringBuffer, sim.cellCount * 4);
    assert(paint.some(rate => rate > 0), 'painted source remains writable');
    sim.setSimulationParams({ sourceLayoutMode: 1 }); sim.step(1);
    assert(sim.getStats().sourceFieldBuildCount === 3, 'random layout rebuilds after painting');
    sim.setSimulationParams({ sourceLayoutMode: 0 }); sim.step(1);
    const restored = await read(sim.customSpringBuffer, sim.cellCount * 4);
    assert(restored.every((rate, i) => rate === paint[i]), 'layout switch retains painting');
    await initialize({ size: 32, water: () => 0, params }); sim.step(1);
    assert(sim.getStats().sourceFieldBuildCount === 1, 'new grid rebuilds source field');
  });
  await test('Spring storms integrate partial final steps without losing sediment', async () => {
    await initialize({ water: () => 0, sediment: () => 0.005, params: { timeStep: 0.03,
      sourceEnabled: true, sourceLayoutMode: 1, sourceSeed: 7, sourceStrength: 0.12,
      sourceRadius: 4, randomSpringCount: 4, rainDuration: 0.071, pulse2Duration: 0.013 } });
    const initial = totals(await state());
    sim.step(10);
    const rates = await read(sim.customSpringBuffer, sim.cellCount * 4);
    const expected = rates.reduce((sum, rate) => sum + rate, 0) * 0.084;
    const actual = totals(await state());
    near(actual.water, expected, 2e-7, 'integrated springs');
    near(actual.solid, initial.solid, 0.00005, 'spring storm solids');
  });
  await test('Gentle nonzero slopes retain hydraulic entrainment precision', async () => {
    const initial = await initialize({ terrain: x => 0.4 + x * 0.00005, water: x => 0.02 + x * 0.005,
      params: { hardnessBase: 1, capacityScale: 10, suspensionRate: 10, sourceEnabled: false } });
    sim.step(3);
    const actual = await state();
    assert(actual[8 * 8] < initial[8 * 8] - 1e-7, 'small grade must not round to zero');
    near(totals(actual).solid, totals(initial).solid, 0.00005, 'gentle slope solids');
  });
  await test('A lake at rest preserves terrain above an uneven lake bed', async () => {
    const initial = await initialize({ terrain: (x, y) => 0.2 + 0.012 * x + 0.009 * y,
      water: (x, y) => 0.8 - (0.2 + 0.012 * x + 0.009 * y),
      params: { timeStep: 0.02, sourceEnabled: false, rainDuration: 0, pulse2Duration: 0 } });
    sim.step(1000);
    const actual = await state();
    for (let i = 0; i < actual.length; i += 8) near(actual[i], initial[i], 2e-6, 'lake bed stability');
    near(totals(actual).water, totals(initial).water, 0.0001, 'resting lake water');
  });
  await test('The displayed lake is planar and dry shore cells remain dry', async () => {
    await initialize({ terrain: x => x < 8 ? 0.2 + x * 0.04 : 0.9,
      water: x => x < 8 ? 0.8 - (0.2 + x * 0.04) : 0,
      params: { renderHeightScale: 0.45, waterHeightScale: 0.45 } });
    const source = await (await fetch('/erosionRender.wgsl')).text();
    const module = sim.device.createShaderModule({ code: source + `
      @group(0) @binding(4) var<storage, read_write> sampledSurface: array<f32>;
      @compute @workgroup_size(8, 8) fn samplePreview(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x >= u32(renderParams.dims.x) || gid.y >= u32(renderParams.dims.y)) { return; }
        sampledSurface[idx(gid.x, gid.y)] = renderWaterSurfaceAt(i32(gid.x), i32(gid.y));
      }` });
    const pipeline = await sim.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'samplePreview' } });
    const output = sim.device.createBuffer({ size: sim.cellCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
      const group = sim.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: sim.renderUniformBufferWater } },
        { binding: 1, resource: { buffer: sim.stateBuffers[0] } }, { binding: 4, resource: { buffer: output } },
      ] });
      const encoder = sim.device.createCommandEncoder(); const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(2, 2); pass.end();
      sim.device.queue.submit([encoder.finish()]);
      const heights = await read(output, sim.cellCount * 4);
      for (let i = 0; i < heights.length; i++) near(heights[i], (i % 16 < 8 ? 0.8 : 0.9) * 0.45, 1e-7, 'displayed water height');
    } finally { output.destroy(); }
  });
  await test('Water and sediment transport remain consistent under a quarter turn', async () => {
    const terrain = (x, y) => 0.25 + 0.003 * x + 0.07 * Math.exp(-((x - 5) ** 2 + (y - 10) ** 2) / 25);
    const water = (x, y) => 0.01 + x * 0.002 + y * 0.0003;
    const sediment = (x, y) => x < 6 && y < 9 ? 0.02 : 0;
    const params = { timeStep: 0.02, thermalRate: 0.3, sourceEnabled: false, rainDuration: 0, pulse2Duration: 0 };
    await initialize({ terrain, water, sediment, params }); sim.step(300); const normal = await state();
    await initialize({ terrain: (x, y) => terrain(y, 15 - x), water: (x, y) => water(y, 15 - x),
      sediment: (x, y) => sediment(y, 15 - x), params });
    sim.step(300); const rotated = await state();
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) for (const field of [0, 1, 2]) {
      near(rotated[(y * 16 + x) * 8 + field], normal[((15 - x) * 16 + y) * 8 + field], 2e-5, 'rotation consistency');
    }
  });
  await test('Closed storm inventories stay consistent at three terrain resolutions', async () => {
    for (const size of [128, 256, 512]) {
      sim.setSimulationParams({ ...base, timeStep: 0.03, rainRate: 0.008, rainDuration: 0.317, pulse2Duration: 0.013,
        thermalRate: 0.25, sourceEnabled: false });
      const raster = await generateTerrainExample(sim.device, 'basin', size, 7);
      await sim.setDEM(raster);
      const initial = await sim.readbackStats();
      sim.step(60);
      const actual = await sim.readbackStats();
      near(actual.totalWater / actual.validCells, 0.008 * 0.33, 2e-9, 'resolution-independent storm depth');
      near(actual.totalSolid, initial.totalSolid, initial.totalSolid * 2e-6, 'resolution solid inventory');
      assert(actual.invalidCells === 0 && actual.terrainRange.min >= 0, 'finite resolved state');
    }
  });
  await test('Full 32-bit spring seeds stay distinct and match their markers', async () => {
    let previous;
    for (const seed of [7, 16777216, 16777217, 2147483647, 4294967295]) {
      await initialize({ water: () => 0, params: { sourceEnabled: true, sourceLayoutMode: 1, sourceSeed: seed,
        randomSpringCount: 4, sourceRadius: 4, sourceStrength: 0.12, rainDuration: 0, pulse2Duration: 0 } });
      const points = sim.getSourcePoints();
      sim.step(3);
      const rates = await read(sim.customSpringBuffer, sim.cellCount * 4);
      for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
        const expected = points.reduce((sum, point) => sum + 0.03 * Math.max(1 - Math.hypot(x - point.x, y - point.y) / 4, 0) ** 2, 0);
        near(rates[y * 16 + x], expected, 1e-7, '32-bit spring footprint');
      }
      if (seed === 16777217) assert(rates.some((rate, i) => Math.abs(rate - previous[i]) > 1e-6), 'adjacent large seeds must remain distinct');
      previous = rates;
    }
  });
  await test('Rain and evaporation follow the analytic balance across timesteps', async () => {
    for (const dt of [0.01, 0.02, 0.04]) {
      await initialize({ water: () => 0.1, params: { timeStep: dt, rainRate: 0.025, evaporationRate: 3,
        rainDuration: 0, pulse2Duration: 0, sourceEnabled: false } });
      sim.step(Math.round(2 / dt));
      const time = sim.getStats().simTime;
      const expected = 0.1 * Math.exp(-3 * time) + 0.025 / 3 * (1 - Math.exp(-3 * time));
      near(totals(await state()).water / 256, expected, 2e-8, 'rain/evaporation balance');
    }
  });
  await test('Partial storms and high evaporation retain the correct remaining water', async () => {
    for (const [dt, evaporationRate] of [[0.01, 4], [0.02, 4], [0.03, 4], [0.03, 200]]) {
      await initialize({ water: () => 0, sediment: () => 0.005, params: { timeStep: dt, rainRate: 0.02,
        evaporationRate, rainDuration: 0.071, pulse2Duration: 0.013, sourceEnabled: false } });
      const initial = totals(await state()); sim.step(Math.round(0.12 / dt));
      const time = sim.getStats().simTime;
      const expected = 0.02 / evaporationRate * (Math.exp(-evaporationRate * (time - 0.084)) - Math.exp(-evaporationRate * time));
      const actual = totals(await state());
      near(actual.water / 256, expected, 2e-9, 'storm evaporation balance');
      near(actual.solid, initial.solid, 0.00005, 'evaporating storm solids');
    }
  });
  await test('Spring sediment survives evaporation and finite partial source steps', async () => {
    for (const dt of [0.02, 0.03, 0.07]) {
      const initial = totals(await initialize({ water: () => 0, params: { timeStep: dt,
        sourceEnabled: true, sourceLayoutMode: 0, sourceSedimentConcentration: 0.12,
        rainRate: 0.01, rainDuration: 0.071, pulse2Duration: 0, sourceTimeOffset: 0,
        evaporationRate: 200, hydraulicErosionEnabled: false } }));
      sim.paintSpringBrush({ x: 8, y: 8, radius: 3, strength: 0.2, hardness: 1 });
      sim.step(1);
      const rates = await read(sim.customSpringBuffer, 256 * 4);
      const rate = rates.reduce((sum, value) => sum + value, 0);
      const final = await advance(1.4);
      near(totals(final).solid, initial.solid + rate * 0.12 * 0.071, 0.00006, 'spring solid budget');
    }
  });
  await test('Bank retreat requires flowing water and conserves closed-basin solids', async () => {
    const initial = await initialize({ terrain: (x, y) => 0.5 + 0.01 * (15 - y) + (x < 6 || x > 9 ? 0.05 : 0),
      water: (x, y) => x >= 6 && x <= 9 ? 0.02 : 0,
      params: { bankErosionRate: 3, bankBendBias: 32, thermalRate: 0.4 } });
    const final = await advance(8);
    near(totals(final).solid, totals(initial).solid, 0.0003, 'bank and thermal solids');
    let cut = 0;
    for (let y = 2; y < 13; y++) for (const x of [5, 10]) cut += Math.max(initial[(y * 16 + x) * 8] - final[(y * 16 + x) * 8], 0);
    assert(cut > 0.001, `active bank retreat ${cut}`);
    const lake = await initialize({ water: () => 0.1, params: { bankErosionRate: 20, bendSedimentRate: 20, riverBendLength: 0.045 } });
    const resting = await advance(3);
    for (let i = 0; i < 256; i++) near(resting[i * 8], lake[i * 8], 2e-7, 'still lake bank terrain');
  });
  await test('Bank retreat moves more material than bed incision alone and respects walls and resistant banks', async () => {
    const mask = new Uint8Array(256).fill(1); mask[8 * 16 + 5] = 0;
    const setup = { mask, terrain: (x, y) => 0.5 + 0.01 * (15 - y) + (x < 6 || x > 9 ? 0.05 : 0),
      water: x => x >= 6 && x <= 9 ? 0.02 : 0 };
    await initialize({ ...setup, params: { bankErosionRate: 0 } });
    const control = await advance(8);
    const initial = await initialize({ ...setup, params: { bankErosionRate: 3 } });
    const active = await advance(8);
    let extraCut = 0;
    for (let y = 2; y < 13; y++) for (const x of [5, 10]) {
      const i = y * 16 + x;
      if (mask[i]) extraCut += control[i * 8] - active[i * 8];
    }
    assert(extraCut > 0.001, `additional bank retreat ${extraCut}`);
    near(totals(active).solid, totals(initial).solid, 0.0003, 'bank solid inventory with a wall');
    const resistant = await initialize({ ...setup, params: { bankErosionRate: 10, hardnessBase: 0, hardnessVariation: 0 } });
    const protectedState = await advance(8);
    for (let i = 0; i < 256; i++) if (mask[i]) near(protectedState[i * 8], resistant[i * 8], 2e-7, 'resistant bank');
  });
  await test('Parallel river rows and cached bends match discharge-derived reference on swapped rectangular grids', async () => {
    for (const [width, height] of [[23, 47], [47, 23]]) {
      sim.setSimulationParams({ ...base, bendSedimentRate: 4, riverBendLength: .06 });
      await sim.setDEM({ width, height, values: Float32Array.from({ length: width * height }, (_, i) => .5 + .001 * (height - Math.floor(i / width))) });
      const initial = await state();
      for (let i = 0; i < width * height; i++) {
        const x = i % width, y = Math.floor(i / width), center = width * .5 + 3 * Math.sin(y * .15);
        initial[i * 8 + 1] = .03 * Math.exp(-(((x - center) / 3) ** 2));
      }
      sim.device.queue.writeBuffer(sim.stateBuffers[0], 0, initial); sim.device.queue.writeBuffer(sim.stateBuffers[1], 0, initial);
      sim.step(1);
      const postFlow = await read(sim.stateBuffers[1], sim.stateByteLength), flux = await read(sim.fluxBuffer, sim.vec4ByteLength);
      const velocity = await read(sim.velocityBuffer, sim.velocityBuffer.size), offset = width * height * 4;
      near(sim.velocityBuffer.size, sim.vec4ByteLength + height * 32, 0, 'rectangular guide allocation');
      for (let y = 0; y < height; y++) {
        let moment = 0, weight = 0;
        for (let x = 0; x < width; x++) {
          const i = y * width + x;
          if (postFlow[i * 8 + 4] < .5 || postFlow[i * 8 + 1] < .0005) continue;
          const q = Math.max(flux[i * 4 + 3] - flux[i * 4 + 2], 0) ** 2;
          moment += x * q; weight += q;
        }
        near(velocity[offset + y * 4], moment / Math.max(weight, 1e-10), 1e-5, 'parallel discharge center');
        near(velocity[offset + y * 4 + 1], weight, Math.max(1e-10, weight * 3e-6), 'parallel discharge weight');
      }
      const center = y => {
        let moment = 0, weight = 0;
        for (let d = -3; d <= 3; d++) {
          const row = Math.max(0, Math.min(height - 1, y + d)), w = velocity[offset + row * 4 + 1] * (4 - Math.abs(d));
          moment += velocity[offset + row * 4] * w; weight += w;
        }
        return weight > 1e-8 ? moment / Math.max(weight, 1e-10) : width * .5;
      };
      const distance = Math.max(4, Math.round(height * .06));
      for (let y = 0; y < height; y++) {
        const above = center(y - distance), below = center(y + distance), slope = (below - above) / (distance * 2);
        const curve = (below - 2 * center(y) + above) / distance ** 2, bend = curve / (1 + slope * slope) ** 2;
        near(velocity[offset + height * 4 + y * 4], bend, 1e-6, 'cached bend x');
        near(velocity[offset + height * 4 + y * 4 + 1], -slope * bend, 1e-6, 'cached bend y');
      }
    }
  });
  await test('Bend sediment drift conserves solids behind masked walls', async () => {
    const size = 64, mask = new Uint8Array(size * size).fill(1);
    for (let y = 25; y < 32; y++) mask[y * size + 35] = 0;
    const center = y => 32 + 6 * Math.sin(y * 0.12);
    const setup = { size, mask,
      terrain: (x, y) => 0.5 + 0.0008 * (size - y) + 0.03 * (1 - Math.exp(-(((x - center(y)) / 3) ** 2))),
      water: (x, y) => Math.abs(x - center(y)) < 3 ? 0.018 : 0,
      sediment: (x, y) => Math.abs(x - center(y)) < 3 ? 0.004 : 0,
      params: { bendSedimentRate: 4, riverBendLength: 0.06, hydraulicErosionEnabled: true,
        bankErosionRate: 3, bankBendBias: 80, thermalRate: 0.1 } };
    const initial = await initialize(setup), final = await advance(15);
    near(totals(final).solid, totals(initial).solid, totals(initial).solid * 2e-6, 'bend drift solid budget');
    for (let i = 0; i < mask.length; i++) if (!mask[i]) {
      near(final[i * 8 + 4], 0, 0, 'masked wall stays invalid');
      near(final[i * 8 + 1] + final[i * 8 + 2], 0, 0, 'masked wall cannot hold flow or sediment');
    }
    await initialize({ ...setup, params: { ...setup.params, bendSedimentRate: 0 } });
    const control = await advance(15);
    let changed = 0;
    for (let i = 0; i < mask.length; i++) if (mask[i]) changed += Math.abs(final[i * 8] - control[i * 8]);
    assert(changed > 1e-4, `bend drift changes deposition: ${changed}`);
  });
  await test('Selected outlet edges export water while other edges stay closed', async () => {
    const dt = 0.02;
    const initial = await initialize({ params: { closedBoundaries: false, openBoundaryEdges: 8,
      hydraulicErosionEnabled: false, timeStep: dt } });
    sim.step(1);
    const flux = await read(sim.fluxBuffer, 256 * 16);
    let outlet = 0;
    for (let i = 0; i < 16; i++) {
      near(flux[i * 16 * 4], 0, 0, 'closed left edge');
      near(flux[(i * 16 + 15) * 4 + 1], 0, 0, 'closed right edge');
      near(flux[i * 4 + 2], 0, 0, 'closed top edge');
      outlet += flux[(15 * 16 + i) * 4 + 3] * dt;
    }
    assert(outlet > 0, 'bottom outlet discharges');
    near(totals(await state()).water, totals(initial).water - outlet, 2e-6, 'outlet water budget');
  });
  await test('Fresh source sediment exports with the same donor flux as its water', async () => {
    const dt = 0.04, k = 5, concentration = 0.2, duration = 0.031;
    const initial = await initialize({ sediment: () => 0.01, params: { timeStep: dt,
      sourceEnabled: true, sourceLayoutMode: 0, sourceSedimentConcentration: concentration,
      rainDuration: duration, pulse2Duration: 0, sourceTimeOffset: 0, evaporationRate: k,
      closedBoundaries: false, hydraulicErosionEnabled: false } });
    sim.paintSpringBrush({ x: 0, y: 8, radius: 3, strength: 0.2, hardness: 1 });
    sim.step(1);
    const rates = await read(sim.customSpringBuffer, 256 * 4);
    const flux = await read(sim.fluxBuffer, 256 * 16);
    let exported = 0;
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
      const i = y * 16 + x;
      const outward = (x === 0 ? flux[i * 4] : 0) + (x === 15 ? flux[i * 4 + 1] : 0)
        + (y === 0 ? flux[i * 4 + 2] : 0) + (y === 15 ? flux[i * 4 + 3] : 0);
      const water = initial[i * 8 + 1] * Math.exp(-k * dt / 2) + rates[i] * -Math.expm1(-k * dt / 2) / k;
      exported += outward * dt * (initial[i * 8 + 2] + rates[i] * concentration * dt / 2) / water;
    }
    const injected = rates.reduce((sum, value) => sum + value, 0) * concentration * duration;
    near(totals(await state()).solid, totals(initial).solid + injected - exported, 0.00003, 'fed source export budget');
  });
  await test('Lowland river changes its banks and keeps a downstream flowing channel', async () => {
    const { raster, initial } = await loadScene('lowland_river', {}, 128);
    const final = await advance(180);
    let bankCut = 0, shiftedRows = 0, downstreamWater = 0;
    for (let y = 16; y < 115; y++) {
      let beforeMoment = 0, afterMoment = 0, beforeSum = 0, afterSum = 0;
      for (let x = 16; x < 112; x++) {
        const i = y * 128 + x, before = initial[i * 8 + 1], after = final[i * 8 + 1];
        beforeMoment += before * x; beforeSum += before; afterMoment += after * x; afterSum += after;
        if (before < 0.0001) bankCut += Math.max(initial[i * 8] - final[i * 8], 0);
        assert(final[i * 8] >= raster.bands[2][i] - 0.002, 'alluvial river respects its resistant substrate');
        if (y > 90) downstreamWater += after;
      }
      if (beforeSum > 1e-5 && afterSum > 1e-5 && Math.abs(beforeMoment / beforeSum - afterMoment / afterSum) > 1) shiftedRows++;
    }
    assert(bankCut > 0.01 && shiftedRows > 5 && downstreamWater > 0.01, `evolving river: bank cut ${bankCut}, shifted rows ${shiftedRows}, downstream water ${downstreamWater}`);
    totals(final);
  });
  await test('River delta carries sediment into its coastal receiving basin', async () => {
    const { raster, initial } = await loadScene('river_delta', { thermalRate: 0 });
    assert(raster.values[5 * 64 + 32] > raster.values[58 * 64 + 32] + 0.1, 'river descends to coast');
    assert(initial[(58 * 64 + 32) * 8 + 1] > 0.025, 'shallow coastal shelf starts submerged');
    const final = await advance(60);
    let upstreamCut = 0, mouthDeposit = 0;
    for (let i = 0; i < 4096; i++) {
      const y = Math.floor(i / 64) / 63, change = final[i * 8] - initial[i * 8];
      if (y < 0.55) upstreamCut += Math.max(-change, 0);
      if (y > 0.55) mouthDeposit += Math.max(change, 0);
    }
    assert(upstreamCut > 1e-4, `upstream incision ${upstreamCut}`);
    assert(mouthDeposit > 1e-5, `coastal deposition ${mouthDeposit}`);
    await loadScene('river_delta', { thermalRate: 0, sourceSedimentConcentration: 0 });
    const clear = await advance(60);
    let fedMouth = 0, clearMouth = 0;
    for (let i = 0; i < 4096; i++) if (Math.floor(i / 64) / 63 > 0.55) {
      fedMouth += final[i * 8] - initial[i * 8]; clearMouth += clear[i * 8] - initial[i * 8];
    }
    assert(fedMouth > clearMouth + 0.001, `source sediment increases coastal buildup: ${fedMouth} vs ${clearMouth}`);
    totals(final);
  });
  await test('Mountain river follows the canyon and respects its resistant floor', async () => {
    const { raster, initial } = await loadScene('mountain_canyon', { thermalRate: 0 });
    const final = await advance(60);
    let canyonCut = 0, wetChannel = 0;
    for (let y = 0; y < 64; y++) {
      const x = Math.round((0.5 + 0.08 * Math.sin(y / 63 * 7)) * 63), i = y * 64 + x;
      assert(raster.values[i] + 0.2 < raster.values[y * 64 + 8], 'canyon below mountain flank');
      canyonCut += Math.max(initial[i * 8] - final[i * 8], 0);
      wetChannel += final[i * 8 + 1];
    }
    assert(canyonCut > 1e-4 && wetChannel > 0.01, `active mountain river: cut ${canyonCut}, water ${wetChannel}`);
    for (let i = 0; i < 4096; i++) assert(final[i * 8] >= raster.bands[2][i] - 0.001, 'resistant rock limits incision');
    totals(final);
  });
  await test('Volcanic crater retains a resting lake without eroding and conserves storm solids', async () => {
    const { raster, initial } = await loadScene('crater_lake', { rainRate: 0, evaporationRate: 0, thermalRate: 0 });
    assert(raster.values[32 * 64 + 44] > raster.values[32 * 64 + 32] + 0.15, 'raised crater rim');
    const resting = await advance(6);
    for (let i = 0; i < 4096; i++) near(resting[i * 8], initial[i * 8], 2e-7, 'resting crater terrain');
    near(totals(resting).water, totals(initial).water, 2e-5, 'retained crater lake');
    const storm = await loadScene('crater_lake');
    const final = await advance(30);
    const before = totals(storm.initial), after = totals(final);
    near(after.solid, before.solid, before.solid * 2e-5, 'volcano closed solids');
    const p = EROSION_PRESETS.volcanic_lake, k = p.evaporationRate;
    const expectedWater = before.water * Math.exp(-k * 30) + 4096 * p.rainRate / k * (Math.exp(-k * (30 - p.rainDuration)) - Math.exp(-k * 30));
    near(after.water, expectedWater, expectedWater * 3e-5, 'crater storm water budget');
    assert(final[(32 * 64 + 32) * 8 + 1] > 0.025, 'crater still holds water');
  });
  await test('Four material layers expose their own erodability and render in relief views', async () => {
    sim.setSimulationParams(base);
    const bands = [0.8, 0.7, 0.6, 0.5].map(h => new Float32Array(256).fill(h));
    const strengths = [0.85, 0.15, 0.6, 0];
    await sim.setDEM({ width: 16, height: 16, values: bands[0], bands }, {
      layerConfig: { layers: strengths.map(s => ({ enabled: true, heightMin: 0, heightMax: 1, hardnessMin: s, hardnessMax: s, thermalEnabled: false })) },
    });
    const initial = await state();
    for (let i = 0; i < 256; i++) initial[i * 8] = 0.75 - Math.floor((i % 16) / 4) * 0.1;
    sim.device.queue.writeBuffer(sim.stateBuffers[0], 0, initial); sim.device.queue.writeBuffer(sim.stateBuffers[1], 0, initial);
    sim.step(1); const actual = await state();
    for (let i = 0; i < 256; i++) near(actual[i * 8 + 3], strengths[Math.floor((i % 16) / 4)], 1e-7, 'exposed layer strength');
    for (const mode of [7, 8, 9, 4]) { sim.setSimulationParams({ renderMode: mode }); sim.render(); }
    await sim.device.queue.onSubmittedWorkDone();
  });
  await test('Volcanic storm deepens radial canyons through deposits while retaining its crater', async () => {
    const { raster, initial } = await loadScene('volcano', {}, 256);
    const final = await advance(120);
    let deepestCut = 0, exposedTephra = 0, exposedTuff = 0, apronDeposit = 0, rimSum = 0, rimCount = 0;
    for (let i = 0; i < 65536; i++) {
      const x = (i % 256) / 255, y = Math.floor(i / 256) / 255;
      const r = Math.hypot((x - 0.51) * 1.06, (y - 0.49) * 0.94), h = final[i * 8], cut = initial[i * 8] - h;
      if (r > 0.23 && r < 0.44) {
        deepestCut = Math.max(deepestCut, cut);
        exposedTephra += h <= raster.bands[1][i] ? 1 : 0;
        exposedTuff += h <= raster.bands[2][i] ? 1 : 0;
      }
      if (r > 0.4 && r < 0.52) apronDeposit += Math.max(-cut, 0);
      if (r > 0.17 && r < 0.19) { rimSum += h; rimCount++; }
    }
    assert(deepestCut > 0.08 && exposedTephra > 25 && exposedTuff > 0, `deep layered gullies: cut ${deepestCut}, tephra ${exposedTephra}, tuff ${exposedTuff}`);
    assert(apronDeposit > 0.1, `sediment apron ${apronDeposit}`);
    assert(rimSum / rimCount > final[(125 * 256 + 130) * 8] + 0.25, 'rim remains above crater floor');
    near(totals(final).solid, totals(initial).solid, totals(initial).solid * 2e-5, 'layered volcano solid inventory');
  });
  const diagnostics = await sim.readbackStats();
  assert(Number.isFinite(diagnostics.totalSolid), 'solid mass diagnostics');
  assert(errors.length === 0, errors.join('\n'));
  window.testResults = { passed: true, results, adapter: { vendor: sim.adapter.info?.vendor, device: sim.adapter.info?.device, description: sim.adapter.info?.description } };
} catch (error) {
  window.testResults = { passed: false, results, error: error.stack, gpuErrors: errors };
}
document.querySelector('pre').textContent = JSON.stringify(window.testResults, null, 2);
