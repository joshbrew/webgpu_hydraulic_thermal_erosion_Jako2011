import Erosion from '../WebGPUTerrainErosion.js';
import { EROSION_PRESETS } from '../erosionPresets.js';
import { generateTerrainExample, seedTerrainExampleSources, terrainExampleOptions } from '../terrainExamples.js';
const sim = new Erosion({ canvas: document.querySelector('canvas') });
let initial, previous, size;
async function read(buffer, bytes) {
  const b = sim.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = sim.device.createCommandEncoder(); e.copyBufferToBuffer(buffer, 0, b, 0, bytes); sim.device.queue.submit([e.finish()]);
  await b.mapAsync(GPUMapMode.READ); const a = new Float32Array(b.getMappedRange()).slice(); b.unmap(); b.destroy(); return a;
}
function metrics(s, reference, flux) {
  const path = [];
  let carved = 0, plugged = 0, abandoned = 0, newlyWet = 0, cut = 0, fill = 0;
  for (let y = Math.round(size * .12); y < size * .88; y++) {
    let moment = 0, sum = 0;
    for (let x = Math.round(size * .15); x < size * .85; x++) {
      const i = y * size + x, h = s[i * 8], w = s[i * 8 + 1], rh = reference[i * 8], rw = reference[i * 8 + 1];
      const weight = flux ? Math.max(flux[i * 4 + 3] - flux[i * 4 + 2], 0) ** 2 : w;
      if (w > .0002) { moment += weight * x; sum += weight; }
      carved += rw < .0005 && w > .001 && rh - h > .003 ? 1 : 0;
      plugged += rw > .001 && h - rh > .003 ? 1 : 0;
      abandoned += rw > .001 && w < .0005 && h - rh > .003 ? 1 : 0;
      newlyWet += rw < .0005 && w > .001 ? 1 : 0;
      cut += Math.max(rh - h, 0); fill += Math.max(h - rh, 0);
    }
    if (sum > 1e-6) path.push({ y, x: moment / sum });
  }
  let length = 0, deviation = 0;
  const smooth = path.map((p, i) => ({ y: p.y, x: path.slice(Math.max(0, i - 3), i + 4).reduce((s, q) => s + q.x, 0) / path.slice(Math.max(0, i - 3), i + 4).length }));
  for (let i = 1; i < smooth.length; i++) length += Math.hypot(smooth[i].x - smooth[i - 1].x, smooth[i].y - smooth[i - 1].y);
  for (const p of path) deviation += (p.x - size / 2) ** 2;
  let outletDischarge = 0;
  if (flux) for (let x = 0; x < size; x++) outletDischarge += flux[((size - 1) * size + x) * 4 + 3];
  return { time: sim.getStats().simTime, carved, plugged, abandoned, newlyWet, cut, fill, outletDischarge,
    sinuosity: length / Math.max(path.at(-1)?.y - path[0]?.y, 1), deviation: Math.sqrt(deviation / Math.max(path.length, 1)), path };
}
window.setupRiver = async (options = {}) => {
  size = options.size || 256;
  const p = { ...EROSION_PRESETS.lowland_meanders, ...options.params };
  sim.setSimulationParams({ ...p, hardnessBase: p.erodabilityMin, hardnessVariation: p.erodabilityMax - p.erodabilityMin,
    talusSlopeCoeff: p.talusCoeff, talusSlopeBias: p.talusBias, cameraAzimuthDeg: 45, cameraElevationDeg: 65,
    cameraPosX: 0.866, cameraPosY: 2.748, cameraPosZ: 0.866 });
  const raster = await generateTerrainExample(sim.device, 'lowland_river', size, options.seed || 7);
  await sim.setDEM(raster, terrainExampleOptions('lowland_river', p)); seedTerrainExampleSources(sim, raster, 'lowland_river');
  initial = previous = await read(sim.stateBuffers[0], sim.stateByteLength); sim.render(); await sim.device.queue.onSubmittedWorkDone();
  return metrics(initial, initial);
};
window.evolveRiver = async seconds => {
  const steps = Math.round(seconds / sim.simulationParams.timeStep);
  for (let i = 0; i < steps; i += 128) { sim.step(Math.min(128, steps - i)); await sim.device.queue.onSubmittedWorkDone(); }
  const s = await read(sim.stateBuffers[0], sim.stateByteLength), flux = await read(sim.fluxBuffer, sim.vec4ByteLength);
  const cumulative = metrics(s, initial, flux), incremental = metrics(s, previous, flux); previous = s;
  sim.render(); await sim.device.queue.onSubmittedWorkDone();
  return { ...cumulative, sinceLast: incremental };
};
await sim.initialize(); window.riverReady = true;
window.riverView = mode => { sim.setSimulationParams({ renderMode: mode }); sim.render(); return sim.device.queue.onSubmittedWorkDone(); };
