import Erosion from '../WebGPUTerrainErosion.js';
import { generateTerrainExample, seedTerrainExampleSources, terrainExampleOptions, TERRAIN_EXAMPLES } from '../terrainExamples.js';
import { EROSION_PRESETS } from '../erosionPresets.js';

export const cases = [
  ['alpine', 'paper_balanced', 60], ['alpine', 'river_cut', 60],
  ['fan', 'delta_depositor', 60], ['badlands', 'flash_flood', 60],
  ['badlands', 'badlands', 60], ['canyon', 'canyon_carver', 90],
  ['talus', 'thermal_heavy', 20], ['basin', 'flash_flood', 60],
  ['alpine', 'gentle_weathering', 90],
  ['river_delta', 'river_delta', 360], ['mountain_canyon', 'mountain_canyon', 90], ['volcano', 'volcanic_cone', 120, 512], ['crater_lake', 'volcanic_lake', 90], ['lowland_river', 'lowland_meanders', 360],
];
const sim = new Erosion({ canvas: document.querySelector('canvas') });
const errors = [];
let initial, initialStats;
simReady();
async function simReady() {
  await sim.initialize();
  sim.device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  window.examplesReady = true;
}
async function readState() {
  const target = sim.device.createBuffer({ size: sim.stateByteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = sim.device.createCommandEncoder();
  encoder.copyBufferToBuffer(sim.stateBuffers[0], 0, target, 0, sim.stateByteLength);
  sim.device.queue.submit([encoder.finish()]);
  await target.mapAsync(GPUMapMode.READ);
  const state = new Float32Array(target.getMappedRange()).slice();
  target.unmap(); target.destroy();
  return state;
}
window.loadExample = async (terrain, presetName, size = 256) => {
  const preset = EROSION_PRESETS[presetName];
  sim.setSimulationParams({ ...preset, closedBoundaries: !!TERRAIN_EXAMPLES[terrain].closed,
    talusSlopeCoeff: preset.talusCoeff, talusSlopeBias: preset.talusBias,
    hardnessBase: preset.erodabilityMin, hardnessVariation: preset.erodabilityMax - preset.erodabilityMin,
    renderGridStep: 1,
    ...(terrain === 'lowland_river' ? { cameraAzimuthDeg: 45, cameraElevationDeg: 65,
      cameraPosX: 0.866, cameraPosY: 2.748, cameraPosZ: 0.866 } : { cameraAzimuthDeg: 45, cameraElevationDeg: 42,
      cameraPosX: 1.524, cameraPosY: 2.060, cameraPosZ: 1.524 }),
  });
  const raster = await generateTerrainExample(sim.device, terrain, size, 7);
  await sim.setDEM(raster, terrainExampleOptions(terrain, preset));
  seedTerrainExampleSources(sim, raster, terrain);
  initial = await readState();
  initialStats = await sim.readbackStats();
  sim.render(); await sim.device.queue.onSubmittedWorkDone();
};
window.advanceExample = async seconds => {
  const steps = Math.round(seconds / sim.simulationParams.timeStep);
  for (let i = 0; i < steps; i += 80) {
    sim.step(Math.min(80, steps - i));
    await sim.device.queue.onSubmittedWorkDone();
  }
  const final = await readState();
  let eroded = 0, deposited = 0, maxChange = 0;
  for (let i = 0; i < final.length; i += 8) {
    if (!final.slice(i, i + 8).every(Number.isFinite) || final[i] < -1e-7 || final[i + 1] < 0 || final[i + 2] < 0) throw Error(`Invalid cell ${i / 8}`);
    const delta = final[i] - initial[i];
    eroded += Math.max(-delta, 0); deposited += Math.max(delta, 0); maxChange = Math.max(maxChange, Math.abs(delta));
  }
  const stats = await sim.readbackStats();
  const drift = (stats.totalSolid - initialStats.totalSolid) / initialStats.totalSolid;
  if (sim.simulationParams.closedBoundaries && Math.abs(drift) > 2e-5) throw Error(`Closed solid drift ${drift}`);
  if (errors.length || stats.invalidCells) throw Error(errors.join('\n') || 'Non-finite cells');
  sim.render(); await sim.device.queue.onSubmittedWorkDone();
  return { seconds, size: sim.width, seed: 7, eroded, deposited, maxChange, solidChangeFraction: drift,
    water: stats.totalWater, sediment: stats.totalSediment, range: stats.terrainRange, closed: sim.simulationParams.closedBoundaries };
};
