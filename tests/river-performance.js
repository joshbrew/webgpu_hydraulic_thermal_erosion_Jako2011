import Current from '../WebGPUTerrainErosion.js';
import { EROSION_PRESETS } from '../erosionPresets.js';
import { generateTerrainExample, terrainExampleOptions, seedTerrainExampleSources } from '../terrainExamples.js';
const median = values => values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
export async function benchmark(Baseline = null) {
  const current = new Current({ canvas: document.createElement('canvas') });
  const baseline = Baseline ? new Baseline({ canvas: document.createElement('canvas') }) : null;
  const versions = [...(baseline ? [['before', baseline]] : []), ['after', current]], report = [];
  try {
    for (const [, sim] of versions) await sim.initialize();
    for (const size of [256, 512, 1024]) {
      const raster = await generateTerrainExample(current.device, 'lowland_river', size, 7), p = EROSION_PRESETS.lowland_meanders;
      for (const [, sim] of versions) {
        sim.setSimulationParams({ ...p, hardnessBase: p.erodabilityMin, hardnessVariation: p.erodabilityMax - p.erodabilityMin, talusSlopeCoeff: p.talusCoeff, talusSlopeBias: p.talusBias });
        await sim.setDEM(raster, terrainExampleOptions('lowland_river', p)); seedTerrainExampleSources(sim, raster, 'lowland_river');
        sim.step(120); await sim.device.queue.onSubmittedWorkDone();
      }
      const measurements = Object.fromEntries(versions.map(([version]) => [version, { gpu: [], wall: [] }]));
      for (let round = 0; round < 5; round++) for (const [version, sim] of round % 2 ? [...versions].reverse() : versions) {
        const start = performance.now(), result = await sim.stepBench(100);
        measurements[version].wall.push((performance.now() - start) / 100);
        if (result.gpuErosionMs != null) measurements[version].gpu.push(result.gpuErosionMs / 100);
      }
      report.push({ size, ...Object.fromEntries(Object.entries(measurements).map(([version, times]) => [version, { gpuMsPerStep: times.gpu.length ? median(times.gpu) : null, completionMsPerStep: median(times.wall) }])) });
    }
    return report;
  } finally { for (const [, sim] of versions) sim.destroy(); }
}
