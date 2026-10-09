import Current from '../WebGPUTerrainErosion.js';
import Baseline from './baseline.js';
import { generateTerrainExample } from '../terrainExamples.js';

const median = values => values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
export async function benchmark() {
  const report = [];
  const before = new Baseline({ canvas: document.createElement('canvas') });
  const after = new Current({ canvas: document.createElement('canvas') });
  for (const sim of [before, after]) { sim.canvas.width = 640; sim.canvas.height = 480; await sim.initialize(); }
  for (const size of [256, 512]) {
    const raster = await generateTerrainExample(after.device, 'alpine', size, 7);
    for (const [thermal, springs] of [[false, false], [true, false], [true, true]]) {
      const times = [];
      for (const sim of [before, after]) {
        sim.setSimulationParams({ rainRate: 0.001, evaporationRate: 0.025, thermalRate: thermal ? 0.25 : 0,
          sourceEnabled: springs, sourceLayoutMode: 1, randomSpringCount: 4,
          timeStep: 0.02, talusSlopeCoeff: 0.012, talusSlopeBias: 0.002 });
        await sim.setDEM(raster);
        sim.step(10); await sim.device.queue.onSubmittedWorkDone();
        const samples = [];
        for (let i = 0; i < 3; i++) {
          const start = performance.now(); sim.step(40); await sim.device.queue.onSubmittedWorkDone();
          samples.push((performance.now() - start) / 40);
        }
        times.push(median(samples));
      }
      const gpuSamples = [];
      for (let i = 0; i < 3; i++) gpuSamples.push((await after.stepBench(40)).gpuErosionMs / 40);
      report.push({ size, thermal, springs, beforeCompletionMsPerStep: times[0], afterCompletionMsPerStep: times[1],
        currentGpuMsPerStep: median(gpuSamples), gpuTimingMode: after.getStats().gpuTimingMode });
    }
  }
  const raster = await generateTerrainExample(after.device, 'alpine', 1024, 7);
  for (const sim of [before, after]) {
    await sim.setDEM(raster); sim.render(); await sim.device.queue.onSubmittedWorkDone();
    const samples = [];
    for (let i = 0; i < 5; i++) {
      const start = performance.now(); sim.render(); await sim.device.queue.onSubmittedWorkDone();
      samples.push(performance.now() - start);
    }
    let gpuRenderMs;
    if (sim === after) {
      const gpuSamples = [];
      for (let i = 0; i < 3; i++) gpuSamples.push((await after.stepAndRenderBench(1)).gpuRenderMs);
      gpuRenderMs = median(gpuSamples);
    }
    report.push({ renderSize: 1024, version: sim === before ? 'before, full mesh' : 'after, auto mesh',
      renderCompletionMs: median(samples), gpuRenderMs, meshCells: sim.drawCount });
  }
  before.destroy(); after.destroy();
  return report;
}
