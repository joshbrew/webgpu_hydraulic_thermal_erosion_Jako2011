import Erosion from '../WebGPUTerrainErosion.js';
import ErosionMeshRenderer from '../ErosionMeshRenderer.js';
import BuildCoordinator from '../vendor/hydraulic-tin/BuildCoordinator.js';
import BatchPacer from '../BatchPacer.js';
import { EROSION_PRESETS } from '../erosionPresets.js';
import { TERRAIN_EXAMPLES, EXAMPLE_MATERIALS, generateTerrainExample, seedTerrainExampleSources, terrainExampleOptions } from '../terrainExamples.js';

const $ = id => document.getElementById(id), canvas = $('terrain');
const times = { lowland_river: 180, river_delta: 90, mountain_canyon: 90, volcano: 90, crater_lake: 90 };
let sim, coordinator, renderer, raster, snapshot, meshInfo, running = false, busy = false, frameHandle = 0, epoch = 0;
let currentTopology=null;
let azimuth = 45, elevation = 52, distance = 2.8, drag;
const pacer = new BatchPacer();
const state = { ready: false, snapshots: 0, meshIteration: null, errors: [] };
const preview = new URLSearchParams(location.search);
if (Object.hasOwn(times,preview.get('scene'))) $('scene').value=preview.get('scene');
if (['256','512','1024'].includes(preview.get('size'))) $('size').value=preview.get('size');

function status(message) {
  const stats = sim?.getStats();
  $('status').textContent = message || `${stats?.simTime.toFixed(2) ?? '0.00'} simulated seconds · ${stats?.iterationCount ?? 0} steps · ${running ? 'running' : 'paused'}`;
  $('run').textContent = running ? 'Pause' : 'Run';
  for (const id of ['scene','size','run','reset','advance','build','mode','min-spacing','max-spacing','guides','color']) $(id).disabled = busy || !state.ready;
  $('surface').disabled = $('wire').disabled = $('wire-opacity').disabled = busy || !renderer;
  $('view-label').textContent = renderer ? ($('surface').value === 'adaptive' ? 'Final mesh · adaptive triangles' : 'Final mesh comparison · full grid') : 'Live erosion · grid surface';
  canvas.dataset.running = String(running);
}
function fail(error) { state.errors.push(error.message); $('error').textContent = error.message; status(); }
function stop() { running = false; epoch++; cancelAnimationFrame(frameHandle); frameHandle = 0; status(); }
function clearMesh() {
  renderer?.destroy(); renderer = null; snapshot = null; meshInfo = null; state.meshIteration = null;
  currentTopology=null;
  $('mesh-stats').textContent = 'Build a mesh to compare triangle counts.';
}
function draw() {
  if (!sim?.ready) return;
  if (renderer) renderer.render({ adaptive: $('surface').value === 'adaptive', wireframe: $('wire').checked, wireOpacity: Number($('wire-opacity').value) });
  else sim.render();
}
function camera() {
  if (!sim?.ready) return;
  const a = azimuth*Math.PI/180, e = elevation*Math.PI/180;
  sim.setSimulationParams({ cameraAzimuthDeg: azimuth, cameraElevationDeg: elevation,
    cameraPosX: Math.cos(e)*Math.cos(a)*distance, cameraPosY: .12+Math.sin(e)*distance, cameraPosZ: Math.cos(e)*Math.sin(a)*distance });
  draw();
}
async function task(work) {
  if (busy) return;
  stop(); busy = true; $('error').textContent = ''; status();
  try { await work(); } catch (error) { fail(error); }
  finally { busy = false; status(); }
}
async function load() {
  clearMesh(); await sim.device.queue.onSubmittedWorkDone();
  status('Preparing layered terrain…');
  const scene = $('scene').value, p = EROSION_PRESETS[TERRAIN_EXAMPLES[scene].preset];
  raster = await generateTerrainExample(sim.device, scene, Number($('size').value), 7);
  sim.setSimulationParams({ ...p, closedBoundaries: !!TERRAIN_EXAMPLES[scene].closed,
    talusSlopeCoeff: p.talusCoeff, talusSlopeBias: p.talusBias,
    hardnessBase: p.erodabilityMin, hardnessVariation: p.erodabilityMax-p.erodabilityMin,
    renderGridStep: 1, renderMode: Number($('color').value), sourceTimeOffset: 0 });
  await sim.setDEM(raster, terrainExampleOptions(scene,p));
  seedTerrainExampleSources(sim,raster,scene);
  $('layers').textContent = 'Material layers · '+EXAMPLE_MATERIALS[scene].map(l => l.label).join(' / ');
  state.ready = true; camera();
}
async function frame(token) {
  if (!running || busy || token !== epoch) return;
  try {
    const steps = pacer.steps, start = performance.now();
    sim.stepAndRender(steps); await sim.device.queue.onSubmittedWorkDone();
    if (!running || token !== epoch) return;
    pacer.observe(steps,performance.now()-start); status(); frameHandle = requestAnimationFrame(() => frame(token));
  } catch (error) { stop(); fail(error); }
}
function start() {
  if (!state.ready || busy || running) return;
  clearMesh(); running = true; const token = ++epoch; pacer.reset(); status();
  frameHandle = requestAnimationFrame(() => frame(token));
}
async function advance(seconds = times[$('scene').value]) {
  clearMesh();
  const dt = sim.simulationParams.timeStep, target = Math.round(seconds/dt);
  pacer.reset(32);let lastPreview=0;
  while (sim.getStats().iterationCount < target) {
    const n = Math.min(pacer.steps,target-sim.getStats().iterationCount), start = performance.now();
    sim.step(n); await sim.device.queue.onSubmittedWorkDone(); pacer.observe(n,performance.now()-start);
    if(performance.now()-lastPreview>=100 || sim.getStats().iterationCount>=target){
      sim.render();status(`Evolving terrain · ${sim.getStats().simTime.toFixed(1)} / ${seconds} seconds`);lastPreview=performance.now();
    }
    await new Promise(requestAnimationFrame);
  }
}
async function capture() {
  if (snapshot) return snapshot;
  const copy = sim.device.createBuffer({ size: sim.cellCount*32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const e = sim.device.createCommandEncoder(); e.copyBufferToBuffer(sim.stateBuffers[0],0,copy,0,copy.size); sim.device.queue.submit([e.finish()]);
    await copy.mapAsync(GPUMapMode.READ);
    const cells = new Float32Array(copy.getMappedRange()), values = new Float32Array(sim.cellCount), mask = new Uint8Array(sim.cellCount);
    for (let i=0;i<sim.cellCount;i++) { values[i]=cells[i*8]; mask[i]=cells[i*8+4]>.5 ? 1 : 0; }
    copy.unmap(); state.snapshots++;
    snapshot = { width: sim.width, height: sim.height, values, mask, iteration: sim.getStats().iterationCount };
    return snapshot;
  } finally { copy.destroy(); }
}
async function buildMesh() {
  status('Capturing paused terrain…');
  const data = await capture(), begin = performance.now();
  coordinator ??= new BuildCoordinator({ debugLogging: false, onstatus: message => status(message) });
  // setRaster transfers ownership to the worker; keep the cached snapshot intact.
  await coordinator.setRaster({ width: data.width, height: data.height, values: data.values.slice(), mask: data.mask.slice() });
  const minSpacing = Math.max(1,Math.min(128,Number($('min-spacing').value)||3));
  const maxSpacing = Math.max(minSpacing,Math.min(256,Number($('max-spacing').value)||28));
  $('min-spacing').value=minSpacing; $('max-spacing').value=maxSpacing;
  let lo=Infinity, hi=-Infinity; for (let i=0;i<data.values.length;i++) if (data.mask[i]) { lo=Math.min(lo,data.values[i]); hi=Math.max(hi,data.values[i]); }
  const rise=Math.max(hi-lo,.001);
  const result = await coordinator.build({
    topologyBuildOptions: { mode: $('mode').value, cellSize: 1, minSpacing, maxSpacing, borderSpacing: maxSpacing, wetnessBins: 21, channelThreshold: 256, candidateStride: 1, relaxIterations: 0, preserveBorder: true, preserveChannels: false, preserveMaskBoundary: true, preserveInnerBorder: true },
    riverOptions: { maxNeighborDistance: Infinity, boundaryEpsilon: .5, minDrop: 1e-5, flatTolerance: 1e-4, sinkMaxRise: rise*.015, sinkClimbPenalty: 2.5 },
    guideOptions: { minWeight: 3, step: Math.max(minSpacing*1.75, Math.min(maxSpacing*.6,minSpacing*5)), maxPoints: Math.min(16000,Math.floor(sim.cellCount*.45)) },
    floodplainOptions: { maxRise: rise*.05, maxClimbStep: rise*.015, maxEdgeSlope: .75, maxWorldDistance: Math.max(minSpacing*6,maxSpacing*1.5), minTriangleFloodplainVerts: 2 },
    minRiverWeight: 6, guideEnabled: $('guides').checked,
  });
  status('Uploading final triangles…');
  const next = new ErosionMeshRenderer(sim);
  try { await next.initialize(result.topology); } catch (error) { next.destroy(); throw error; }
  renderer?.destroy(); renderer=next; state.meshIteration=data.iteration;
  currentTopology=result.topology;
  meshInfo = { vertices: result.topology.positionsGrid.length/2, triangles: result.topology.indices.length/3, gridTriangles: (sim.width-1)*(sim.height-1)*2, buildMs: performance.now()-begin };
  $('mesh-stats').textContent = `${meshInfo.vertices.toLocaleString()} vertices · ${meshInfo.triangles.toLocaleString()} triangles\nFull grid: ${meshInfo.gridTriangles.toLocaleString()} triangles\n${(100*(1-meshInfo.triangles/meshInfo.gridTriangles)).toFixed(1)}% fewer · ${(meshInfo.buildMs/1000).toFixed(2)} s build\nFrozen at ${sim.getStats().simTime.toFixed(2)} simulated seconds`;
  $('surface').value='adaptive'; draw(); await sim.device.queue.onSubmittedWorkDone();
}
$('run').onclick = () => running ? stop() : start();
$('reset').onclick = () => task(load);
$('advance').onclick = () => task(() => advance());
$('build').onclick = () => task(buildMesh);
for (const id of ['scene','size']) $(id).onchange = () => task(load);
for (const id of ['surface','wire']) $(id).onchange = () => { draw(); status(); };
$('wire-opacity').oninput = draw;
$('color').onchange = () => { sim.setSimulationParams({ renderMode: Number($('color').value) }); draw(); };
canvas.onpointerdown = e => { drag=[e.clientX,e.clientY]; canvas.setPointerCapture(e.pointerId); };
canvas.onpointermove = e => { if (!drag || busy) return; azimuth+=(e.clientX-drag[0])*.35; elevation=Math.max(10,Math.min(85,elevation+(e.clientY-drag[1])*.25)); drag=[e.clientX,e.clientY]; camera(); };
canvas.onpointerup = canvas.onpointercancel = () => { drag=null; };
canvas.addEventListener('wheel',e => { e.preventDefault(); if (busy) return; distance=Math.max(1.1,Math.min(8,distance*Math.exp(e.deltaY*.001))); camera(); },{ passive:false });
window.addEventListener('pagehide', () => { stop(); coordinator?.dispose(); renderer?.destroy(); sim?.destroy(); });
// Diagnostics also support conservation checks without injecting hooks into the solver.
window.erosionMeshStudio = { state, get sim() { return sim; }, get mesh() { return meshInfo; }, get renderer(){return renderer;},get topology(){return currentTopology;},get busy() { return busy; }, advance: seconds => task(() => advance(seconds)) };
async function initialize() {
  busy=true; status(); sim=new Erosion({canvas}); await sim.initialize();
  sim.device.addEventListener('uncapturederror',e => fail(e.error));
  sim.device.lost.then(info => { if (info.reason==='destroyed') return; stop(); state.ready=false; fail(Error(info.message || 'GPU device lost. Reload to reconnect.')); });
  await load();
  const seconds=Number(preview.get('seconds'));
  if (Number.isFinite(seconds) && seconds>0 && seconds<=720) await advance(seconds);
  if (preview.get('mesh')==='1') await buildMesh();
  busy=false; status();
}
initialize().catch(error => { busy=false; fail(error); });
