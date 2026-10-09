import Erosion from '../WebGPUTerrainErosion.js';
import BatchPacer from '../BatchPacer.js';
import MeshPostProcess from '../MeshPostProcess.js';
import createMeshControls from '../MeshControls.js';
import { EROSION_PRESETS } from '../erosionPresets.js';
import { EXAMPLE_MATERIALS, TERRAIN_EXAMPLES, generateTerrainExample, seedTerrainExampleSources, terrainExampleOptions } from '../terrainExamples.js';

// One reusable live viewport/device, including when opened directly from disk.
// The classic browser bundle contains all shaders and generator code.
const canvas = document.createElement('canvas');
canvas.width = 800; canvas.height = 600;
canvas.setAttribute('aria-label', 'Live erosion preview');
canvas.style.touchAction = 'none';
const panel = document.createElement('div');
panel.className = 'live-panel';
const toolbar = document.createElement('div');
toolbar.className = 'live-toolbar';
const button = text => { const b = document.createElement('button'); b.type = 'button'; b.textContent = text; return b; };
const play = button('Pause'), reset = button('Reset'), result = button('View result'), close = button('Close');
const view = document.createElement('select');
const loadLabel = document.createElement('label'); loadLabel.textContent = 'River sediment load ';
const sedimentLoad = document.createElement('input'); sedimentLoad.type = 'number';
sedimentLoad.min = '0'; sedimentLoad.max = '1'; sedimentLoad.step = '0.01'; sedimentLoad.style.width = '80px';
sedimentLoad.title = 'Sediment supplied per unit of spring water. Zero supplies clear water.';
loadLabel.append(sedimentLoad);
view.setAttribute('aria-label', 'Live view');
for (const [value, label] of [[7, 'Natural'], [8, 'Slope relief'], [9, 'River flow'], [4, 'Erodability'], [6, 'Erosion and deposition']]) {
  const option = document.createElement('option'); option.value = value; option.textContent = label; view.append(option);
}
const status = document.createElement('span'); status.setAttribute('role', 'status');
status.className = 'live-status';
toolbar.append(play, reset, result, view, loadLabel, close, status);
const hint = document.createElement('p'); hint.textContent = 'Drag to orbit · scroll to zoom. Run continues until paused. View result advances to the pictured time. River sediment load controls material arriving with spring water.';
const legend = document.createElement('div'); legend.className = 'layer-legend';
const meshControls=createMeshControls({
  onbuild:async options=>{
    if(running||loading||fastForward)return;loading=true;updateStatus('Building hydraulic mesh…');
    let failure;
    try{mesh??=new MeshPostProcess(sim,message=>meshControls.update({message,busy:true}));await mesh.build(options);}
    catch(error){failure=error.message;}
    finally{loading=false;updateStatus(failure);}
  },
  onview:next=>{mesh?.render(next);},
  onwater:async(action,options)=>{
    if(loading||fastForward)return;
    if(action==='inspect'){try{const w=await mesh?.swe?.readStats();if(!w)throw Error('Run mesh water first.');
      meshControls.update({message:`Mesh water: ${w.time.toFixed(2)} s · ${w.steps} steps · stable step ${w.dt.toFixed(5)} s\nWater volume ${w.totalWater.toPrecision(6)} · depth max ${w.maxDepth.toFixed(4)} · ${w.invalid} invalid cells`});}catch(error){meshControls.update({message:error.message});}return;}
    if(action==='pause'){waterRunning=false;stop();return;}
    if(action==='reset'){waterRunning=false;stop();mesh?.resetWater();updateStatus();return;}
    if(!mesh?.renderer)return;
    loading=true;updateStatus('Preparing water over the fixed mesh…');
    let failure;
    try{await mesh.startWater(options);waterRunning=true;running=true;const token=++playEpoch;pacer.reset(8);frameHandle=requestAnimationFrame(()=>frame(token));}
    catch(error){failure=error.message;}
    finally{loading=false;updateStatus(failure);}
  },
});
panel.append(toolbar, canvas, hint, legend, meshControls.root);
let sim, active, raster, loading = false, running = false, fastForward = false;
let epoch = 0, frameHandle = 0, playEpoch = 0;
let mesh=null,waterRunning=false;
function discardMesh(){waterRunning=false;mesh?.destroy();mesh=null;}
function draw(){if(mesh?.renderer)mesh.render();else sim.render();}
const pacer = new BatchPacer();
let azimuth = 45, elevation = 42, distance = 2.9;
let pointer = null;

function updateStatus(message) {
  status.textContent = message || `${sim?.getStats().simTime.toFixed(2) ?? '0.00'} simulated seconds${running ? ' · running' : ' · paused'}`;
  play.textContent = running ? 'Pause' : 'Run';
  play.disabled = reset.disabled = result.disabled = view.disabled = loading || fastForward || !sim?.ready || !active;
  sedimentLoad.disabled = view.disabled || !active?.preset.sourceEnabled;
  result.disabled ||= !sim?.ready || sim.getStats().iterationCount >= (active?.picturedSteps ?? 0);
  panel.dataset.running = String(running);
  panel.dataset.time = String(sim?.getStats().simTime ?? 0);
  meshControls.update({info:mesh?.info||null,busy:loading||fastForward,running:running&&!waterRunning,waterRunning,waterSteps:mesh?.swe?.steps||0,ready:!!sim?.ready&&!!active,message});
}
function stop() {
  running = false; waterRunning=false; playEpoch++;
  cancelAnimationFrame(frameHandle); frameHandle = 0;
  if (active) updateStatus();
}
function camera() {
  if (!sim?.ready) return;
  const a = azimuth * Math.PI / 180, e = elevation * Math.PI / 180;
  sim.setSimulationParams({ cameraAzimuthDeg: azimuth, cameraElevationDeg: elevation,
    cameraPosX: Math.cos(e) * Math.cos(a) * distance,
    cameraPosY: 0.12 + Math.sin(e) * distance, cameraPosZ: Math.cos(e) * Math.sin(a) * distance });
  draw();
}
async function frame(token) {
  frameHandle = 0;
  if (!running || token !== playEpoch || loading || fastForward) return;
  try {
    const steps = pacer.steps, start = performance.now();
    if(waterRunning)mesh.stepWater(steps);else sim.stepAndRender(steps);
    await sim.device.queue.onSubmittedWorkDone();
    if (!running || token !== playEpoch) return;
    pacer.observe(steps, performance.now() - start);
    updateStatus(); frameHandle = requestAnimationFrame(() => frame(token));
  } catch (error) { stop(); updateStatus(error.message); }
}
function start() {
  if (!sim?.ready || loading || fastForward || running) return;
  discardMesh();running = true; const token = ++playEpoch;
  pacer.reset(active.preset.iterationsPerFrame);
  updateStatus(); frameHandle = requestAnimationFrame(() => frame(token));
}
async function restore() {
  discardMesh();
  await sim.device.queue.onSubmittedWorkDone();
  const p = active.preset;
  sim.setSimulationParams({ ...p, closedBoundaries: !!TERRAIN_EXAMPLES[active.terrain].closed,
    talusSlopeCoeff: p.talusCoeff, talusSlopeBias: p.talusBias,
    hardnessBase: p.erodabilityMin, hardnessVariation: p.erodabilityMax - p.erodabilityMin,
    renderGridStep: 0, renderMode: Number(view.value), sourceTimeOffset: 0,
    sourceSedimentConcentration: Math.max(0, Math.min(1, Number(sedimentLoad.value) || 0)) });
  await sim.setDEM(raster, terrainExampleOptions(active.terrain, p));
  seedTerrainExampleSources(sim, raster, active.terrain);
  active.seconds = active.picturedSeconds;
  active.picturedSteps = Math.round(active.picturedSeconds / p.timeStep);
  camera(); updateStatus();
}
async function load(card) {
  stop(); const token = ++epoch; loading = true;
  const launch = card.querySelector('.run-sim');
  launch.disabled = true;
  card.append(panel); panel.hidden = false;
  updateStatus('Preparing terrain…');
  try {
    if (!sim) { sim = new Erosion({ canvas }); await sim.initialize(); }
    await sim.device.queue.onSubmittedWorkDone();
    if (token !== epoch) return;
    const terrain = card.dataset.terrain, preset = EROSION_PRESETS[card.dataset.preset];
    if (!preset || !TERRAIN_EXAMPLES[terrain]) throw Error('Unknown example.');
    active = { card, terrain, preset, picturedSeconds: Number(card.dataset.seconds), seconds: Number(card.dataset.seconds) };
    raster = await generateTerrainExample(sim.device, terrain, Number(card.dataset.size) || 256, Number(card.dataset.seed) || 7);
    if (token !== epoch) return;
    azimuth = 45; elevation = terrain === 'lowland_river' ? 65 : 42; distance = 2.9; view.value = '8';
    sedimentLoad.value = String(preset.sourceSedimentConcentration || 0);
    legend.replaceChildren();
    EXAMPLE_MATERIALS[terrain].forEach((material, i) => {
      const item = document.createElement('span'); item.textContent = material.label;
      item.style.borderColor = ['#9c8a66', '#856647', '#6e7066', '#4a4f4d'][i]; legend.append(item);
    });
    await restore();
    loading = false; start();
  } catch (error) { loading = false; updateStatus(error.message); }
  finally { launch.disabled = false; if (token !== epoch) { loading = false; updateStatus(); } }
}
document.querySelectorAll('.run-sim').forEach(launch => launch.addEventListener('click', () => {
  if (loading || fastForward) return;
  const card = launch.closest('article');
  if (active?.card === card && sim?.ready) { card.append(panel); panel.hidden = false; start(); }
  else void load(card);
}));
play.addEventListener('click', () => running ? stop() : start());
reset.addEventListener('click', async () => {
  stop(); loading = true; updateStatus('Restoring terrain…');
  let failure;
  try { await restore(); } catch (error) { failure = error.message; }
  finally { loading = false; updateStatus(failure); }
});
result.addEventListener('click', async () => {
  stop(); discardMesh();fastForward = true; updateStatus('Advancing to the pictured result…');
  const token = epoch;
  let failure;
  try {
    while (token === epoch && sim.getStats().iterationCount < active.picturedSteps) {
      const steps = Math.min(128, active.picturedSteps - sim.getStats().iterationCount);
      sim.step(steps); sim.render(); await sim.device.queue.onSubmittedWorkDone();
      updateStatus(`Advancing… ${sim.getStats().simTime.toFixed(1)} / ${active.picturedSeconds} seconds`);
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
  } catch (error) { failure = error.message; }
  finally { fastForward = false; updateStatus(failure); }
});
view.addEventListener('change', () => { sim.setSimulationParams({ renderMode: Number(view.value) }); draw(); });
sedimentLoad.addEventListener('input', () => {
  sim.setSimulationParams({ sourceSedimentConcentration: Math.max(0, Math.min(1, Number(sedimentLoad.value) || 0)) });
});
close.addEventListener('click', () => { stop(); epoch++; panel.hidden = true; });
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
window.addEventListener('pagehide', () => { stop();discardMesh(); sim?.destroy(); });
canvas.addEventListener('pointerdown', event => {
  pointer = { id: event.pointerId, x: event.clientX, y: event.clientY }; canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', event => {
  if (!pointer || event.pointerId !== pointer.id) return;
  azimuth += (event.clientX - pointer.x) * 0.35;
  elevation = Math.max(12, Math.min(85, elevation + (event.clientY - pointer.y) * 0.25));
  pointer.x = event.clientX; pointer.y = event.clientY; camera();
});
for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(event, () => { pointer = null; });
canvas.addEventListener('wheel', event => {
  event.preventDefault(); distance = Math.max(1.1, Math.min(5, distance * Math.exp(event.deltaY * 0.001))); camera();
}, { passive: false });
