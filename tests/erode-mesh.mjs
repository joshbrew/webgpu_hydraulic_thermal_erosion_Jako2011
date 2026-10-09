import assert from 'node:assert/strict';
import http from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import { pathToFileURL } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ? pathToFileURL(process.env.PLAYWRIGHT_MODULE).href : 'playwright');
const root = resolve('.');
const server = http.createServer(async (req,res) => {
  try {
    const p = resolve(root,'.'+req.url.split('?')[0]);
    if (!p.startsWith(root+'\\') && !p.startsWith(root+'/')) throw Error('Outside root');
    res.writeHead(200,{'Content-Type':extname(p)==='.html'?'text/html':'text/javascript'}); res.end(await readFile(p));
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => server.listen(0,'127.0.0.1',r));
const browser = await chromium.launch({ channel:'chrome',headless:true,args:['--enable-unsafe-webgpu'] });
const reports=[];
try {
  const page = await browser.newPage({viewport:{width:1450,height:1100}}), errors=[];
  page.on('pageerror',e => errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/examples/erode-mesh.html`);
  const ready = () => page.waitForFunction(() => window.erosionMeshStudio?.state.ready && !window.erosionMeshStudio.busy,undefined,{timeout:90000});
  await ready();
  await page.locator('#size').selectOption('256'); await ready();
  for (const scene of ['lowland_river','river_delta','volcano','mountain_canyon','crater_lake']) {
    await page.locator('#scene').selectOption(scene); await ready();
    await page.evaluate(() => window.erosionMeshStudio.advance(3)); await ready();
    // Test-only state inspection: all eight cell components must remain bit-identical.
    await page.evaluate(async () => {
      const s=window.erosionMeshStudio.sim;
      window.readMeshTestState=async () => {
        const b=s.device.createBuffer({size:s.cellCount*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const e=s.device.createCommandEncoder();e.copyBufferToBuffer(s.stateBuffers[0],0,b,0,b.size);s.device.queue.submit([e.finish()]);
        await b.mapAsync(GPUMapMode.READ);const a=new Uint32Array(b.getMappedRange()).slice();b.unmap();b.destroy();return a;
      };
      window.meshStateBefore=await window.readMeshTestState();window.meshTimeBefore=s.getStats().iterationCount;
    });
    const snapshotsBefore=await page.evaluate(() => window.erosionMeshStudio.state.snapshots);
    await page.locator('#build').click(); await ready();
    const result=await page.evaluate(async () => {
      const s=window.erosionMeshStudio, after=await window.readMeshTestState();
      return { ...s.mesh,scene:document.getElementById('scene').value,stateUnchanged:after.every((v,i)=>v===window.meshStateBefore[i]),iterationUnchanged:s.sim.getStats().iterationCount===window.meshTimeBefore,snapshots:s.state.snapshots,errors:s.state.errors };
    });
    assert(result.stateUnchanged && result.iterationUnchanged,'Mesh build must preserve the complete simulation state and time');
    assert(result.triangles>10 && result.triangles<result.gridTriangles*.5,'Adaptive mesh should substantially reduce triangle count');
    assert.equal(result.snapshots,snapshotsBefore+1); assert.deepEqual(result.errors,[]);
    const water=await page.evaluate(async()=>{
      const {default:Water}=await import('/StaticMeshWater.js'),a=window.erosionMeshStudio;
      const swe=await new Water(a.sim,a.topology,a.renderer).initialize({rainRate:.001});
      try{
        for(let i=0;i<4;i++)swe.step(128);
        a.renderer.render({swe});await a.sim.device.queue.onSubmittedWorkDone();
        const stats=await swe.readStats(),after=await window.readMeshTestState();
        return {invalid:stats.invalid,steps:stats.steps,dt:stats.dt,unchanged:after.every((v,i)=>v===window.meshStateBefore[i])};
      }finally{swe.destroy();a.renderer.render();}
    });
    assert(water.invalid===0&&water.dt>0&&water.steps===512&&water.unchanged,'Fixed mesh water must stay finite and preserve the full raster on every landscape');
    const withWire=await page.locator('canvas').screenshot();
    await page.locator('#wire-opacity').fill('0.2'); const lightWire=await page.locator('canvas').screenshot();
    assert.notEqual(Buffer.compare(withWire,lightWire),0,'Wireframe opacity must affect the rendered lines');
    await page.locator('#wire-opacity').fill('0.55');
    await page.locator('#wire').uncheck(); const plain=await page.locator('canvas').screenshot();
    assert.notEqual(Buffer.compare(withWire,plain),0,'Actual triangle edges must change the rendered image');
    await page.locator('#surface').selectOption('grid'); const grid=await page.locator('canvas').screenshot();
    assert.notEqual(Buffer.compare(plain,grid),0,'Grid/adaptive comparison must use distinct geometry');
    for (const color of ['7','9','6','8']) await page.locator('#color').selectOption(color);
    await page.locator('#min-spacing').fill('4'); await page.locator('#build').click(); await ready();
    assert.equal(await page.evaluate(() => window.erosionMeshStudio.state.snapshots),result.snapshots,'Spacing rebuild should reuse the cached snapshot');
    await page.locator('#wire').check();
    if (scene==='volcano') await page.screenshot({path:resolve(root,'tests/erode-mesh-preview.png'),fullPage:true});
    await page.locator('#run').click(); await page.waitForFunction(() => window.erosionMeshStudio.sim.getStats().iterationCount>window.meshTimeBefore+64);
    assert.equal(await page.evaluate(() => window.erosionMeshStudio.mesh),null,'Resuming must invalidate the final mesh');
    await page.locator('#run').click(); await ready();
    await page.locator('#build').click(); await ready();
    assert.equal(await page.evaluate(() => window.erosionMeshStudio.state.snapshots),result.snapshots+1,'Evolved terrain requires a fresh snapshot');
    reports.push(result); console.log(JSON.stringify(result));
  }
  // Full-size, evolved example with a worker blob opened directly from disk.
  await page.goto(pathToFileURL(resolve(root,'examples/erode-mesh.html')).href+'?scene=lowland_river&seconds=180&mesh=1'); await ready();
  const evolved=await page.evaluate(() => ({...window.erosionMeshStudio.mesh,time:window.erosionMeshStudio.sim.getStats().simTime,errors:window.erosionMeshStudio.state.errors}));
  console.log('Evolved file preview:',JSON.stringify(evolved));
  // The solver integrates with the float32 time step uploaded to the GPU.
  assert(Math.abs(evolved.time-180)<1e-4); assert.deepEqual(evolved.errors,[]);
  await page.screenshot({path:resolve(root,'tests/erode-mesh-river.png'),fullPage:true});
  await page.setViewportSize({width:430,height:900});
  assert(await page.evaluate(() => document.documentElement.scrollWidth<=innerWidth),'Mobile controls should fit the viewport');
  assert.deepEqual(errors,[]);
  reports.push({scene:'evolved lowland / file',...evolved});
  await writeFile(resolve(root,'tests/erode-mesh-report.json'),JSON.stringify(reports,null,2)+'\n');
  console.log('Post-process state conservation, fixed-mesh GPU water, comparison/wireframes, cache/rebuild/resume, five scenes, evolved 512 river, mobile and file:// passed.');
} finally {await browser.close();server.close();}
