import http from 'node:http';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,extname} from 'node:path';
import {pathToFileURL} from 'node:url';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE?pathToFileURL(process.env.PLAYWRIGHT_MODULE).href:'playwright');
const root=resolve('.'),server=http.createServer(async(req,res)=>{try{const p=resolve(root,'.'+req.url.split('?')[0]);if(!p.startsWith(root+'\\')&&!p.startsWith(root+'/'))throw Error('path');res.writeHead(200,{'Content-Type':extname(p)==='.html'?'text/html':'text/javascript'});res.end(await readFile(p));}catch{res.writeHead(404);res.end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const browser=await chromium.launch({channel:'chrome',headless:true,args:['--enable-unsafe-webgpu']});
try{const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/examples/erode-mesh.html?size=512&seconds=4&mesh=1`);
  await page.waitForFunction(()=>window.erosionMeshStudio?.mesh&&!window.erosionMeshStudio.busy,undefined,{timeout:90000});
  const report=await page.evaluate(async()=>{
    const {default:Water}=await import('/StaticMeshWater.js'),{default:Renderer}=await import('/ErosionMeshRenderer.js');
    const a=window.erosionMeshStudio,sim=a.sim,errors=[];sim.device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
    const median=values=>values.slice().sort((a,b)=>a-b)[Math.floor(values.length/2)];
    async function measure(topology,renderer){const water=await new Water(sim,topology,renderer).initialize({roughness:.025,rainRate:.001});
      water.step(200);await sim.device.queue.onSubmittedWorkDone();const batches=[];for(let i=0;i<5;i++)batches.push(await water.benchmark(100));
      const stats=await water.readStats();water.destroy();if(stats.invalid)throw Error('Non-finite benchmark state');
      return {triangles:topology.indices.length/3,gpuMsPerStep:batches[0].gpuMs===null?null:median(batches.map(b=>b.gpuMsPerStep)),completionMsPerStep:median(batches.map(b=>b.completionMs/100)),batches,dt:stats.dt};}
    const adaptive=await measure(a.topology,a.renderer),n=sim.cellCount,p=new Float32Array(n*2),indices=new Uint32Array((sim.width-1)*(sim.height-1)*6);
    for(let y=0;y<sim.height;y++)for(let x=0;x<sim.width;x++){const i=y*sim.width+x;p[i*2]=x;p[i*2+1]=y;}
    let k=0;for(let y=0;y<sim.height-1;y++)for(let x=0;x<sim.width-1;x++){const i=y*sim.width+x;indices.set([i,i+1,i+sim.width,i+1,i+sim.width+1,i+sim.width],k);k+=6;}
    const topology={positionsGrid:p,indices},renderer=await new Renderer(sim).initialize(topology),fullGrid=await measure(topology,renderer);renderer.destroy();
    if(errors.length)throw Error(errors.join('\n'));
    return {adapter:sim.adapter.info?.vendor||'unavailable',resolution:[sim.width,sim.height],method:'Same fixed-bed SWE implementation: one shared edge flux, GPU CFL and four dispatches per step. 200 warmup steps; median of five 100-step batches. Compute only; meshes have different spatial fidelity and may select different timesteps.',adaptive,fullGrid};
  });
  console.log(JSON.stringify(report,null,2));await writeFile(resolve(root,'tests/mesh-water-performance-report.json'),JSON.stringify(report,null,2)+'\n');
}finally{await browser.close();server.close();}
