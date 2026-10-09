import Erosion from '../WebGPUTerrainErosion.js';
import Renderer from '../ErosionMeshRenderer.js';
import Water from '../StaticMeshWater.js';
const results=[],errors=[];
const assert=(yes,text)=>{if(!yes)throw Error(text);};
const sim=new Erosion({canvas:document.querySelector('canvas')});
const width=23,height=17,positionsGrid=new Float32Array(width*height*2),indices=[];
for(let y=0;y<height;y++)for(let x=0;x<width;x++){positionsGrid[(y*width+x)*2]=x;positionsGrid[(y*width+x)*2+1]=y;}
for(let y=0;y<height-1;y++)for(let x=0;x<width-1;x++){const i=y*width+x;indices.push(i,i+1,i+width,i+1,i+width+1,i+width);}
const topology={positionsGrid,indices:new Uint32Array(indices)};
async function read(buffer){const b=sim.device.createBuffer({size:buffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const e=sim.device.createCommandEncoder();e.copyBufferToBuffer(buffer,0,b,0,b.size);sim.device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);const f=new Float32Array(b.getMappedRange()).slice();b.unmap();return f;}finally{b.destroy();}}
async function scene(bed,h,options={}){const values=new Float32Array(width*height),water=new Float32Array(width*height);for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=y*width+x;values[i]=bed(x,y);water[i]=h(x,y,values[i]);}
  sim.setSimulationParams({closedBoundaries:true,cellSize:1,gravity:9.81,sourceEnabled:false,renderMode:8,...options.simulation});await sim.setDEM({width,height,values,water,mask:new Uint8Array(values.length).fill(1)});
  const mesh=options.topology||topology;
  const renderer=await new Renderer(sim).initialize(mesh),swe=await new Water(sim,mesh,renderer).initialize({roughness:0,...options});
  return {renderer,swe,geometry:await read(swe.geometry)};}
const mass=(s,g)=>{let sum=0;for(let i=0;i<s.length;i+=4)sum+=s[i]*g[(i/4)*12];return sum;};
const evolve=(s,n)=>{while(n>0){const batch=Math.min(256,n);s.step(batch);n-=batch;}};
async function check(name,fn){try{await fn();results.push({name,passed:true});}catch(e){results.push({name,passed:false,error:e.message});}}
try{await sim.initialize();sim.device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
await check('A curved raster lake remains at rest after coarse mesh projection',async()=>{
  const coarse={positionsGrid:new Float32Array([0,0,width-1,0,0,height-1,width-1,height-1]),indices:new Uint32Array([0,1,2,1,3,2])};
  const a=await scene((x,y)=>.02+.0004*x*y+.005*Math.sin(x*.4)*Math.cos(y*.3),(x,y,z)=>.3-z,{topology:coarse,maxTimeStep:.05});
  const initial=await a.swe.readStats();
  assert(initial.values.every((v,i)=>i%4!==0||Math.abs(v+a.geometry[(i/4)*12+1]-.3)<1e-7),'Projected free surface is not level');
  evolve(a.swe,1000);const final=await a.swe.readStats();
  assert(final.values.every((v,i)=>Math.abs(v-initial.values[i])<2e-6),'Projection created spurious lake motion');
  a.swe.destroy();a.renderer.destroy();
});
await check('Lake at rest on a stepped bed and a wet/dry shore',async()=>{
  const a=await scene((x,y)=>.03*(Math.floor(x/3)+Math.floor(y/5)),()=>0,{maxTimeStep:1});
  const initial=new Float32Array(a.swe.triangles*4);for(let t=0;t<a.swe.triangles;t++)initial[t*4]=Math.max(0,.15-a.geometry[t*12+1]);
  sim.device.queue.writeBuffer(a.swe.states[0],0,initial);evolve(a.swe,1000);const final=await a.swe.readStats();
  let motion=0,change=0;for(let i=0;i<initial.length;i+=4){change=Math.max(change,Math.abs(final.values[i]-initial[i]));motion=Math.max(motion,Math.hypot(final.values[i+1],final.values[i+2]));}
  assert(change<2e-6&&motion<2e-6,`Resting lake disturbed: ${change}, ${motion}`);assert(Math.abs(mass(final.values,a.geometry)-mass(initial,a.geometry))<1e-4,'Lake mass changed');
  a.renderer.render({swe:a.swe});await sim.device.queue.onSubmittedWorkDone();a.swe.destroy();a.renderer.destroy();
});
await check('Closed dam break conserves water and momentum on mixed triangle orientations',async()=>{
  const a=await scene(()=>0,x=>x<10?.2:.01,{maxTimeStep:1});const initial=await a.swe.readStats();evolve(a.swe,1200);const final=await a.swe.readStats();
  assert(Math.abs(mass(final.values,a.geometry)/mass(initial.values,a.geometry)-1)<2e-5,'Dam break lost water');
  assert(final.dt<1&&final.dt>0,'GPU CFL must shorten an unsafe requested timestep');
  assert(final.values.every(v=>Number.isFinite(v)),'Non-finite dam break state');assert(final.values.some((v,i)=>i%4!==0&&Math.abs(v)>.001),'Dam break did not move');
  assert(final.values.every((v,i)=>i%4!==0||v>=0),'Negative water depth');a.swe.destroy();a.renderer.destroy();
});
await check('Dry-bed flooding stays positive and conserves water',async()=>{
  const a=await scene(()=>0,(x,y)=>x<5&&y>4&&y<12?.3:0,{maxTimeStep:.5});const initial=await a.swe.readStats();evolve(a.swe,1500);const final=await a.swe.readStats();
  assert(final.values.every((v,i)=>Number.isFinite(v)&&(i%4!==0||v>=0)),'Dry flooding is invalid');assert(Math.abs(mass(final.values,a.geometry)/mass(initial.values,a.geometry)-1)<3e-5,'Dry front lost water');
  assert(final.values.filter((v,i)=>i%4===0&&v>1e-5).length>initial.values.filter((v,i)=>i%4===0&&v>1e-5).length,'Wet front failed to advance');a.swe.destroy();a.renderer.destroy();
});
await check('Uniform rain and evaporation follow analytic water balance',async()=>{
  const a=await scene(()=>0,()=>.05,{rainRate:.01,evaporationRate:.1,maxTimeStep:.02});evolve(a.swe,600);const final=await a.swe.readStats(),expected=.05*Math.exp(-.1*final.time)+.1*(1-Math.exp(-.1*final.time));
  assert(final.values.every((v,i)=>i%4!==0||Math.abs(v-expected)<2e-6),'Weather balance differs from analytic solution');a.swe.destroy();a.renderer.destroy();
});
await check('An open outlet exports exactly its shared edge flux without boundary inflow',async()=>{
  const a=await scene(()=>0,()=>.05,{maxTimeStep:.02,simulation:{closedBoundaries:false,openBoundaryEdges:2}}),initial=await a.swe.readStats();
  for(let i=0;i<initial.values.length;i+=4)initial.values[i+1]=.05;
  sim.device.queue.writeBuffer(a.swe.states[0],0,initial.values);a.swe.step(1);
  const final=await a.swe.readStats(),edges=await read(a.swe.edges),ei=new Uint32Array(edges.buffer),fluxes=await read(a.swe.fluxes);let exported=0;
  for(let i=0;i<a.swe.edgeCount;i++)if(ei[i*8+1]===0xffffffff&&ei[i*8+2])exported+=fluxes[i*4]*final.dt;
  assert(Math.abs(mass(initial.values,a.geometry)-mass(final.values,a.geometry)-exported)<1e-6,'Outlet export differs from mass loss');
  evolve(a.swe,1000);const evolved=await a.swe.readStats();assert(evolved.invalid===0&&evolved.totalWater<final.totalWater,'Outlet flow should drain finite water');a.swe.destroy();a.renderer.destroy();
});
await check('Normal water steps perform no readbacks and preserve every raster erosion value',async()=>{
  const a=await scene(x=>x*.002,()=>.02),before=await read(sim.stateBuffers[0]);let maps=0;const original=GPUBuffer.prototype.mapAsync;
  GPUBuffer.prototype.mapAsync=function(...args){maps++;return original.apply(this,args);};
  try{evolve(a.swe,301);a.renderer.render({swe:a.swe});await sim.device.queue.onSubmittedWorkDone();assert(maps===0,'Water step mapped a GPU buffer');}
  finally{GPUBuffer.prototype.mapAsync=original;}
  const after=await read(sim.stateBuffers[0]);assert(after.every((v,i)=>Object.is(v,before[i])),'Mesh water changed the erosion state');a.swe.destroy();a.renderer.destroy();
});
window.meshWaterResults={passed:results.every(r=>r.passed)&&errors.length===0,results,errors};
}catch(error){window.meshWaterResults={passed:false,error:error.message,results,errors};}
