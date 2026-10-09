// First-order finite-volume shallow-water flow on a fixed triangular mesh.
// Audusse et al. (2004) hydrostatic reconstruction + Rusanov shared edge flux.
// Geometry is built once. CFL reduction, fluxes and updates stay entirely on GPU.
import GpuTiming from './GpuTiming.js';
const code=/* wgsl */ `
struct Params { counts:vec4<u32>, physics:vec4<f32>, weather:vec4<f32> }
struct Geometry { metrics:vec4<f32>, links:vec4<u32>, forcing:vec4<f32> }
struct Edge { cells:vec4<u32>, normal:vec4<f32> }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> geometry:array<Geometry>;
@group(0) @binding(2) var<storage,read> edges:array<Edge>;
@group(0) @binding(3) var<storage,read> input:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> output:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> fluxes:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> partials:array<f32>;
@group(0) @binding(7) var<storage,read_write> clock:vec4<f32>;
fn velocity(s:vec4<f32>)->vec2<f32>{return select(vec2<f32>(0.0),s.yz/max(s.x,1e-7),s.x>1e-7);}
fn physical(h:f32,q:vec2<f32>,n:vec2<f32>)->vec3<f32>{let un=dot(q,n)/max(h,1e-7);return vec3<f32>(dot(q,n),q*un+0.5*params.physics.x*h*h*n);}
@compute @workgroup_size(128) fn edgeFlux(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=params.counts.y){return;}let e=edges[i];let a=e.cells.x;let b=e.cells.y;let n=e.normal.xy;
  let l=input[a];var r=l;let zl=geometry[a].metrics.y;var zr=zl;
  if(b!=0xffffffffu){r=input[b];zr=geometry[b].metrics.y;}
  else if(e.cells.z==0u || dot(l.yz,n)<0.0){r=vec4<f32>(l.x,l.yz-2.0*dot(l.yz,n)*n,0.0);}
  let top=max(zl,zr);let hl=max(0.0,l.x+zl-top);let hr=max(0.0,r.x+zr-top);
  let ul=velocity(l);let ur=velocity(r);let ql=hl*ul;let qr=hr*ur;
  let speed=max(abs(dot(ul,n))+sqrt(params.physics.x*hl),abs(dot(ur,n))+sqrt(params.physics.x*hr));
  let f=0.5*(physical(hl,ql,n)+physical(hr,qr,n))-0.5*speed*(vec3<f32>(hr,qr)-vec3<f32>(hl,ql));
  fluxes[i]=vec4<f32>(f*e.normal.z,speed*e.normal.z);
}
var<workgroup> minima:array<f32,128>;
@compute @workgroup_size(128) fn localCFL(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
  var dt=params.physics.y;
  if(id.x<params.counts.x){let g=geometry[id.x];let sum=fluxes[g.links.x].w+fluxes[g.links.y].w+fluxes[g.links.z].w;dt=min(dt,params.physics.z*g.metrics.x/max(sum,1e-12));}
  minima[lane]=dt;workgroupBarrier();
  for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){minima[lane]=min(minima[lane],minima[lane+stride]);}workgroupBarrier();}
  if(lane==0u){partials[group.x]=minima[0];}
}
@compute @workgroup_size(128) fn globalCFL(@builtin(local_invocation_index) lane:u32){
  var dt=params.physics.y;for(var i=lane;i<params.counts.z;i+=128u){dt=min(dt,partials[i]);}
  minima[lane]=dt;workgroupBarrier();
  for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){minima[lane]=min(minima[lane],minima[lane+stride]);}workgroupBarrier();}
  if(lane==0u){clock.y=minima[0];clock.x+=minima[0];}
}
@compute @workgroup_size(128) fn update(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=params.counts.x){return;}let g=geometry[i];let s=input[i];var sum=vec3<f32>(0.0);
  for(var j=0u;j<3u;j++){let ei=g.links[j];let e=edges[ei];let left=e.cells.x==i;
    let other=select(e.cells.x,e.cells.y,left);var z=g.metrics.y;if(other!=0xffffffffu){z=geometry[other].metrics.y;}
    let hs=max(0.0,s.x+g.metrics.y-max(g.metrics.y,z));
    let correction=0.5*params.physics.x*(s.x*s.x-hs*hs)*e.normal.xy*e.normal.z;
    sum+=select(-1.0,1.0,left)*(fluxes[ei].xyz+vec3<f32>(0.0,correction));
  }
  let dt=clock.y;let u=vec3<f32>(s.x,s.yz)-dt/g.metrics.x*sum;
  let decay=exp(-params.weather.y*dt);
  let duration=select(dt,(1.0-decay)/max(params.weather.y,1e-8),params.weather.y>1e-8);
  let h=max(0.0,u.x)*decay+(params.weather.x+g.forcing.x)*duration;
  var q=u.yz*decay;
  if(h<=1e-7){q=vec2<f32>(0.0);}else{
    let friction=params.physics.x*params.physics.w*params.physics.w*length(q)/pow(max(h,1e-7),7.0/3.0);
    q/=1.0+dt*friction;
  }
  output[i]=vec4<f32>(h,q,0.0);
}
`;
const initCode=/* wgsl */ `
struct P { counts:vec4<u32>, dims:vec4<f32> }
struct G { metrics:vec4<f32>, links:vec4<u32>, forcing:vec4<f32> }
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read_write> geometry:array<G>;
@group(0) @binding(2) var<storage,read> positions:array<vec2<f32>>;
@group(0) @binding(3) var<storage,read> indices:array<u32>;
@group(0) @binding(4) var<storage,read> raster:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> state:array<vec4<f32>>;
fn cell(q:vec2<i32>)->vec2<f32>{let xy=clamp(q,vec2<i32>(0),vec2<i32>(p.counts.yz)-1);let c=raster[(u32(xy.y)*p.counts.y+u32(xy.x))*2u];return c.xy;}
fn sample(q:vec2<f32>)->vec2<f32>{let b=vec2<i32>(floor(q));let f=fract(q);return mix(mix(cell(b),cell(b+vec2<i32>(1,0)),f.x),mix(cell(b+vec2<i32>(0,1)),cell(b+vec2<i32>(1,1)),f.x),f.y);}
@compute @workgroup_size(128) fn initialize(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=p.counts.x){return;}
  let a=positions[indices[i*3u]];let b=positions[indices[i*3u+1u]];let c=positions[indices[i*3u+2u]];
  // Project bed and water with the SAME quadrature. Different samples invent
  // free-surface gradients when a curved raster lake is transferred to triangles.
  let mean=(sample(a*(2.0/3.0)+(b+c)/6.0)+sample(b*(2.0/3.0)+(a+c)/6.0)+sample(c*(2.0/3.0)+(a+b)/6.0))/3.0;
  geometry[i].metrics.y=mean.x;state[i]=vec4<f32>(max(mean.y,0.0),0.0,0.0,0.0);
}
`;

export function buildWaterGeometry(topology,{cellSize=1,closedBoundaries=true,openBoundaryEdges=15,width=0,height=0,sources=[]}={}){
  const pos=topology.positionsGrid,ind=topology.indices,n=ind.length/3,bytes=new ArrayBuffer(n*48),floats=new Float32Array(bytes),uints=new Uint32Array(bytes),edgeList=[],map=new Map();
  for(let t=0;t<n;t++){
    const ids=[ind[t*3],ind[t*3+1],ind[t*3+2]],p=ids.map(i=>[pos[i*2],pos[i*2+1]]);
    const cross=(p[1][0]-p[0][0])*(p[2][1]-p[0][1])-(p[2][0]-p[0][0])*(p[1][1]-p[0][1]);
    if(!Number.isFinite(cross)||Math.abs(cross)<1e-10)throw Error('Water mesh contains a degenerate triangle.');
    const cx=(p[0][0]+p[1][0]+p[2][0])/3,cy=(p[0][1]+p[1][1]+p[2][1])/3,offset=t*12;
    floats[offset]=Math.abs(cross)*.5*cellSize*cellSize;floats[offset+2]=cx;floats[offset+3]=cy;
    let spring=0;for(const s of sources){const r=Math.max(.5,s.radius||1),d2=((cx-s.x)**2+(cy-s.y)**2)/(r*r);if(d2<1)spring+=(s.strength||0)*Math.exp(-d2*4);}
    floats[offset+8]=spring;
    for(let k=0;k<3;k++){
      const a=ids[k],b=ids[(k+1)%3],key=Math.min(a,b)* (pos.length/2)+Math.max(a,b);
      let e=map.get(key);
      if(e!==undefined){if(edgeList[e].right!==0xffffffff)throw Error('Water mesh has a non-manifold edge.');edgeList[e].right=t;}
      else{const pa=p[k],pb=p[(k+1)%3],dx=pb[0]-pa[0],dy=pb[1]-pa[1],len=Math.hypot(dx,dy),sign=Math.sign(cross);
        let side=0;if(Math.max(pa[0],pb[0])<1e-4)side=1;else if(Math.min(pa[0],pb[0])>width-1-1e-4)side=2;else if(Math.max(pa[1],pb[1])<1e-4)side=4;else if(Math.min(pa[1],pb[1])>height-1-1e-4)side=8;
        e=edgeList.length;edgeList.push({left:t,right:0xffffffff,open:!closedBoundaries&&!!(side&openBoundaryEdges),nx:sign*dy/len,ny:-sign*dx/len,len:len*cellSize});map.set(key,e);}
      uints[offset+4+k]=e;
    }
  }
  const eb=new ArrayBuffer(edgeList.length*32),ef=new Float32Array(eb),eu=new Uint32Array(eb);
  edgeList.forEach((e,i)=>{eu[i*8]=e.left;eu[i*8+1]=e.right;eu[i*8+2]=e.open?1:0;ef[i*8+4]=e.nx;ef[i*8+5]=e.ny;ef[i*8+6]=e.len;});
  return {geometry:bytes,edges:eb,triangles:n,edgeCount:edgeList.length};
}

export default class StaticMeshWater {
  constructor(sim,topology,renderer){this.sim=sim;this.topology=topology;this.renderer=renderer;this.buffers=[];this.current=0;this.steps=0;}
  async initialize(options={}){
    const s=this.sim,d=s.device,p=s.simulationParams;
    this.options={maxTimeStep:.02,cfl:.35,roughness:.025,rainRate:0,evaporationRate:0,...options};
    const data=buildWaterGeometry(this.topology,{cellSize:p.cellSize||1,width:s.width,height:s.height,closedBoundaries:p.closedBoundaries,openBoundaryEdges:p.openBoundaryEdges??15,sources:p.sourceEnabled?s.getSourcePoints():[]});
    this.triangles=data.triangles;this.edgeCount=data.edgeCount;this.groups=Math.ceil(this.triangles/128);
    const geometryFloats=new Float32Array(data.geometry);
    this.areas=Float32Array.from({length:this.triangles},(_,i)=>geometryFloats[i*12]);
    this.dispatchCounts=[Math.ceil(this.edgeCount/128),this.groups,1,this.groups];
    const upload=(data,usage=GPUBufferUsage.STORAGE)=>{const size=typeof data==='number'?data:data.byteLength;const b=d.createBuffer({size,usage:usage|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});if(typeof data!=='number')d.queue.writeBuffer(b,0,data);this.buffers.push(b);return b;};
    this.geometry=upload(data.geometry);this.edges=upload(data.edges);this.states=[upload(this.triangles*16),upload(this.triangles*16)];
    this.fluxes=upload(this.edgeCount*16);this.partials=upload(this.groups*4);this.clock=upload(16);this.uniform=upload(48,GPUBufferUsage.UNIFORM);
    const m=d.createShaderModule({code});
    const layout=d.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[1,2,3,4,5,6,7].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding<4?'read-only-storage':'storage'}}))]});
    const pl=d.createPipelineLayout({bindGroupLayouts:[layout]});
    this.pipelines=await Promise.all(['edgeFlux','localCFL','globalCFL','update'].map(entryPoint=>d.createComputePipelineAsync({layout:pl,compute:{module:m,entryPoint}})));
    this.bindings=[0,1].map(i=>d.createBindGroup({layout,entries:[this.uniform,this.geometry,this.edges,this.states[i],this.states[1-i],this.fluxes,this.partials,this.clock].map((buffer,binding)=>({binding,resource:{buffer}}))}));
    const ip=await d.createComputePipelineAsync({layout:'auto',compute:{module:d.createShaderModule({code:initCode}),entryPoint:'initialize'}});
    const iu=upload(new Uint32Array([this.triangles,s.width,s.height,0,0,0,0,0]),GPUBufferUsage.UNIFORM);
    const ig=d.createBindGroup({layout:ip.getBindGroupLayout(0),entries:[iu,this.geometry,this.renderer.positions,this.renderer.indices,s.stateBuffers[0],this.states[0]].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const enc=d.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(ip);pass.setBindGroup(0,ig);pass.dispatchWorkgroups(this.groups);pass.end();d.queue.submit([enc.finish()]);
    this.updateOptions({});return this;
  }
  updateOptions(options={}){
    Object.assign(this.options,options);const o=this.options,bytes=new ArrayBuffer(48),u=new Uint32Array(bytes),f=new Float32Array(bytes);
    u.set([this.triangles,this.edgeCount,this.groups,0]);f.set([Math.max(.001,this.sim.simulationParams.gravity||9.81),Math.max(.00001,Math.min(1,o.maxTimeStep)),Math.max(.05,Math.min(.45,o.cfl)),Math.max(0,o.roughness)],4);
    f.set([Math.max(0,o.rainRate),Math.max(0,o.evaporationRate),0,0],8);this.sim.device.queue.writeBuffer(this.uniform,0,bytes);
  }
  step(n=1,timing=null){const d=this.sim.device,e=d.createCommandEncoder(),pass=e.beginComputePass({timestampWrites:timing?.writes()});
    for(let i=0;i<Math.max(1,Math.min(256,n|0));i++){
      pass.setBindGroup(0,this.bindings[this.current]);
      for(let j=0;j<4;j++){pass.setPipeline(this.pipelines[j]);pass.dispatchWorkgroups(this.dispatchCounts[j]);}
      this.current=1-this.current;this.steps++;
    }pass.end();timing?.resolve(e);d.queue.submit([e.finish()]);
  }
  async benchmark(n=100){n=Math.max(1,Math.min(256,n|0));const d=this.sim.device,timing=d.features.has('timestamp-query')?new GpuTiming(d,1):null,start=performance.now();
    this.step(n,timing);await d.queue.onSubmittedWorkDone();const completionMs=performance.now()-start,gpuMs=timing?await timing.read():null;return{steps:n,completionMs,gpuMs,gpuMsPerStep:gpuMs===null?null:gpuMs/n};}
  async readStats(){const capturedSteps=this.steps,d=this.sim.device,b=d.createBuffer({size:this.triangles*16+16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    try{const e=d.createCommandEncoder();e.copyBufferToBuffer(this.states[this.current],0,b,0,this.triangles*16);e.copyBufferToBuffer(this.clock,0,b,this.triangles*16,16);d.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);const values=new Float32Array(b.getMappedRange()).slice();b.unmap();
      let totalWater=0,maxDepth=0,maxSpeed=0,invalid=0;for(let i=0;i<this.triangles;i++){const h=values[i*4],qx=values[i*4+1],qy=values[i*4+2];if(!Number.isFinite(h+qx+qy)||h<0){invalid++;continue;}totalWater+=h*this.areas[i];maxDepth=Math.max(maxDepth,h);if(h>1e-7)maxSpeed=Math.max(maxSpeed,Math.hypot(qx,qy)/h);}
      return {values:values.slice(0,-4),time:values.at(-4),dt:values.at(-3),steps:capturedSteps,totalWater,maxDepth,maxSpeed,invalid};}finally{b.destroy();}}
  destroy(){for(const b of this.buffers)b.destroy();this.buffers=[];}
}
