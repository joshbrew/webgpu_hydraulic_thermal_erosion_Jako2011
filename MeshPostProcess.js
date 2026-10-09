import ErosionMeshRenderer from './ErosionMeshRenderer.js';
import StaticMeshWater from './StaticMeshWater.js';
import HeightfieldTINBuilder from './vendor/hydraulic-tin/HeightfieldTINBuilder.ts';
import RiverNetworkBuilder from './vendor/hydraulic-tin/RiverNetworkBuilder.ts';

export { MESH_FIDELITY } from './MeshSettings.js';

export default class MeshPostProcess {
  constructor(sim, onstatus = () => {}, { execution = 'worker' } = {}) {
    this.sim = sim; this.onstatus = onstatus; this.snapshots = 0;
    this.execution=execution;
    this.view = { adaptive: true, wireframe: true, wireOpacity: .55 };
  }
  async capture() {
    const s = this.sim;
    if (this.snapshot?.iteration === s.iterationCount && this.snapshot.buffer === s.stateBuffers[0]) return this.snapshot;
    const copy = s.device.createBuffer({ size: s.cellCount*32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const e = s.device.createCommandEncoder(); e.copyBufferToBuffer(s.stateBuffers[0],0,copy,0,copy.size); s.device.queue.submit([e.finish()]);
      await copy.mapAsync(GPUMapMode.READ);
      const cells = new Float32Array(copy.getMappedRange()), values = new Float32Array(s.cellCount), mask = new Uint8Array(s.cellCount);
      for (let i=0;i<s.cellCount;i++) { values[i]=cells[i*8]; mask[i]=cells[i*8+4]>.5?1:0; }
      copy.unmap(); this.snapshots++;
      return this.snapshot = { width:s.width,height:s.height,values,mask,iteration:s.iterationCount,buffer:s.stateBuffers[0] };
    } finally { copy.destroy(); }
  }
  async build(options = {}) {
    this.swe?.destroy();this.swe=null;
    this.onstatus('Capturing paused terrain…');
    const data = await this.capture(), start = performance.now();
    if(this.execution==='worker'){
      // Never initialize the nested topology worker module in the erosion worker:
      // some worker bundlers execute its entry point against the current global.
      const {default:BuildCoordinator}=await import('./vendor/hydraulic-tin/BuildCoordinator.js');
      this.coordinator ??= new BuildCoordinator({ debugLogging:false,onstatus:this.onstatus });
      await this.coordinator.setRaster({ width:data.width,height:data.height,values:data.values.slice(),mask:data.mask.slice() });
    }
    let minSpacing=Math.max(1,Math.min(256,Number(options.minSpacing)||3));
    let maxSpacing=Math.max(minSpacing,Math.min(512,Number(options.maxSpacing)||20));
    const budget=Math.max(128,Math.min(250000,Math.floor(Number(options.maxTriangles)||16000)));
    let lo=Infinity,hi=-Infinity;for(let i=0;i<data.values.length;i++)if(data.mask[i]){lo=Math.min(lo,data.values[i]);hi=Math.max(hi,data.values[i]);}
    const rise=Math.max(hi-lo,.001); let result;
    // Hard budget: coarsen and rebuild from the SAME snapshot, never truncate triangles.
    for(let attempt=0;attempt<8;attempt++) {
      const settings={
        topologyBuildOptions:{ mode:options.mode||'hybrid',cellSize:1,minSpacing,maxSpacing,borderSpacing:maxSpacing,wetnessBins:21,channelThreshold:256,candidateStride:1,relaxIterations:0,preserveBorder:true,preserveChannels:true,preserveMaskBoundary:true,preserveInnerBorder:true },
        riverOptions:{ maxNeighborDistance:Infinity,boundaryEpsilon:.5,minDrop:1e-5,flatTolerance:1e-4,sinkMaxRise:rise*.015,sinkClimbPenalty:2.5 },
        guideOptions:{ minWeight:3,step:Math.max(minSpacing*1.75,Math.min(maxSpacing*.6,minSpacing*5)),maxPoints:Math.min(16000,Math.floor(data.values.length*.45)) },
        floodplainOptions:{ maxRise:rise*.05,maxClimbStep:rise*.015,maxEdgeSlope:.75,maxWorldDistance:Math.max(minSpacing*6,maxSpacing*1.5),minTriangleFloodplainVerts:2 },
        minRiverWeight:6,guideEnabled:options.guideEnabled!==false,
      };
      if(this.execution==='worker')result=await this.coordinator.build(settings);
      else{
        // The original demo already owns a background worker. Build there while
        // paused instead of nesting another worker inside the bundled worker.
        this.onstatus('Building hydraulic topology in the background…');
        const builder=new HeightfieldTINBuilder(settings.topologyBuildOptions);builder.setRaster(data);
        let topology=await builder.build();
        if(settings.guideEnabled){const river=new RiverNetworkBuilder(settings.riverOptions),network=river.buildFromTopology(topology);
          const guides=river.buildGuidePoints(topology,network,settings.guideOptions);
          if(guides.length)topology=await builder.buildWithForcedWorldPoints(guides);}
        result={topology};
      }
      const count=result.topology.indices.length/3;if(count<=budget)break;
      if(attempt===7)throw Error('This boundary needs more triangles. Increase the triangle limit.');
      const scale=Math.max(1.2,Math.sqrt(count/budget)*1.1);minSpacing*=scale;maxSpacing*=scale;
      this.onstatus(`Fitting the ${budget.toLocaleString()} triangle limit…`);
    }
    const next=new ErosionMeshRenderer(this.sim);
    try{await next.initialize(result.topology);}catch(error){next.destroy();throw error;}
    this.renderer?.destroy();this.renderer=next;this.topology=result.topology;
    const fidelity=measureFidelity(data,result.topology);
    this.info={ vertices:result.topology.positionsGrid.length/2,triangles:result.topology.indices.length/3,gridTriangles:(data.width-1)*(data.height-1)*2,
      buildMs:performance.now()-start,minSpacing,maxSpacing,maxTriangles:budget,iteration:data.iteration,time:this.sim.getStats().simTime,snapshots:this.snapshots,...fidelity };
    this.render();return this.info;
  }
  render(view={}) { Object.assign(this.view,view); if(this.renderer)this.renderer.render({...this.view,swe:this.swe});else this.sim.render(); }
  async startWater(options={}) {
    if(!this.renderer)throw Error('Build a hydraulic mesh first.');
    if(!this.swe){const next=new StaticMeshWater(this.sim,this.topology,this.renderer);
      try{await next.initialize(options);}catch(error){next.destroy();throw error;}this.swe=next;
    }else this.swe.updateOptions(options);
    this.view.adaptive=true;this.render();
  }
  resetWater() {this.swe?.destroy();this.swe=null;this.render();}
  stepWater(n) {this.swe?.step(n);this.render();}
  invalidate() { this.swe?.destroy();this.swe=null;this.renderer?.destroy();this.renderer=null;this.topology=null;this.snapshot=null;this.info=null; }
  destroy() { this.invalidate();this.coordinator?.dispose();this.coordinator=null; }
}

// Seven samples per triangle measure approximation error, not a global error bound.
export function measureFidelity(raster,topology) {
  const {width,height,values}=raster,p=topology.positionsGrid,ind=topology.indices;
  const sample=(x,y)=>{x=Math.max(0,Math.min(width-1,x));y=Math.max(0,Math.min(height-1,y));const ix=Math.floor(x),iy=Math.floor(y),fx=x-ix,fy=y-iy;
    const a=values[iy*width+ix],b=values[iy*width+Math.min(ix+1,width-1)],c=values[Math.min(iy+1,height-1)*width+ix],d=values[Math.min(iy+1,height-1)*width+Math.min(ix+1,width-1)];
    return (a*(1-fx)+b*fx)*(1-fy)+(c*(1-fx)+d*fx)*fy;};
  let error=0,areaSum=0,maxError=0;
  const weights=[[1/3,1/3,1/3],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6],[.5,.5,0],[.5,0,.5],[0,.5,.5]];
  for(let i=0;i<ind.length;i+=3){const a=ind[i]*2,b=ind[i+1]*2,c=ind[i+2]*2;
    const area=Math.abs((p[b]-p[a])*(p[c+1]-p[a+1])-(p[c]-p[a])*(p[b+1]-p[a+1]))*.5;
    const heights=[sample(p[a],p[a+1]),sample(p[b],p[b+1]),sample(p[c],p[c+1])];
    for(const w of weights){const x=w[0]*p[a]+w[1]*p[b]+w[2]*p[c],y=w[0]*p[a+1]+w[1]*p[b+1]+w[2]*p[c+1];
      const delta=Math.abs(w[0]*heights[0]+w[1]*heights[1]+w[2]*heights[2]-sample(x,y));error+=delta*delta*area;areaSum+=area;maxError=Math.max(maxError,delta);}
  }
  return { sampledRmsError:Math.sqrt(error/Math.max(areaSum,1e-12)),sampledMaxError:maxError };
}
