(()=>{var qe=`
struct Cell {
  terrain: f32, water: f32, sediment: f32, erodability: f32,
  mask: f32, thermalOut: f32, thermalIn: f32, history: f32,
}
struct Partial {
  totals: vec4<f32>,
  minimum: vec4<f32>,
  maximum: vec4<f32>,
  history: vec4<f32>,
}
@group(0) @binding(0) var<uniform> counts: vec4<u32>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<storage, read> input: array<Partial>;
@group(0) @binding(3) var<storage, read_write> output: array<Partial>;
var<workgroup> scratch: array<Partial, 256>;
fn empty() -> Partial {
  return Partial(vec4<f32>(0.0), vec4<f32>(1e30), vec4<f32>(-1e30), vec4<f32>(1e30, -1e30, 0.0, 0.0));
}
fn merge(a: Partial, b: Partial) -> Partial {
  return Partial(a.totals + b.totals, min(a.minimum, b.minimum), max(a.maximum, b.maximum),
    vec4<f32>(min(a.history.x, b.history.x), max(a.history.y, b.history.y), a.history.zw + b.history.zw));
}
fn reduce(local: u32, group: u32) {
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (local < stride) { scratch[local] = merge(scratch[local], scratch[local + stride]); }
    workgroupBarrier();
  }
  if (local == 0u) { output[group] = scratch[0]; }
}
@compute @workgroup_size(256)
fn reduceCells(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
  var value = empty();
  if (gid.x < counts.x) {
    let cell = cells[gid.x];
    if (cell.mask >= 0.5) {
      let physical = vec4<f32>(cell.terrain, cell.water, cell.sediment, cell.erodability);
      let valid = all(physical == physical) && all(abs(physical) < vec4<f32>(1e30)) && cell.history == cell.history && abs(cell.history) < 1e30;
      if (valid) {
        value = Partial(vec4<f32>(physical.xyz, 1.0), physical, physical, vec4<f32>(cell.history, cell.history, cell.erodability, 0.0));
      } else { value.history.w = 1.0; }
    }
  }
  scratch[local] = value;
  reduce(local, group.x);
}
@compute @workgroup_size(256)
fn reducePartials(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
  var value = empty();
  if (gid.x < counts.x) { value = input[gid.x]; }
  scratch[local] = value;
  reduce(local, group.x);
}
`,q=class{constructor(e){this.device=e,this.pipelines=null,this.resources=[],this.count=0,this.pending=Promise.resolve()}async initialize(){if(!this.pipelines){let e=this.device.createShaderModule({code:qe,label:"erosion-mass-reduction"});this.pipelines=Promise.all(["reduceCells","reducePartials"].map(t=>this.device.createComputePipelineAsync({layout:"auto",compute:{module:e,entryPoint:t}})))}return this.pipelines}read(e,t,i=()=>({})){let a=this.pending.then(()=>this.reduce(e,t,i));return this.pending=a.catch(()=>{}),a}async reduce(e,t,i){let[a,r]=await this.initialize();if(this.count!==t){this.destroyResources(),this.count=t;let f=Math.max(1,Math.ceil(t/256));this.buffers=[0,1].map(()=>this.device.createBuffer({size:f*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,label:"erosion-mass-partials"})),this.staging=this.device.createBuffer({size:64,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),this.resources.push(...this.buffers,this.staging),this.uniforms=[];for(let u=t;;u=Math.ceil(u/256)){let S=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});if(this.device.queue.writeBuffer(S,0,new Uint32Array([u,0,0,0])),this.resources.push(S),this.uniforms.push(S),Math.ceil(u/256)<=1)break}}let n=this.device.createCommandEncoder({label:"erosion-inspect-mass"}),l=n.beginComputePass(),o=t,c=0,d=0;for(;;){let f=d===0?a:r,u=this.device.createBindGroup({layout:f.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniforms[d]}},{binding:d===0?1:2,resource:{buffer:d===0?e:this.buffers[1-c]}},{binding:3,resource:{buffer:this.buffers[c]}}]});if(l.setPipeline(f),l.setBindGroup(0,u),o=Math.ceil(o/256),l.dispatchWorkgroups(o),o<=1)break;c=1-c,d++}l.end(),n.copyBufferToBuffer(this.buffers[c],0,this.staging,0,64);let p=i();this.device.queue.submit([n.finish()]),await this.staging.mapAsync(GPUMapMode.READ);let h;try{h=new Float32Array(this.staging.getMappedRange()).slice()}finally{this.staging.unmap()}let v=h[3],m=f=>v?{min:h[4+f],max:h[8+f]}:null;return{...p,validCells:v,invalidCells:h[15],totalTerrain:h[0],totalWater:h[1],totalSediment:h[2],totalSolid:h[0]+h[2],terrainRange:m(0),waterRange:m(1),sedimentRange:m(2),historyRange:v?{min:h[12],max:h[13]}:null,averageHardness:v?h[14]/v:0,statisticsReadbackBytes:64}}destroyResources(){for(let e of this.resources)e.destroy();this.resources=[],this.count=0}destroy(){this.destroyResources()}};var I=class{constructor(e,t){this.count=t*2,this.querySet=e.createQuerySet({type:"timestamp",count:this.count}),this.resolveBuffer=e.createBuffer({size:this.count*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),this.readBuffer=e.createBuffer({size:this.count*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ})}writes(e=0){return{querySet:this.querySet,beginningOfPassWriteIndex:e*2,endOfPassWriteIndex:e*2+1}}resolve(e){e.resolveQuerySet(this.querySet,0,this.count,this.resolveBuffer,0),e.copyBufferToBuffer(this.resolveBuffer,0,this.readBuffer,0,this.count*8)}async read(){try{await this.readBuffer.mapAsync(GPUMapMode.READ);let e=new BigUint64Array(this.readBuffer.getMappedRange()),t=0n;for(let i=0;i<this.count;i+=2)e[i+1]>=e[i]&&(t+=e[i+1]-e[i]);return Number(t)/1e6}finally{this.readBuffer.unmap(),this.readBuffer.destroy(),this.resolveBuffer.destroy(),this.querySet.destroy()}}};var W=8,Ye=16,Re=8,Ae=8;function M(s,e,t){return s<e?e:s>t?t:s}function ne(s,e){return Math.ceil(s/e)*e}function Ee(s){let e=s>>>0;return e=Math.imul(e^e>>>16,2146121005),e=Math.imul(e^e>>>15,2221713035),(e^e>>>16)>>>0}function fe(s){return s*Math.PI/180}function oe(s){let e=Math.hypot(s[0],s[1],s[2])||1;return[s[0]/e,s[1]/e,s[2]/e]}function je(s,e){return[s[0]-e[0],s[1]-e[1],s[2]-e[2]]}function Oe(s,e){return[s[1]*e[2]-s[2]*e[1],s[2]*e[0]-s[0]*e[2],s[0]*e[1]-s[1]*e[0]]}function Xe(s,e){let t=new Float32Array(16);for(let i=0;i<4;i++)for(let a=0;a<4;a++)t[i*4+a]=s[0+a]*e[i*4+0]+s[4+a]*e[i*4+1]+s[8+a]*e[i*4+2]+s[12+a]*e[i*4+3];return t}function Ke(s,e,t,i){let a=1/Math.tan(s*.5),r=1/(t-i),n=new Float32Array(16);return n[0]=a/Math.max(e,1e-6),n[5]=a,n[10]=(i+t)*r,n[11]=-1,n[14]=2*i*t*r,n}function Ze(s,e,t){let i=oe(je(s,e)),a=oe(Oe(t,i)),r=Oe(i,a),n=new Float32Array(16);return n[0]=a[0],n[1]=r[0],n[2]=i[0],n[3]=0,n[4]=a[1],n[5]=r[1],n[6]=i[1],n[7]=0,n[8]=a[2],n[9]=r[2],n[10]=i[2],n[11]=0,n[12]=-(a[0]*s[0]+a[1]*s[1]+a[2]*s[2]),n[13]=-(r[0]*s[0]+r[1]*s[1]+r[2]*s[2]),n[14]=-(i[0]*s[0]+i[1]*s[1]+i[2]*s[2]),n[15]=1,n}var Y=class{constructor(e={}){this.canvas=e.canvas??null,this.adapter=null,this.device=null,this.context=null,this.presentationFormat=null,this.width=0,this.height=0,this.cellCount=0,this.stateByteLength=0,this.vec4ByteLength=0,this.iterationCount=0,this.simulationTime=0,this.allocatedStateByteLength=0,this.allocatedVec4ByteLength=0,this.dispatchX=0,this.dispatchY=0,this.drawCount=0,this.stateBuffers=[],this.fluxBuffer=null,this.velocityBuffer=null,this.thermalPipeBufferA=null,this.thermalPipeBufferB=null,this.readbackBuffer=null,this.customSpringBuffer=null,this.layerTopBuffer=null,this.layerSurfaceValues=new Float32Array(0),this.layerMaterialValues=new Float32Array(16),this.layerMaterialBuffer=null,this.paintedSpringMap=new Float32Array(0),this.sourceFieldDirty=!0,this.sourceFieldBuildCount=0,this.allocatedPaintedSpringByteLength=0,this.allocatedReadbackByteLength=0,this.renderUniformValues=new Float32Array(40),this.renderUniformBufferTerrain=null,this.renderUniformBufferWater=null,this.depthTexture=null,this.depthTextureView=null,this.depthTextureSize={width:0,height:0},this.sceneColorTexture=null,this.sceneColorView=null,this.sceneColorSize={width:0,height:0},this.sceneSampler=null,this.dummySceneTexture=null,this.dummySceneView=null,this.paramValues=new Float32Array(52),this.paramBits=new Uint32Array(this.paramValues.buffer),this.paramBuffer=null,this.stepParamBuffer=null,this.stepParamStride=256,this.stepParamCapacity=0,this.stepParamValuesCpu=null,this.stepParamValuesCpuCapacity=0,this.stepParamTemplateDirty=!0,this.stepParamPreparedIterations=0,this.computeDynamicOffset=new Uint32Array(1),this.terrainRenderBundle=null,this.finalRenderBundle=null,this.terrainRenderBundleList=[],this.finalRenderBundleList=[],this.computeBindGroupLayout=null,this.renderBindGroupLayout=null,this.compositeBindGroupLayout=null,this.computeBindGroup01=null,this.computeBindGroup10=null,this.renderBindGroupTerrain=null,this.renderBindGroupWater=null,this.compositeBindGroup=null,this.fluxPipeline=null,this.sourcePipeline=null,this.flowPipeline=null,this.erosionPipeline=null,this.thermalOutflowPipeline=null,this.channelPipeline=null,this.channelBendPipeline=null,this.renderPipeline=null,this.waterRenderPipeline=null,this.compositePipeline=null,this.initialized=!1,this.ready=!1,this.latestReadbackStats=null,this.gpuStatistics=null,this.lastErosionPassMs=0,this.lastRenderPassMs=0,this.avgErosionPassMs=0,this.avgRenderPassMs=0,this.lastCpuErosionPassMs=0,this.lastCpuRenderPassMs=0,this.avgCpuErosionPassMs=0,this.avgCpuRenderPassMs=0,this.lastGpuErosionPassMs=0,this.lastGpuRenderPassMs=0,this.avgGpuErosionPassMs=0,this.avgGpuRenderPassMs=0,this.layeredDemEnabled=!1,this.layerConfigMode="single",this.simulationParams={cellSize:1,timeStep:.02,rainRate:.001,evaporationRate:.025,pipeArea:16,gravity:9.81,capacityScale:1.2,suspensionRate:.35,depositionRate:1.4,softeningRate:1.2,maxErosionDepth:.04,thermalRate:.25,talusSlopeCoeff:.012,talusSlopeBias:.002,renderHeightScale:.08,waterOpacity:.28,sedimentTint:.35,hardnessBase:.24,hardnessVariation:.12,sourceCenterX:.5,sourceCenterY:.5,sourceRadius:4,sourceStrength:.06,sourceEnabled:!1,rainDuration:0,pulse2Duration:0,sourceLayoutMode:0,randomSpringCount:4,sourceSeed:1,metersPerPixel:100,sourceTimeOffset:0,historyDecay:.9997,edgeDrainStrength:.08,edgeWaterFloor:0,renderMode:8,cameraAzimuthDeg:45,cameraElevationDeg:42,cameraDistance:2.9,cameraPosX:1.5238999619464006,cameraPosY:1.9404787584406888,cameraPosZ:1.5238999619464006,waterHeightScale:.12,thermalVisualizationScale:160,hydraulicErosionEnabled:!0,closedBoundaries:!1,renderGridStep:0,flowDrag:2,maxOutflowFraction:.5,bankErosionRate:0,sourceSedimentConcentration:0,bankBendBias:24,bendSedimentRate:0,riverBendLength:0,openBoundaryEdges:15,straightBankErosionFactor:.08},this.setSimulationParams(e.simulationParams??{})}async initialize(e=this.canvas){if(!e)throw new Error("A canvas is required to initialize WebGPU erosion.");if(this.canvas=e,this.#x(),!navigator.gpu)throw new Error("WebGPU is not available in this browser.");if(this.adapter=await navigator.gpu.requestAdapter({powerPreference:"high-performance"}),!this.adapter)throw new Error("Failed to acquire a WebGPU adapter.");if(this.device=await this.adapter.requestDevice({requiredFeatures:this.adapter.features.has("timestamp-query")?["timestamp-query"]:[]}),this.stepParamStride=Math.max(256,this.device.limits?.minUniformBufferOffsetAlignment||256),this.context=e.getContext("webgpu"),!this.context)throw new Error("Failed to acquire a WebGPU canvas context.");this.presentationFormat=navigator.gpu.getPreferredCanvasFormat(),this.context.configure({device:this.device,format:this.presentationFormat,alphaMode:"opaque"}),this.paramBuffer=this.device.createBuffer({size:ne(this.paramValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-erosion-params"}),this.renderUniformBufferTerrain=this.device.createBuffer({size:ne(this.renderUniformValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-render-params-terrain"}),this.renderUniformBufferWater=this.device.createBuffer({size:ne(this.renderUniformValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-render-params-water"}),this.layerMaterialBuffer=this.device.createBuffer({size:ne(this.layerMaterialValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-layer-material-params"}),this.sceneSampler=this.device.createSampler({addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge",magFilter:"linear",minFilter:"linear",mipmapFilter:"linear"}),this.#m(),this.#w(),await this.#U(),this.#s(),this.initialized=!0}destroy(){this.gpuStatistics?.destroy(),this.gpuStatistics=null;for(let e of this.stateBuffers)e?.destroy?.();this.stateBuffers=[],this.fluxBuffer?.destroy?.(),this.velocityBuffer?.destroy?.(),this.thermalPipeBufferA?.destroy?.(),this.thermalPipeBufferB?.destroy?.(),this.readbackBuffer?.destroy?.(),this.customSpringBuffer?.destroy?.(),this.layerTopBuffer?.destroy?.(),this.depthTexture?.destroy?.(),this.sceneColorTexture?.destroy?.(),this.dummySceneTexture?.destroy?.(),this.renderUniformBufferTerrain?.destroy?.(),this.renderUniformBufferWater?.destroy?.(),this.layerMaterialBuffer?.destroy?.(),this.paramBuffer?.destroy?.(),this.stepParamBuffer?.destroy?.(),this.fluxBuffer=null,this.velocityBuffer=null,this.thermalPipeBufferA=null,this.thermalPipeBufferB=null,this.readbackBuffer=null,this.customSpringBuffer=null,this.layerTopBuffer=null,this.layerSurfaceValues=new Float32Array(0),this.layerMaterialBuffer=null,this.layerMaterialValues=new Float32Array(16),this.paintedSpringMap=new Float32Array(0),this.allocatedPaintedSpringByteLength=0,this.allocatedReadbackByteLength=0,this.depthTexture=null,this.depthTextureView=null,this.depthTextureSize={width:0,height:0},this.sceneColorTexture=null,this.sceneColorView=null,this.sceneColorSize={width:0,height:0},this.dummySceneTexture=null,this.dummySceneView=null,this.renderUniformBufferTerrain=null,this.renderUniformBufferWater=null,this.layerMaterialBuffer=null,this.paramBuffer=null,this.stepParamBuffer=null,this.stepParamCapacity=0,this.stepParamValuesCpu=null,this.stepParamValuesCpuCapacity=0,this.#f(),this.computeBindGroup01=null,this.computeBindGroup10=null,this.renderBindGroupTerrain=null,this.renderBindGroupWater=null,this.compositeBindGroup=null,this.ready=!1,this.width=0,this.height=0,this.cellCount=0,this.iterationCount=0,this.simulationTime=0,this.dispatchX=0,this.dispatchY=0,this.drawCount=0,this.latestReadbackStats=null,this.lastErosionPassMs=0,this.lastRenderPassMs=0,this.avgErosionPassMs=0,this.avgRenderPassMs=0,this.lastCpuErosionPassMs=0,this.lastCpuRenderPassMs=0,this.avgCpuErosionPassMs=0,this.avgCpuRenderPassMs=0,this.lastGpuErosionPassMs=0,this.lastGpuRenderPassMs=0,this.avgGpuErosionPassMs=0,this.avgGpuRenderPassMs=0,this.paintedSpringMap=new Float32Array(0)}setSimulationParams(e={}){let t={...e};Number.isFinite(t.erodabilityBase)&&!Number.isFinite(t.hardnessBase)&&(t.hardnessBase=t.erodabilityBase),Number.isFinite(t.erodabilityVariation)&&!Number.isFinite(t.hardnessVariation)&&(t.hardnessVariation=t.erodabilityVariation),["sourceRadius","sourceStrength","sourceSeed","randomSpringCount","sourceLayoutMode"].some(r=>r in t&&t[r]!==this.simulationParams[r])&&(this.sourceFieldDirty=!0);let a="sourceLayoutMode"in t&&t.sourceLayoutMode!==this.simulationParams.sourceLayoutMode;Object.assign(this.simulationParams,t),a&&this.simulationParams.sourceLayoutMode<.5&&this.device&&this.#p(),this.layeredDemEnabled&&(Number.isFinite(t.hardnessBase)||Number.isFinite(t.hardnessVariation))&&(this.layerMaterialValues[0]=this.simulationParams.hardnessBase,this.layerMaterialValues[1]=M(this.simulationParams.hardnessBase+this.simulationParams.hardnessVariation,0,1),this.#i()),this.paramBuffer&&this.device&&this.#s(),(this.renderUniformBufferTerrain||this.renderUniformBufferWater)&&this.device&&this.#h()}resetRainTimer(){this.simulationParams.sourceTimeOffset=this.simulationTime,this.paramBuffer&&this.device&&this.#s()}restartSources(){this.resetRainTimer()}async applyTerrainBrush(e={}){if(!this.ready||this.width<=0||this.height<=0)throw new Error("Terrain painting is only available after the simulation is initialized.");let t=M(Number(e.x)||0,0,Math.max(this.width-1,0)),i=M(Number(e.y)||0,0,Math.max(this.height-1,0)),a=Math.max(.5,Number(e.radius)||1),r=Math.max(1e-4,Number(e.amount)||.01),n=M(Number(e.hardness)||.5,.01,1),l=!!e.subtract,o=await this.#v(),c=Math.max(0,Math.floor(t-a-1)),d=Math.min(this.width-1,Math.ceil(t+a+1)),p=Math.max(0,Math.floor(i-a-1)),h=Math.min(this.height-1,Math.ceil(i+a+1)),v=2.65-n*2.2,m=l?-1:1;for(let f=p;f<=h;f++)for(let u=c;u<=d;u++){let S=u-t,b=f-i,w=Math.hypot(S,b);if(w>a)continue;let y=1-w/a,P=Math.pow(M(y,0,1),v),T=f*this.width+u,R=T*W;if(o[R+4]<.5)continue;let D=M(o[R]+m*r*P,0,2);o[R]=D;let se=Number.isFinite(o[R+3])?o[R+3]:this.simulationParams.hardnessBase;o[R+3]=this.layerSurfaceValues.length===this.cellCount*4?this.#u(this.layerSurfaceValues,T*4,D,se):se}return this.device.queue.writeBuffer(this.stateBuffers[0],0,o),this.device.queue.writeBuffer(this.stateBuffers[1],0,o),this.latestReadbackStats=null,this.render(),this.getStats()}paintSpringBrush(e={}){if(!this.ready||this.width<=0||this.height<=0)throw new Error("Spring painting is only available after the simulation is initialized.");let t=M(Number(e.x)||0,0,Math.max(this.width-1,0)),i=M(Number(e.y)||0,0,Math.max(this.height-1,0)),a=Math.max(.5,Number(e.radius)||1),r=Math.max(0,Number(e.strength)||0),n=M(Number(e.hardness)||.5,.01,1),l=!!e.erase;this.#A();let o=Math.max(0,Math.floor(t-a-1)),c=Math.min(this.width-1,Math.ceil(t+a+1)),d=Math.max(0,Math.floor(i-a-1)),p=Math.min(this.height-1,Math.ceil(i+a+1)),h=2.65-n*2.2;for(let v=d;v<=p;v++)for(let m=o;m<=c;m++){let f=m-t,u=v-i,S=Math.hypot(f,u);if(S>a)continue;let b=1-S/a,w=Math.pow(M(b,0,1),h),y=v*this.width+m;l?this.paintedSpringMap[y]=Math.max(0,this.paintedSpringMap[y]-Math.max(r,1)*w):this.paintedSpringMap[y]=M(this.paintedSpringMap[y]+r*w,0,1)}return this.#p(),this.#s(),this.render(),this.getSourcePoints()}clearPaintedSprings(){return this.paintedSpringMap instanceof Float32Array&&this.paintedSpringMap.length===this.cellCount&&this.paintedSpringMap.fill(0),this.#p(),this.#s(),this.render(),this.getSourcePoints()}#d(e={}){let t=M(this.simulationParams.hardnessBase??.16,0,1),i=M(t+Math.max(0,this.simulationParams.hardnessVariation??.03),0,1),a=[{enabled:!0,label:"Base",hardnessMin:t,hardnessMax:i,thermalEnabled:!0},{enabled:!1,label:"Silt",hardnessMin:.24,hardnessMax:.42,thermalEnabled:!0},{enabled:!1,label:"Clay",hardnessMin:.1,hardnessMax:.22,thermalEnabled:!0},{enabled:!1,label:"Bedrock",hardnessMin:0,hardnessMax:.02,thermalEnabled:!0}],r=Array.isArray(e.layerConfig?.layers)?e.layerConfig.layers:[];return a.map((n,l)=>{let o=r[l]||{},c=Number.isFinite(o.heightMin)?o.heightMin:0,d=Number.isFinite(o.heightMax)?o.heightMax:1,p=Number.isFinite(o.hardnessMin)?o.hardnessMin:Number.isFinite(o.erodabilityMin)?o.erodabilityMin:n.hardnessMin,h=Number.isFinite(o.hardnessMax)?o.hardnessMax:Number.isFinite(o.erodabilityMax)?o.erodabilityMax:n.hardnessMax;return{enabled:!!o.enabled,label:o.label||n.label,preset:o.preset||"custom",heightMin:c,heightMax:d,hardnessMin:M(Math.min(p,h),0,1),hardnessMax:M(Math.max(p,h),0,1),thermalEnabled:o.thermalEnabled!==!1}})}#u(e,t,i,a){let r=-1,n=1/0,l=-1/0,o=-1;for(let m=0;m<4;m++){let f=m*4;if(this.layerMaterialValues[f+2]<.5)continue;let u=e[t+m];!Number.isFinite(u)||u<-1e8||(u>=l&&(l=u,o=m),u>=i&&u<=n&&(n=u,r=m))}if(r<0&&(r=o,n=l),r<0||!Number.isFinite(n))return M(a,0,1);let c=-1/0;for(let m=0;m<4;m++){let f=m*4;if(this.layerMaterialValues[f+2]<.5)continue;let u=e[t+m];!Number.isFinite(u)||u<-1e8||u>=n||u>c&&(c=u)}let d=0;Number.isFinite(c)&&c>-1e8?d=M((n-i)/Math.max(n-c,1e-6),0,1):i<n&&(d=M((n-i)/Math.max(n,1e-6),0,1));let p=r*4,h=this.layerMaterialValues[p],v=this.layerMaterialValues[p+1];return M(h+(v-h)*d,0,1)}#i(){this.layerMaterialBuffer&&this.device&&this.device.queue.writeBuffer(this.layerMaterialBuffer,0,this.layerMaterialValues)}getSourcePoints(){if(this.width<=0||this.height<=0)return[];let e=this.simulationParams.sourceLayoutMode|0,t=!!this.simulationParams.sourceEnabled;if(e===0){let r=this.paintedSpringMap;if(!(r instanceof Float32Array)||r.length!==this.cellCount)return[];let n=[],l=.0025,o=0;for(let h=0;h<r.length;h++)r[h]>l&&o++;let d=Math.max(1,Math.ceil(o/8192)),p=0;for(let h=0;h<this.height;h++)for(let v=0;v<this.width;v++){let m=h*this.width+v,f=r[m];f<=l||p++%d===0&&n.push({x:v,y:h,radius:.7,strength:f,painted:!0,active:t})}return n}if(!t)return[];let i=[],a=Math.min(16,Math.max(1,this.simulationParams.randomSpringCount|0));for(let r=0;r<a;r++){let n=Math.imul(Math.floor(Math.max(0,this.simulationParams.sourceSeed??1))>>>0,37)+r*17+1,l=(Ee(n)&16777215)/16777216,o=(Ee(n^2654435769)&16777215)/16777216;i.push({x:l*Math.max(this.width-1,0),y:o*Math.max(this.height-1,0),radius:this.simulationParams.sourceRadius,strength:this.simulationParams.sourceStrength,painted:!1,active:!0})}return i}async setDEM(e,t={}){if(this.initialized||await this.initialize(this.canvas),!e||!(e.values instanceof Float32Array))throw new Error("setDEM expects a raster with Float32Array values.");let i=e.width|0,a=e.height|0;if(i<=0||a<=0)throw new Error("Raster width and height must be positive.");if(e.values.length!==i*a)throw new Error("Raster value count does not match width * height.");let r=Number.isFinite(t.minHeight)?t.minHeight:0,n=Number.isFinite(t.maxHeight)?t.maxHeight:1,l=Math.max(1e-6,n-r),o=e.mask instanceof Uint8Array&&e.mask.length===i*a?e.mask:null,c=Array.isArray(e.bands)?e.bands:null,d=e.water instanceof Float32Array&&e.water.length===i*a?e.water:null,p=this.#d(t),h=t.demSourceMode||t.layerConfig?.mode||"single",v=Array.isArray(c)&&c.some(y=>y instanceof Float32Array&&y.length===i*a),m=v||h!=="single";this.width=i,this.height=a,this.cellCount=i*a,this.stateByteLength=this.cellCount*W*4,this.vec4ByteLength=this.cellCount*Ye,this.iterationCount=0,this.simulationTime=0,this.simulationParams.sourceTimeOffset=0,this.sourceFieldDirty=!0,this.sourceFieldBuildCount=0,this.dispatchX=Math.ceil(this.width/Re),this.dispatchY=Math.ceil(this.height/Ae),this.drawCount=Math.max(1,(this.width-1)*(this.height-1)),this.#f(),this.#l(),this.#c(),this.#O(),this.#s();let f=new Float32Array(this.cellCount*W),u=m?new Float32Array(this.cellCount*4):null;u&&u.fill(-1e9);let S=this.simulationParams.hardnessBase,b=this.simulationParams.hardnessVariation;this.layerMaterialValues.fill(0);for(let y=0;y<4;y++){let P=p[y],T=y*4;this.layerMaterialValues[T]=P.hardnessMin,this.layerMaterialValues[T+1]=P.hardnessMax;let R=!!(c&&c[y]instanceof Float32Array&&c[y].length===this.cellCount),D=m&&!v&&y===0;this.layerMaterialValues[T+2]=P.enabled&&(R||D)?1:0,this.layerMaterialValues[T+3]=P.thermalEnabled?1:0}for(let y=0;y<this.cellCount;y++){let P=y*W,T=y*4,R=o?o[y]?1:0:1,D=r+e.values[y]*l,se=M(e.values[y],0,1),Ce=Math.sin(y*12.9898+i*.031+a*.017)*43758.5453,Ve=((Ce-Math.floor(Ce))*2-1)*b*.25,Be=M(S+(1-se)*b+Ve,0,1),re=D,ke=Be;if(m){let H=-1/0;for(let O=0;O<4;O++){let Ne=O*4;if(this.layerMaterialValues[Ne+2]<.5)continue;let Te=c&&c[O]instanceof Float32Array&&c[O].length===this.cellCount?c[O]:O===0?e.values:null;if(!Te)continue;let me=p[O],He=M(Te[y],0,1),pe=me.heightMin+He*(me.heightMax-me.heightMin);u[T+O]=pe,pe>H&&(H=pe)}Number.isFinite(H)&&H>-1e8?re=H:(u[T]=re,this.layerMaterialValues[2]=1),ke=this.#u(u,T,re,Be)}f[P]=re,f[P+1]=R&&Number.isFinite(d?.[y])?Math.max(d[y],0):0,f[P+2]=0,f[P+3]=ke,f[P+4]=R,f[P+5]=0,f[P+6]=0,f[P+7]=0}let w=this.device.createCommandEncoder({label:"paper-reset-sim-scratch"});w.clearBuffer(this.fluxBuffer),w.clearBuffer(this.velocityBuffer),w.clearBuffer(this.thermalPipeBufferA),w.clearBuffer(this.thermalPipeBufferB),m||w.clearBuffer(this.layerTopBuffer),this.device.queue.submit([w.finish()]),this.device.queue.writeBuffer(this.stateBuffers[0],0,f),this.device.queue.writeBuffer(this.stateBuffers[1],0,f),m&&u&&this.device.queue.writeBuffer(this.layerTopBuffer,0,u),this.layeredDemEnabled=m,this.layerSurfaceValues=this.layeredDemEnabled&&u?u:new Float32Array(0),this.#i(),this.layerConfigMode=h,this.ready=!0,this.latestReadbackStats=null,this.paintedSpringMap=new Float32Array(0),this.#p(),this.#h(),this.#l(),this.#c(),this.render()}resize(){!this.context||!this.device||!this.canvas||(this.#x(),this.context.configure({device:this.device,format:this.presentationFormat,alphaMode:"opaque"}),this.#l(),this.#c(),this.#h())}#r(e=1,t="paper-erosion-step",i=null){let a=this.device.createCommandEncoder({label:t});return this.#M(a,Math.max(1,e|0),i),i?.resolve(a),a.finish()}#n(e="paper-erosion-render",t=null){this.#l(),this.#c(),this.#h();let i=this.context.getCurrentTexture().createView(),a=this.device.createCommandEncoder({label:e});return this.#S(a,i,t),t?.resolve(a),a.finish()}async#o(e,t){let i=performance.now();return this.device.queue.submit([e]),t?t.read():(await this.device.queue.onSubmittedWorkDone(),Math.max(0,performance.now()-i))}async stepBench(e=1){if(!this.ready)return null;let t=this.device.features.has("timestamp-query")?new I(this.device,1):null,i=performance.now(),a=this.#r(Math.max(1,e|0),"paper-erosion-step-bench",t),r=Math.max(0,performance.now()-i),n=await this.#o(a,t);return this.#a(r),this.#y(n),{cpuErosionMs:r,gpuErosionMs:n,cpuRenderMs:0,gpuRenderMs:0,gpuTimingMode:this.getStats().gpuTimingMode}}async stepAndRenderBench(e=1){if(!this.ready)return null;let t=!!(this.context&&this.width>=2&&this.height>=2),i=this.device.features.has("timestamp-query")?new I(this.device,1):null,a=performance.now(),r=this.#r(Math.max(1,e|0),t?"paper-erosion-step-bench":"paper-erosion-step-render-bench",i),n=Math.max(0,performance.now()-a),l=await this.#o(r,i);this.#a(n),this.#y(l);let o=0,c=0;if(t){let d=this.device.features.has("timestamp-query")?new I(this.device,2):null,p=performance.now(),h=this.#n("paper-erosion-render-bench",d);o=Math.max(0,performance.now()-p),c=await this.#o(h,d),this.#e(o),this.#T(c)}else this.#e(0);return{cpuErosionMs:n,gpuErosionMs:l,cpuRenderMs:o,gpuRenderMs:c,gpuTimingMode:this.getStats().gpuTimingMode}}step(e=1){if(!this.ready)return;let t=performance.now(),i=this.#r(Math.max(1,e|0),"paper-erosion-step");this.device.queue.submit([i]),this.#a(Math.max(0,performance.now()-t))}stepAndRender(e=1){if(!this.ready)return;let t=!!(this.context&&this.width>=2&&this.height>=2),i=this.device.createCommandEncoder({label:t?"paper-erosion-step-render":"paper-erosion-step"}),a=performance.now();this.#M(i,Math.max(1,e|0));let r=performance.now(),n=r,l=r;if(t){n=performance.now(),this.#l(),this.#c(),this.#h();let h=this.context.getCurrentTexture().createView();this.#S(i,h),l=performance.now()}let o=performance.now();this.device.queue.submit([i.finish()]);let c=performance.now()-o,d=Math.max(0,r-a),p=Math.max(0,l-n);if(t&&d+p>1e-6){let h=d+p;d+=c*(d/h),p+=c*(p/h)}else d+=c,p=0;this.#a(d),this.#e(p)}render(){if(!this.ready||!this.context||this.width<2||this.height<2)return;let e=performance.now(),t=this.#n("paper-erosion-render");this.device.queue.submit([t]),this.#e(Math.max(0,performance.now()-e))}async readbackStats(){return this.ready?(this.gpuStatistics??=new q(this.device),this.latestReadbackStats=await this.gpuStatistics.read(this.stateBuffers[0],this.cellCount,()=>({statisticsIteration:this.iterationCount,statisticsSimTime:this.simulationTime})),this.getStats()):this.getStats()}async exportTerrainPng(){if(!this.ready||this.width<=0||this.height<=0)throw new Error("Terrain export is only available after the simulation is initialized.");if(typeof OffscreenCanvas!="function")throw new Error("OffscreenCanvas is not available for terrain export.");let e=await this.#v(),t=1/0,i=-1/0;for(let d=0;d<this.cellCount;d++){let p=d*W;if(e[p+4]<.5)continue;let h=e[p];h<t&&(t=h),h>i&&(i=h)}(!Number.isFinite(t)||!Number.isFinite(i))&&(t=0,i=1);let a=Math.max(i-t,1e-6),r=new Uint8ClampedArray(this.width*this.height*4);for(let d=0;d<this.height;d++){let p=this.height-1-d;for(let h=0;h<this.width;h++){let m=(d*this.width+h)*W,f=(p*this.width+h)*4,u=e[m+4]>=.5,S=e[m],b=u?M((S-t)/a,0,1):0,w=Math.max(0,Math.min(255,Math.round(b*255)));r[f]=w,r[f+1]=w,r[f+2]=w,r[f+3]=u?255:0}}let n=new OffscreenCanvas(this.width,this.height),l=n.getContext("2d");if(!l)throw new Error("Failed to create a terrain export canvas.");l.putImageData(new ImageData(r,this.width,this.height),0,0);let c=await(await n.convertToBlob({type:"image/png"})).arrayBuffer();return{width:this.width,height:this.height,minTerrain:t,maxTerrain:i,data:c}}#t(e,t){return e>0?e*.88+t*.12:t}#a(e){let t=Math.max(0,Number(e)||0);this.lastErosionPassMs=t,this.lastCpuErosionPassMs=t,this.avgErosionPassMs=this.#t(this.avgErosionPassMs,t),this.avgCpuErosionPassMs=this.#t(this.avgCpuErosionPassMs,t)}#e(e){let t=Math.max(0,Number(e)||0);this.lastRenderPassMs=t,this.lastCpuRenderPassMs=t,this.avgRenderPassMs=this.#t(this.avgRenderPassMs,t),this.avgCpuRenderPassMs=this.#t(this.avgCpuRenderPassMs,t)}#y(e){let t=Math.max(0,Number(e)||0);this.lastGpuErosionPassMs=t,this.avgGpuErosionPassMs=this.#t(this.avgGpuErosionPassMs,t)}#T(e){let t=Math.max(0,Number(e)||0);this.lastGpuRenderPassMs=t,this.avgGpuRenderPassMs=this.#t(this.avgGpuRenderPassMs,t)}getStats(){let e=this.simulationTime,t=Math.max(0,this.simulationParams.rainDuration??0),i=Math.max(0,e-this.simulationParams.sourceTimeOffset),a=t<=0||i<t||this.simulationParams.pulse2Duration>0&&i<t+this.simulationParams.pulse2Duration;return{width:this.width,height:this.height,cellCount:this.cellCount,ready:this.ready,iterationCount:this.iterationCount,simTime:e,rainActive:a,rainDuration:t,sourceFieldBuildCount:this.sourceFieldBuildCount,layeredDemEnabled:this.layeredDemEnabled,layerConfigMode:this.layerConfigMode,activeLayerCount:[0,1,2,3].reduce((r,n)=>r+(this.layerMaterialValues[n*4+2]>.5?1:0),0),lastErosionPassMs:this.lastErosionPassMs,lastRenderPassMs:this.lastRenderPassMs,avgErosionPassMs:this.avgErosionPassMs,avgRenderPassMs:this.avgRenderPassMs,lastCpuErosionPassMs:this.lastCpuErosionPassMs,lastCpuRenderPassMs:this.lastCpuRenderPassMs,avgCpuErosionPassMs:this.avgCpuErosionPassMs,avgCpuRenderPassMs:this.avgCpuRenderPassMs,gpuTimingMode:this.device?.features.has("timestamp-query")?"timestamp-query":"completion-time",lastGpuErosionPassMs:this.lastGpuErosionPassMs,lastGpuRenderPassMs:this.lastGpuRenderPassMs,avgGpuErosionPassMs:this.avgGpuErosionPassMs,avgGpuRenderPassMs:this.avgGpuRenderPassMs,...this.latestReadbackStats??{}}}#x(){if(!this.canvas)return;if(typeof this.canvas.getBoundingClientRect!="function"){let l=Math.max(1,Math.round(this.canvas.width||1)),o=Math.max(1,Math.round(this.canvas.height||1));this.canvas.width!==l&&(this.canvas.width=l),this.canvas.height!==o&&(this.canvas.height=o);return}let e=this.canvas.getBoundingClientRect(),t=Math.max(1,Math.round(e.width||this.canvas.width||1)),i=Math.max(1,Math.round(e.height||this.canvas.height||1)),a=Math.max(1,Math.min(globalThis.devicePixelRatio||1,2)),r=Math.max(1,Math.round(t*a)),n=Math.max(1,Math.round(i*a));this.canvas.width!==r&&(this.canvas.width=r),this.canvas.height!==n&&(this.canvas.height=n)}#l(){if(!this.device||!this.canvas)return;let e=Math.max(1,this.canvas.width||this.canvas.clientWidth||1),t=Math.max(1,this.canvas.height||this.canvas.clientHeight||1);this.depthTexture&&this.depthTextureSize.width===e&&this.depthTextureSize.height===t||(this.depthTexture?.destroy?.(),this.depthTexture=this.device.createTexture({size:{width:e,height:t},format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT,label:"paper-depth"}),this.depthTextureView=this.depthTexture.createView(),this.depthTextureSize={width:e,height:t})}#c(){if(!this.device||!this.canvas)return;let e=Math.max(1,this.canvas.width||this.canvas.clientWidth||1),t=Math.max(1,this.canvas.height||this.canvas.clientHeight||1);this.sceneColorTexture&&this.sceneColorSize.width===e&&this.sceneColorSize.height===t||(this.sceneColorTexture?.destroy?.(),this.sceneColorTexture=this.device.createTexture({size:{width:e,height:t},format:this.presentationFormat,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,label:"paper-scene-color"}),this.sceneColorView=this.sceneColorTexture.createView(),this.sceneColorSize={width:e,height:t},this.stateBuffers.length===2&&this.sceneSampler&&this.#g())}async#v(){this.#R();let e=this.device.createCommandEncoder({label:"paper-erosion-readback"});e.copyBufferToBuffer(this.stateBuffers[0],0,this.readbackBuffer,0,this.stateByteLength),this.device.queue.submit([e.finish()]),await this.readbackBuffer.mapAsync(GPUMapMode.READ);let i=this.readbackBuffer.getMappedRange().slice(0);return this.readbackBuffer.unmap(),new Float32Array(i)}#m(){if(!this.device)return;let e=Math.max(4,this.cellCount*4);this.customSpringBuffer&&this.allocatedPaintedSpringByteLength===e||(this.customSpringBuffer?.destroy?.(),this.customSpringBuffer=this.device.createBuffer({size:e,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:"paper-cached-source-rates"}),this.allocatedPaintedSpringByteLength=e,this.#b())}#p(){if(this.sourceFieldDirty=!0,!!this.device&&(this.#m(),!!this.customSpringBuffer)){if(!(this.paintedSpringMap instanceof Float32Array)||this.paintedSpringMap.length!==this.cellCount){this.#b();return}this.device.queue.writeBuffer(this.customSpringBuffer,0,this.paintedSpringMap)}}#R(){this.device&&(this.readbackBuffer&&this.allocatedReadbackByteLength===this.stateByteLength||(this.readbackBuffer?.destroy?.(),this.readbackBuffer=this.device.createBuffer({size:this.stateByteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,label:"paper-readback"}),this.allocatedReadbackByteLength=this.stateByteLength))}#A(){(!(this.paintedSpringMap instanceof Float32Array)||this.paintedSpringMap.length!==this.cellCount)&&(this.paintedSpringMap=new Float32Array(this.cellCount))}#b(){if(this.sourceFieldDirty=!0,!this.device||(this.#m(),!this.customSpringBuffer))return;let e=this.device.createCommandEncoder({label:"paper-clear-painted-springs"});e.clearBuffer(this.customSpringBuffer),this.device.queue.submit([e.finish()])}#w(){!this.device||!this.presentationFormat||this.dummySceneTexture||(this.dummySceneTexture=this.device.createTexture({size:{width:1,height:1},format:this.presentationFormat,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST,label:"paper-dummy-scene-color"}),this.dummySceneView=this.dummySceneTexture.createView(),this.device.queue.writeTexture({texture:this.dummySceneTexture},new Uint8Array([14,17,20,255]),{bytesPerRow:256},{width:1,height:1,depthOrArrayLayers:1}))}#f(){this.terrainRenderBundle=null,this.finalRenderBundle=null,this.terrainRenderBundleList.length=0,this.finalRenderBundleList.length=0}#E(){if(!this.device||!this.renderPipeline||!this.compositePipeline||!this.waterRenderPipeline||!this.renderBindGroupTerrain||!this.renderBindGroupWater||!this.compositeBindGroup||this.terrainRenderBundle&&this.finalRenderBundle)return;let e=this.drawCount||Math.max(1,(this.width-1)*(this.height-1)),t=this.device.createRenderBundleEncoder({colorFormats:[this.presentationFormat],depthStencilFormat:"depth24plus"});t.setPipeline(this.renderPipeline),t.setBindGroup(0,this.renderBindGroupTerrain),t.draw(6,e,0,0),this.terrainRenderBundle=t.finish({label:"paper-render-terrain-bundle"}),this.terrainRenderBundleList[0]=this.terrainRenderBundle;let i=this.device.createRenderBundleEncoder({colorFormats:[this.presentationFormat],depthStencilFormat:"depth24plus"});i.setPipeline(this.compositePipeline),i.setBindGroup(0,this.compositeBindGroup),i.draw(3,1,0,0),i.setPipeline(this.waterRenderPipeline),i.setBindGroup(0,this.renderBindGroupWater),i.draw(6,e,0,0),this.finalRenderBundle=i.finish({label:"paper-render-final-bundle"}),this.finalRenderBundleList[0]=this.finalRenderBundle}#M(e,t,i=null){if(!this.ready)return;let a=this.dispatchX||Math.ceil(this.width/Re),r=this.dispatchY||Math.ceil(this.height/Ae);this.#F(t);let n=this.stepParamStride>>>2,l=this.stepParamValuesCpu,o=this.paramValues[3],c=this.simulationTime,d=this.#P(o);for(let w=0;w<t;w++){let y=w*n;l[y+25]=c+w*o,this.#C(l,y,c+w*o,d)}this.device.queue.writeBuffer(this.stepParamBuffer,0,l.buffer,0,this.stepParamStride*t);let p=this.fluxPipeline,h=this.flowPipeline,v=this.thermalOutflowPipeline,m=this.erosionPipeline,f=this.computeBindGroup01,u=this.computeBindGroup10,S=this.computeDynamicOffset,b=e.beginComputePass({label:"paper-erosion-compute",timestampWrites:i?.writes()});this.sourceFieldDirty&&this.simulationParams.sourceEnabled&&this.simulationParams.sourceLayoutMode>=.5&&(S[0]=0,b.setBindGroup(0,f,S),b.setPipeline(this.sourcePipeline),b.dispatchWorkgroups(a,r),this.sourceFieldDirty=!1,this.sourceFieldBuildCount++);for(let w=0;w<t;w++)S[0]=w*this.stepParamStride,b.setBindGroup(0,f,S),b.setPipeline(p),b.dispatchWorkgroups(a,r),b.setPipeline(h),b.dispatchWorkgroups(a,r),b.setBindGroup(0,u,S),this.simulationParams.bendSedimentRate>0&&this.simulationParams.riverBendLength>0&&this.simulationParams.hydraulicErosionEnabled&&(b.setPipeline(this.channelPipeline),b.dispatchWorkgroups(this.height),b.setPipeline(this.channelBendPipeline),b.dispatchWorkgroups(Math.ceil(this.height/64))),(this.simulationParams.thermalRate>0||(this.simulationParams.bankErosionRate>0||this.simulationParams.bendSedimentRate>0)&&this.simulationParams.hydraulicErosionEnabled)&&(b.setPipeline(v),b.dispatchWorkgroups(a,r)),b.setPipeline(m),b.dispatchWorkgroups(a,r);b.end(),this.iterationCount+=t,this.simulationTime+=t*o,this.#B(this.iterationCount,this.paramValues)}#S(e,t,i=null){this.#E();{let a=e.beginRenderPass({timestampWrites:i?.writes(0),colorAttachments:[{view:this.sceneColorView,clearValue:{r:.055,g:.065,b:.08,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:this.depthTextureView?{view:this.depthTextureView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}:void 0});if(this.terrainRenderBundle)a.executeBundles(this.terrainRenderBundleList);else{let r=this.drawCount||Math.max(1,(this.width-1)*(this.height-1));a.setPipeline(this.renderPipeline),a.setBindGroup(0,this.renderBindGroupTerrain),a.draw(6,r,0,0)}a.end()}{let a=e.beginRenderPass({timestampWrites:i?.writes(1),colorAttachments:[{view:t,clearValue:{r:.055,g:.065,b:.08,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:this.depthTextureView?{view:this.depthTextureView,depthLoadOp:"load",depthStoreOp:"discard"}:void 0});if(this.finalRenderBundle)a.executeBundles(this.finalRenderBundleList);else{let r=this.drawCount||Math.max(1,(this.width-1)*(this.height-1));a.setPipeline(this.compositePipeline),a.setBindGroup(0,this.compositeBindGroup),a.draw(3,1,0,0),a.setPipeline(this.waterRenderPipeline),a.setBindGroup(0,this.renderBindGroupWater),a.draw(6,r,0,0)}a.end()}}#h(){if(!this.renderUniformBufferTerrain&&!this.renderUniformBufferWater||!this.device)return;let e=this.canvas?Math.max((this.canvas.width||1)/Math.max(this.canvas.height||1,1),1e-6):1,t=fe(this.simulationParams.cameraAzimuthDeg),i=fe(this.simulationParams.cameraElevationDeg),a=this.width>1||this.height>1?2/Math.max(this.width-1,this.height-1,1):1,r=[this.simulationParams.cameraPosX??1.5238999619464006,this.simulationParams.cameraPosY??1.9404787584406888,this.simulationParams.cameraPosZ??1.5238999619464006],n=oe([-Math.cos(i)*Math.cos(t),-Math.sin(i),-Math.cos(i)*Math.sin(t)]),l=[r[0]+n[0],r[1]+n[1],r[2]+n[2]],o=Ze(r,l,[0,1,0]),c=Ke(fe(50),e,.01,32),d=Xe(c,o),p=Math.max(.25,(this.simulationParams.metersPerPixel??100)/100);this.renderUniformValues.set(d,0),this.renderUniformValues[16]=this.width,this.renderUniformValues[17]=this.height,this.renderUniformValues[18]=p,this.renderUniformValues[19]=this.simulationParams.renderHeightScale,this.renderUniformValues[20]=this.simulationParams.waterOpacity,this.renderUniformValues[21]=this.simulationParams.sedimentTint,this.renderUniformValues[22]=this.simulationParams.renderMode,this.renderUniformValues[23]=this.simulationParams.thermalVisualizationScale,this.renderUniformValues[24]=a,this.renderUniformValues[25]=this.simulationParams.waterHeightScale,this.renderUniformValues[27]=5e-4;let h=oe([.45,.82,.36]);this.renderUniformValues[28]=h[0],this.renderUniformValues[29]=h[1],this.renderUniformValues[30]=h[2];let v=Math.max(1,Math.floor(this.simulationParams.renderGridStep)||Math.ceil(Math.max(this.width-1,this.height-1,1)/512)),m=Math.max(1,Math.ceil((this.width-1)/v)*Math.ceil((this.height-1)/v));this.drawCount!==m&&(this.drawCount=m,this.#f()),this.renderUniformValues[31]=v,this.renderUniformValues[32]=r[0],this.renderUniformValues[33]=r[1],this.renderUniformValues[34]=r[2],this.renderUniformValues[35]=this.layeredDemEnabled?1:0,this.renderUniformValues[36]=this.simulationTime,this.renderUniformValues[37]=this.iterationCount,this.renderUniformValues[38]=this.canvas?.width||1,this.renderUniformValues[39]=this.canvas?.height||1,this.renderUniformValues[26]=0,this.device.queue.writeBuffer(this.renderUniformBufferTerrain,0,this.renderUniformValues),this.renderUniformValues[26]=1,this.device.queue.writeBuffer(this.renderUniformBufferWater,0,this.renderUniformValues)}#O(){if(this.stateBuffers.length!==2||!this.fluxBuffer||!this.velocityBuffer||!this.thermalPipeBufferA||!this.thermalPipeBufferB||!this.layerTopBuffer||this.allocatedStateByteLength!==this.stateByteLength||this.allocatedVec4ByteLength!==this.vec4ByteLength||this.velocityBuffer?.size!==this.vec4ByteLength+this.height*32){for(let t of this.stateBuffers)t?.destroy?.();this.fluxBuffer?.destroy?.(),this.velocityBuffer?.destroy?.(),this.thermalPipeBufferA?.destroy?.(),this.thermalPipeBufferB?.destroy?.(),this.readbackBuffer?.destroy?.(),this.customSpringBuffer?.destroy?.(),this.layerTopBuffer?.destroy?.(),this.customSpringBuffer=null,this.layerTopBuffer=null,this.stateBuffers=[0,1].map(t=>this.device.createBuffer({size:this.stateByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:`paper-state-${t}`})),this.fluxBuffer=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:"paper-flux"}),this.velocityBuffer=this.device.createBuffer({size:this.vec4ByteLength+this.height*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:"paper-velocity"}),this.thermalPipeBufferA=this.device.createBuffer({size:this.vec4ByteLength*2,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:"paper-thermal-pipes-a"}),this.thermalPipeBufferB=this.device.createBuffer({size:this.vec4ByteLength*2,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-thermal-pipes-b"}),this.layerTopBuffer=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-layer-top-heights"}),this.readbackBuffer=null,this.allocatedReadbackByteLength=0,this.allocatedStateByteLength=this.stateByteLength,this.allocatedVec4ByteLength=this.vec4ByteLength,this.#g()}}#P(e){let t=Math.max(e,0)*.5,i=Math.max(this.simulationParams.evaporationRate,0),a=Math.max(this.simulationParams.rainDuration??0,0);return{half:t,rate:i,decay:Math.exp(-i*t),fullWeight:i>0?-Math.expm1(-i*t)/i:t,end:a>0?this.simulationParams.sourceTimeOffset+a+Math.max(this.simulationParams.pulse2Duration??0,0):1/0}}#C(e,t,i,a){let{half:r,rate:n,decay:l,fullWeight:o,end:c}=a,d=M(c-i,0,r*2),p=Math.min(d,r),h=Math.max(d-r,0);e[t+40]=p===r?o:n>0?-Math.expm1(-n*p)/n*Math.exp(-n*(r-p)):p,e[t+41]=l,e[t+42]=h===r?o:n>0?-Math.expm1(-n*h)/n*Math.exp(-n*(r-h)):h,e[t+43]=p,e[t+47]=h}#B(e,t=this.paramValues){let i=Math.max(.25,(this.simulationParams.metersPerPixel??100)/100);t[0]=this.width,t[1]=this.height,t[2]=i,t[3]=this.simulationParams.timeStep,t[4]=this.simulationParams.rainRate,t[5]=this.simulationParams.evaporationRate,t[6]=this.simulationParams.pipeArea,t[7]=this.simulationParams.gravity,t[8]=this.simulationParams.capacityScale,t[9]=this.simulationParams.suspensionRate,t[10]=this.simulationParams.depositionRate,t[11]=this.simulationParams.softeningRate,t[12]=this.simulationParams.maxErosionDepth,t[13]=this.simulationParams.thermalRate,t[14]=this.simulationParams.talusSlopeCoeff,t[15]=this.simulationParams.talusSlopeBias,t[16]=this.simulationParams.sourceLayoutMode,t[17]=this.simulationParams.randomSpringCount,t[18]=this.simulationParams.pulse2Duration,t[19]=this.simulationParams.edgeDrainStrength,t[20]=this.simulationParams.sourceCenterX,t[21]=this.simulationParams.sourceCenterY,t[22]=this.simulationParams.sourceRadius,t[23]=this.simulationParams.sourceStrength,t[24]=this.simulationParams.sourceEnabled?1:0,t[25]=this.simulationTime,t[26]=Math.max(0,this.simulationParams.rainDuration??0),t[27]=this.simulationParams.sourceTimeOffset,t[28]=this.simulationParams.historyDecay,t[29]=this.simulationParams.hardnessBase,t[30]=this.simulationParams.closedBoundaries?1:0,t[31]=4;let a=t===this.paramValues?this.paramBits:new Uint32Array(t.buffer,t.byteOffset,t.length);return a[32]=Math.floor(Math.max(0,this.simulationParams.sourceSeed??1))>>>0,t[33]=this.simulationParams.metersPerPixel??100,t[34]=this.simulationParams.hydraulicErosionEnabled?1:0,t[35]=Math.max(0,this.simulationParams.edgeWaterFloor??0),t[36]=this.simulationParams.hardnessBase,t[37]=this.simulationParams.hardnessVariation,t[38]=Math.max(0,this.simulationParams.flowDrag??2),t[39]=M(this.simulationParams.maxOutflowFraction??.5,.05,1),t[44]=Math.max(0,this.simulationParams.bankErosionRate??0),t[45]=Math.max(0,this.simulationParams.bankBendBias??24),t[46]=Math.max(0,this.simulationParams.sourceSedimentConcentration??0),t[47]=0,t[48]=Math.max(0,this.simulationParams.bendSedimentRate??0),t[49]=Math.max(0,this.simulationParams.straightBankErosionFactor??.08),t[50]=Math.max(0,this.simulationParams.riverBendLength??0),t[51]=Math.max(0,Math.min(15,Math.floor(this.simulationParams.openBoundaryEdges??15))),this.#C(t,0,this.simulationTime,this.#P(t[3])),t}#s(){this.#B(this.iterationCount,this.paramValues),this.stepParamTemplateDirty=!0,this.stepParamPreparedIterations=0,this.paramBuffer&&this.device.queue.writeBuffer(this.paramBuffer,0,this.paramValues)}#F(e=1){if(this.#k(e),!this.stepParamValuesCpu||!this.stepParamTemplateDirty&&this.stepParamPreparedIterations>=e)return;let t=this.stepParamStride>>>2,i=this.stepParamTemplateDirty?0:this.stepParamPreparedIterations;for(let a=i;a<e;a++)this.stepParamValuesCpu.set(this.paramValues,a*t);this.stepParamTemplateDirty=!1,this.stepParamPreparedIterations=Math.max(this.stepParamPreparedIterations,e)}#k(e=1){let t=Math.max(1,e|0);if(this.stepParamBuffer&&this.stepParamCapacity>=t&&this.stepParamValuesCpu&&this.stepParamValuesCpuCapacity>=t)return;let i=Math.max(8,t,this.stepParamCapacity>0?this.stepParamCapacity*2:0),a=this.stepParamStride>>>2;(!this.stepParamValuesCpu||this.stepParamValuesCpuCapacity<t)&&(this.stepParamValuesCpu=new Float32Array(a*i),this.stepParamValuesCpuCapacity=i,this.stepParamTemplateDirty=!0,this.stepParamPreparedIterations=0),(!this.stepParamBuffer||this.stepParamCapacity<t)&&(this.stepParamBuffer?.destroy?.(),this.stepParamBuffer=this.device.createBuffer({size:this.stepParamStride*i,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-step-params"}),this.stepParamCapacity=i,this.stepParamTemplateDirty=!0,this.stepParamPreparedIterations=0,this.stateBuffers.length===2&&this.#g())}async#U(){let e=this.device.createShaderModule({code:$e,label:"paper-erosion-compute-wgsl"}),t=this.device.createShaderModule({code:Ue,label:"paper-erosion-render-wgsl"}),i=this.device.createShaderModule({code:Qe,label:"paper-erosion-composite-wgsl"}),a=typeof this.device.createComputePipelineAsync=="function"?c=>this.device.createComputePipelineAsync(c):async c=>this.device.createComputePipeline(c),r=typeof this.device.createRenderPipelineAsync=="function"?c=>this.device.createRenderPipelineAsync(c):async c=>this.device.createRenderPipeline(c);this.computeBindGroupLayout=this.device.createBindGroupLayout({label:"paper-compute-bgl",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"uniform",hasDynamicOffset:!0}},{binding:1,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),this.renderBindGroupLayout=this.device.createBindGroupLayout({label:"paper-render-bgl",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:4,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:5,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}},{binding:6,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]}),this.compositeBindGroupLayout=this.device.createBindGroupLayout({label:"paper-composite-bgl",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});let n=this.device.createPipelineLayout({label:"paper-compute-layout",bindGroupLayouts:[this.computeBindGroupLayout]}),l=this.device.createPipelineLayout({label:"paper-render-layout",bindGroupLayouts:[this.renderBindGroupLayout]}),o=this.device.createPipelineLayout({label:"paper-composite-layout",bindGroupLayouts:[this.compositeBindGroupLayout]});[this.fluxPipeline,this.sourcePipeline,this.flowPipeline,this.erosionPipeline,this.thermalOutflowPipeline,this.channelPipeline,this.channelBendPipeline,this.renderPipeline,this.waterRenderPipeline,this.compositePipeline]=await Promise.all([a({label:"paper-flux-pipeline",layout:n,compute:{module:e,entryPoint:"fluxMain"}}),a({label:"paper-source-pipeline",layout:n,compute:{module:e,entryPoint:"sourceMain"}}),a({label:"paper-flow-pipeline",layout:n,compute:{module:e,entryPoint:"flowMain"}}),a({label:"paper-erosion-pipeline",layout:n,compute:{module:e,entryPoint:"erosionMain"}}),a({label:"paper-thermal-outflow-pipeline",layout:n,compute:{module:e,entryPoint:"thermalOutflowMain"}}),a({label:"river-channel-guide-pipeline",layout:n,compute:{module:e,entryPoint:"channelMain"}}),a({label:"paper-channel-bend-pipeline",layout:n,compute:{module:e,entryPoint:"channelBendMain"}}),r({label:"paper-render-pipeline",layout:l,vertex:{module:t,entryPoint:"vsMesh"},fragment:{module:t,entryPoint:"fsMesh",targets:[{format:this.presentationFormat}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),r({label:"paper-water-render-pipeline",layout:l,vertex:{module:t,entryPoint:"vsMesh"},fragment:{module:t,entryPoint:"fsMesh",targets:[{format:this.presentationFormat,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:!1,depthCompare:"less-equal"}}),r({label:"paper-composite-pipeline",layout:o,vertex:{module:i,entryPoint:"vsComposite"},fragment:{module:i,entryPoint:"fsComposite",targets:[{format:this.presentationFormat}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:!1,depthCompare:"always"}})])}#g(){!this.sceneColorView||!this.sceneSampler||(this.#f(),this.#w(),this.#m(),this.#k(1),!(!this.dummySceneView||!this.stepParamBuffer||!this.customSpringBuffer||!this.layerTopBuffer||!this.layerMaterialBuffer)&&(this.computeBindGroup01=this.device.createBindGroup({label:"paper-compute-0to1",layout:this.computeBindGroupLayout,entries:[{binding:0,resource:{buffer:this.stepParamBuffer,size:this.paramValues.byteLength}},{binding:1,resource:{buffer:this.stateBuffers[0]}},{binding:2,resource:{buffer:this.stateBuffers[1]}},{binding:3,resource:{buffer:this.fluxBuffer}},{binding:4,resource:{buffer:this.velocityBuffer}},{binding:5,resource:{buffer:this.thermalPipeBufferA}},{binding:6,resource:{buffer:this.thermalPipeBufferB}},{binding:7,resource:{buffer:this.customSpringBuffer}},{binding:8,resource:{buffer:this.layerTopBuffer}},{binding:9,resource:{buffer:this.layerMaterialBuffer}}]}),this.computeBindGroup10=this.device.createBindGroup({label:"paper-compute-1to0",layout:this.computeBindGroupLayout,entries:[{binding:0,resource:{buffer:this.stepParamBuffer,size:this.paramValues.byteLength}},{binding:1,resource:{buffer:this.stateBuffers[1]}},{binding:2,resource:{buffer:this.stateBuffers[0]}},{binding:3,resource:{buffer:this.fluxBuffer}},{binding:4,resource:{buffer:this.velocityBuffer}},{binding:5,resource:{buffer:this.thermalPipeBufferA}},{binding:6,resource:{buffer:this.thermalPipeBufferB}},{binding:7,resource:{buffer:this.customSpringBuffer}},{binding:8,resource:{buffer:this.layerTopBuffer}},{binding:9,resource:{buffer:this.layerMaterialBuffer}}]}),this.renderBindGroupTerrain=this.device.createBindGroup({label:"paper-render-terrain",layout:this.renderBindGroupLayout,entries:[{binding:0,resource:{buffer:this.renderUniformBufferTerrain}},{binding:1,resource:{buffer:this.stateBuffers[0]}},{binding:2,resource:this.sceneSampler},{binding:3,resource:this.dummySceneView},{binding:4,resource:{buffer:this.layerTopBuffer}},{binding:5,resource:{buffer:this.layerMaterialBuffer}},{binding:6,resource:{buffer:this.velocityBuffer}}]}),this.renderBindGroupWater=this.device.createBindGroup({label:"paper-render-water",layout:this.renderBindGroupLayout,entries:[{binding:0,resource:{buffer:this.renderUniformBufferWater}},{binding:1,resource:{buffer:this.stateBuffers[0]}},{binding:2,resource:this.sceneSampler},{binding:3,resource:this.sceneColorView},{binding:4,resource:{buffer:this.layerTopBuffer}},{binding:5,resource:{buffer:this.layerMaterialBuffer}},{binding:6,resource:{buffer:this.velocityBuffer}}]}),this.compositeBindGroup=this.device.createBindGroup({label:"paper-composite",layout:this.compositeBindGroupLayout,entries:[{binding:0,resource:this.sceneSampler},{binding:1,resource:this.sceneColorView}]})))}},$e=`
struct SimParams {
  dims: vec4<f32>,
  hydro0: vec4<f32>,
  hydro1: vec4<f32>,
  thermal0: vec4<f32>,
  render0: vec4<f32>,
  source0: vec4<f32>,
  source1: vec4<f32>,
  misc0: vec4<f32>,
  sourceSeed: u32,
  sourceMetersPerPixel: f32,
  hydraulicEnabled: f32,
  edgeWaterFloor: f32,
  material0: vec4<f32>,
  integration: vec4<f32>,
  banks: vec4<f32>,
  meander: vec4<f32>,
}

struct CellState {
  terrain: f32,
  water: f32,
  sediment: f32,
  hardness: f32,
  mask: f32,
  aux0: f32,
  aux1: f32,
  aux2: f32,
}

struct StateBuffer {
  cells: array<CellState>,
}

struct Vec4Buffer {
  values: array<vec4<f32>>,
}

struct FloatBuffer {
  values: array<f32>,
}

struct LayerMaterialParams {
  layers: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> srcState: StateBuffer;
@group(0) @binding(2) var<storage, read_write> dstState: StateBuffer;
@group(0) @binding(3) var<storage, read_write> fluxState: Vec4Buffer;
@group(0) @binding(4) var<storage, read_write> velocityState: Vec4Buffer;
@group(0) @binding(5) var<storage, read_write> thermalPipeA: Vec4Buffer;
@group(0) @binding(6) var<storage, read_write> thermalPipeB: Vec4Buffer;
@group(0) @binding(7) var<storage, read_write> paintedSourceState: FloatBuffer;
@group(0) @binding(8) var<storage, read> layerTopState: Vec4Buffer;
@group(0) @binding(9) var<uniform> layerMaterials: LayerMaterialParams;

fn gridWidth() -> u32 { return u32(params.dims.x); }
fn gridHeight() -> u32 { return u32(params.dims.y); }
fn timeStep() -> f32 { return params.dims.w; }
fn cellSize() -> f32 { return max(params.dims.z, 1e-6); }
fn cellArea() -> f32 { let c = cellSize(); return c * c; }
fn idx(x: u32, y: u32) -> u32 { return y * gridWidth() + x; }
fn hydraulicErosionEnabled() -> bool { return params.hydraulicEnabled > 0.5; }

fn clampCoord(v: i32, dim: u32) -> u32 {
  return u32(clamp(v, 0, max(i32(dim) - 1, 0)));
}

fn inBounds(x: i32, y: i32) -> bool {
  return x >= 0 && y >= 0 && x < i32(gridWidth()) && y < i32(gridHeight());
}

fn readStateClamped(x: i32, y: i32) -> CellState {
  return srcState.cells[idx(clampCoord(x, gridWidth()), clampCoord(y, gridHeight()))];
}

fn readFluxOrZero(x: i32, y: i32) -> vec4<f32> {
  if (!inBounds(x, y)) { return vec4<f32>(0.0); }
  return fluxState.values[idx(u32(x), u32(y))];
}

fn readVelocityClamped(x: i32, y: i32) -> vec4<f32> {
  return velocityState.values[idx(clampCoord(x, gridWidth()), clampCoord(y, gridHeight()))];
}

fn readThermalAOrZero(x: i32, y: i32) -> vec4<f32> {
  if (!inBounds(x, y)) { return vec4<f32>(0.0); }
  return thermalPipeA.values[idx(u32(x), u32(y))];
}

fn readThermalBOrZero(x: i32, y: i32) -> vec4<f32> {
  if (!inBounds(x, y)) { return vec4<f32>(0.0); }
  return thermalPipeB.values[idx(u32(x), u32(y))];
}

fn totalHeight(cell: CellState) -> f32 { return cell.terrain + cell.water; }

fn finiteOr(value: f32, fallback: f32) -> f32 {
  if (value == value && abs(value) < 1e30) { return value; }
  return fallback;
}

fn resolveMaterialInfo(i: u32, terrain: f32) -> vec4<f32> {
  let tops = layerTopState.values[i];
  var chosenIndex: u32 = 0u;
  var foundChoice = false;
  var bestAbove = 1e30;
  var highestTop = -1e30;
  var highestIndex: u32 = 0u;

  for (var layerIndex: u32 = 0u; layerIndex < 4u; layerIndex = layerIndex + 1u) {
    let material = layerMaterials.layers[layerIndex];
    if (material.z < 0.5) { continue; }
    let top = tops[layerIndex];
    if (!(top > -1e8)) { continue; }
    if (top >= highestTop) {
      highestTop = top;
      highestIndex = layerIndex;
    }
    // Later layers are deeper: coincident tops mean no upper-layer thickness.
    if (top >= terrain && top <= bestAbove) {
      bestAbove = top;
      chosenIndex = layerIndex;
      foundChoice = true;
    }
  }

  if (!foundChoice) {
    if (!(highestTop > -1e8)) {
      return vec4<f32>(-1.0, 1.0, -1.0, 0.0);
    }
    chosenIndex = highestIndex;
  }

  let chosenTop = tops[chosenIndex];
  var lowerTop = -1e30;
  for (var layerIndex: u32 = 0u; layerIndex < 4u; layerIndex = layerIndex + 1u) {
    let material = layerMaterials.layers[layerIndex];
    if (material.z < 0.5) { continue; }
    let top = tops[layerIndex];
    if (!(top > -1e8) || top >= chosenTop) { continue; }
    if (top > lowerTop) { lowerTop = top; }
  }

  var depthT = 0.0;
  if (lowerTop > -1e8) {
    depthT = clamp((chosenTop - terrain) / max(chosenTop - lowerTop, 1e-6), 0.0, 1.0);
  } else if (terrain < chosenTop) {
    depthT = clamp((chosenTop - terrain) / max(chosenTop, 1e-6), 0.0, 1.0);
  }

  let selectedMaterial = layerMaterials.layers[chosenIndex];
  return vec4<f32>(clamp(mix(selectedMaterial.x, selectedMaterial.y, depthT), 0.0, 1.0), selectedMaterial.w, f32(chosenIndex), depthT);
}

fn resolveMaterialHardness(i: u32, terrain: f32, fallbackHardness: f32) -> f32 {
  let info = resolveMaterialInfo(i, terrain);
  if (info.z < 0.0) {
    return clamp(params.material0.x + (1.0 - clamp(terrain, 0.0, 1.0)) * params.material0.y, 0.0, 1.0);
  }
  return info.x;
}

fn resolveMaterialThermalEnabled(i: u32, terrain: f32) -> bool {
  let info = resolveMaterialInfo(i, terrain);
  if (info.z < 0.0) {
    return true;
  }
  return info.y >= 0.5;
}

// Stop at the next material interface so a large cut cannot skip a resistant
// stratum. The following step resolves that newly exposed material's rates.
fn materialCutLimit(i: u32, terrain: f32) -> f32 {
  let tops = layerTopState.values[i];
  var floorHeight = 0.0;
  for (var layerIndex = 0u; layerIndex < 4u; layerIndex++) {
    if (layerMaterials.layers[layerIndex].z < 0.5) { continue; }
    let top = tops[layerIndex];
    if (top < terrain && top > floorHeight) { floorHeight = top; }
  }
  return max(terrain - floorHeight, 0.0);
}

fn bankTransfersEnabled() -> bool { return params.banks.x > 0.0 && hydraulicErosionEnabled(); }
fn readBankOrZero(x: i32, y: i32) -> vec4<f32> {
  if (!inBounds(x, y)) { return vec4<f32>(0.0); }
  return thermalPipeA.values[gridWidth() * gridHeight() + idx(u32(x), u32(y))];
}
fn directionAt(x: i32, y: i32, fallback: vec2<f32>) -> vec2<f32> {
  if (!inBounds(x, y)) { return fallback; }
  let cell = srcState.cells[idx(u32(x), u32(y))];
  let v = velocityState.values[idx(u32(x), u32(y))];
  if (cell.mask < 0.5 || cell.water < 0.0005 || v.z < 0.01) { return fallback; }
  return v.xy / max(length(v.xy), 1e-6);
}
// Empirical bank undercutting. Flow-direction change along the current gives
// the bend's outward side; straight banks receive only a small baseline rate.
// Every donor transfer is later received as suspended solid, never deleted.
fn averagedFlowDirection(x: i32, y: i32, fallback: vec2<f32>) -> vec2<f32> {
  var momentum = vec2<f32>(0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (!inBounds(x + dx, y + dy)) { continue; }
      let i = idx(u32(x + dx), u32(y + dy));
      let cell = srcState.cells[i];
      let v = velocityState.values[i];
      if (cell.mask > 0.5 && cell.water > 0.0005 && v.w > 1e-5) { momentum += v.xy * cell.water; }
    }
  }
  if (length(momentum) < 1e-6) { return fallback; }
  return normalize(momentum);
}
fn flowTurn(nx: i32, ny: i32, direction: vec2<f32>) -> vec2<f32> {
  if (params.meander.x > 0.0 && params.meander.z > 0.0) {
    let row = u32(clamp(ny, 0, i32(gridHeight()) - 1));
    return velocityState.values[gridWidth() * gridHeight() + gridHeight() + row].xy;
  }
  if (params.meander.x > 0.0) {
    // Sample several cells along the current, suppressing cell-scale bends.
    let step = vec2<i32>(round(direction * 3.0));
    return (averagedFlowDirection(nx + step.x, ny + step.y, direction)
      - averagedFlowDirection(nx - step.x, ny - step.y, direction)) / 6.0;
  }
  let dx = (directionAt(nx + 1, ny, direction) - directionAt(nx - 1, ny, direction)) * 0.5;
  let dy = (directionAt(nx, ny + 1, direction) - directionAt(nx, ny - 1, direction)) * 0.5;
  return direction.x * dx + direction.y * dy;
}
fn channelCenter(y: i32) -> f32 {
  var moment = 0.0;
  var weight = 0.0;
  for (var d = -3; d <= 3; d++) {
    let row = u32(clamp(y + d, 0, i32(gridHeight()) - 1));
    let data = velocityState.values[gridWidth() * gridHeight() + row];
    let w = data.y * f32(4 - abs(d));
    moment += data.x * w; weight += w;
  }
  return select(f32(gridWidth()) * 0.5, moment / max(weight, 1e-10), weight > 1e-8);
}
// Optional streamwise guide for north-to-south lowland rivers. It comes from
// current discharge, not an imposed curve or an elapsed-time animation.
var<workgroup> channelMoments: array<f32, 64>;
var<workgroup> channelWeights: array<f32, 64>;
@compute @workgroup_size(64)
fn channelMain(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let row = group.x;
  var moment = 0.0;
  var weight = 0.0;
  for (var x = local; x < gridWidth(); x += 64u) {
    let i = idx(x, row);
    let cell = srcState.cells[i];
    if (cell.mask < 0.5 || cell.water < 0.0005) { continue; }
    let f = fluxState.values[i];
    let discharge = max(f.w - f.z, 0.0);
    // Prefer the active channel over broad, slowly moving floodplain films.
    let q = discharge * discharge;
    moment += f32(x) * q; weight += q;
  }
  channelMoments[local] = moment; channelWeights[local] = weight;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local < stride) { channelMoments[local] += channelMoments[local + stride]; channelWeights[local] += channelWeights[local + stride]; }
    workgroupBarrier();
  }
  if (local == 0u) {
    velocityState.values[gridWidth() * gridHeight() + row] = vec4<f32>(channelMoments[0] / max(channelWeights[0], 1e-10), channelWeights[0], 0.0, 0.0);
  }
}
// Cache the smoothed bend once per row instead of repeating its 21 guide
// reads for every wet receiver and adjacent bank in the erosion pass.
@compute @workgroup_size(64)
fn channelBendMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridHeight()) { return; }
  let y = i32(gid.x);
  let distance = max(4, i32(round(f32(gridHeight()) * params.meander.z)));
  let center = channelCenter(y);
  let above = channelCenter(y - distance);
  let below = channelCenter(y + distance);
  let slope = (below - above) / f32(distance * 2);
  let curve = (below - 2.0 * center + above) / f32(distance * distance);
  let turn = vec2<f32>(1.0, -slope) * curve / pow(1.0 + slope * slope, 2.0);
  velocityState.values[gridWidth() * gridHeight() + gridHeight() + gid.x] = vec4<f32>(turn, 0.0, 0.0);
}
fn bankWeight(bank: CellState, nx: i32, ny: i32, toBank: vec2<f32>) -> f32 {
  if (!inBounds(nx, ny)) { return 0.0; }
  let receiver = srcState.cells[idx(u32(nx), u32(ny))];
  let v = velocityState.values[idx(u32(nx), u32(ny))];
  if (receiver.mask < 0.5 || receiver.water < 0.001 || v.z < 0.02 || v.w < 1e-5) { return 0.0; }
  let direction = v.xy / max(length(v.xy), 1e-6);
  let turn = flowTurn(nx, ny, direction);
  let side = pow(max(1.0 - abs(dot(direction, toBank)), 0.0), 2.0);
  let bend = clamp(-dot(turn, toBank) * params.banks.y, 0.0, 3.0);
  let relief = max(bank.terrain - receiver.terrain - 0.0005, 0.0);
  return relief * side * (params.meander.y + bend) * min(v.z, 2.0) * smoothstep(0.001, 0.008, receiver.water);
}

fn bendSedimentEnabled() -> bool { return params.meander.x > 0.0 && hydraulicErosionEnabled(); }
fn readBendSediment(x: i32, y: i32) -> vec4<f32> {
  if (!inBounds(x, y)) { return vec4<f32>(0.0); }
  return thermalPipeB.values[gridWidth() * gridHeight() + idx(u32(x), u32(y))];
}
fn sedimentReceiver(nx: i32, ny: i32, cell: CellState) -> f32 {
  if (!inBounds(nx, ny)) { return 0.0; }
  let receiver = srcState.cells[idx(u32(nx), u32(ny))];
  return select(0.0, 1.0, receiver.mask > 0.5 && receiver.terrain < cell.terrain + cell.water);
}
// Empirical transverse bed-load drift toward a bend's inner side. These
// donor-limited internal transfers help point bars deflect the current.
fn writeBendSediment(i: u32, x: i32, y: i32, cell: CellState) {
  if (!bendSedimentEnabled()) { return; }
  let offset = gridWidth() * gridHeight() + i;
  thermalPipeB.values[offset] = vec4<f32>(0.0);
  let v = velocityState.values[i];
  if (cell.mask < 0.5 || cell.water < 0.001 || v.z < 0.02 || v.w < 1e-5 || cell.sediment <= 0.0) { return; }
  let direction = v.xy / max(length(v.xy), 1e-6);
  let turn = flowTurn(x, y, direction);
  let drift = turn - direction * dot(turn, direction);
  let weights = max(vec4<f32>(-drift.x, drift.x, -drift.y, drift.y), vec4<f32>(0.0))
    * vec4<f32>(sedimentReceiver(x - 1, y, cell), sedimentReceiver(x + 1, y, cell), sedimentReceiver(x, y - 1, cell), sedimentReceiver(x, y + 1, cell));
  let sum = dot(weights, vec4<f32>(1.0));
  if (sum < 1e-8) { return; }
  let fraction = min(0.5, 1.0 - exp(-params.meander.x * min(sum * params.banks.y, 3.0) * timeStep()));
  thermalPipeB.values[offset] = weights * (cell.sediment * fraction / sum);
}

fn writeBankTransfers(i: u32, x: i32, y: i32, cell: CellState) -> f32 {
  if (!bankTransfersEnabled()) { return 0.0; }
  let offset = gridWidth() * gridHeight() + i;
  thermalPipeA.values[offset] = vec4<f32>(0.0);
  if (cell.mask < 0.5) { return 0.0; }
  let weights = vec4<f32>(bankWeight(cell, x - 1, y, vec2<f32>(1.0, 0.0)), bankWeight(cell, x + 1, y, vec2<f32>(-1.0, 0.0)),
    bankWeight(cell, x, y - 1, vec2<f32>(0.0, 1.0)), bankWeight(cell, x, y + 1, vec2<f32>(0.0, -1.0)));
  let sum = dot(weights, vec4<f32>(1.0));
  if (sum <= 1e-10) { return 0.0; }
  let erodability = resolveMaterialHardness(i, cell.terrain, cell.hardness);
  let response = 1.0 - exp(-params.banks.x * erodability * timeStep());
  let amount = min(materialCutLimit(i, cell.terrain), min(max(params.thermal0.x, 0.0), sum * response));
  thermalPipeA.values[offset] = weights * (amount / sum);
  return amount;
}

fn neighborTerrain(x: i32, y: i32, center: f32) -> f32 {
  let cell = readStateClamped(x, y);
  return select(center, finiteOr(cell.terrain, center), cell.mask > 0.5);
}

fn terrainNormal(x: i32, y: i32) -> vec3<f32> {
  let center = finiteOr(readStateClamped(x, y).terrain, 0.0);
  let tl = neighborTerrain(x - 1, y - 1, center);
  let tc = neighborTerrain(x, y - 1, center);
  let tr = neighborTerrain(x + 1, y - 1, center);
  let ml = neighborTerrain(x - 1, y, center);
  let mr = neighborTerrain(x + 1, y, center);
  let bl = neighborTerrain(x - 1, y + 1, center);
  let bc = neighborTerrain(x, y + 1, center);
  let br = neighborTerrain(x + 1, y + 1, center);
  let inv = 1.0 / max(32.0 * cellSize(), 1e-6);
  let dzdx = ((3.0 * tr + 10.0 * mr + 3.0 * br) - (3.0 * tl + 10.0 * ml + 3.0 * bl)) * inv;
  let dzdy = ((3.0 * bl + 10.0 * bc + 3.0 * br) - (3.0 * tl + 10.0 * tc + 3.0 * tr)) * inv;
  return normalize(vec3<f32>(-dzdx, 1.0, -dzdy));
}

fn terrainSinAlpha(x: i32, y: i32) -> f32 {
  let n = terrainNormal(x, y);
  return length(n.xz);
}

fn depthLimiter(water: f32) -> f32 {
  let limit = max(params.thermal0.x, 1e-6);
  if (water <= 0.0) { return 0.0; }
  if (water >= limit) { return 1.0; }
  return 1.0 - (limit - water) / limit;
}

fn sourceHash(seed: u32) -> u32 {
  var h = seed;
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  return h ^ (h >> 16u);
}
fn springHash(seed: u32) -> vec2<f32> {
  return vec2<f32>(f32(sourceHash(seed) & 0xffffffu), f32(sourceHash(seed ^ 0x9e3779b9u) & 0xffffffu)) / 16777216.0;
}

fn rainAmountAt(x: u32, y: u32) -> f32 {
  var rate = max(params.hydro0.x, 0.0);
  if (params.source1.x > 0.5) { rate += max(paintedSourceState.values[idx(x, y)], 0.0); }
  return rate;
}

fn availableWater(x: u32, y: u32) -> f32 {
  // Integrate sources and evaporation for half a step before transport.
  return max(srcState.cells[idx(x, y)].water * params.integration.y + params.integration.x * rainAmountAt(x, y), 0.0);
}

// Rebuild only when spring settings or the grid change. Painted sources reuse
// this buffer directly; CPU paint data is retained when switching layouts.
@compute @workgroup_size(8, 8)
fn sourceMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }
  let pos = vec2<f32>(gid.xy);
  let radius = max(params.source0.z, 1.0);
  let count = clamp(u32(max(params.render0.y, 1.0)), 1u, 16u);
  let perSpring = max(params.source0.w, 0.0) / f32(count);
  var rate = 0.0;
  for (var j = 0u; j < count; j++) {
    let uv = springHash(params.sourceSeed * 37u + j * 17u + 1u);
    let center = uv * vec2<f32>(f32(gridWidth() - 1u), f32(gridHeight() - 1u));
    let falloff = max(1.0 - distance(pos, center) / radius, 0.0);
    rate += perSpring * falloff * falloff;
  }
  paintedSourceState.values[idx(gid.x, gid.y)] = rate;
}

// Sediment is a solid thickness per cell. Each face exports the same
// fraction of the donor's sediment as its available water volume.
fn springSedimentRate(x: u32, y: u32) -> f32 {
  return select(0.0, max(paintedSourceState.values[idx(x, y)], 0.0) * params.banks.z, params.source1.x > 0.5);
}
fn availableSediment(x: u32, y: u32, sediment: f32) -> f32 {
  return max(sediment, 0.0) + params.integration.w * springSedimentRate(x, y);
}
fn sedimentFlux(x: i32, y: i32) -> vec4<f32> {
  if (!inBounds(x, y)) { return vec4<f32>(0.0); }
  let i = idx(u32(x), u32(y));
  let cell = srcState.cells[i];
  if (cell.mask < 0.5) { return vec4<f32>(0.0); }
  let wetDepth = availableWater(u32(x), u32(y));
  if (wetDepth <= 1e-8) { return vec4<f32>(0.0); }
  return fluxState.values[i] * (timeStep() / cellArea()) * (availableSediment(u32(x), u32(y), cell.sediment) / wetDepth);
}

fn thermalDistanceScale(dx: i32, dy: i32) -> f32 {
  if (dx == 0 || dy == 0) {
    return 1.0;
  }
  return 1.41421356;
}

@compute @workgroup_size(8, 8)
fn fluxMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }

  let i = idx(gid.x, gid.y);
  let cell = srcState.cells[i];
  if (cell.mask < 0.5) {
    fluxState.values[i] = vec4<f32>(0.0);
    velocityState.values[i] = vec4<f32>(0.0);
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let dt = timeStep();
  let rainWater = availableWater(gid.x, gid.y);
  let centerTotal = cell.terrain + rainWater;

  let leftCell = readStateClamped(x - 1, y);
  let rightCell = readStateClamped(x + 1, y);
  let topCell = readStateClamped(x, y - 1);
  let bottomCell = readStateClamped(x, y + 1);

  let edgeDrain = max(params.render0.w, 0.0);
  let retainedEdgeWater = max(params.edgeWaterFloor, 0.0);
  let outsideTotal = select(centerTotal, cell.terrain + retainedEdgeWater - edgeDrain, rainWater > retainedEdgeWater + 1e-6);
  let leftTotal = select(outsideTotal, leftCell.terrain + availableWater(u32(max(x - 1, 0)), gid.y), inBounds(x - 1, y));
  let rightTotal = select(outsideTotal, rightCell.terrain + availableWater(u32(min(x + 1, i32(gridWidth()) - 1)), gid.y), inBounds(x + 1, y));
  let topTotal = select(outsideTotal, topCell.terrain + availableWater(gid.x, u32(max(y - 1, 0))), inBounds(x, y - 1));
  let bottomTotal = select(outsideTotal, bottomCell.terrain + availableWater(gid.x, u32(min(y + 1, i32(gridHeight()) - 1))), inBounds(x, y + 1));

  let oldFlux = fluxState.values[i] * exp(-max(params.material0.z, 0.0) * dt);
  let flowScale = dt * params.hydro0.z * params.hydro0.w / cellSize();
  var nextFlux = vec4<f32>(
    max(0.0, oldFlux.x + flowScale * (centerTotal - leftTotal)),
    max(0.0, oldFlux.y + flowScale * (centerTotal - rightTotal)),
    max(0.0, oldFlux.z + flowScale * (centerTotal - topTotal)),
    max(0.0, oldFlux.w + flowScale * (centerTotal - bottomTotal))
  );

  if (inBounds(x - 1, y) && leftCell.mask < 0.5) { nextFlux.x = 0.0; }
  if (inBounds(x + 1, y) && rightCell.mask < 0.5) { nextFlux.y = 0.0; }
  if (inBounds(x, y - 1) && topCell.mask < 0.5) { nextFlux.z = 0.0; }
  if (inBounds(x, y + 1) && bottomCell.mask < 0.5) { nextFlux.w = 0.0; }
  if (params.misc0.z > 0.5) {
    if (x == 0) { nextFlux.x = 0.0; }
    if (x == i32(gridWidth()) - 1) { nextFlux.y = 0.0; }
    if (y == 0) { nextFlux.z = 0.0; }
    if (y == i32(gridHeight()) - 1) { nextFlux.w = 0.0; }
  }
  let edges = u32(params.meander.w);
  if (x == 0 && (edges & 1u) == 0u) { nextFlux.x = 0.0; }
  if (x == i32(gridWidth()) - 1 && (edges & 2u) == 0u) { nextFlux.y = 0.0; }
  if (y == 0 && (edges & 4u) == 0u) { nextFlux.z = 0.0; }
  if (y == i32(gridHeight()) - 1 && (edges & 8u) == 0u) { nextFlux.w = 0.0; }

  nextFlux = vec4<f32>(
    finiteOr(max(0.0, nextFlux.x), 0.0),
    finiteOr(max(0.0, nextFlux.y), 0.0),
    finiteOr(max(0.0, nextFlux.z), 0.0),
    finiteOr(max(0.0, nextFlux.w), 0.0)
  );
  let sumOut = finiteOr(nextFlux.x + nextFlux.y + nextFlux.z + nextFlux.w, 0.0);
  let maxOut = finiteOr(rainWater * cellArea() / max(dt, 1e-6), 0.0) * clamp(params.material0.w, 0.05, 1.0);
  if (sumOut > maxOut && sumOut > 0.0) {
    nextFlux *= maxOut / sumOut;
  }

  let maxOutsideOut = max(rainWater - retainedEdgeWater, 0.0) * cellArea() / max(dt, 1e-6);
  let outsideFluxX = select(0.0, nextFlux.x, !inBounds(x - 1, y));
  let outsideFluxY = select(0.0, nextFlux.y, !inBounds(x + 1, y));
  let outsideFluxZ = select(0.0, nextFlux.z, !inBounds(x, y - 1));
  let outsideFluxW = select(0.0, nextFlux.w, !inBounds(x, y + 1));
  let sumOutsideOut = outsideFluxX + outsideFluxY + outsideFluxZ + outsideFluxW;
  if (sumOutsideOut > maxOutsideOut && sumOutsideOut > 0.0) {
    let outsideScale = maxOutsideOut / sumOutsideOut;
    if (!inBounds(x - 1, y)) { nextFlux.x *= outsideScale; }
    if (!inBounds(x + 1, y)) { nextFlux.y *= outsideScale; }
    if (!inBounds(x, y - 1)) { nextFlux.z *= outsideScale; }
    if (!inBounds(x, y + 1)) { nextFlux.w *= outsideScale; }
  }
  fluxState.values[i] = vec4<f32>(
    finiteOr(nextFlux.x, 0.0),
    finiteOr(nextFlux.y, 0.0),
    finiteOr(nextFlux.z, 0.0),
    finiteOr(nextFlux.w, 0.0)
  );
}

@compute @workgroup_size(8, 8)
fn flowMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }

  let i = idx(gid.x, gid.y);
  let cell = srcState.cells[i];
  if (cell.mask < 0.5) {
    dstState.cells[i] = CellState(0.0, 0.0, 0.0, cell.hardness, 0.0, 0.0, 0.0, 0.0);
    velocityState.values[i] = vec4<f32>(0.0);
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let dt = timeStep();
  let rainWater = availableWater(gid.x, gid.y);

  let localFlux = fluxState.values[i];
  let inflow =
    readFluxOrZero(x - 1, y).y +
    readFluxOrZero(x + 1, y).x +
    readFluxOrZero(x, y - 1).w +
    readFluxOrZero(x, y + 1).z;

  let outflow = localFlux.x + localFlux.y + localFlux.z + localFlux.w;
  let deltaVolume = dt * (inflow - outflow);
  let water = max(0.0, rainWater + deltaVolume / cellArea());

  let deltaWx = 0.5 * (
    readFluxOrZero(x - 1, y).y - localFlux.x +
    localFlux.y - readFluxOrZero(x + 1, y).x
  );
  let deltaWy = 0.5 * (
    readFluxOrZero(x, y - 1).w - localFlux.z +
    localFlux.w - readFluxOrZero(x, y + 1).z
  );
  let avgWater = max(0.5 * (rainWater + water), 1e-4);
  var vel = vec2<f32>(deltaWx, deltaWy) / max(cellSize() * avgWater, 1e-4);
  vel = vec2<f32>(finiteOr(vel.x, 0.0), finiteOr(vel.y, 0.0));
  var speed = finiteOr(length(vel), 0.0);
  let maxVelocity = max(params.misc0.w, 0.05);
  if (speed > maxVelocity && speed > 1e-6) {
    vel *= maxVelocity / speed;
    speed = maxVelocity;
  }
  velocityState.values[i] = vec4<f32>(finiteOr(vel.x, 0.0), finiteOr(vel.y, 0.0), finiteOr(speed, 0.0), clamp(dt * outflow / max(cellArea() * rainWater, 1e-8), 0.0, 1.0));

  let selfSedimentFlux = sedimentFlux(x, y);
  let sedimentOut = dot(selfSedimentFlux, vec4<f32>(1.0));
  let sedimentIn = sedimentFlux(x - 1, y).y + sedimentFlux(x + 1, y).x
    + sedimentFlux(x, y - 1).w + sedimentFlux(x, y + 1).z;
  let sediment = max(0.0, availableSediment(gid.x, gid.y, cell.sediment) - sedimentOut + sedimentIn);
  let activeHardness = resolveMaterialHardness(i, finiteOr(cell.terrain, 0.0), finiteOr(cell.hardness, 0.1));
  dstState.cells[i] = CellState(finiteOr(cell.terrain, 0.0), finiteOr(water, 0.0), finiteOr(sediment, 0.0), activeHardness, cell.mask, 0.0, 0.0, finiteOr(cell.aux2, 0.0));
}

// Exchange, evaporation, settling and thermal application share one state write.
@compute @workgroup_size(8, 8)
fn erosionMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }
  let i = idx(gid.x, gid.y);
  let cell = srcState.cells[i];
  if (cell.mask < 0.5) { dstState.cells[i] = cell; return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let dt = timeStep();
  let velocity = velocityState.values[i];
  let speed = max(finiteOr(velocity.z, 0.0), 0.0);
  var terrain = finiteOr(cell.terrain, 0.0);
  var sediment = max(finiteOr(cell.sediment, 0.0), 0.0) + params.banks.w * springSedimentRate(gid.x, gid.y);
  // Finish the source/evaporation split after transport. Sediment is unaffected.
  var water = max(finiteOr(cell.water, 0.0), 0.0) * params.integration.y
    + params.integration.z * rainAmountAt(gid.x, gid.y);
  var erodability = resolveMaterialHardness(i, terrain, cell.hardness);
  var history = finiteOr(cell.aux2, 0.0) * pow(clamp(params.misc0.x, 0.0, 1.0), dt / 0.02);
  var selfOut = 0.0;
  var incoming = 0.0;
  var bankOut = 0.0;
  if (bendSedimentEnabled()) {
    sediment = max(sediment - dot(readBendSediment(x, y), vec4<f32>(1.0)), 0.0)
      + readBendSediment(x - 1, y).y + readBendSediment(x + 1, y).x
      + readBendSediment(x, y - 1).w + readBendSediment(x, y + 1).z;
  }
  if (bankTransfersEnabled()) {
    bankOut = dot(readBankOrZero(x, y), vec4<f32>(1.0));
    sediment += readBankOrZero(x - 1, y).y + readBankOrZero(x + 1, y).x
      + readBankOrZero(x, y - 1).w + readBankOrZero(x, y + 1).z;
    history -= bankOut * 240.0;
  }
  if (params.thermal0.y > 0.0) {
    selfOut = dot(thermalPipeA.values[i] + thermalPipeB.values[i], vec4<f32>(1.0));
    incoming = readThermalAOrZero(x - 1, y).y + readThermalAOrZero(x + 1, y).x
      + readThermalAOrZero(x, y - 1).w + readThermalAOrZero(x, y + 1).z
      + readThermalBOrZero(x - 1, y - 1).w + readThermalBOrZero(x + 1, y - 1).z
      + readThermalBOrZero(x - 1, y + 1).y + readThermalBOrZero(x + 1, y + 1).x;
  }
  var capacity = 0.0;
  if (hydraulicErosionEnabled()) {
    // A tangent/normal collision dot product is identically zero. Use the
    // actual terrain grade and require outgoing water before entrainment.
    let slope = terrainSinAlpha(x, y);
    let movingWater = select(0.0, 1.0, velocity.w > 1e-6 && water > 1e-6);
    capacity = max(params.hydro1.x, 0.0) * slope * speed * depthLimiter(water) * movingWater;
    if (capacity > sediment) {
      // Recent deposition is a proxy for loose soil above the active material.
      let looseSoil = clamp(erodability * (1.0 + max(cell.aux2, 0.0) * max(params.hydro1.w, 0.0)), 0.0, 1.0);
      let response = 1.0 - exp(-max(params.hydro1.y, 0.0) * looseSoil * dt);
      var downhillFloor = terrain;
      let left = readStateClamped(x - 1, y);
      let right = readStateClamped(x + 1, y);
      let top = readStateClamped(x, y - 1);
      let bottom = readStateClamped(x, y + 1);
      if (left.mask > 0.5) { downhillFloor = min(downhillFloor, left.terrain); }
      if (right.mask > 0.5) { downhillFloor = min(downhillFloor, right.terrain); }
      if (top.mask > 0.5) { downhillFloor = min(downhillFloor, top.terrain); }
      if (bottom.mask > 0.5) { downhillFloor = min(downhillFloor, bottom.terrain); }
      var cutLimit = max(terrain - downhillFloor, 0.0) * 0.5;
      // An open edge has an outlet beyond the sampled height field.
      if (params.misc0.z < 0.5 && (x == 0 || y == 0 || x == i32(gridWidth()) - 1 || y == i32(gridHeight()) - 1)) {
        cutLimit = max(params.thermal0.x, 0.0);
      }
      // Reserve the soil already committed to talus transfer and limit each
      // cut by water depth and the user-selected maximum cut depth.
      let eroded = min((capacity - sediment) * response,
        min(max(materialCutLimit(i, terrain) - selfOut - bankOut, 0.0), min(cutLimit, min(water, max(params.thermal0.x, 0.0)))));
      terrain -= eroded;
      sediment += eroded;
      history -= eroded * 240.0;
    } else {
      let deposited = (sediment - capacity) * (1.0 - exp(-max(params.hydro1.z, 0.0) * dt));
      terrain += deposited;
      sediment -= deposited;
      history += deposited * 180.0;
    }
  }
  // A dry cell cannot carry suspended solid, even with hydraulic erosion off.
  if (water <= 1e-6) {
    terrain += sediment;
    history += sediment * 180.0;
    sediment = 0.0;
  }
  terrain = terrain - selfOut - bankOut + incoming;
  erodability = resolveMaterialHardness(i, terrain, erodability);
  // No upper clamps: clipping terrain, water or sediment destroys mass.
  dstState.cells[i] = CellState(finiteOr(terrain, cell.terrain), finiteOr(water, 0.0),
    finiteOr(sediment, 0.0), erodability, cell.mask, selfOut, incoming, clamp(history, -1.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn thermalOutflowMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }

  let i = idx(gid.x, gid.y);
  let cell = srcState.cells[i];
  let bankOut = writeBankTransfers(i, i32(gid.x), i32(gid.y), cell);
  writeBendSediment(i, i32(gid.x), i32(gid.y), cell);
  if (cell.mask < 0.5) {
    thermalPipeA.values[i] = vec4<f32>(0.0);
    thermalPipeB.values[i] = vec4<f32>(0.0);
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let activeHardness = resolveMaterialHardness(i, finiteOr(cell.terrain, 0.0), finiteOr(cell.hardness, params.misc0.y));
  let thermalAllowed = resolveMaterialThermalEnabled(i, finiteOr(cell.terrain, 0.0));
  if (!thermalAllowed || params.thermal0.y <= 1e-6) {
    thermalPipeA.values[i] = vec4<f32>(0.0);
    thermalPipeB.values[i] = vec4<f32>(0.0);
    return;
  }
  let softness = clamp(activeHardness, 0.0, 1.0);
  let thresholdBase = cellSize() * max(0.0, params.thermal0.w + params.thermal0.z * (1.0 - softness));

  var rawCard = vec4<f32>(0.0);
  var rawDiag = vec4<f32>(0.0);
  var weightSum = 0.0;
  var maxExcess = 0.0;

  let left = readStateClamped(x - 1, y);
  let dropL = cell.terrain - left.terrain;
  let thrL = thresholdBase * thermalDistanceScale(-1, 0);
  let excessL = max(0.0, dropL - thrL);
  if (inBounds(x - 1, y) && left.mask > 0.5 && excessL > 0.0) { rawCard.x = excessL; weightSum += excessL; maxExcess = max(maxExcess, excessL); }

  let right = readStateClamped(x + 1, y);
  let dropR = cell.terrain - right.terrain;
  let thrR = thresholdBase * thermalDistanceScale(1, 0);
  let excessR = max(0.0, dropR - thrR);
  if (inBounds(x + 1, y) && right.mask > 0.5 && excessR > 0.0) { rawCard.y = excessR; weightSum += excessR; maxExcess = max(maxExcess, excessR); }

  let top = readStateClamped(x, y - 1);
  let dropT = cell.terrain - top.terrain;
  let thrT = thresholdBase * thermalDistanceScale(0, -1);
  let excessT = max(0.0, dropT - thrT);
  if (inBounds(x, y - 1) && top.mask > 0.5 && excessT > 0.0) { rawCard.z = excessT; weightSum += excessT; maxExcess = max(maxExcess, excessT); }

  let bottom = readStateClamped(x, y + 1);
  let dropB = cell.terrain - bottom.terrain;
  let thrB = thresholdBase * thermalDistanceScale(0, 1);
  let excessB = max(0.0, dropB - thrB);
  if (inBounds(x, y + 1) && bottom.mask > 0.5 && excessB > 0.0) { rawCard.w = excessB; weightSum += excessB; maxExcess = max(maxExcess, excessB); }

  let tl = readStateClamped(x - 1, y - 1);
  let dropTL = cell.terrain - tl.terrain;
  let thrTL = thresholdBase * thermalDistanceScale(-1, -1);
  let excessTL = max(0.0, dropTL - thrTL);
  if (inBounds(x - 1, y - 1) && tl.mask > 0.5 && excessTL > 0.0) { rawDiag.x = excessTL; weightSum += excessTL; maxExcess = max(maxExcess, excessTL); }

  let tr = readStateClamped(x + 1, y - 1);
  let dropTR = cell.terrain - tr.terrain;
  let thrTR = thresholdBase * thermalDistanceScale(1, -1);
  let excessTR = max(0.0, dropTR - thrTR);
  if (inBounds(x + 1, y - 1) && tr.mask > 0.5 && excessTR > 0.0) { rawDiag.y = excessTR; weightSum += excessTR; maxExcess = max(maxExcess, excessTR); }

  let bl = readStateClamped(x - 1, y + 1);
  let dropBL = cell.terrain - bl.terrain;
  let thrBL = thresholdBase * thermalDistanceScale(-1, 1);
  let excessBL = max(0.0, dropBL - thrBL);
  if (inBounds(x - 1, y + 1) && bl.mask > 0.5 && excessBL > 0.0) { rawDiag.z = excessBL; weightSum += excessBL; maxExcess = max(maxExcess, excessBL); }

  let br = readStateClamped(x + 1, y + 1);
  let dropBR = cell.terrain - br.terrain;
  let thrBR = thresholdBase * thermalDistanceScale(1, 1);
  let excessBR = max(0.0, dropBR - thrBR);
  if (inBounds(x + 1, y + 1) && br.mask > 0.5 && excessBR > 0.0) { rawDiag.w = excessBR; weightSum += excessBR; maxExcess = max(maxExcess, excessBR); }

  if (weightSum <= 1e-6 || maxExcess <= 1e-6) {
    thermalPipeA.values[i] = vec4<f32>(0.0);
    thermalPipeB.values[i] = vec4<f32>(0.0);
    return;
  }

  let response = 1.0 - exp(-max(params.thermal0.y, 0.0) * softness * timeStep());
  let totalOut = min(max(materialCutLimit(i, cell.terrain) - bankOut, 0.0), 0.5 * maxExcess * response);
  thermalPipeA.values[i] = rawCard * (totalOut / weightSum);
  thermalPipeB.values[i] = rawDiag * (totalOut / weightSum);
}
`,Qe=`
@group(0) @binding(0) var sceneSampler: sampler;
@group(0) @binding(1) var sceneTexture: texture_2d<f32>;

struct CompositeOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vsComposite(@builtin(vertex_index) vertexIndex: u32) -> CompositeOut {
  var out: CompositeOut;
  let pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(3.0, 1.0)
  );
  let uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 2.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(2.0, 0.0)
  );
  out.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
  out.uv = uv[vertexIndex];
  return out;
}

@fragment
fn fsComposite(in: CompositeOut) -> @location(0) vec4<f32> {
  return textureSampleLevel(sceneTexture, sceneSampler, clamp(in.uv, vec2<f32>(0.001), vec2<f32>(0.999)), 0.0);
}
`;function Fe(){return Ue}var Ue=`
struct RenderParams {
  viewProj: mat4x4<f32>,
  dims: vec4<f32>,
  shading: vec4<f32>,
  misc: vec4<f32>,
  lightDir: vec4<f32>,
  cameraPos: vec4<f32>,
  timeData: vec4<f32>,
}

struct CellState {
  terrain: f32,
  water: f32,
  sediment: f32,
  hardness: f32,
  mask: f32,
  aux0: f32,
  aux1: f32,
  aux2: f32,
}

struct StateBuffer {
  cells: array<CellState>,
}

fn finiteOr(value: f32, fallback: f32) -> f32 {
  if (abs(value) < 1e30) { return value; }
  return fallback;
}

struct RenderVertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) terrain: f32,
  @location(2) water: f32,
  @location(3) sediment: f32,
  @location(4) hardness: f32,
  @location(5) thermal: f32,
  @location(6) history: f32,
  @location(7) mask: f32,
  @location(8) worldPos: vec3<f32>,
  @location(9) gridPos: vec2<f32>,
  @location(10) materialColor: vec3<f32>,
  @location(11) cavity: f32,
  @location(12) flow: vec3<f32>,
}

@group(0) @binding(0) var<uniform> renderParams: RenderParams;
@group(0) @binding(1) var<storage, read> renderState: StateBuffer;
@group(0) @binding(2) var sceneSampler: sampler;
@group(0) @binding(3) var sceneTexture: texture_2d<f32>;
struct LayerTops { values: array<vec4<f32>> }
struct LayerMaterials { layers: array<vec4<f32>, 4> }
@group(0) @binding(4) var<storage, read> renderLayerTops: LayerTops;
@group(0) @binding(5) var<uniform> renderLayerMaterials: LayerMaterials;
struct Velocities { values: array<vec4<f32>> }
@group(0) @binding(6) var<storage, read> renderVelocity: Velocities;

fn exposedMaterialColor(i: u32, terrain: f32) -> vec3<f32> {
  if (renderParams.cameraPos.w < 0.5) { return vec3<f32>(0.51, 0.45, 0.34); }
  let tops = renderLayerTops.values[i];
  var chosen = 0u;
  var bestAbove = 1e30;
  var highest = -1e30;
  var highestIndex = 0u;
  var found = false;
  for (var layer = 0u; layer < 4u; layer++) {
    if (renderLayerMaterials.layers[layer].z < 0.5 || tops[layer] < -1e8) { continue; }
    if (tops[layer] >= highest) { highest = tops[layer]; highestIndex = layer; }
    if (tops[layer] >= terrain && tops[layer] <= bestAbove) {
      bestAbove = tops[layer]; chosen = layer; found = true;
    }
  }
  if (!found) { chosen = highestIndex; }
  switch chosen {
    case 0u: { return vec3<f32>(0.61, 0.54, 0.40); }
    case 1u: { return vec3<f32>(0.52, 0.40, 0.28); }
    case 2u: { return vec3<f32>(0.43, 0.44, 0.40); }
    default: { return vec3<f32>(0.29, 0.31, 0.30); }
  }
}

fn idx(x: u32, y: u32) -> u32 {
  return y * u32(renderParams.dims.x) + x;
}

fn renderStateAt(x: i32, y: i32) -> CellState {
  let maxX = max(i32(renderParams.dims.x) - 1, 0);
  let maxY = max(i32(renderParams.dims.y) - 1, 0);
  let cx = clamp(x, 0, maxX);
  let cy = clamp(y, 0, maxY);
  return renderState.cells[idx(u32(cx), u32(cy))];
}

fn renderTerrainHeightAt(x: i32, y: i32) -> f32 {
  return max(finiteOr(renderStateAt(x, y).terrain, 0.0), 0.0) * renderParams.dims.w;
}

fn renderWaterDepthAt(x: i32, y: i32) -> f32 {
  return max(finiteOr(renderStateAt(x, y).water, 0.0), 0.0) * renderParams.misc.y;
}

fn renderRelief(x: i32, y: i32, centerHeight: f32) -> vec4<f32> {
  let worldScale = max(renderParams.misc.x, 1e-5);
  let tl = renderTerrainHeightAt(x - 1, y - 1);
  let tc = renderTerrainHeightAt(x, y - 1);
  let tr = renderTerrainHeightAt(x + 1, y - 1);
  let ml = renderTerrainHeightAt(x - 1, y);
  let mr = renderTerrainHeightAt(x + 1, y);
  let bl = renderTerrainHeightAt(x - 1, y + 1);
  let bc = renderTerrainHeightAt(x, y + 1);
  let br = renderTerrainHeightAt(x + 1, y + 1);
  let inv = 1.0 / max(32.0 * worldScale, 1e-5);
  let dzdx = ((3.0 * tr + 10.0 * mr + 3.0 * br) - (3.0 * tl + 10.0 * ml + 3.0 * bl)) * inv;
  let dzdz = ((3.0 * tl + 10.0 * tc + 3.0 * tr) - (3.0 * bl + 10.0 * bc + 3.0 * br)) * inv;
  let cavity = clamp((ml + mr + tc + bc - 4.0 * centerHeight) / worldScale * 1.8, -1.0, 1.0);
  return vec4<f32>(normalize(vec3<f32>(-dzdx, 1.0, -dzdz)), cavity);
}

// Display the solver's water surface. Smoothing depth across unequal terrain
// creates artificial waves in a resting lake and spills water onto dry cells.
fn renderWaterSurfaceAt(x: i32, y: i32) -> f32 {
  return renderTerrainHeightAt(x, y) + renderWaterDepthAt(x, y);
}

fn cornerCoord(localVertex: u32) -> vec2<u32> {
  switch localVertex {
    case 0u: { return vec2<u32>(0u, 0u); }
    case 1u: { return vec2<u32>(1u, 0u); }
    case 2u: { return vec2<u32>(0u, 1u); }
    case 3u: { return vec2<u32>(0u, 1u); }
    case 4u: { return vec2<u32>(1u, 0u); }
    default: { return vec2<u32>(1u, 1u); }
  }
}

@vertex
fn vsMesh(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> RenderVertexOut {
  var out: RenderVertexOut;
  let width = u32(renderParams.dims.x);
  let height = u32(renderParams.dims.y);
  let meshStep = max(u32(renderParams.lightDir.w), 1u);
  let cellsWide = max((width - 1u + meshStep - 1u) / meshStep, 1u);
  let cellX = (instanceIndex % cellsWide) * meshStep;
  let cellY = (instanceIndex / cellsWide) * meshStep;
  if (cellY >= max(height - 1u, 1u)) {
    out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
    out.normal = vec3<f32>(0.0, 1.0, 0.0);
    out.terrain = 0.0;
    out.water = 0.0;
    out.sediment = 0.0;
    out.hardness = 0.0;
    out.thermal = 0.0;
    out.history = 0.0;
    out.mask = 0.0;
    out.worldPos = vec3<f32>(0.0);
    out.gridPos = vec2<f32>(0.0);
    out.materialColor = vec3<f32>(0.0);
    out.cavity = 0.0;
    return out;
  }

  let corner = cornerCoord(vertexIndex % 6u);
  let gx = min(cellX + corner.x * meshStep, width - 1u);
  let gy = min(cellY + corner.y * meshStep, height - 1u);
  let cell = renderState.cells[idx(gx, gy)];
  let halfW = (renderParams.dims.x - 1.0) * 0.5;
  let halfH = (renderParams.dims.y - 1.0) * 0.5;
  let worldScale = renderParams.misc.x;
  let worldX = (f32(gx) - halfW) * worldScale;
  let worldZ = (halfH - f32(gy)) * worldScale;
  let terrainY = renderTerrainHeightAt(i32(gx), i32(gy));
  let waterSurfaceY = renderWaterSurfaceAt(i32(gx), i32(gy));
  let waterY = max(waterSurfaceY - terrainY, 0.0);
  let waterPass = renderParams.misc.z > 0.5;
  let worldY = select(terrainY, waterSurfaceY, waterPass);

  out.position = renderParams.viewProj * vec4<f32>(worldX, worldY, worldZ, 1.0);
  out.normal = vec3<f32>(0.0, 1.0, 0.0);
  out.cavity = 0.0;
  out.materialColor = vec3<f32>(0.51, 0.45, 0.34);
  if (!waterPass) {
    let relief = renderRelief(i32(gx), i32(gy), terrainY);
    out.normal = relief.xyz;
    out.cavity = relief.w;
    out.materialColor = exposedMaterialColor(idx(gx, gy), cell.terrain);
  }
  out.terrain = finiteOr(cell.terrain, 0.0);
  out.water = finiteOr(waterY / max(renderParams.misc.y, 1e-6), 0.0);
  out.sediment = finiteOr(cell.sediment, 0.0);
  out.hardness = finiteOr(cell.hardness, 0.1);
  out.thermal = clamp(pow(max(finiteOr((cell.aux0 + cell.aux1) * renderParams.shading.w, 0.0), 0.0), 0.45), 0.0, 1.0);
  out.history = clamp(finiteOr(cell.aux2, 0.0), -1.0, 1.0);
  out.mask = cell.mask;
  out.worldPos = vec3<f32>(worldX, worldY, worldZ);
  out.gridPos = vec2<f32>(f32(gx), f32(gy));
  out.flow = vec3<f32>(0.0);
  if (waterPass && renderParams.shading.z == 9.0) {
    let v = renderVelocity.values[idx(gx, gy)];
    out.flow = vec3<f32>(v.xy, v.w);
  }
  return out;
}

@fragment
fn fsMesh(in: RenderVertexOut) -> @location(0) vec4<f32> {
  // Derivatives of the displayed surface avoid repeated per-pixel storage
  // reads and keep reflections consistent with the actual water geometry.
  let faceNormal = cross(dpdx(in.worldPos), dpdy(in.worldPos));
  let faceLength = max(length(faceNormal), 1e-12);
  let waterNormal = faceNormal / faceLength * select(-1.0, 1.0, faceNormal.y >= 0.0);
  if (in.mask < 0.5) {
    return vec4<f32>(0.03, 0.04, 0.07, 1.0);
  }

  let waterPass = renderParams.misc.z > 0.5;
  let terrainNormal = normalize(mix(in.normal, waterNormal, 0.2));
  let lambert = clamp(dot(terrainNormal, normalize(renderParams.lightDir.xyz)), 0.0, 1.0);
  let hemi = 0.80 + 0.20 * clamp(terrainNormal.y * 0.5 + 0.5, 0.0, 1.0);

  let h = clamp(in.terrain, 0.0, 1.0);
  var topo = mix(vec3<f32>(0.05, 0.18, 0.50), vec3<f32>(0.10, 0.68, 0.82), smoothstep(0.00, 0.20, h));
  topo = mix(topo, vec3<f32>(0.18, 0.74, 0.34), smoothstep(0.16, 0.36, h));
  topo = mix(topo, vec3<f32>(0.88, 0.84, 0.20), smoothstep(0.34, 0.58, h));
  topo = mix(topo, vec3<f32>(0.90, 0.62, 0.18), smoothstep(0.56, 0.76, h));
  topo = mix(topo, vec3<f32>(0.95, 0.94, 0.90), smoothstep(0.78, 1.00, h));
  let litTopo = topo * (0.72 + 0.20 * lambert + 0.16 * hemi);
  let contour = 0.92 + 0.08 * abs(sin(h * 44.0));
  let terrainColor = clamp(litTopo * contour, vec3<f32>(0.0), vec3<f32>(1.0));

  let shorelineFade = smoothstep(0.0002, 0.004, in.water);
  // Depth-dependent absorption keeps shallow rivers visible while leaving
  // thin wet films and dry shore cells transparent.
  let waterAlpha = clamp((1.0 - exp(-in.water * max(renderParams.shading.x, 0.0) * 96.0)) * shorelineFade, 0.0, 0.97);
  let mode = i32(round(renderParams.shading.z));

  if (waterPass) {
    if (mode == 9) {
      let wet = smoothstep(0.0002, 0.0015, in.water);
      let speed = length(in.flow.xy);
      let direction = in.flow.xy / max(speed, 1e-6);
      let tile = fract(in.gridPos / 6.0) - 0.5;
      let along = dot(tile, direction);
      let across = abs(dot(tile, vec2<f32>(-direction.y, direction.x)));
      let stem = select(0.0, 1.0, along > -0.28 && along < 0.13 && across < 0.028);
      let head = select(0.0, 1.0, along > 0.02 && along < 0.29 && across < (0.29 - along) * 0.7);
      let arrow = max(stem, head) * smoothstep(0.02, 0.15, speed) * smoothstep(1e-5, 0.001, in.flow.z);
      let flowColor = mix(vec3<f32>(0.025, 0.22, 0.34), vec3<f32>(0.24, 0.60, 0.72), clamp(speed * 0.25, 0.0, 1.0));
      return vec4<f32>(mix(flowColor, vec3<f32>(0.88, 0.99, 1.0), arrow), wet * 0.88);
    }
    if (waterAlpha <= max(renderParams.misc.w, 0.01)) {
      return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let w = clamp(in.water * 2.2, 0.0, 1.0);
    let s = clamp(in.sediment * renderParams.shading.y * 2.7, 0.0, 1.0);
    let viewDir = normalize(renderParams.cameraPos.xyz - in.worldPos);
    let lightDir = normalize(renderParams.lightDir.xyz);
    let phase = renderParams.timeData.x;
    let ripple = vec2<f32>(sin(in.worldPos.x * 95.0 + phase * 1.7), cos(in.worldPos.z * 87.0 - phase * 1.3)) * 0.008;
    let combinedNormal = normalize(waterNormal + vec3<f32>(ripple.x, 0.0, ripple.y));
    let ndv = clamp(dot(combinedNormal, viewDir), 0.0, 1.0);
    let fresnel = pow(1.0 - ndv, 3.8);
    let sunFacing = clamp(dot(combinedNormal, lightDir), 0.0, 1.0);
    let halfDir = normalize(lightDir + viewDir);
    let sunSpecTight = pow(clamp(dot(combinedNormal, halfDir), 0.0, 1.0), 54.0);
    let sunSpecBroad = pow(clamp(dot(combinedNormal, halfDir), 0.0, 1.0), 16.0);
    let rippleGlow = pow(clamp(1.0 - combinedNormal.y, 0.0, 1.0), 0.75);

    if (mode == 3) {
      let deepTint = mix(vec3<f32>(0.002, 0.008, 0.016), vec3<f32>(0.006, 0.024, 0.040), w);
      let shallowTint = mix(vec3<f32>(0.008, 0.030, 0.050), vec3<f32>(0.018, 0.060, 0.080), w);
      let mutedWater = mix(deepTint, shallowTint, clamp(0.22 + 0.48 * w, 0.0, 1.0));
      let suspendedSedimentGlow = mix(vec3<f32>(0.16, 0.12, 0.05), vec3<f32>(0.52, 0.36, 0.10), s);
      var waterColorOnly = mutedWater * (0.82 + 0.06 * sunFacing);
      waterColorOnly += suspendedSedimentGlow * (0.05 + 0.12 * s);
      waterColorOnly += vec3<f32>(0.000, 0.010, 0.018) * fresnel;
      let finalAlpha = clamp(waterAlpha * (0.18 + 0.14 * w), 0.0, 0.32);
      return vec4<f32>(clamp(waterColorOnly, vec3<f32>(0.0), vec3<f32>(1.0)), finalAlpha);
    }

    let deepTint = mix(vec3<f32>(0.003, 0.018, 0.060), vec3<f32>(0.008, 0.050, 0.135), w);
    let shallowTint = mix(vec3<f32>(0.015, 0.120, 0.220), vec3<f32>(0.045, 0.280, 0.380), w);
    let waterBase = mix(deepTint, shallowTint, clamp(0.20 + 0.58 * w, 0.0, 1.0));
    var waterColorOnly = waterBase * (0.78 + 0.12 * sunFacing);
    waterColorOnly += vec3<f32>(0.010, 0.095, 0.145) * (0.24 + 0.76 * w);
    waterColorOnly += vec3<f32>(0.020, 0.220, 0.280) * rippleGlow * (0.18 + 0.42 * w);
    waterColorOnly += vec3<f32>(1.30, 1.18, 0.96) * sunSpecTight * (0.10 + 0.32 * w);
    waterColorOnly += vec3<f32>(0.18, 0.16, 0.13) * sunSpecBroad * (0.06 + 0.12 * w);
    waterColorOnly += vec3<f32>(0.000, 0.028, 0.050) * fresnel;
    let finalAlpha = clamp(waterAlpha * (0.95 + 0.05 * fresnel), 0.0, 0.97);
    return vec4<f32>(clamp(waterColorOnly, vec3<f32>(0.0), vec3<f32>(1.0)), finalAlpha);
  }

  if (mode == 0) {
    return vec4<f32>(terrainColor, 1.0);
  }
  if (mode == 2) {
    let w = clamp(in.water * 1.8, 0.0, 1.0);
    let abyss = mix(vec3<f32>(0.004, 0.012, 0.026), vec3<f32>(0.010, 0.030, 0.052), h);
    let shelf = mix(vec3<f32>(0.018, 0.040, 0.060), vec3<f32>(0.050, 0.100, 0.110), smoothstep(0.02, 0.32, h));
    let seabed = mix(abyss, shelf, smoothstep(0.0, 0.65, w));
    let wetSediment = mix(vec3<f32>(0.08, 0.22, 0.18), vec3<f32>(0.24, 0.52, 0.30), clamp(in.sediment * 2.2, 0.0, 1.0));
    let seabedLit = mix(seabed, wetSediment, clamp(in.sediment * 0.22 + (1.0 - w) * 0.08, 0.0, 0.32));
    return vec4<f32>(clamp(seabedLit * (0.78 + 0.10 * lambert + 0.12 * hemi), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
  }
  if (mode == 3) {
    let s = clamp(in.sediment * renderParams.shading.y * 2.7, 0.0, 1.0);
    let matteLambert = 0.80 + 0.18 * lambert + 0.14 * hemi;
    let sedimentBase = mix(vec3<f32>(0.18, 0.14, 0.10), vec3<f32>(1.00, 0.86, 0.46), s);
    var sedimentColor = sedimentBase * matteLambert;
    sedimentColor = mix(sedimentColor, vec3<f32>(1.00, 0.94, 0.70), s * 0.42);
    sedimentColor += vec3<f32>(0.04, 0.06, 0.03) * pow(s, 0.72);
    return vec4<f32>(clamp(sedimentColor, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
  }
  if (mode == 4) {
    return vec4<f32>(mix(vec3<f32>(0.25, 0.00, 0.45), vec3<f32>(0.95, 0.95, 1.00), clamp(in.hardness, 0.0, 1.0)), 1.0);
  }
  if (mode == 5) {
    return vec4<f32>(mix(vec3<f32>(0.08, 0.05, 0.18), vec3<f32>(1.00, 0.88, 0.05), in.thermal), 1.0);
  }
  if (mode == 6) {
    let signedHistory = clamp(in.history, -1.0, 1.0);
    let mag = pow(abs(signedHistory), 0.55);
    let erosionColor = vec3<f32>(1.00, 0.08, 0.05);
    let depositionColor = vec3<f32>(0.08, 0.95, 0.20);
    let diagColor = select(mix(terrainColor * 0.18, depositionColor, mag), mix(terrainColor * 0.18, erosionColor, mag), signedHistory < 0.0);
    return vec4<f32>(diagColor, 1.0);
  }
  let slope = length(terrainNormal.xz);
  let steepness = smoothstep(0.1, 0.88, slope);
  let reliefLight = (0.24 + 0.64 * lambert + 0.12 * hemi) * (1.0 - 0.2 * max(in.cavity, 0.0) + 0.05 * max(-in.cavity, 0.0));
  if (mode == 8 || mode == 9) {
    let slopeColor = mix(mix(vec3<f32>(0.38, 0.47, 0.29), vec3<f32>(0.72, 0.57, 0.34), smoothstep(0.05, 0.55, slope)), vec3<f32>(0.45, 0.40, 0.37), smoothstep(0.55, 0.93, slope));
    return vec4<f32>(slopeColor * reliefLight, 1.0);
  }
  if (mode == 7) {
    let erosion = smoothstep(0.03, 0.85, max(-in.history, 0.0));
    let deposition = smoothstep(0.03, 0.85, max(in.history, 0.0));
    let moisture = smoothstep(0.0005, 0.035, in.water);
    let lowland = 1.0 - smoothstep(0.18, 0.65, h);
    let vegetation = moisture * lowland * (1.0 - steepness) * 0.38;
    let soil = mix(vec3<f32>(0.40, 0.35, 0.26), vec3<f32>(0.55, 0.49, 0.39), smoothstep(0.18, 0.85, h));
    let rock = mix(vec3<f32>(0.36, 0.34, 0.31), vec3<f32>(0.61, 0.59, 0.54), h);
    let alluvium = vec3<f32>(0.55, 0.45, 0.31);
    let stratum = mix(soil, in.materialColor, select(0.0, 0.7, renderParams.cameraPos.w > 0.5));
    let slopeTint = mix(vec3<f32>(0.64, 0.50, 0.33), rock * 0.86, smoothstep(0.45, 0.9, slope));
    var natural = mix(stratum, slopeTint, steepness * 0.48 + erosion * 0.12);
    natural = mix(natural, alluvium, deposition * (1.0 - steepness) * 0.55);
    natural = mix(natural, vec3<f32>(0.25, 0.33, 0.21), vegetation);
    natural *= 1.0 - moisture * 0.18;
    // Subtle spatial grain, without height bands or diagnostic history stripes.
    let grain = 0.98 + 0.04 * fract(sin(dot(in.worldPos.xz, vec2<f32>(127.1, 311.7))) * 43758.5453);
    return vec4<f32>(clamp(natural * reliefLight * grain, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
  }

  return vec4<f32>(terrainColor, 1.0);
}
`;var Je=`
@group(0) @binding(7) var<storage, read> meshPositions: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read> meshIndices: array<u32>;
@group(0) @binding(9) var<uniform> wireStyle: vec4<f32>;
struct WaterGeometry { metrics:vec4<f32>, links:vec4<u32>, forcing:vec4<f32> }
@group(0) @binding(10) var<storage, read> meshWater:array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> waterGeometry:array<WaterGeometry>;
fn postVertex(p: vec2<f32>) -> RenderVertexOut {
  let base = vec2<i32>(floor(p));
  let f = fract(p);
  let a = renderStateAt(base.x, base.y);
  let b = renderStateAt(base.x + 1, base.y);
  let c = renderStateAt(base.x, base.y + 1);
  let d = renderStateAt(base.x + 1, base.y + 1);
  let weights = vec4<f32>((1.0-f.x)*(1.0-f.y), f.x*(1.0-f.y), (1.0-f.x)*f.y, f.x*f.y);
  let h = dot(vec4<f32>(a.terrain,b.terrain,c.terrain,d.terrain), weights);
  let water = dot(vec4<f32>(a.water,b.water,c.water,d.water), weights);
  let nearest = vec2<i32>(round(p));
  let cell = renderStateAt(nearest.x, nearest.y);
  let i = idx(u32(nearest.x),u32(nearest.y));
  let waterPass = renderParams.misc.z > 0.5;
  let y = max(h,0.0)*renderParams.dims.w + select(0.0,max(water,0.0)*renderParams.misc.y,waterPass);
  let world = vec3<f32>((p.x-(renderParams.dims.x-1.0)*0.5)*renderParams.misc.x,y,((renderParams.dims.y-1.0)*0.5-p.y)*renderParams.misc.x);
  let relief = renderRelief(nearest.x,nearest.y,renderTerrainHeightAt(nearest.x,nearest.y));
  var out: RenderVertexOut;
  out.position = renderParams.viewProj*vec4<f32>(world,1.0);
  out.normal = select(relief.xyz,vec3<f32>(0.0,1.0,0.0),waterPass);
  out.terrain = h; out.water = water;
  out.sediment = dot(vec4<f32>(a.sediment,b.sediment,c.sediment,d.sediment),weights);
  out.hardness = cell.hardness;
  out.thermal = clamp(pow(max((cell.aux0+cell.aux1)*renderParams.shading.w,0.0),0.45),0.0,1.0);
  out.history = clamp(cell.aux2,-1.0,1.0); out.mask = cell.mask;
  out.worldPos = world; out.gridPos = p;
  out.materialColor = exposedMaterialColor(i,h); out.cavity = relief.w;
  out.flow = vec3<f32>(0.0);
  if (waterPass && renderParams.shading.z == 9.0) { let v = renderVelocity.values[i]; out.flow = vec3<f32>(v.xy,v.w); }
  return out;
}
@vertex fn vsPost(@builtin(vertex_index) i: u32) -> RenderVertexOut { return postVertex(meshPositions[i]); }
@vertex fn vsSWE(@builtin(vertex_index) i:u32)->RenderVertexOut {
  var out=postVertex(meshPositions[meshIndices[i]]);let tri=i/3u;let s=meshWater[tri];
  let bed=waterGeometry[tri].metrics.y;
  let eta=bed*renderParams.dims.w+s.x*renderParams.misc.y;
  out.worldPos.y=eta;out.position=renderParams.viewProj*vec4<f32>(out.worldPos,1.0);
  out.water=(eta-out.terrain*renderParams.dims.w)/max(renderParams.misc.y,1e-6);
  out.flow=vec3<f32>(s.yz/max(s.x,1e-7),length(s.yz)/max(s.x,1e-7));return out;
}
struct WireOut { @builtin(position) position: vec4<f32>, @location(0) bary: vec3<f32>, @location(1) mask: f32 }
@vertex fn vsWire(@builtin(vertex_index) i: u32) -> WireOut {
  let v = postVertex(meshPositions[meshIndices[i]]);
  var out: WireOut;
  out.position = v.position; out.position.z -= 0.00008*out.position.w;
  out.mask = v.mask;
  out.bary = vec3<f32>(0.0); out.bary[i % 3u] = 1.0;
  return out;
}
@fragment fn fsWire(v: WireOut) -> @location(0) vec4<f32> {
  let edge = v.bary / max(fwidth(v.bary),vec3<f32>(0.00001));
  let alpha = 1.0-smoothstep(0.4,1.15,min(edge.x,min(edge.y,edge.z)));
  if (alpha < 0.01 || v.mask < 0.5) { discard; }
  return vec4<f32>(0.07,0.16,0.18,alpha*wireStyle.x);
}
`,j=class{constructor(e){this.sim=e,this.buffers=[]}async initialize(e){let{sim:t}=this,i=t.device,a=e.positionsGrid,r=e.indices;if(!a?.length||!r?.length)throw Error("No triangles were produced. Try a finer spacing.");for(let m of r)if(m>=a.length/2)throw Error("Invalid mesh index.");for(let m of a)if(!Number.isFinite(m))throw Error("Invalid mesh position.");this.indexCount=r.length;let n=(m,f)=>{let u=i.createBuffer({size:m.byteLength,usage:f|GPUBufferUsage.COPY_DST});return i.queue.writeBuffer(u,0,m),this.buffers.push(u),u};this.positions=n(a,GPUBufferUsage.STORAGE),this.indices=n(r,GPUBufferUsage.STORAGE|GPUBufferUsage.INDEX),this.style=n(new Float32Array([.55,0,0,0]),GPUBufferUsage.UNIFORM),this.dummyWater=n(new Float32Array(4),GPUBufferUsage.STORAGE),this.dummyGeometry=n(new Float32Array(12),GPUBufferUsage.STORAGE);let l=GPUShaderStage.VERTEX,o=GPUShaderStage.FRAGMENT,c=i.createBindGroupLayout({entries:[{binding:0,visibility:l|o,buffer:{type:"uniform"}},{binding:1,visibility:l|o,buffer:{type:"read-only-storage"}},{binding:2,visibility:o,sampler:{type:"filtering"}},{binding:3,visibility:o,texture:{sampleType:"float"}},{binding:4,visibility:l,buffer:{type:"read-only-storage"}},{binding:5,visibility:l,buffer:{type:"uniform"}},{binding:6,visibility:l,buffer:{type:"read-only-storage"}},{binding:7,visibility:l,buffer:{type:"read-only-storage"}},{binding:8,visibility:l,buffer:{type:"read-only-storage"}},{binding:9,visibility:o,buffer:{type:"uniform"}},{binding:10,visibility:l,buffer:{type:"read-only-storage"}},{binding:11,visibility:l,buffer:{type:"read-only-storage"}}]}),d=i.createPipelineLayout({bindGroupLayouts:[c]}),p=i.createShaderModule({code:Fe()+Je}),h={color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha"}},v=(m,f,u,S)=>i.createRenderPipelineAsync({layout:d,vertex:{module:p,entryPoint:m},fragment:{module:p,entryPoint:f,targets:[{format:t.presentationFormat,...u?{blend:h}:{}}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:S,depthCompare:"less-equal"}});return[this.terrain,this.water,this.wire,this.sweWater]=await Promise.all([v("vsPost","fsMesh",!1,!0),v("vsPost","fsMesh",!0,!1),v("vsWire","fsWire",!0,!1),v("vsSWE","fsMesh",!0,!1)]),this.groupLayout=c,this.refreshGroups(),this}refreshGroups(e=null){let t=this.sim,i=t.device,a=[t.stateBuffers[0],t.sceneColorView,t.layerTopBuffer,t.velocityBuffer,e?.states[0],e?.geometry];if(this.groupKey?.every((n,l)=>n===a[l]))return;this.groupKey=a;let r=(n,l=this.dummyWater)=>i.createBindGroup({layout:this.groupLayout,entries:[{binding:0,resource:{buffer:n?t.renderUniformBufferWater:t.renderUniformBufferTerrain}},{binding:1,resource:{buffer:t.stateBuffers[0]}},{binding:2,resource:t.sceneSampler},{binding:3,resource:n?t.sceneColorView:t.dummySceneView},{binding:4,resource:{buffer:t.layerTopBuffer}},{binding:5,resource:{buffer:t.layerMaterialBuffer}},{binding:6,resource:{buffer:t.velocityBuffer}},{binding:7,resource:{buffer:this.positions}},{binding:8,resource:{buffer:this.indices}},{binding:9,resource:{buffer:this.style}},{binding:10,resource:{buffer:l}},{binding:11,resource:{buffer:e?.geometry||this.dummyGeometry}}]});this.terrainGroup=r(!1),this.waterGroup=r(!0),this.sweGroups=e?e.states.map(n=>r(!0,n)):null}render({adaptive:e=!0,wireframe:t=!0,wireOpacity:i=.55,swe:a=null}={}){let r=this.sim,n=r.device.createCommandEncoder();this.refreshGroups(a),r.device.queue.writeBuffer(this.style,0,new Float32Array([Math.max(0,Math.min(1,i)),0,0,0])),r.setSimulationParams({});let l=n.beginRenderPass({colorAttachments:[{view:r.sceneColorView,clearValue:{r:.055,g:.065,b:.08,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:r.depthTextureView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});l.setPipeline(e?this.terrain:r.renderPipeline),l.setBindGroup(0,e?this.terrainGroup:r.renderBindGroupTerrain),e?(l.setIndexBuffer(this.indices,"uint32"),l.drawIndexed(this.indexCount)):l.draw(6,r.drawCount),l.end();let o=n.beginRenderPass({colorAttachments:[{view:r.context.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:.055,g:.065,b:.08,a:1}}],depthStencilAttachment:{view:r.depthTextureView,depthLoadOp:"load",depthStoreOp:"discard"}});o.setPipeline(r.compositePipeline),o.setBindGroup(0,r.compositeBindGroup),o.draw(3),o.setPipeline(a?this.sweWater:e?this.water:r.waterRenderPipeline),o.setBindGroup(0,a?this.sweGroups[a.current]:e?this.waterGroup:r.renderBindGroupWater),a?o.draw(this.indexCount):e?(o.setIndexBuffer(this.indices,"uint32"),o.drawIndexed(this.indexCount)):o.draw(6,r.drawCount),t&&(o.setPipeline(this.wire),o.setBindGroup(0,this.terrainGroup),o.draw(this.indexCount)),o.end(),r.device.queue.submit([n.finish()])}destroy(){for(let e of this.buffers)e.destroy();this.buffers=[]}};var et='(()=>{function computeLocalRelief(heights,width,height,radius=2,mask=null){if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||heights.length!==width*height||mask&&mask.length!==heights.length||!Number.isFinite(radius))throw new Error("Invalid relief raster dimensions or radius.");radius=Math.max(0,Math.floor(radius));const low=new Float32Array(heights.length),high=new Float32Array(heights.length),result=new Float32Array(heights.length);const minQueue=new Int32Array(Math.max(width,height)),maxQueue=new Int32Array(minQueue.length);for(let y=0;y<height;y++){let a=0,b=0,c=0,d=0,next=0;for(let x=0;x<width;x++){const end=Math.min(width-1,x+radius);while(next<=end){const i=y*width+next,v=heights[i];if((!mask||mask[i])&&Number.isFinite(v)){while(b>a&&heights[y*width+minQueue[b-1]]>=v)b--;while(d>c&&heights[y*width+maxQueue[d-1]]<=v)d--;minQueue[b++]=next;maxQueue[d++]=next}next++}while(b>a&&minQueue[a]<x-radius)a++;while(d>c&&maxQueue[c]<x-radius)c++;low[y*width+x]=b>a?heights[y*width+minQueue[a]]:Infinity;high[y*width+x]=d>c?heights[y*width+maxQueue[c]]:-Infinity}}for(let x=0;x<width;x++){let a=0,b=0,c=0,d=0,next=0;for(let y=0;y<height;y++){const end=Math.min(height-1,y+radius);while(next<=end){const i2=next*width+x;while(b>a&&low[minQueue[b-1]*width+x]>=low[i2])b--;while(d>c&&high[maxQueue[d-1]*width+x]<=high[i2])d--;minQueue[b++]=next;maxQueue[d++]=next;next++}while(b>a&&minQueue[a]<y-radius)a++;while(d>c&&maxQueue[c]<y-radius)c++;const i=y*width+x;if((!mask||mask[i])&&Number.isFinite(heights[i]))result[i]=high[maxQueue[c]*width+x]-low[minQueue[a]*width+x]}}return result}var EPSILON=Math.pow(2,-52);var EDGE_STACK=new Uint32Array(512);function orient2d(ax,ay,bx,by,cx,cy){const acx=ax-cx;const acy=ay-cy;const bcx=bx-cx;const bcy=by-cy;return acy*bcx-acx*bcy}function circumradius(ax,ay,bx,by,cx,cy){const dx=bx-ax;const dy=by-ay;const ex=cx-ax;const ey=cy-ay;const bl=dx*dx+dy*dy;const cl=ex*ex+ey*ey;const determinant=dx*ey-dy*ex;if(determinant===0)return Infinity;const factor=.5/determinant;const x=(ey*bl-dy*cl)*factor;const y=(dx*cl-ex*bl)*factor;return x*x+y*y}function hashKey(dx,dy,hashSize){const absSum=Math.abs(dx)+Math.abs(dy);if(absSum===0)return 0;const p=dx/absSum;return((dy>0?3-p:1+p)*hashSize*.25|0)%hashSize}function insertionSort(ids,dists,left,right){for(let i=left+1;i<=right;i++){const id=ids[i];const d=dists[id];let j=i-1;while(j>=left&&dists[ids[j]]>d){ids[j+1]=ids[j];j--}ids[j+1]=id}}function swap(arr,i,j){const temp=arr[i];arr[i]=arr[j];arr[j]=temp}function quicksort(ids,dists,left,right){if(right-left<=20){insertionSort(ids,dists,left,right);return}const median=left+right>>1;if(dists[ids[left]]>dists[ids[right]])swap(ids,left,right);if(dists[ids[left]]>dists[ids[median]])swap(ids,left,median);if(dists[ids[median]]>dists[ids[right]])swap(ids,median,right);const pivot=ids[median];const pivotDist=dists[pivot];swap(ids,median,right);let partitionIndex=left;for(let i=left;i<right;i++){if(dists[ids[i]]<pivotDist){swap(ids,i,partitionIndex);partitionIndex++}}swap(ids,partitionIndex,right);const leftSize=partitionIndex-left;const rightSize=right-partitionIndex;if(leftSize<rightSize){quicksort(ids,dists,left,partitionIndex-1);quicksort(ids,dists,partitionIndex+1,right)}else{quicksort(ids,dists,partitionIndex+1,right);quicksort(ids,dists,left,partitionIndex-1)}}function defaultGetX(p){return p[0]}function defaultGetY(p){return p[1]}var Delaunator=class _Delaunator{coords;triangles;halfedges;hull;pointCount;capacity;_cx;_cy;_hullStart;_hullPrev;_hullNext;_hullTri;_hullHash;_ids;_dists;_triangles;_halfedges;_hashSize;trianglesLen;static from(points,getX=defaultGetX,getY=defaultGetY,capacity=points.length){const n=points.length;if(capacity<n){throw new RangeError(`capacity ${capacity} is smaller than point count ${n}`)}const coords=new Float64Array(capacity*2);for(let i=0,j=0;i<n;i++,j+=2){const p=points[i];coords[j]=getX(p);coords[j+1]=getY(p)}return new _Delaunator(coords,n)}static withCapacity(capacity){return new _Delaunator(capacity,0)}constructor(coordsOrCapacity,pointCount){if(typeof coordsOrCapacity==="number"){const capacity=Math.max(0,Math.floor(coordsOrCapacity));this.coords=new Float64Array(capacity*2);this.capacity=capacity;this.pointCount=pointCount===void 0?0:Math.max(0,Math.floor(pointCount));if(this.pointCount>this.capacity){throw new RangeError(`pointCount ${this.pointCount} exceeds capacity ${this.capacity}`)}}else{this.coords=coordsOrCapacity;this.capacity=coordsOrCapacity.length>>1;this.pointCount=pointCount===void 0?this.capacity:Math.max(0,Math.floor(pointCount));if(this.pointCount>this.capacity){throw new RangeError(`pointCount ${this.pointCount} exceeds capacity ${this.capacity}`)}}this._allocate(this.capacity);this.update()}get activeCoords(){return this.coords.subarray(0,this.pointCount<<1)}clear(){this.pointCount=0;this.update()}reserve(minCapacity){const targetCapacity=Math.max(0,Math.floor(minCapacity));if(targetCapacity<=this.capacity)return;const nextCoords=new Float64Array(targetCapacity*2);nextCoords.set(this.coords.subarray(0,this.pointCount<<1));this.coords=nextCoords;this.capacity=targetCapacity;this._allocate(targetCapacity)}ensureCapacity(minCapacity){const needed=Math.max(0,Math.floor(minCapacity));if(needed<=this.capacity)return;let next=this.capacity>0?this.capacity:1;while(next<needed)next*=2;this.reserve(next)}setPointCount(pointCount){const count=Math.max(0,Math.floor(pointCount));this.ensureCapacity(count);this.pointCount=count}setPoint(index,x,y){const i=Math.floor(index);if(i<0){throw new RangeError(`index ${index} must be >= 0`)}this.ensureCapacity(i+1);const j=i<<1;this.coords[j]=x;this.coords[j+1]=y;if(i>=this.pointCount){this.pointCount=i+1}}setPointsFlat(source,pointCount=source.length>>1){const count=Math.max(0,Math.floor(pointCount));this.setPointCount(count);const coordCount=count<<1;for(let i=0;i<coordCount;i++){this.coords[i]=source[i]}}update(pointCount=this.pointCount){this.setPointCount(pointCount);const coords=this.coords;const hullPrev=this._hullPrev;const hullNext=this._hullNext;const hullTri=this._hullTri;const hullHash=this._hullHash;const ids=this._ids;const dists=this._dists;const n=this.pointCount;if(n===0){this.hull=new Uint32Array(0);this.trianglesLen=0;this.triangles=this._triangles.subarray(0,0);this.halfedges=this._halfedges.subarray(0,0);return}if(n<3){const hull2=new Uint32Array(n);for(let i=0;i<n;i++)hull2[i]=i;this.hull=hull2;this.trianglesLen=0;this.triangles=this._triangles.subarray(0,0);this.halfedges=this._halfedges.subarray(0,0);return}let minX=Infinity;let minY=Infinity;let maxX=-Infinity;let maxY=-Infinity;for(let i=0,j=0;i<n;i++,j+=2){const x=coords[j];const y=coords[j+1];if(x<minX)minX=x;if(y<minY)minY=y;if(x>maxX)maxX=x;if(y>maxY)maxY=y;ids[i]=i}const centerX=(minX+maxX)*.5;const centerY=(minY+maxY)*.5;let i0=0;let minDist=Infinity;for(let i=0,j=0;i<n;i++,j+=2){const dx2=centerX-coords[j];const dy2=centerY-coords[j+1];const d=dx2*dx2+dy2*dy2;if(d<minDist){i0=i;minDist=d}}const i0c=i0<<1;const i0x=coords[i0c];const i0y=coords[i0c+1];let i1=-1;minDist=Infinity;for(let i=0,j=0;i<n;i++,j+=2){if(i===i0)continue;const dx2=i0x-coords[j];const dy2=i0y-coords[j+1];const d=dx2*dx2+dy2*dy2;if(d>0&&d<minDist){i1=i;minDist=d}}if(i1===-1){this.hull=new Uint32Array([i0]);this.trianglesLen=0;this.triangles=this._triangles.subarray(0,0);this.halfedges=this._halfedges.subarray(0,0);return}let i1c=i1<<1;let i1x=coords[i1c];let i1y=coords[i1c+1];let i2=-1;let minRadius=Infinity;for(let i=0,j=0;i<n;i++,j+=2){if(i===i0||i===i1)continue;const r=circumradius(i0x,i0y,i1x,i1y,coords[j],coords[j+1]);if(r<minRadius){i2=i;minRadius=r}}if(i2===-1||minRadius===Infinity){const baseX=coords[0];const baseY=coords[1];for(let i=0,j2=0;i<n;i++,j2+=2){dists[i]=coords[j2]-baseX||coords[j2+1]-baseY}quicksort(ids,dists,0,n-1);const hull2=new Uint32Array(n);let j=0;let d0=-Infinity;for(let i=0;i<n;i++){const id=ids[i];const d=dists[id];if(d>d0){hull2[j++]=id;d0=d}}this.hull=hull2.subarray(0,j);this.trianglesLen=0;this.triangles=this._triangles.subarray(0,0);this.halfedges=this._halfedges.subarray(0,0);return}let i2c=i2<<1;let i2x=coords[i2c];let i2y=coords[i2c+1];if(orient2d(i0x,i0y,i1x,i1y,i2x,i2y)<0){const tempI=i1;i1=i2;i2=tempI;const tempX=i1x;i1x=i2x;i2x=tempX;const tempY=i1y;i1y=i2y;i2y=tempY;i1c=i1<<1;i2c=i2<<1}const dx=i1x-i0x;const dy=i1y-i0y;const ex=i2x-i0x;const ey=i2y-i0y;const bl=dx*dx+dy*dy;const cl=ex*ex+ey*ey;const determinant=dx*ey-dy*ex;const factor=.5/determinant;const cx=i0x+(ey*bl-dy*cl)*factor;const cy=i0y+(dx*cl-ex*bl)*factor;this._cx=cx;this._cy=cy;for(let i=0,j=0;i<n;i++,j+=2){const ddx=coords[j]-cx;const ddy=coords[j+1]-cy;dists[i]=ddx*ddx+ddy*ddy}quicksort(ids,dists,0,n-1);this._hullStart=i0;let hullSize=3;hullNext[i0]=i1;hullPrev[i2]=i1;hullNext[i1]=i2;hullPrev[i0]=i2;hullNext[i2]=i0;hullPrev[i1]=i0;hullTri[i0]=0;hullTri[i1]=1;hullTri[i2]=2;hullHash.fill(-1);const hashSize=this._hashSize;hullHash[hashKey(i0x-cx,i0y-cy,hashSize)]=i0;hullHash[hashKey(i1x-cx,i1y-cy,hashSize)]=i1;hullHash[hashKey(i2x-cx,i2y-cy,hashSize)]=i2;this.trianglesLen=0;this._addTriangle(i0,i1,i2,-1,-1,-1);let xp=0;let yp=0;for(let k=0;k<n;k++){const i=ids[k];const ic=i<<1;const x=coords[ic];const y=coords[ic+1];if(k>0&&Math.abs(x-xp)<=EPSILON&&Math.abs(y-yp)<=EPSILON)continue;xp=x;yp=y;if(i===i0||i===i1||i===i2)continue;let start=-1;let key=hashKey(x-cx,y-cy,hashSize);for(let j=0;j<hashSize;j++){start=hullHash[key];if(start!==-1&&start!==hullNext[start])break;key++;if(key===hashSize)key=0}if(start===-1)continue;start=hullPrev[start];let e=start;let q=0;while(true){q=hullNext[e];const ec2=e<<1;const qc=q<<1;const qx=coords[qc];const qy=coords[qc+1];if((y-qy)*(coords[ec2]-qx)-(x-qx)*(coords[ec2+1]-qy)<0)break;e=q;if(e===start){e=-1;break}}if(e===-1)continue;let t=this._addTriangle(e,i,hullNext[e],-1,-1,hullTri[e]);hullTri[i]=this._legalize(t+2);hullTri[e]=t;hullSize++;let next=hullNext[e];while(true){q=hullNext[next];const nc=next<<1;const qc=q<<1;const qx=coords[qc];const qy=coords[qc+1];if((y-qy)*(coords[nc]-qx)-(x-qx)*(coords[nc+1]-qy)>=0)break;t=this._addTriangle(next,i,q,hullTri[i],-1,hullTri[next]);hullTri[i]=this._legalize(t+2);hullNext[next]=next;hullSize--;next=q}if(e===start){while(true){q=hullPrev[e];const qc=q<<1;const ec2=e<<1;const ex2=coords[ec2];const ey2=coords[ec2+1];if((y-ey2)*(coords[qc]-ex2)-(x-ex2)*(coords[qc+1]-ey2)>=0)break;t=this._addTriangle(q,i,e,-1,hullTri[e],hullTri[q]);this._legalize(t+2);hullTri[q]=t;hullNext[e]=e;hullSize--;e=q}}this._hullStart=hullPrev[i]=e;hullNext[e]=hullPrev[next]=i;hullNext[i]=next;hullHash[hashKey(x-cx,y-cy,hashSize)]=i;const ec=e<<1;hullHash[hashKey(coords[ec]-cx,coords[ec+1]-cy,hashSize)]=e}const hull=new Uint32Array(hullSize);for(let i=0,e=this._hullStart;i<hullSize;i++){hull[i]=e;e=hullNext[e]}this.hull=hull;this.triangles=this._triangles.subarray(0,this.trianglesLen);this.halfedges=this._halfedges.subarray(0,this.trianglesLen)}_allocate(capacity){const hashSize=Math.max(1,Math.ceil(Math.sqrt(capacity)));this._hashSize=hashSize;this._hullPrev=new Uint32Array(capacity);this._hullNext=new Uint32Array(capacity);this._hullTri=new Uint32Array(capacity);this._hullHash=new Int32Array(hashSize);this._ids=new Uint32Array(capacity);this._dists=new Float64Array(capacity);const maxTriangles=Math.max(2*capacity-5,0);this._triangles=new Uint32Array(maxTriangles*3);this._halfedges=new Int32Array(maxTriangles*3);this.trianglesLen=0}_legalize(a){const triangles=this._triangles;const halfedges=this._halfedges;const coords=this.coords;const hullStart=this._hullStart;const hullTri=this._hullTri;const hullPrev=this._hullPrev;let stackLen=0;let ar=0;while(true){const b=halfedges[a];if(b===-1){if(stackLen===0)break;a=EDGE_STACK[--stackLen];continue}const a0=a-a%3;const aLocal=a-a0;ar=aLocal===0?a0+2:a-1;const al=aLocal===2?a0:a+1;const b0=b-b%3;const bLocal=b-b0;const bl=bLocal===0?b0+2:b-1;const br=bLocal===2?b0:b+1;const p0=triangles[ar];const pr=triangles[a];const pl=triangles[al];const p1=triangles[bl];const p0c=p0<<1;const prc=pr<<1;const plc=pl<<1;const p1c=p1<<1;const ax=coords[p0c];const ay=coords[p0c+1];const bx=coords[prc];const by=coords[prc+1];const cx=coords[plc];const cy=coords[plc+1];const px=coords[p1c];const py=coords[p1c+1];const dx=ax-px;const dy=ay-py;const ex=bx-px;const ey=by-py;const fx=cx-px;const fy=cy-py;const ap=dx*dx+dy*dy;const bp=ex*ex+ey*ey;const cp=fx*fx+fy*fy;const illegal=dx*(ey*cp-bp*fy)-dy*(ex*cp-bp*fx)+ap*(ex*fy-ey*fx)<0;if(illegal){triangles[a]=p1;triangles[b]=p0;const hbl=halfedges[bl];if(hbl===-1){let e=hullStart;do{if(hullTri[e]===bl){hullTri[e]=a;break}e=hullPrev[e]}while(e!==hullStart)}halfedges[a]=hbl;if(hbl!==-1)halfedges[hbl]=a;const har=halfedges[ar];halfedges[b]=har;if(har!==-1)halfedges[har]=b;halfedges[ar]=bl;halfedges[bl]=ar;if(stackLen<EDGE_STACK.length)EDGE_STACK[stackLen++]=br}else{if(stackLen===0)break;a=EDGE_STACK[--stackLen]}}return ar}_addTriangle(i0,i1,i2,a,b,c){const t=this.trianglesLen;const triangles=this._triangles;const halfedges=this._halfedges;triangles[t]=i0;triangles[t+1]=i1;triangles[t+2]=i2;halfedges[t]=a;if(a!==-1)halfedges[a]=t;halfedges[t+1]=b;if(b!==-1)halfedges[b]=t+1;halfedges[t+2]=c;if(c!==-1)halfedges[c]=t+2;this.trianglesLen=t+3;return t}};var EPSILON2=1e-6;var DEFAULT_BUCKET_COUNT=256;var LARGE_DISTANCE=1e9;function clamp(value,min,max){return value<min?min:value>max?max:value}function lerp(a,b,t){return a+(b-a)*t}function smoothstep(edge0,edge1,x){const t=clamp((x-edge0)/Math.max(edge1-edge0,EPSILON2),0,1);return t*t*(3-2*t)}function quantizeRadius(radius,minRadius,maxRadius,bins=16){if(bins<=1||maxRadius<=minRadius+EPSILON2)return radius;const t=clamp((radius-minRadius)/Math.max(maxRadius-minRadius,EPSILON2),0,1);const q=Math.round(t*(bins-1))/(bins-1);return lerp(minRadius,maxRadius,q)}function isFiniteNumber(value){return Number.isFinite(value)}function indexOfCell(x,y,width){return y*width+x}function makeFullMask(width,height){const mask=new Uint8Array(width*height);mask.fill(1);return mask}function isBoundaryCell(mask,width,height,x,y){const i=indexOfCell(x,y,width);if(mask[i]===0)return false;if(x===0||y===0||x===width-1||y===height-1)return true;const up=indexOfCell(x,y-1,width);const down=indexOfCell(x,y+1,width);const left=indexOfCell(x-1,y,width);const right=indexOfCell(x+1,y,width);return mask[up]===0||mask[down]===0||mask[left]===0||mask[right]===0}function scanMinMax(values,mask){let min=Infinity;let max=-Infinity;for(let i=0;i<values.length;i++){if(mask&&mask[i]===0)continue;const v=values[i];if(!isFiniteNumber(v))continue;if(v<min)min=v;if(v>max)max=v}if(!isFiniteNumber(min)||!isFiniteNumber(max)){return{min:0,max:1}}if(Math.abs(max-min)<=EPSILON2){return{min,max:min+1}}return{min,max}}function normalizeArray(values,mask){const out=new Float32Array(values.length);const{min,max}=scanMinMax(values,mask);const inv=1/Math.max(max-min,EPSILON2);for(let i=0;i<values.length;i++){if(mask&&mask[i]===0)continue;out[i]=clamp((values[i]-min)*inv,0,1)}return out}function normalizeAbsArray(values,mask){const absValues=new Float32Array(values.length);for(let i=0;i<values.length;i++){absValues[i]=Math.abs(values[i])}return normalizeArray(absValues,mask)}function bilinearSample(values,width,height,x,y){const x0=clamp(Math.floor(x),0,width-1);const y0=clamp(Math.floor(y),0,height-1);const x1=clamp(x0+1,0,width-1);const y1=clamp(y0+1,0,height-1);const tx=clamp(x-x0,0,1);const ty=clamp(y-y0,0,1);const v00=values[indexOfCell(x0,y0,width)];const v10=values[indexOfCell(x1,y0,width)];const v01=values[indexOfCell(x0,y1,width)];const v11=values[indexOfCell(x1,y1,width)];const a=lerp(v00,v10,tx);const b=lerp(v01,v11,tx);return lerp(a,b,ty)}function nearestMaskSample(mask,width,height,x,y){const gx=clamp(Math.round(x),0,width-1);const gy=clamp(Math.round(y),0,height-1);return mask[indexOfCell(gx,gy,width)]}function ensureImageReady(image){if(image.complete&&image.naturalWidth>0&&image.naturalHeight>0){return Promise.resolve()}if(typeof image.decode==="function"){return image.decode().catch(()=>{return new Promise((resolve,reject)=>{image.onload=()=>resolve();image.onerror=()=>reject(new Error("Failed to decode image source."))})})}return new Promise((resolve,reject)=>{image.onload=()=>resolve();image.onerror=()=>reject(new Error("Failed to load image source."))})}async function rasterFromImage(source,options={}){const valueFromRGBA=options.valueFromRGBA??((r,g,b)=>{return(r+g+b)/(3*255)});let width=0;let height=0;let imageData;if(typeof ImageData!=="undefined"&&source instanceof ImageData){width=source.width;height=source.height;imageData=source}else{if(typeof HTMLImageElement!=="undefined"&&source instanceof HTMLImageElement){await ensureImageReady(source);width=source.naturalWidth||source.width;height=source.naturalHeight||source.height}else if(typeof ImageBitmap!=="undefined"&&source instanceof ImageBitmap){width=source.width;height=source.height}else if(typeof HTMLCanvasElement!=="undefined"&&source instanceof HTMLCanvasElement){width=source.width;height=source.height}else{throw new Error("Unsupported image source.")}let canvas;let ctx;if(typeof OffscreenCanvas!=="undefined"){canvas=new OffscreenCanvas(width,height);ctx=canvas.getContext("2d")}else if(typeof document!=="undefined"){canvas=document.createElement("canvas");canvas.width=width;canvas.height=height;ctx=canvas.getContext("2d")}else{throw new Error("No canvas implementation available for image rasterization.")}if(!ctx){throw new Error("Failed to acquire a 2D canvas context.")}ctx.drawImage(source,0,0,width,height);imageData=ctx.getImageData(0,0,width,height)}const values=new Float32Array(width*height);const mask=new Uint8Array(width*height);const rgba=imageData.data;const flipY=!!options.flipY;const alphaMeansInvalid=!!options.alphaMeansInvalid;for(let y=0;y<height;y++){const srcY=flipY?height-1-y:y;for(let x=0;x<width;x++){const srcIndex=(srcY*width+x)*4;const dstIndex=indexOfCell(x,y,width);const r=rgba[srcIndex];const g=rgba[srcIndex+1];const b=rgba[srcIndex+2];const a=rgba[srcIndex+3];values[dstIndex]=valueFromRGBA(r,g,b,a,dstIndex);mask[dstIndex]=alphaMeansInvalid&&a===0?0:1}}return{width,height,values,mask}}var HeightfieldTINBuilder=class{options;width=0;height=0;heights=null;mask=null;analysis=null;topology=null;constructor(options={}){const cellSize=Math.max(options.cellSize??1,EPSILON2);const minSpacing=Math.max(options.minSpacing??cellSize,EPSILON2);const maxSpacing=Math.max(options.maxSpacing??cellSize*16,minSpacing);this.options={mode:options.mode??"hybrid",cellSize,worldMinX:options.worldMinX??0,worldMinY:options.worldMinY??0,minSpacing,maxSpacing,borderSpacing:Math.max(options.borderSpacing??maxSpacing,EPSILON2),wetnessBins:Math.max(4,Math.floor(options.wetnessBins??21)),channelThreshold:Math.max(1,Math.floor(options.channelThreshold??128)),floodplainChannelDistanceCells:Math.max(1,Math.floor(options.floodplainChannelDistanceCells??6)),candidateStride:Math.max(1,Math.floor(options.candidateStride??1)),maxPoints:Math.max(3,Math.floor(options.maxPoints??Number.MAX_SAFE_INTEGER)),preserveBorder:options.preserveBorder??true,preserveChannels:options.preserveChannels??true,preserveMaskBoundary:options.preserveMaskBoundary??true,preserveInnerBorder:options.preserveInnerBorder??true,slopeWeight:Math.max(0,options.slopeWeight??.35),curvatureWeight:Math.max(0,options.curvatureWeight??.15),reliefWeight:Math.max(0,options.reliefWeight??.2),wetnessWeight:Math.max(0,options.wetnessWeight??.4),channelWeight:Math.max(0,options.channelWeight??.2),floodplainWeight:Math.max(0,options.floodplainWeight??.25),relaxIterations:Math.max(0,Math.floor(options.relaxIterations??0)),forcedWorldPoints:options.forcedWorldPoints??null}}setRaster(raster){if(!raster||raster.width<=0||raster.height<=0){throw new Error("Raster must have positive width and height.")}if(!(raster.values instanceof Float32Array)){throw new Error("Raster values must be a Float32Array.")}if(raster.values.length!==raster.width*raster.height){throw new Error("Raster value count does not match width * height.")}const mask=raster.mask&&raster.mask.length===raster.width*raster.height?raster.mask instanceof Uint8Array?raster.mask:new Uint8Array(raster.mask):makeFullMask(raster.width,raster.height);this.width=raster.width;this.height=raster.height;this.heights=raster.values;this.mask=mask;this.invalidate()}async setImage(source,options={}){const raster=await rasterFromImage(source,options);this.setRaster(raster)}setHeightValue(x,y,value){this.ensureRaster();const gx=clamp(Math.floor(x),0,this.width-1);const gy=clamp(Math.floor(y),0,this.height-1);const i=indexOfCell(gx,gy,this.width);this.heights[i]=value;this.analysis=null;this.topology=null}async buildWithForcedWorldPoints(forcedWorldPoints){this.ensureRaster();const analysis=this.analysis??this.buildAnalysis();const previousForced=this.options.forcedWorldPoints;try{this.options.forcedWorldPoints=forcedWorldPoints??previousForced??null;const topology=this.buildTopology(analysis);this.analysis=analysis;this.topology=topology;return topology}finally{this.options.forcedWorldPoints=previousForced}}getAnalysis(){return this.analysis}getTopology(){return this.topology}async build(){this.ensureRaster();const analysis=this.buildAnalysis();const topology=this.buildTopology(analysis);this.analysis=analysis;this.topology=topology;return topology}async rebuild(){return this.build()}resampleVertexHeights(){this.ensureRaster();if(!this.topology){throw new Error("Build topology before resampling heights.")}const heights=this.sampleVertexHeights(this.topology.positionsGrid);this.topology.heights=heights;return heights}invalidate(){this.analysis=null;this.topology=null}ensureRaster(){if(!this.heights||!this.mask||this.width<=0||this.height<=0){throw new Error("No heightfield raster is loaded.")}}buildAnalysis(){const width=this.width;const height=this.height;const cellSize=this.options.cellSize;const heights=this.heights;const mask=this.mask;const slopeTan=this.computeSlopeTan(heights,width,height,cellSize,mask);const slopeDegrees=new Float32Array(width*height);for(let i=0;i<slopeTan.length;i++){slopeDegrees[i]=Math.atan(slopeTan[i])*(180/Math.PI)}const curvature=this.computeCurvature(heights,width,height,cellSize,mask);const localRelief=this.computeLocalRelief(heights,width,height,mask);const flowReceiver=this.computeFlowReceiver(heights,width,height,cellSize,mask);const flowAccumulation=this.computeFlowAccumulation(flowReceiver,width,height,mask);const wetness=this.computeWetness(flowAccumulation,slopeTan,width,height,cellSize,mask);const channels=this.extractChannels(flowAccumulation,width,height,mask);const channelDistance=this.computeChannelDistance(channels,width,height,mask);const floodplain=this.extractFloodplain(wetness,slopeTan,channelDistance,width,height,mask);const terrainPriority=this.computeTerrainPriority(slopeTan,curvature,localRelief,mask);const hydroPriority=this.computeHydroPriority(wetness,channels,channelDistance,floodplain,mask);let validCellCount=0;let channelCellCount=0;for(let i=0;i<mask.length;i++){if(mask[i])validCellCount++;if(channels[i])channelCellCount++}const totalArea=validCellCount*cellSize*cellSize;const totalChannelLength=channelCellCount*cellSize;const hillslopeLength=totalChannelLength>EPSILON2?totalArea/(2*totalChannelLength):this.options.maxSpacing;const importance=this.computeImportance(terrainPriority,hydroPriority,mask);const spacing=this.computeSpacing(terrainPriority,hydroPriority,channels,floodplain,mask,hillslopeLength);return{width,height,cellSize,heights,mask,slopeTan,slopeDegrees,curvature,localRelief,flowReceiver,flowAccumulation,wetness,channels,channelDistance,floodplain,terrainPriority,hydroPriority,importance,spacing,validCellCount,totalArea,totalChannelLength,hillslopeLength}}computeSlopeTan(heights,width,height,cellSize,mask){const slope=new Float32Array(width*height);const inv2=1/(2*cellSize);for(let y=0;y<height;y++){const ym1=y>0?y-1:y;const yp1=y<height-1?y+1:y;for(let x=0;x<width;x++){const i=indexOfCell(x,y,width);if(!mask[i])continue;const xm1=x>0?x-1:x;const xp1=x<width-1?x+1:x;const left=heights[indexOfCell(xm1,y,width)];const right=heights[indexOfCell(xp1,y,width)];const up=heights[indexOfCell(x,ym1,width)];const down=heights[indexOfCell(x,yp1,width)];const dzdx=(right-left)*inv2;const dzdy=(down-up)*inv2;slope[i]=Math.hypot(dzdx,dzdy)}}return slope}computeCurvature(heights,width,height,cellSize,mask){const curvature=new Float32Array(width*height);const inv=1/(cellSize*cellSize);for(let y=0;y<height;y++){const ym1=y>0?y-1:y;const yp1=y<height-1?y+1:y;for(let x=0;x<width;x++){const i=indexOfCell(x,y,width);if(!mask[i])continue;const xm1=x>0?x-1:x;const xp1=x<width-1?x+1:x;const c=heights[i];const left=heights[indexOfCell(xm1,y,width)];const right=heights[indexOfCell(xp1,y,width)];const up=heights[indexOfCell(x,ym1,width)];const down=heights[indexOfCell(x,yp1,width)];curvature[i]=(left+right+up+down-4*c)*inv}}return curvature}computeLocalRelief(heights,width,height,mask){return computeLocalRelief(heights,width,height,1,mask)}computeFlowReceiver(heights,width,height,cellSize,mask){const receiver=new Int32Array(width*height);receiver.fill(-1);const diag=Math.SQRT2*cellSize;for(let y=0;y<height;y++){for(let x=0;x<width;x++){const i=indexOfCell(x,y,width);if(!mask[i])continue;const z=heights[i];let bestIndex=-1;let bestSlope=0;for(let oy=-1;oy<=1;oy++){for(let ox=-1;ox<=1;ox++){if(ox===0&&oy===0)continue;const nx=x+ox;const ny=y+oy;if(nx<0||ny<0||nx>=width||ny>=height)continue;const ni=indexOfCell(nx,ny,width);if(!mask[ni])continue;const dz=z-heights[ni];if(dz<=0)continue;const dist=ox!==0&&oy!==0?diag:cellSize;const localSlope=dz/dist;if(localSlope>bestSlope){bestSlope=localSlope;bestIndex=ni}}}receiver[i]=bestIndex}}return receiver}computeFlowAccumulation(receiver,width,height,mask){const count=width*height;const indegree=new Int32Array(count);const accumulation=new Float32Array(count);const queue=new Int32Array(count);for(let i=0;i<count;i++){if(!mask[i])continue;accumulation[i]=1;const r=receiver[i];if(r>=0)indegree[r]++}let head=0;let tail=0;for(let i=0;i<count;i++){if(mask[i]&&indegree[i]===0){queue[tail++]=i}}while(head<tail){const i=queue[head++];const r=receiver[i];if(r>=0){accumulation[r]+=accumulation[i];indegree[r]--;if(indegree[r]===0){queue[tail++]=r}}}return accumulation}computeWetness(accumulation,slopeTan,width,height,cellSize,mask){const wetness=new Float32Array(width*height);for(let i=0;i<wetness.length;i++){if(!mask[i])continue;const ai=accumulation[i]*cellSize;wetness[i]=Math.log(Math.max(ai,EPSILON2)/Math.max(slopeTan[i],EPSILON2))}return wetness}extractChannels(accumulation,width,height,mask){const channels=new Uint8Array(width*height);const threshold=this.options.channelThreshold;for(let i=0;i<channels.length;i++){if(!mask[i])continue;if(accumulation[i]>=threshold){channels[i]=1}}return channels}computeChannelDistance(channels,width,height,mask){const distance=new Float32Array(width*height);const queue=new Int32Array(width*height);let head=0;let tail=0;for(let i=0;i<distance.length;i++){if(!mask[i]){distance[i]=LARGE_DISTANCE;continue}if(channels[i]){distance[i]=0;queue[tail++]=i}else{distance[i]=LARGE_DISTANCE}}while(head<tail){const i=queue[head++];const x=i%width;const y=i/width|0;const base=distance[i];for(let oy=-1;oy<=1;oy++){for(let ox=-1;ox<=1;ox++){if(ox===0&&oy===0)continue;const nx=x+ox;const ny=y+oy;if(nx<0||ny<0||nx>=width||ny>=height)continue;const ni=indexOfCell(nx,ny,width);if(!mask[ni])continue;const step=ox!==0&&oy!==0?Math.SQRT2:1;const next=base+step;if(next<distance[ni]){distance[ni]=next;queue[tail++]=ni}}}}return distance}extractFloodplain(wetness,slopeTan,channelDistance,width,height,mask){const wetnessNorm=normalizeArray(wetness,mask);const slopeNorm=normalizeArray(slopeTan,mask);const floodplain=new Uint8Array(width*height);const maxDist=this.options.floodplainChannelDistanceCells;for(let i=0;i<floodplain.length;i++){if(!mask[i])continue;const nearChannel=channelDistance[i]<=maxDist;const wetEnough=wetnessNorm[i]>=.58;const flatEnough=slopeNorm[i]<=.42;if(nearChannel&&wetEnough&&flatEnough){floodplain[i]=1}}return floodplain}computeTerrainPriority(slopeTan,curvature,localRelief,mask){const slopeNorm=normalizeArray(slopeTan,mask);const curvatureNorm=normalizeAbsArray(curvature,mask);const reliefNorm=normalizeArray(localRelief,mask);const out=new Float32Array(slopeTan.length);for(let i=0;i<out.length;i++){if(!mask[i])continue;const ruggedness=slopeNorm[i]*.48+reliefNorm[i]*.34+curvatureNorm[i]*.18;const majorRelief=Math.max(slopeNorm[i],reliefNorm[i]);const sharpBreak=Math.max(curvatureNorm[i],(slopeNorm[i]+reliefNorm[i])*.5);const terrain=Math.max(ruggedness,majorRelief*.97,sharpBreak*.9);out[i]=clamp(Math.pow(terrain,.7),0,1)}return normalizeArray(out,mask)}computeHydroPriority(wetness,channels,channelDistance,floodplain,mask){const wetnessNorm=normalizeArray(wetness,mask);const out=new Float32Array(wetness.length);let distanceMax=0;for(let i=0;i<channelDistance.length;i++){if(!mask[i])continue;if(channelDistance[i]<LARGE_DISTANCE&&channelDistance[i]>distanceMax){distanceMax=channelDistance[i]}}for(let i=0;i<out.length;i++){if(!mask[i])continue;const d=channelDistance[i]>=LARGE_DISTANCE?distanceMax:channelDistance[i];const proximity=1-clamp(d/Math.max(distanceMax,1),0,1);const hydro=wetnessNorm[i]*.46+proximity*.18+channels[i]*.22+floodplain[i]*.14;out[i]=clamp(hydro,0,1)}return normalizeArray(out,mask)}computeImportance(terrainPriority,hydroPriority,mask){const out=new Float32Array(terrainPriority.length);const mode=this.options.mode;for(let i=0;i<out.length;i++){if(!mask[i])continue;const terrain=terrainPriority[i];const hydro=hydroPriority[i];if(mode==="traditional"){out[i]=terrain}else if(mode==="hydrological"){out[i]=Math.max(terrain*.8,hydro)}else if(mode==="hydrographic"){out[i]=Math.max(terrain,hydro*.55)*.96+Math.min(terrain,hydro)*.04}else{out[i]=Math.max(terrain,hydro*.5)*.97+Math.min(terrain,hydro)*.03}out[i]=clamp(out[i],0,1)}return normalizeArray(out,mask)}computeSpacing(terrainPriority,hydroPriority,channels,floodplain,mask,hillslopeLength){const out=new Float32Array(terrainPriority.length);const cellSize=this.options.cellSize;const minSpacing=Math.max(this.options.minSpacing,cellSize);const maxSpacing=Math.max(minSpacing,Math.min(this.options.maxSpacing,Math.max(hillslopeLength,minSpacing)));const mode=this.options.mode;for(let i=0;i<out.length;i++){if(!mask[i])continue;const terrain=clamp(terrainPriority[i],0,1);const hydro=clamp(hydroPriority[i],0,1);const terrainT=smoothstep(.08,.94,terrain);const terrainSpacing=lerp(maxSpacing,minSpacing,Math.pow(terrainT,.62));let spacing=terrainSpacing;if(mode==="hydrological"){const hydroT=smoothstep(.42,.97,hydro);const hydroSpacing=lerp(maxSpacing,minSpacing*1.15,Math.pow(hydroT,1.15));spacing=Math.min(terrainSpacing,hydroSpacing)}else{if(channels[i]){const channelT=Math.max(.82,smoothstep(.45,.98,hydro));const channelSpacing=lerp(maxSpacing,minSpacing*1.02,Math.pow(channelT,1.05));spacing=Math.min(spacing,channelSpacing)}else if(floodplain[i]){const floodT=Math.max(.62,smoothstep(.45,.95,hydro));const floodSpacing=lerp(maxSpacing,minSpacing*1.28,Math.pow(floodT,1.15));spacing=Math.min(spacing,floodSpacing)}else if(hydro>=.9&&terrain>=.35){const hydroSpacing=lerp(maxSpacing,minSpacing*1.6,Math.pow(hydro,1.35));spacing=Math.min(spacing,hydroSpacing)}}if(terrain>=.82){spacing=Math.min(spacing,lerp(maxSpacing,minSpacing,.92))}else if(terrain>=.68){spacing=Math.min(spacing,lerp(maxSpacing,minSpacing,.8))}else if(terrain>=.52){spacing=Math.min(spacing,lerp(maxSpacing,minSpacing,.64))}out[i]=clamp(spacing,minSpacing,maxSpacing)}return out}buildTopology(analysis){const accepted=[];const acceptedMask=new Uint8Array(analysis.width*analysis.height);const hashCellSize=Math.max(analysis.cellSize,this.options.minSpacing*.5,this.options.maxSpacing*.25);const minWorldX=this.options.worldMinX;const minWorldY=this.options.worldMinY;const maxWorldX=minWorldX+Math.max(0,analysis.width-1)*analysis.cellSize;const maxWorldY=minWorldY+Math.max(0,analysis.height-1)*analysis.cellSize;const bucketWidth=Math.max(1,Math.floor((maxWorldX-minWorldX)/hashCellSize)+1);const bucketHeight=Math.max(1,Math.floor((maxWorldY-minWorldY)/hashCellSize)+1);const spatial=Array.from({length:bucketWidth*bucketHeight},()=>[]);const tryAccept=(gx,gy,radius,fixed)=>{if(gx<0||gy<0||gx>=analysis.width||gy>=analysis.height)return false;const cellIndex=indexOfCell(gx,gy,analysis.width);if(!analysis.mask[cellIndex])return false;if(acceptedMask[cellIndex])return false;const x=this.options.worldMinX+gx*analysis.cellSize;const y=this.options.worldMinY+gy*analysis.cellSize;const hx=clamp(Math.floor((x-minWorldX)/hashCellSize),0,bucketWidth-1);const hy=clamp(Math.floor((y-minWorldY)/hashCellSize),0,bucketHeight-1);const queryRadius=Math.max(radius,hashCellSize*.75);const ring=Math.max(1,Math.ceil(queryRadius/hashCellSize));const minHX=Math.max(0,hx-ring);const maxHX=Math.min(bucketWidth-1,hx+ring);const minHY=Math.max(0,hy-ring);const maxHY=Math.min(bucketHeight-1,hy+ring);for(let by=minHY;by<=maxHY;by++){const rowBase=by*bucketWidth;for(let bx=minHX;bx<=maxHX;bx++){const bucket=spatial[rowBase+bx];for(let i=0;i<bucket.length;i++){const other=accepted[bucket[i]];const dx=x-other.x;const dy=y-other.y;const minDist=Math.max(radius,other.radius);if(dx*dx+dy*dy<minDist*minDist){return false}}}}const pointIndex=accepted.length;accepted.push({x,y,gx,gy,radius,fixed});acceptedMask[cellIndex]=1;spatial[hy*bucketWidth+hx].push(pointIndex);return true};this.emitForcedPoints(analysis,tryAccept);this.emitAdaptivePoints(analysis,tryAccept);if(accepted.length<3){throw new Error("Adaptive sampling retained fewer than 3 valid points.")}if(this.options.relaxIterations>0){this.relaxAcceptedPoints(accepted,analysis)}const pointCount=accepted.length;const positions2D=new Float32Array(pointCount*2);const positionsGrid=new Float32Array(pointCount*2);const uvs=new Float32Array(pointCount*2);const vertexImportance=new Float32Array(pointCount);const vertexWetness=new Float32Array(pointCount);const vertexBoundaryMask=new Uint8Array(pointCount);for(let i=0;i<pointCount;i++){const p=accepted[i];const j=i*2;positions2D[j]=p.x;positions2D[j+1]=p.y;positionsGrid[j]=p.gx;positionsGrid[j+1]=p.gy;uvs[j]=analysis.width>1?p.gx/(analysis.width-1):0;uvs[j+1]=analysis.height>1?p.gy/(analysis.height-1):0;const gx=clamp(Math.round(p.gx),0,analysis.width-1);const gy=clamp(Math.round(p.gy),0,analysis.height-1);const cellIndex=indexOfCell(gx,gy,analysis.width);vertexImportance[i]=analysis.importance[cellIndex];vertexWetness[i]=analysis.wetness[cellIndex];vertexBoundaryMask[i]=isBoundaryCell(analysis.mask,analysis.width,analysis.height,gx,gy)?1:0}const triangulationCoords=new Float64Array(pointCount*2);for(let i=0;i<positions2D.length;i++){triangulationCoords[i]=positions2D[i]}const delaunay=new Delaunator(triangulationCoords);const filtered=this.filterTrianglesToMask(delaunay.triangles,positionsGrid,analysis);const halfedges=this.buildHalfedges(filtered);const{centers,areas}=this.buildTriangleGeometry(filtered,positions2D);const{offsets,neighbors}=this.buildVertexAdjacency(filtered,pointCount);const heights=this.sampleVertexHeights(positionsGrid);return{positions2D,positionsGrid,uvs,heights,indices:filtered,halfedges,triangleCenters:centers,triangleAreas:areas,vertexNeighborOffsets:offsets,vertexNeighbors:neighbors,vertexBoundaryMask,vertexImportance,vertexWetness,analysis}}emitForcedPoints(analysis,tryAccept){const minSpacing=Math.max(this.options.minSpacing,analysis.cellSize);const maxSpacing=Math.max(this.options.maxSpacing,minSpacing);const borderStepCells=Math.max(1,Math.round(this.options.borderSpacing/analysis.cellSize));const boundaryStepCells=Math.max(1,Math.round(maxSpacing/analysis.cellSize));const channelStepCells=Math.max(2,Math.round(minSpacing*2.25/analysis.cellSize));const innerRingOffset=Math.max(1,Math.round(maxSpacing/analysis.cellSize));const borderRadius=Math.max(maxSpacing*.5,analysis.cellSize);const boundaryRadius=Math.max(maxSpacing*.5,analysis.cellSize);const channelRadius=Math.max(minSpacing*1,analysis.cellSize);const forcedPointRadius=Math.max(minSpacing*.35,analysis.cellSize*.5);if(this.options.preserveBorder){for(let x=0;x<analysis.width;x+=borderStepCells){tryAccept(x,0,borderRadius,true);tryAccept(x,analysis.height-1,borderRadius,true)}for(let y=0;y<analysis.height;y+=borderStepCells){tryAccept(0,y,borderRadius,true);tryAccept(analysis.width-1,y,borderRadius,true)}tryAccept(0,0,borderRadius,true);tryAccept(analysis.width-1,0,borderRadius,true);tryAccept(0,analysis.height-1,borderRadius,true);tryAccept(analysis.width-1,analysis.height-1,borderRadius,true);if(this.options.preserveInnerBorder&&analysis.width>2&&analysis.height>2){const left=clamp(innerRingOffset,1,analysis.width-2);const right=clamp(analysis.width-1-innerRingOffset,1,analysis.width-2);const top=clamp(innerRingOffset,1,analysis.height-2);const bottom=clamp(analysis.height-1-innerRingOffset,1,analysis.height-2);for(let x=left;x<=right;x+=boundaryStepCells){tryAccept(x,top,boundaryRadius,true);tryAccept(x,bottom,boundaryRadius,true)}for(let y=top;y<=bottom;y+=boundaryStepCells){tryAccept(left,y,boundaryRadius,true);tryAccept(right,y,boundaryRadius,true)}}}if(this.options.preserveMaskBoundary){for(let y=0;y<analysis.height;y++){for(let x=0;x<analysis.width;x++){if(!isBoundaryCell(analysis.mask,analysis.width,analysis.height,x,y))continue;const keep=x===0||y===0||x===analysis.width-1||y===analysis.height-1||x%boundaryStepCells===0||y%boundaryStepCells===0;if(keep){tryAccept(x,y,boundaryRadius,true)}}}}if(this.options.preserveChannels){for(let y=0;y<analysis.height;y++){for(let x=0;x<analysis.width;x++){const i=indexOfCell(x,y,analysis.width);if(!analysis.channels[i])continue;const neighborCount=this.countMarkedNeighbors(analysis.channels,analysis.width,analysis.height,x,y);const isEndpoint=neighborCount<=1;const isJunction=neighborCount>=3;const isSampledRun=x%channelStepCells===0&&y%channelStepCells===0||(x+y)%channelStepCells===0&&neighborCount===2;if(isEndpoint||isJunction||isSampledRun){tryAccept(x,y,channelRadius,true)}}}}const forcedWorldPoints=this.options.forcedWorldPoints;if(forcedWorldPoints&&forcedWorldPoints.length>=2){for(let i=0;i+1<forcedWorldPoints.length;i+=2){const x=Number(forcedWorldPoints[i]);const y=Number(forcedWorldPoints[i+1]);if(!isFiniteNumber(x)||!isFiniteNumber(y))continue;const gx=Math.round((x-this.options.worldMinX)/analysis.cellSize);const gy=Math.round((y-this.options.worldMinY)/analysis.cellSize);tryAccept(gx,gy,forcedPointRadius,true)}}}countMarkedNeighbors(field,width,height,x,y){let count=0;for(let oy=-1;oy<=1;oy++){for(let ox=-1;ox<=1;ox++){if(ox===0&&oy===0)continue;const nx=x+ox;const ny=y+oy;if(nx<0||ny<0||nx>=width||ny>=height)continue;if(field[indexOfCell(nx,ny,width)]!==0){count++}}}return count}emitAdaptivePoints(analysis,tryAccept){const stride=this.options.candidateStride;const terrainBuckets=Array.from({length:DEFAULT_BUCKET_COUNT},()=>[]);const hydroBuckets=Array.from({length:DEFAULT_BUCKET_COUNT},()=>[]);const blendBuckets=Array.from({length:DEFAULT_BUCKET_COUNT},()=>[]);for(let y=0;y<analysis.height;y+=stride){for(let x=0;x<analysis.width;x+=stride){const i=indexOfCell(x,y,analysis.width);if(!analysis.mask[i])continue;const terrain=clamp(analysis.terrainPriority[i],0,1);const hydro=clamp(analysis.hydroPriority[i],0,1);const blend=clamp(Math.max(terrain,hydro*.45),0,1);const tb=clamp(Math.floor(Math.pow(terrain,.85)*(DEFAULT_BUCKET_COUNT-1)),0,DEFAULT_BUCKET_COUNT-1);const hb=clamp(Math.floor(Math.pow(hydro,1.55)*(DEFAULT_BUCKET_COUNT-1)),0,DEFAULT_BUCKET_COUNT-1);const bb=clamp(Math.floor(Math.pow(blend,.95)*(DEFAULT_BUCKET_COUNT-1)),0,DEFAULT_BUCKET_COUNT-1);terrainBuckets[tb].push(i);hydroBuckets[hb].push(i);blendBuckets[bb].push(i)}}const acceptedFlags=new Uint8Array(analysis.width*analysis.height);let acceptedCount=0;const processBucketSet=(buckets,radiusScale,allowHydroBoost,hardLimit)=>{for(let bucketIndex=DEFAULT_BUCKET_COUNT-1;bucketIndex>=0;bucketIndex--){const bucket=buckets[bucketIndex];for(let k=0;k<bucket.length;k++){if(acceptedCount>=this.options.maxPoints||acceptedCount>=hardLimit)return;const i=bucket[k];if(acceptedFlags[i])continue;const x=i%analysis.width;const y=i/analysis.width|0;const spacing=analysis.spacing[i];const terrain=clamp(analysis.terrainPriority[i],0,1);const hydro=clamp(analysis.hydroPriority[i],0,1);let radius=Math.max(spacing*radiusScale,analysis.cellSize*.7);if(terrain>=.85){radius*=.64}else if(terrain>=.7){radius*=.74}else if(terrain>=.55){radius*=.84}if(allowHydroBoost&&(analysis.channels[i]||analysis.floodplain[i])){radius*=.95}radius=quantizeRadius(radius,Math.max(analysis.cellSize*.7,this.options.minSpacing*.32),Math.max(this.options.minSpacing,this.options.maxSpacing*.62),16);if(tryAccept(x,y,radius,false)){acceptedFlags[i]=1;acceptedCount++}}}};const maxPoints=this.options.maxPoints;const terrainLimit=Math.floor(maxPoints*.82);const hydroLimit=Math.floor(maxPoints*.92);processBucketSet(terrainBuckets,.54,false,terrainLimit);processBucketSet(blendBuckets,.56,false,hydroLimit);processBucketSet(hydroBuckets,.6,true,maxPoints)}relaxAcceptedPoints(accepted,analysis){if(accepted.length<3)return;for(let iter=0;iter<this.options.relaxIterations;iter++){const positions=new Float64Array(accepted.length*2);for(let i=0;i<accepted.length;i++){positions[i*2]=accepted[i].x;positions[i*2+1]=accepted[i].y}const delaunay=new Delaunator(positions);const neighborSumsX=new Float64Array(accepted.length);const neighborSumsY=new Float64Array(accepted.length);const neighborCounts=new Uint32Array(accepted.length);for(let i=0;i<delaunay.triangles.length;i+=3){const a=delaunay.triangles[i];const b=delaunay.triangles[i+1];const c=delaunay.triangles[i+2];this.accumulateNeighbor(a,b,accepted,neighborSumsX,neighborSumsY,neighborCounts);this.accumulateNeighbor(a,c,accepted,neighborSumsX,neighborSumsY,neighborCounts);this.accumulateNeighbor(b,a,accepted,neighborSumsX,neighborSumsY,neighborCounts);this.accumulateNeighbor(b,c,accepted,neighborSumsX,neighborSumsY,neighborCounts);this.accumulateNeighbor(c,a,accepted,neighborSumsX,neighborSumsY,neighborCounts);this.accumulateNeighbor(c,b,accepted,neighborSumsX,neighborSumsY,neighborCounts)}for(let i=0;i<accepted.length;i++){if(accepted[i].fixed)continue;if(neighborCounts[i]===0)continue;const targetX=neighborSumsX[i]/neighborCounts[i];const targetY=neighborSumsY[i]/neighborCounts[i];const gx=clamp(Math.round((targetX-this.options.worldMinX)/analysis.cellSize),0,analysis.width-1);const gy=clamp(Math.round((targetY-this.options.worldMinY)/analysis.cellSize),0,analysis.height-1);if(analysis.mask[indexOfCell(gx,gy,analysis.width)]){accepted[i].gx=gx;accepted[i].gy=gy;accepted[i].x=this.options.worldMinX+gx*analysis.cellSize;accepted[i].y=this.options.worldMinY+gy*analysis.cellSize}}}}accumulateNeighbor(a,b,accepted,sumX,sumY,counts){sumX[a]+=accepted[b].x;sumY[a]+=accepted[b].y;counts[a]++}filterTrianglesToMask(triangles,positionsGrid,analysis){const kept=[];for(let i=0;i<triangles.length;i+=3){const a=triangles[i];const b=triangles[i+1];const c=triangles[i+2];const a2=a*2;const b2=b*2;const c2=c*2;const cx=(positionsGrid[a2]+positionsGrid[b2]+positionsGrid[c2])/3;const cy=(positionsGrid[a2+1]+positionsGrid[b2+1]+positionsGrid[c2+1])/3;if(!nearestMaskSample(analysis.mask,analysis.width,analysis.height,cx,cy)){continue}kept.push(a,b,c)}return Uint32Array.from(kept)}buildHalfedges(indices){const halfedges=new Int32Array(indices.length);halfedges.fill(-1);const edgeMap=new Map;for(let i=0;i<indices.length;i+=3){for(let e=0;e<3;e++){const a=indices[i+e];const b=indices[i+(e+1)%3];const key=`${a}:${b}`;const rev=`${b}:${a}`;if(edgeMap.has(rev)){const opposite=edgeMap.get(rev);halfedges[i+e]=opposite;halfedges[opposite]=i+e}else{edgeMap.set(key,i+e)}}}return halfedges}buildTriangleGeometry(indices,positions2D){const triCount=indices.length/3;const centers=new Float32Array(triCount*2);const areas=new Float32Array(triCount);for(let t=0;t<triCount;t++){const i0=indices[t*3]*2;const i1=indices[t*3+1]*2;const i2=indices[t*3+2]*2;const ax=positions2D[i0];const ay=positions2D[i0+1];const bx=positions2D[i1];const by=positions2D[i1+1];const cx=positions2D[i2];const cy=positions2D[i2+1];centers[t*2]=(ax+bx+cx)/3;centers[t*2+1]=(ay+by+cy)/3;const twiceArea=Math.abs((bx-ax)*(cy-ay)-(by-ay)*(cx-ax));areas[t]=.5*twiceArea}return{centers,areas}}buildVertexAdjacency(indices,vertexCount){const sets=Array.from({length:vertexCount},()=>new Set);for(let i=0;i<indices.length;i+=3){const a=indices[i];const b=indices[i+1];const c=indices[i+2];sets[a].add(b);sets[a].add(c);sets[b].add(a);sets[b].add(c);sets[c].add(a);sets[c].add(b)}const offsets=new Uint32Array(vertexCount+1);let total=0;for(let i=0;i<vertexCount;i++){offsets[i]=total;total+=sets[i].size}offsets[vertexCount]=total;const neighbors=new Uint32Array(total);let write=0;for(let i=0;i<vertexCount;i++){const list=Array.from(sets[i]);list.sort((a,b)=>a-b);for(let j=0;j<list.length;j++){neighbors[write++]=list[j]}}return{offsets,neighbors}}sampleVertexHeights(positionsGrid){const heights=new Float32Array(positionsGrid.length>>1);for(let i=0,j=0;i<heights.length;i++,j+=2){heights[i]=bilinearSample(this.heights,this.width,this.height,positionsGrid[j],positionsGrid[j+1])}return heights}};var EPSILON3=1e-6;var INF=1e30;function clamp2(value,min,max){return value<min?min:value>max?max:value}function distance2D(ax,ay,bx,by){const dx=bx-ax;const dy=by-ay;return Math.hypot(dx,dy)}function sampleGridField(field,width,height,x,y){const gx=clamp2(Math.round(x),0,width-1);const gy=clamp2(Math.round(y),0,height-1);return field[gy*width+gx]}var MinHeap=class{ids;keys;size;constructor(capacity){this.ids=new Int32Array(capacity);this.keys=new Float64Array(capacity);this.size=0}push(id,key){let i=this.size++;this.ids[i]=id;this.keys[i]=key;while(i>0){const p=i-1>>1;if(this.keys[p]<=key)break;this.ids[i]=this.ids[p];this.keys[i]=this.keys[p];i=p}this.ids[i]=id;this.keys[i]=key}pop(){if(this.size===0)return null;const rootId=this.ids[0];const rootKey=this.keys[0];this.size--;if(this.size>0){const lastId=this.ids[this.size];const lastKey=this.keys[this.size];let i=0;while(true){let c=i*2+1;if(c>=this.size)break;if(c+1<this.size&&this.keys[c+1]<this.keys[c])c++;if(this.keys[c]>=lastKey)break;this.ids[i]=this.ids[c];this.keys[i]=this.keys[c];i=c}this.ids[i]=lastId;this.keys[i]=lastKey}return{id:rootId,key:rootKey}}};var RiverNetworkBuilder=class{options;constructor(options={}){this.options={maxNeighborDistance:Math.max(options.maxNeighborDistance??Number.POSITIVE_INFINITY,0),boundaryEpsilon:Math.max(options.boundaryEpsilon??.5,0),minDrop:Math.max(options.minDrop??1e-5,0),flatTolerance:Math.max(options.flatTolerance??1e-4,0),sinkMaxRise:Math.max(options.sinkMaxRise??.02,0),sinkClimbPenalty:Math.max(options.sinkClimbPenalty??2.5,0)}}buildFromTopology(topology){const vertexCount=topology.heights.length;if(vertexCount===0){return{edges:[],downstream:new Int32Array(0),edgeIndexBySource:new Int32Array(0),vertexFlowWeight:new Float32Array(0),upstreamCount:new Uint32Array(0),boundaryMask:new Uint8Array(0),maxWeight:0,minWeight:0}}const neighbors=this.buildNeighbors(topology);const boundaryMask=this.computeBoundaryMask(topology.positionsGrid,topology.vertexBoundaryMask);const downstream=this.chooseDownstream(topology,neighbors,boundaryMask);const{edges,edgeIndexBySource,vertexFlowWeight,upstreamCount,minWeight,maxWeight}=this.accumulateFlow(topology,downstream);return{edges,downstream,edgeIndexBySource,vertexFlowWeight,upstreamCount,boundaryMask,minWeight,maxWeight}}buildSegments(topology,result,minWeight){const mask=this.collectActiveEdgeMask(result,minWeight);return this.buildSegmentsFromMask(topology,result,mask)}buildSegmentsFromMask(topology,result,mask){let activeCount=0;for(let i=0;i<mask.length;i++){if(mask[i])activeCount++}const positions=topology.positionsGrid;const segments=new Float32Array(activeCount*4);let write=0;for(let i=0;i<result.edges.length;i++){if(!mask[i])continue;const edge=result.edges[i];const a2=edge.source*2;const b2=edge.target*2;segments[write++]=positions[a2];segments[write++]=positions[a2+1];segments[write++]=positions[b2];segments[write++]=positions[b2+1]}return segments}buildActiveArtifacts(topology,result,minWeight){const mask=this.collectActiveEdgeMask(result,minWeight);const vertexActiveMask=new Uint8Array(result.vertexFlowWeight.length);let activeVertexCount=0;let activeEdgeCount=0;for(let i=0;i<result.edges.length;i++){if(!mask[i])continue;activeEdgeCount++;const edge=result.edges[i];if(!vertexActiveMask[edge.source]){vertexActiveMask[edge.source]=1;activeVertexCount++}if(!vertexActiveMask[edge.target]){vertexActiveMask[edge.target]=1;activeVertexCount++}}const activeEdges=new Array(activeEdgeCount);const segments=new Float32Array(activeEdgeCount*4);const positions=topology.positionsGrid;let edgeWrite=0;let segmentWrite=0;for(let i=0;i<result.edges.length;i++){if(!mask[i])continue;const edge=result.edges[i];activeEdges[edgeWrite++]=edge;const a2=edge.source*2;const b2=edge.target*2;segments[segmentWrite++]=positions[a2];segments[segmentWrite++]=positions[a2+1];segments[segmentWrite++]=positions[b2];segments[segmentWrite++]=positions[b2+1]}const activeVertices=new Uint32Array(activeVertexCount);let vertexWrite=0;for(let i=0;i<vertexActiveMask.length;i++){if(vertexActiveMask[i]){activeVertices[vertexWrite++]=i}}return{mask,activeEdges,activeVertices,segments}}collectActiveEdges(result,minWeight){const mask=this.collectActiveEdgeMask(result,minWeight);const out=[];for(let i=0;i<result.edges.length;i++){if(mask[i])out.push(result.edges[i])}return out}collectActiveVertices(result,minWeight){const mask=this.collectActiveEdgeMask(result,minWeight);return this.collectActiveVerticesFromMask(result,mask)}collectActiveVerticesFromMask(result,mask){const activeMask=new Uint8Array(result.vertexFlowWeight.length);let count=0;for(let i=0;i<result.edges.length;i++){if(!mask[i])continue;const edge=result.edges[i];if(!activeMask[edge.source]){activeMask[edge.source]=1;count++}if(!activeMask[edge.target]){activeMask[edge.target]=1;count++}}const out=new Uint32Array(count);let write=0;for(let i=0;i<activeMask.length;i++){if(activeMask[i])out[write++]=i}return out}buildGuidePoints(topology,result,options){const minWeight=Math.max(options.minWeight,0);const step=Math.max(options.step,.5);const maxPoints=Math.max(0,Math.floor(options.maxPoints??6e4));const activeEdges=this.collectActiveEdges(result,minWeight);const positions=topology.positionsGrid;const points=[];let minX=Infinity;let minY=Infinity;let maxX=-Infinity;let maxY=-Infinity;for(let i=0;i<positions.length;i+=2){const x=positions[i];const y=positions[i+1];if(x<minX)minX=x;if(y<minY)minY=y;if(x>maxX)maxX=x;if(y>maxY)maxY=y}const qScale=4;const quantWidth=Math.max(1,Math.floor((maxX-minX)*qScale)+3);const seen=new Set;const pushPoint=(x,y)=>{const qx=Math.round((x-minX)*qScale);const qy=Math.round((y-minY)*qScale);const key=qy*quantWidth+qx;if(seen.has(key))return;seen.add(key);points.push(x,y)};for(let i=0;i<activeEdges.length;i++){if(points.length>>1>=maxPoints)break;const edge=activeEdges[i];const a2=edge.source*2;const b2=edge.target*2;const ax=positions[a2];const ay=positions[a2+1];const bx=positions[b2];const by=positions[b2+1];const len=Math.max(distance2D(ax,ay,bx,by),EPSILON3);const steps=Math.max(1,Math.ceil(len/step));for(let s=0;s<=steps;s++){if(points.length>>1>=maxPoints)break;const t=s/steps;pushPoint(ax+(bx-ax)*t,ay+(by-ay)*t)}}return Float32Array.from(points)}buildNeighbors(topology){const vertexCount=topology.heights.length;const neighbors=Array.from({length:vertexCount},()=>[]);const positions=topology.positions2D;const offsets=topology.vertexNeighborOffsets;const verts=topology.vertexNeighbors;const maxDistance=this.options.maxNeighborDistance;for(let i=0;i<vertexCount;i++){for(let p=offsets[i];p<offsets[i+1];p++){const j=verts[p];if(j===i)continue;const a2=i*2;const b2=j*2;const d=distance2D(positions[a2],positions[a2+1],positions[b2],positions[b2+1]);if(d<=maxDistance+EPSILON3){neighbors[i].push(j)}}}return neighbors.map(list=>Uint32Array.from(list))}computeBoundaryMask(positionsGrid,provided){if(provided&&provided.length===positionsGrid.length>>1){return new Uint8Array(provided)}const vertexCount=positionsGrid.length>>1;const boundaryMask=new Uint8Array(vertexCount);let minX=Infinity;let minY=Infinity;let maxX=-Infinity;let maxY=-Infinity;for(let i=0;i<positionsGrid.length;i+=2){const x=positionsGrid[i];const y=positionsGrid[i+1];if(x<minX)minX=x;if(y<minY)minY=y;if(x>maxX)maxX=x;if(y>maxY)maxY=y}const eps=this.options.boundaryEpsilon;for(let i=0;i<vertexCount;i++){const x=positionsGrid[i*2];const y=positionsGrid[i*2+1];if(x<=minX+eps||x>=maxX-eps||y<=minY+eps||y>=maxY-eps){boundaryMask[i]=1}}return boundaryMask}chooseDownstream(topology,neighbors,boundaryMask){const vertexCount=topology.heights.length;const downstream=new Int32Array(vertexCount);downstream.fill(-1);const heights=topology.heights;const positions2D=topology.positions2D;const positionsGrid=topology.positionsGrid;const wetness=topology.vertexWetness;const analysis=topology.analysis;const width=analysis.width;const height=analysis.height;const flowAccumulation=analysis.flowAccumulation;const flowReceiver=analysis.flowReceiver;for(let i=0;i<vertexCount;i++){const a2=i*2;const gx=positionsGrid[a2];const gy=positionsGrid[a2+1];const h=heights[i];const localAccum=sampleGridField(flowAccumulation,width,height,gx,gy);const receiverCell=sampleGridField(flowReceiver,width,height,gx,gy);const rx=receiverCell>=0?receiverCell%width:-1;const ry=receiverCell>=0?Math.floor(receiverCell/width):-1;const rvx=rx>=0?rx-gx:0;const rvy=ry>=0?ry-gy:0;const rvLen=Math.hypot(rvx,rvy);let best=-1;let bestScore=-INF;for(let k=0;k<neighbors[i].length;k++){const j=neighbors[i][k];const j2=j*2;const nh=heights[j];const drop=h-nh;const downhill=drop>this.options.minDrop;const flat=nh<=h+this.options.flatTolerance&&j>i;if(!downhill&&!flat)continue;const dx=positions2D[j2]-positions2D[a2];const dy=positions2D[j2+1]-positions2D[a2+1];const dist=Math.max(Math.hypot(dx,dy),EPSILON3);const jgx=positionsGrid[j2];const jgy=positionsGrid[j2+1];const nbrAccum=sampleGridField(flowAccumulation,width,height,jgx,jgy);let alignment=0;if(rvLen>EPSILON3){alignment=(dx*rvx+dy*rvy)/(dist*rvLen)}const slopeTerm=downhill?drop/dist*6:.05/dist;const accumTerm=Math.log1p(Math.max(0,nbrAccum-localAccum))*.12;const wetnessTerm=wetness[j]*.035;const alignTerm=Math.max(0,alignment)*.7;const boundaryTerm=boundaryMask[j]?.08:0;const lowerTerm=downhill?.08:0;const score=slopeTerm+accumTerm+wetnessTerm+alignTerm+boundaryTerm+lowerTerm;if(score>bestScore){bestScore=score;best=j}}downstream[i]=best}const escape=this.computeEscapeDownstream(topology,neighbors,boundaryMask);for(let i=0;i<vertexCount;i++){if(downstream[i]<0&&!boundaryMask[i]){downstream[i]=escape[i]}}return downstream}computeEscapeDownstream(topology,neighbors,boundaryMask){const vertexCount=topology.heights.length;const bestNext=new Int32Array(vertexCount);bestNext.fill(-1);const bestCost=new Float64Array(vertexCount);bestCost.fill(INF);const heights=topology.heights;const positions2D=topology.positions2D;const positionsGrid=topology.positionsGrid;const analysis=topology.analysis;const width=analysis.width;const height=analysis.height;const flowAccumulation=analysis.flowAccumulation;const flowReceiver=analysis.flowReceiver;const maxRise=Math.max(this.options.sinkMaxRise,this.options.flatTolerance*64);const climbPenalty=this.options.sinkClimbPenalty;const heap=new MinHeap(vertexCount*4+16);for(let i=0;i<vertexCount;i++){if(boundaryMask[i]){bestCost[i]=0;heap.push(i,0)}}while(heap.size>0){const item=heap.pop();if(!item)break;const u=item.id;const costU=item.key;if(costU>bestCost[u]+EPSILON3)continue;for(let p=0;p<neighbors[u].length;p++){const v=neighbors[u][p];const stepCost=this.transitionCost(v,u,topology,flowAccumulation,flowReceiver,width,height,maxRise,climbPenalty);if(!isFinite(stepCost))continue;const candidate=costU+stepCost;if(candidate+EPSILON3<bestCost[v]){bestCost[v]=candidate;bestNext[v]=u;heap.push(v,candidate)}}}return bestNext}transitionCost(from,to,topology,flowAccumulation,flowReceiver,width,height,maxRise,climbPenalty){const heights=topology.heights;const positions2D=topology.positions2D;const positionsGrid=topology.positionsGrid;const a2=from*2;const b2=to*2;const ax=positions2D[a2];const ay=positions2D[a2+1];const bx=positions2D[b2];const by=positions2D[b2+1];const dist=Math.max(distance2D(ax,ay,bx,by),EPSILON3);const h0=heights[from];const h1=heights[to];const climb=Math.max(0,h1-h0);if(climb>maxRise+EPSILON3)return INF;const gx=positionsGrid[a2];const gy=positionsGrid[a2+1];const localAccum=sampleGridField(flowAccumulation,width,height,gx,gy);const receiverCell=sampleGridField(flowReceiver,width,height,gx,gy);const rx=receiverCell>=0?receiverCell%width:-1;const ry=receiverCell>=0?Math.floor(receiverCell/width):-1;const rvx=rx>=0?rx-gx:0;const rvy=ry>=0?ry-gy:0;const rvLen=Math.hypot(rvx,rvy);let alignment=0;if(rvLen>EPSILON3){const dx=positionsGrid[b2]-gx;const dy=positionsGrid[b2+1]-gy;alignment=(dx*rvx+dy*rvy)/(Math.max(Math.hypot(dx,dy),EPSILON3)*rvLen)}const jgx=positionsGrid[b2];const jgy=positionsGrid[b2+1];const nbrAccum=sampleGridField(flowAccumulation,width,height,jgx,jgy);const accumPenalty=nbrAccum+EPSILON3>=localAccum?0:.35*(1-nbrAccum/Math.max(localAccum,1));const alignPenalty=rvLen>EPSILON3?.5*(1-Math.max(-.25,alignment)):.2;const risePenalty=climbPenalty*(climb/Math.max(dist,EPSILON3));return dist*(1+alignPenalty+accumPenalty)+risePenalty}accumulateFlow(topology,downstream){const vertexCount=topology.heights.length;const vertexFlowWeight=new Float32Array(vertexCount);vertexFlowWeight.fill(1);const upstreamCount=new Uint32Array(vertexCount);const edgeIndexBySource=new Int32Array(vertexCount);edgeIndexBySource.fill(-1);const order=new Uint32Array(vertexCount);for(let i=0;i<vertexCount;i++)order[i]=i;const heights=topology.heights;const wetness=topology.vertexWetness;order.sort((a,b)=>{const dz=heights[b]-heights[a];if(Math.abs(dz)>EPSILON3)return dz;const dw=wetness[b]-wetness[a];if(Math.abs(dw)>EPSILON3)return dw;return b-a});for(let i=0;i<order.length;i++){const v=order[i];const d=downstream[v];if(d>=0&&d!==v){vertexFlowWeight[d]+=vertexFlowWeight[v];upstreamCount[d]++}}const edges=[];const positions=topology.positions2D;let minWeight=Infinity;let maxWeight=0;for(let v=0;v<vertexCount;v++){const d=downstream[v];if(d<0||d===v)continue;const a2=v*2;const b2=d*2;const dx=positions[b2]-positions[a2];const dy=positions[b2+1]-positions[a2+1];const length=Math.max(Math.hypot(dx,dy),EPSILON3);const weight=vertexFlowWeight[v];const edge={source:v,target:d,weight,length};edgeIndexBySource[v]=edges.length;edges.push(edge);if(weight<minWeight)minWeight=weight;if(weight>maxWeight)maxWeight=weight}if(!isFinite(minWeight))minWeight=0;return{edges,edgeIndexBySource,vertexFlowWeight,upstreamCount,minWeight,maxWeight}}collectActiveEdgeMask(result,minWeight){const threshold=Math.max(minWeight,0);const mask=new Uint8Array(result.edges.length);const vertexCount=result.vertexFlowWeight.length;for(let v=0;v<vertexCount;v++){if(result.vertexFlowWeight[v]+EPSILON3<threshold)continue;let current=v;let steps=0;while(current>=0&&steps<vertexCount){const edgeIndex=result.edgeIndexBySource[current];if(edgeIndex<0)break;if(mask[edgeIndex])break;mask[edgeIndex]=1;current=result.downstream[current];steps++}}return mask}};var EPSILON4=1e-6;var LARGE_VALUE=1e30;function clamp3(value,min,max){return value<min?min:value>max?max:value}function distance2D2(ax,ay,bx,by){const dx=bx-ax;const dy=by-ay;return Math.hypot(dx,dy)}var FloodplainClassifier=class{options;bestDistance=new Float32Array(0);channelRefHeight=new Float32Array(0);channelCeiling=new Float32Array(0);vertexMaskScratch=new Uint8Array(0);triangleMaskScratch=new Uint8Array(0);queue=new Int32Array(0);constructor(options={}){this.options={maxRise:Math.max(options.maxRise??.06,0),maxClimbStep:Math.max(options.maxClimbStep??.02,0),maxEdgeSlope:Math.max(options.maxEdgeSlope??.75,0),maxWorldDistance:Math.max(options.maxWorldDistance??Number.POSITIVE_INFINITY,0),minTriangleFloodplainVerts:clamp3(Math.floor(options.minTriangleFloodplainVerts??2),1,3)}}classify(topology,seedVerticesInput){const vertexCount=topology.heights.length;const triangleCount=topology.indices.length/3;this.ensureScratch(vertexCount,triangleCount);const positions=topology.positions2D;const heights=topology.heights;const offsets=topology.vertexNeighborOffsets;const neighbors=topology.vertexNeighbors;const seedVertices=seedVerticesInput instanceof Uint32Array?seedVerticesInput:Uint32Array.from(Array.from(seedVerticesInput));const vertexMask=this.vertexMaskScratch;const triangleMask=this.triangleMaskScratch;vertexMask.fill(0,0,vertexCount);triangleMask.fill(0,0,triangleCount);if(seedVertices.length===0){return{vertexMask:vertexMask.slice(0,vertexCount),triangleMask:triangleMask.slice(0,triangleCount),seedVertices}}const bestDistance=this.bestDistance;const channelRefHeight=this.channelRefHeight;const channelCeiling=this.channelCeiling;for(let i=0;i<vertexCount;i++){bestDistance[i]=LARGE_VALUE;channelRefHeight[i]=LARGE_VALUE;channelCeiling[i]=LARGE_VALUE}let head=0;let tail=0;for(let i=0;i<seedVertices.length;i++){const v=seedVertices[i];const refHeight=heights[v];vertexMask[v]=1;bestDistance[v]=0;channelRefHeight[v]=refHeight;channelCeiling[v]=refHeight+this.options.maxRise;this.queue[tail++]=v}while(head<tail){const vertex=this.queue[head++];const vx=positions[vertex*2];const vy=positions[vertex*2+1];const vh=heights[vertex];const activeDistance=bestDistance[vertex];const activeRefHeight=channelRefHeight[vertex];const activeCeiling=channelCeiling[vertex];for(let p=offsets[vertex];p<offsets[vertex+1];p++){const nbr=neighbors[p];const nx=positions[nbr*2];const ny=positions[nbr*2+1];const nh=heights[nbr];const edgeLen=Math.max(distance2D2(vx,vy,nx,ny),EPSILON4);const nextDistance=activeDistance+edgeLen;const localSlope=Math.abs(nh-vh)/edgeLen;const climb=nh-vh;if(nextDistance>this.options.maxWorldDistance+EPSILON4)continue;if(nh>activeCeiling+EPSILON4)continue;if(nh>activeRefHeight+this.options.maxRise+EPSILON4)continue;if(climb>this.options.maxClimbStep+EPSILON4)continue;if(localSlope>this.options.maxEdgeSlope+EPSILON4)continue;const improvesDistance=nextDistance<bestDistance[nbr]-EPSILON4;const improvesRef=activeRefHeight<channelRefHeight[nbr]-EPSILON4;if(!vertexMask[nbr]||improvesDistance||improvesRef){vertexMask[nbr]=1;bestDistance[nbr]=Math.min(bestDistance[nbr],nextDistance);channelRefHeight[nbr]=Math.min(channelRefHeight[nbr],activeRefHeight);channelCeiling[nbr]=Math.min(channelCeiling[nbr],activeCeiling);this.queue[tail++]=nbr}}}for(let triIndex=0,t=0;triIndex<triangleCount;triIndex++,t+=3){const a=topology.indices[t];const b=topology.indices[t+1];const c=topology.indices[t+2];const count=vertexMask[a]+vertexMask[b]+vertexMask[c];if(count>=this.options.minTriangleFloodplainVerts){triangleMask[triIndex]=1}}return{vertexMask:vertexMask.slice(0,vertexCount),triangleMask:triangleMask.slice(0,triangleCount),seedVertices}}ensureScratch(vertexCount,triangleCount){if(this.bestDistance.length<vertexCount){this.bestDistance=new Float32Array(vertexCount);this.channelRefHeight=new Float32Array(vertexCount);this.channelCeiling=new Float32Array(vertexCount);this.vertexMaskScratch=new Uint8Array(vertexCount);this.queue=new Int32Array(vertexCount)}if(this.triangleMaskScratch.length<triangleCount){this.triangleMaskScratch=new Uint8Array(triangleCount)}}};var DEBUG_LOGGING=false;var WORKER_NAME="Topology worker";var cachedRaster=null;var cachedBuilder=null;var cachedBuildOptionsKey="";var cachedTopology=null;var cachedRiverBuilder=null;var cachedRiverResult=null;var cachedFloodplainClassifier=null;self.addEventListener("error",event=>{logError("global error",{message:event?.message||"Unknown worker error",filename:event?.filename||null,lineno:event?.lineno||null,colno:event?.colno||null})});self.addEventListener("unhandledrejection",event=>{const reason=event?.reason instanceof Error?event.reason.stack||event.reason.message:String(event?.reason);logError("unhandled rejection",{reason})});function emitWorkerLog(level,...args){if(!DEBUG_LOGGING)return;const fn=typeof console[level]==="function"?console[level]:console.log;fn.call(console,"[TopologyWorker]",...args)}function log(...args){emitWorkerLog("log",...args)}function logError(...args){emitWorkerLog("error",...args)}function summarizeTopology(topology){if(!topology)return null;return{vertices:topology.positionsGrid?topology.positionsGrid.length>>1:0,triangles:topology.indices?Math.floor(topology.indices.length/3):0,width:topology.analysis?.width??0,height:topology.analysis?.height??0}}function summarizeRiverResult(result){if(!result)return null;return{edgeCount:result.edges?result.edges.length:0,minWeight:result.minWeight??0,maxWeight:result.maxWeight??0}}function summarizeView(view){if(!view)return null;return{segmentCount:view.segments?view.segments.length>>2:0,activeVertexCount:view.activeVertexCount??0,activeEdgeCount:view.activeEdgeCount??0,floodplainSeedCount:view.floodplainResult?.seedVertices?view.floodplainResult.seedVertices.length:0}}function cloneTypedArray(array){return new array.constructor(array)}function collectUiTopologyTransferables(topology){return[topology.positionsGrid.buffer,topology.indices.buffer,topology.analysis.mask.buffer,topology.analysis.importance.buffer]}function collectHydroTransferables(view){const transferables=[];if(view.segments)transferables.push(view.segments.buffer);if(view.floodplainResult){transferables.push(view.floodplainResult.vertexMask.buffer,view.floodplainResult.triangleMask.buffer,view.floodplainResult.seedVertices.buffer)}return transferables}function buildOptionsKey(buildOptions){return JSON.stringify(buildOptions)}function resetDerivedCaches(){cachedTopology=null;cachedRiverBuilder=null;cachedRiverResult=null;cachedFloodplainClassifier=null}function makeRiverBuilder(options){cachedRiverBuilder=new RiverNetworkBuilder(options);return cachedRiverBuilder}function makeFloodplainClassifier(options){cachedFloodplainClassifier=new FloodplainClassifier(options);return cachedFloodplainClassifier}function scanRange(values,mask=null){let min=Infinity;let max=-Infinity;for(let i=0;i<values.length;i++){if(mask&&!mask[i])continue;const v=values[i];if(!Number.isFinite(v))continue;if(v<min)min=v;if(v>max)max=v}if(!Number.isFinite(min)||!Number.isFinite(max)){return{min:null,max:null}}return{min,max}}function maxVertexDegree(offsets){let max=0;for(let i=0;i+1<offsets.length;i++){const degree=offsets[i+1]-offsets[i];if(degree>max)max=degree}return max}function makeUiTopology(topology){const analysis=topology.analysis;return{positionsGrid:cloneTypedArray(topology.positionsGrid),indices:cloneTypedArray(topology.indices),analysis:{mask:cloneTypedArray(analysis.mask),importance:cloneTypedArray(analysis.importance)},statsSummary:{validCellCount:analysis.validCellCount,hillslopeLength:analysis.hillslopeLength,wetnessRange:scanRange(analysis.wetness,analysis.mask),importanceRange:scanRange(analysis.importance,analysis.mask),spacingRange:scanRange(analysis.spacing,analysis.mask),triangleAreaRange:scanRange(topology.triangleAreas),maxVertexDegree:maxVertexDegree(topology.vertexNeighborOffsets)}}}async function ensureBuilder(buildOptions){if(!cachedRaster)throw new Error("No raster in topology worker.");const key=buildOptionsKey(buildOptions);if(!cachedBuilder||cachedBuildOptionsKey!==key){log("creating builder",{keyChanged:cachedBuildOptionsKey!==key,width:cachedRaster.width,height:cachedRaster.height});cachedBuilder=new HeightfieldTINBuilder(buildOptions);cachedBuilder.setRaster(cachedRaster);cachedBuildOptionsKey=key;resetDerivedCaches()}else{log("reusing cached builder",{width:cachedRaster.width,height:cachedRaster.height})}return cachedBuilder}async function ensureFirstTopology(buildOptions){const builder=await ensureBuilder(buildOptions);if(cachedTopology){return{topology:cachedTopology,buildMs:0,fromCache:true}}const t0=performance.now();const topology=await builder.build();const buildMs=performance.now()-t0;cachedTopology=topology;return{topology,buildMs,fromCache:false}}function deriveHydroView(minRiverWeight,floodplainOptions){if(!cachedTopology||!cachedRiverBuilder||!cachedRiverResult||!cachedFloodplainClassifier){return{segments:new Float32Array(0),activeVertexCount:0,activeEdgeCount:0,riverStats:{totalEdgeCount:0,minWeight:0,maxWeight:0},floodplainResult:{vertexMask:new Uint8Array(0),triangleMask:new Uint8Array(0),seedVertices:new Uint32Array(0)},timings:{deriveMs:0}}}const t0=performance.now();const artifacts=cachedRiverBuilder.buildActiveArtifacts(cachedTopology,cachedRiverResult,minRiverWeight);const classifier=floodplainOptions?makeFloodplainClassifier(floodplainOptions):cachedFloodplainClassifier;const floodplainResult=classifier.classify(cachedTopology,artifacts.activeVertices);return{segments:artifacts.segments,activeVertexCount:artifacts.activeVertices.length,activeEdgeCount:artifacts.activeEdges.length,riverStats:{totalEdgeCount:cachedRiverResult.edges.length,minWeight:cachedRiverResult.minWeight,maxWeight:cachedRiverResult.maxWeight},floodplainResult,timings:{deriveMs:performance.now()-t0}}}log("module loaded",{href:typeof self.location!=="undefined"?self.location.href:null,worker:WORKER_NAME});self.onmessage=async event=>{const data=event&&typeof event.data==="object"&&event.data!==null?event.data:null;const id=data?.id;const type=data?.type;const payload=data?.payload;const startedAt=performance.now();if(typeof type!=="string"){log("ignoring non-request message",{dataType:typeof event?.data,hasData:!!data,keys:data?Object.keys(data):null});return}log("message received",{id,type});try{if(type==="ping"){self.postMessage({id,ok:true,payload:{ok:true,worker:"topology"}});log("ping complete",{id,elapsedMs:performance.now()-startedAt});return}if(type==="setRaster"){cachedRaster={width:payload.width,height:payload.height,values:payload.values,mask:payload.mask};cachedBuilder=null;cachedBuildOptionsKey="";resetDerivedCaches();self.postMessage({id,ok:true,payload:{ok:true}});return}if(type==="buildGuide"){const first=await ensureFirstTopology(payload.buildOptions);const topology=first.topology;const riverBuilder=makeRiverBuilder(payload.riverOptions);const t0=performance.now();const riverResult=riverBuilder.buildFromTopology(topology);cachedRiverResult=riverResult;const guidePoints=riverBuilder.buildGuidePoints(topology,riverResult,payload.guideOptions);const activeVertexCount=riverBuilder.collectActiveVertices(riverResult,payload.guideOptions.minWeight).length;const buildGuideMs=performance.now()-t0;self.postMessage({id,ok:true,payload:{firstVertexCount:topology.positionsGrid.length>>1,firstTriangleCount:Math.floor(topology.indices.length/3),activeVertexCount,guidePoints,timings:{buildMs:first.buildMs,buildGuideMs}}},[guidePoints.buffer]);return}if(type==="buildFinal"){const builder=await ensureBuilder(payload.buildOptions);let topology=null;let buildMs=0;if(payload.reuseCachedTopology&&cachedTopology){topology=cachedTopology}else if(payload.guidePoints&&payload.guidePoints.length>=2){const t0=performance.now();topology=await builder.buildWithForcedWorldPoints(payload.guidePoints);buildMs=performance.now()-t0;cachedTopology=topology;cachedRiverResult=null}else{const first=await ensureFirstTopology(payload.buildOptions);topology=first.topology;buildMs=first.buildMs}let riverResult=cachedRiverResult;if(!riverResult){const riverBuilder=makeRiverBuilder(payload.riverOptions);riverResult=riverBuilder.buildFromTopology(topology);cachedRiverResult=riverResult}else if(!cachedRiverBuilder){makeRiverBuilder(payload.riverOptions)}cachedFloodplainClassifier=makeFloodplainClassifier(payload.floodplainOptions);const hydroView=deriveHydroView(payload.minRiverWeight,payload.floodplainOptions);const uiTopology=makeUiTopology(topology);const transferables=collectUiTopologyTransferables(uiTopology).concat(collectHydroTransferables(hydroView));self.postMessage({id,ok:true,payload:{topology:uiTopology,hydroView,timings:{buildMs,deriveMs:hydroView.timings.deriveMs}}},transferables);cachedBuilder=null;cachedBuildOptionsKey="";log("buildFinal complete",{id,topology:summarizeTopology(topology),river:summarizeRiverResult(riverResult),hydroView:summarizeView(hydroView),timings:{buildMs,deriveMs:hydroView.timings.deriveMs},elapsedMs:performance.now()-startedAt});return}if(type==="updateDerived"){const hydroView=deriveHydroView(payload.minRiverWeight,payload.floodplainOptions);const transferables=collectHydroTransferables(hydroView);self.postMessage({id,ok:true,payload:hydroView},transferables);return}throw new Error(`Unknown topology worker message: ${type}`)}catch(error){const message=error instanceof Error?error.message:String(error);const stack=error instanceof Error?error.stack:null;logError("error",{id,type,message,stack,elapsedMs:performance.now()-startedAt});self.postMessage({id,ok:false,error:stack?`${message}\n${stack}`:message})}};var Topology_worker_default=self;;if(typeof import_meta !== \'undefined\')import_meta.url=location.origin+"/dist/";})();\n',tt=URL.createObjectURL(new globalThis.Blob([et],{type:"text/javascript"})),le=tt;var it=15e3,at=5e3,st=12e4,rt=3e4,nt=!0;function ot(){let s=1;return()=>s++}function ge(s){return s?{vertices:s.positionsGrid?s.positionsGrid.length>>1:0,triangles:s.indices?Math.floor(s.indices.length/3):0,analysisWidth:s.analysis?.width??0,analysisHeight:s.analysis?.height??0}:null}function X(s){return s?{activeVertexCount:s.activeVertexCount??0,activeEdgeCount:s.activeEdgeCount??0,segmentCount:s.segments?s.segments.length>>2:0,floodplainSeedCount:s.floodplainResult?.seedVertices?s.floodplainResult.seedVertices.length:0}:null}function ye(s,e){return e?s==="setRaster"?{width:e.width??0,height:e.height??0,valueCount:e.values?e.values.length:0,maskCount:e.mask?e.mask.length:0}:s==="buildGuide"?{hasBuildOptions:!!e.buildOptions,hasRiverOptions:!!e.riverOptions,hasGuideOptions:!!e.guideOptions}:s==="buildFinal"?{hasBuildOptions:!!e.buildOptions,guidePointCount:e.guidePoints?e.guidePoints.length>>1:0,reuseCachedTopology:!!e.reuseCachedTopology,minRiverWeight:e.minRiverWeight??null}:s==="updateDerived"?{minRiverWeight:e.minRiverWeight??null}:null:null}var K=class{constructor(e={}){this.nextId=ot(),this.pending=new Map,this.requestTimeoutMs=Math.max(1e3,Math.floor(e.requestTimeoutMs??it)),this.readyTimeoutMs=Math.max(1e3,Math.floor(e.readyTimeoutMs??at)),this.heavyPhaseTimeoutMs=Math.max(this.requestTimeoutMs,Math.floor(e.heavyPhaseTimeoutMs??st)),this.hydroUpdateTimeoutMs=Math.max(this.requestTimeoutMs,Math.floor(e.hydroUpdateTimeoutMs??rt)),this.debugLogging=e.debugLogging??nt,this.onstatus=typeof e.onstatus=="function"?e.onstatus:null,this.topologyWorker=new Worker(le,{type:String(le).startsWith("blob:")?"classic":"module",name:"TopologyWorker"}),this.#u(this.topologyWorker,"Topology worker"),this.#e("worker instance created",{topologyWorkerScript:String(le)}),this.topologyReady=!1,this.currentRaster=null,this.currentTopology=null,this.currentHydroView=null,this.#e("constructed",{requestTimeoutMs:this.requestTimeoutMs,readyTimeoutMs:this.readyTimeoutMs,heavyPhaseTimeoutMs:this.heavyPhaseTimeoutMs,hydroUpdateTimeoutMs:this.hydroUpdateTimeoutMs})}dispose(){this.#e("dispose"),this.#o(new Error("Build coordinator disposed.")),this.topologyWorker.terminate(),this.currentRaster=null,this.currentTopology=null,this.currentHydroView=null,this.topologyReady=!1}async#d(){this.topologyReady||(this.#i("Starting topology worker\u2026"),await this.#t(this.topologyWorker,"ping",null,[],this.readyTimeoutMs),this.topologyReady=!0,this.#e("topology worker ready"))}async setRaster(e){await this.#d(),this.currentRaster={width:e.width,height:e.height};let t={width:e.width,height:e.height,values:e.values,mask:e.mask};this.#e("setRaster begin",ye("setRaster",t)),this.#i("Uploading raster\u2026"),await this.#t(this.topologyWorker,"setRaster",t,[t.values.buffer,t.mask.buffer]),this.#e("setRaster complete",{width:e.width,height:e.height})}async build(e){if(!this.currentRaster)throw new Error("No raster has been set.");this.#e("build begin",{raster:{width:this.currentRaster.width,height:this.currentRaster.height},guideEnabled:!!e.guideEnabled,minRiverWeight:e.minRiverWeight});let t=0,i=1,a=0,r={firstTopologyMs:0,firstGuideMs:0,secondTopologyMs:0,finalHydroMs:0};if(!e.guideEnabled){this.#i("Building topology\u2026");let c=await this.#t(this.topologyWorker,"buildFinal",{buildOptions:e.topologyBuildOptions,guidePoints:null,reuseCachedTopology:!1,riverOptions:e.riverOptions,floodplainOptions:e.floodplainOptions,minRiverWeight:e.minRiverWeight},[],this.heavyPhaseTimeoutMs);return r.firstTopologyMs=c.timings?.buildMs??0,r.finalHydroMs=c.timings?.deriveMs??0,this.currentTopology=c.topology,this.currentHydroView=c.hydroView,this.#i("Build complete"),this.#e("build complete",{topology:ge(c.topology),hydroView:X(c.hydroView),guidePointCount:t,passCount:i,phaseTimings:r}),{topology:c.topology,hydroView:c.hydroView,guidePointCount:t,passCount:i,phaseTimings:r,buildMetrics:{firstGuideActiveVertices:a}}}this.#i("Building topology pass 1\u2026");let n=await this.#t(this.topologyWorker,"buildGuide",{buildOptions:e.topologyBuildOptions,riverOptions:e.riverOptions,guideOptions:e.guideOptions},[],this.heavyPhaseTimeoutMs);this.#e("buildGuide complete",{firstVertexCount:n.firstVertexCount||0,firstTriangleCount:n.firstTriangleCount||0,activeVertexCount:n.activeVertexCount||0,guidePointCount:n.guidePoints?n.guidePoints.length>>1:0,timings:n.timings}),t=n.guidePoints.length>>1,a=n.activeVertexCount||0,r.firstTopologyMs=n.timings?.buildMs??0,r.firstGuideMs=n.timings?.buildGuideMs??0;let l=t>=Math.max(128,Math.floor((n.firstVertexCount||0)*.015))&&t>=Math.max(64,Math.floor((n.activeVertexCount||0)*.2));i=l?2:1,this.#e("second pass decision",{shouldRunSecondPass:l,guideEnabled:!!e.guideEnabled,guidePointCount:t,firstVertexCount:n.firstVertexCount||0,firstGuideActiveVertices:a}),this.#i(l?"Building topology pass 2\u2026":"Deriving hydro overlays\u2026");let o=await this.#t(this.topologyWorker,"buildFinal",{buildOptions:e.topologyBuildOptions,guidePoints:l?n.guidePoints:null,reuseCachedTopology:!l,riverOptions:e.riverOptions,floodplainOptions:e.floodplainOptions,minRiverWeight:e.minRiverWeight},l?[n.guidePoints.buffer]:[],this.heavyPhaseTimeoutMs);return r.secondTopologyMs=l?o.timings?.buildMs??0:0,r.finalHydroMs=o.timings?.deriveMs??0,this.currentTopology=o.topology,this.currentHydroView=o.hydroView,this.#i("Build complete"),this.#e("build complete",{topology:ge(o.topology),hydroView:X(o.hydroView),guidePointCount:t,passCount:i,phaseTimings:r}),{topology:o.topology,hydroView:o.hydroView,guidePointCount:t,passCount:i,phaseTimings:r,buildMetrics:{firstGuideActiveVertices:a}}}async updateHydroOnly({minRiverWeight:e,floodplainOptions:t}){await this.#d(),this.#i("Updating hydro overlays\u2026"),this.#e("updateHydroOnly begin",{minRiverWeight:e});let i=await this.#t(this.topologyWorker,"updateDerived",{minRiverWeight:e,floodplainOptions:t},[],this.hydroUpdateTimeoutMs);return this.currentHydroView=i,this.#i("Hydro updated"),this.#e("updateHydroOnly complete",X(i)),i}#u(e,t){e.onmessage=i=>this.#r(i,t),e.onerror=i=>{let a=i?.message?`${t}: ${i.message}`:`${t} failed to load or crashed.`;this.#e("worker error",{label:t,message:a,filename:i?.filename||null,lineno:i?.lineno||null,colno:i?.colno||null}),this.#n(e,new Error(a))},e.onmessageerror=i=>{this.#e("worker messageerror",{label:t,event:i}),this.#n(e,new Error(`${t}: message serialization failed.`))}}#i(e){this.#e("status",e),this.onstatus&&this.onstatus(e)}#r(e,t="Worker"){let i=e.data||{},{id:a,ok:r,payload:n,error:l}=i;if(!this.pending.has(a)){this.#e("received response for unknown request",{label:t,id:a,ok:r,error:l,dataKeys:i?Object.keys(i):null});return}let o=this.pending.get(a);this.pending.delete(a),clearTimeout(o.timer);let c=performance.now()-o.startedAt;r?(this.#e("response ok",{label:t,id:a,type:o.type,elapsedMs:c,payloadSummary:this.#a(o.type,n)}),o.resolve(n)):(this.#e("response error",{label:t,id:a,type:o.type,elapsedMs:c,error:l}),o.reject(new Error(l||"Worker request failed.")))}#n(e,t){for(let[i,a]of this.pending.entries())a.worker===e&&(clearTimeout(a.timer),this.pending.delete(i),this.#e("reject pending for worker",{id:i,type:a.type,error:t.message}),a.reject(t))}#o(e){for(let[t,i]of this.pending.entries())clearTimeout(i.timer),this.pending.delete(t),this.#e("reject all pending",{id:t,type:i.type,error:e.message}),i.reject(e)}#t(e,t,i,a=[],r=this.requestTimeoutMs){return new Promise((n,l)=>{let o=this.nextId(),c=performance.now(),d="Topology worker",p=setTimeout(()=>{if(!this.pending.has(o))return;this.pending.delete(o);let h=performance.now()-c,v=new Error(`Timed out waiting for worker response: ${t}`);this.#e("request timeout",{worker:d,id:o,type:t,timeoutMs:r,elapsedMs:h,payloadSummary:ye(t,i),hint:t==="ping"?"Worker module may still be starting, blocked on import, or failed before onmessage was installed.":null}),l(v)},r);this.pending.set(o,{worker:e,resolve:n,reject:l,timer:p,type:t,startedAt:c}),this.#e("send request",{worker:d,id:o,type:t,timeoutMs:r,transferables:a.length,payloadSummary:ye(t,i)});try{e.postMessage({id:o,type:t,payload:i},a)}catch(h){clearTimeout(p),this.pending.delete(o),this.#e("postMessage failed",{worker:d,id:o,type:t,error:h instanceof Error?h.stack||h.message:String(h)}),l(h)}})}#a(e,t){return t?e==="ping"||e==="setRaster"?t:e==="buildGuide"?{firstVertexCount:t.firstVertexCount||0,firstTriangleCount:t.firstTriangleCount||0,activeVertexCount:t.activeVertexCount||0,guidePointCount:t.guidePoints?t.guidePoints.length>>1:0,timings:t.timings??null}:e==="buildFinal"?{topology:ge(t.topology),hydroView:X(t.hydroView),timings:t.timings??null}:e==="updateDerived"?{hydroView:X(t),timings:t.timings??null}:null:null}#e(...e){this.debugLogging&&console.log("[BuildCoordinator]",...e)}};var Z=class{constructor({budgetMs:e=8,maximumSteps:t=64}={}){this.budgetMs=e,this.maximumSteps=t,this.reset()}reset(e=4){this.steps=Math.max(1,Math.min(this.maximumSteps,Math.floor(Number(e))||1))}observe(e,t){if(!Number.isFinite(t)||t<=0)return;let i=Math.max(.5,Math.min(1.25,this.budgetMs/Math.max(t,.25))),a=Math.round(e*i);this.reset(i>=1.2?Math.max(e+1,a):a)}};var B={iterationsPerFrame:4,stepIterations:160,timeStep:.02,rainRate:.001,evaporationRate:.025,pipeArea:16,gravity:9.81,capacityScale:1.2,flowDrag:2,maxOutflowFraction:.5,suspensionRate:.35,depositionRate:1.4,softeningRate:1.2,maxErosionDepth:.04,thermalRate:.25,talusCoeff:.012,talusBias:.002,erodabilityMin:.24,erodabilityMax:.36,sourceStrength:.06,sourceRadius:4,sourceEnabled:!1,sourceSedimentConcentration:0,bankErosionRate:0,bankBendBias:24,bendSedimentRate:0,straightBankErosionFactor:.08,riverBendLength:0,openBoundaryEdges:15,sourceLayoutMode:1,randomSpringCount:3,sourceSeed:7,rainDuration:0,pulse2Duration:0,metersPerPixel:100,renderHeightScale:.45,waterHeightScale:.45,waterOpacity:.28,edgeWaterFloor:0,renderMode:8,hydraulicErosionEnabled:!0,closedBoundaries:!1},Ge={paper_balanced:{...B,label:"Balanced drainage",description:"Steady light rain cuts connected drainage networks. Start with the alpine catchment and compare after 20\u201360 simulated seconds."},river_cut:{...B,label:"Spring-fed channels",rainRate:25e-5,flowDrag:1.4,sourceEnabled:!0,sourceStrength:.12,randomSpringCount:3,capacityScale:1.8,suspensionRate:.55,depositionRate:1.2,thermalRate:.18,description:"Localized springs carve channels with sediment bars downstream. Use the alpine catchment; change Spring seed for a different source layout."},delta_depositor:{...B,label:"Alluvial fan",rainRate:.0025,evaporationRate:.012,flowDrag:3.5,capacityScale:.7,suspensionRate:.4,depositionRate:3,erodabilityMin:.4,erodabilityMax:.6,thermalRate:.08,rainDuration:25,edgeWaterFloor:.015,description:"Runoff leaves a narrow mountain outlet and spreads across a gentle plain. Use the fan example; the 25-second storm is followed by settling."},flash_flood:{...B,label:"Flash flood and dry-down",rainRate:.018,flowDrag:2.5,evaporationRate:.18,capacityScale:3.2,suspensionRate:1.2,depositionRate:2.4,rainDuration:6,thermalRate:.08,erodabilityMin:.35,erodabilityMax:.55,description:"A six-second storm strips loose soil, then deposits its load as water recedes. Run the fan or badlands for 40\u201380 simulated seconds."},badlands:{...B,label:"Soft badlands",rainRate:.002,evaporationRate:.065,capacityScale:1.8,suspensionRate:.55,depositionRate:1.8,erodabilityMin:.55,erodabilityMax:.75,thermalRate:.45,talusCoeff:.006,talusBias:.001,description:"Soft terrain develops rills while unstable banks slump. Use the badlands example and compare after 30\u201390 simulated seconds."},canyon_carver:{...B,label:"Layered canyon",rainRate:6e-4,sourceEnabled:!0,sourceStrength:.16,randomSpringCount:2,capacityScale:2.8,suspensionRate:.7,depositionRate:1.1,thermalRate:.12,erodabilityMin:.12,erodabilityMax:.22,description:"Slow incision preserves resistant shelves. The canyon example includes a softer upper layer over resistant bedrock; run for 60\u2013180 simulated seconds."},river_delta:{...B,label:"River delta",rainRate:0,evaporationRate:0,sourceEnabled:!0,sourceLayoutMode:0,sourceStrength:.45,sourceSedimentConcentration:.12,flowDrag:3,capacityScale:1.2,suspensionRate:.3,depositionRate:1.4,thermalRate:.04,erodabilityMin:.45,erodabilityMax:.6,edgeWaterFloor:.035,waterOpacity:.6,description:"A sediment-fed river enters a shallow coastal basin. Upstream sediment replenishes the bed and builds mouth bars. Adjust River sediment load in the live preview; Run continues until paused."},mountain_canyon:{...B,label:"Mountain river canyon",rainRate:1e-4,evaporationRate:.012,sourceEnabled:!0,sourceLayoutMode:0,sourceStrength:.4,flowDrag:1.4,capacityScale:2.8,suspensionRate:.75,depositionRate:1.2,thermalRate:.08,erodabilityMin:.25,erodabilityMax:.4,waterOpacity:.6,description:"A headwater spring follows a descending mountain canyon. A thin erodible surface overlies resistant rock; compare incision and downstream bars after 60\u2013120 simulated seconds."},volcanic_cone:{...B,label:"Volcanic radial canyons",rainRate:.006,rainDuration:35,closedBoundaries:!0,evaporationRate:.075,flowDrag:1.8,capacityScale:3.4,suspensionRate:1.25,depositionRate:2.5,thermalRate:.06,erodabilityMin:.74,erodabilityMax:.9,talusCoeff:.016,waterOpacity:.45,description:"Longonot-inspired broad crater and radial drainage. A 35-second storm incises loose ash and thick tephra above resistant tuff and lava. Run 90\u2013150 seconds to see deepening gullies, exposed layers and sediment aprons."},volcanic_lake:{...B,label:"Crater lake and flank erosion",rainRate:.006,rainDuration:20,closedBoundaries:!0,evaporationRate:.035,flowDrag:2,capacityScale:2.8,suspensionRate:.95,depositionRate:2,thermalRate:.04,erodabilityMin:.65,erodabilityMax:.8,talusCoeff:.02,waterOpacity:.6,description:"A twenty-second storm cuts the layered volcanic flanks while the crater collects runoff. Compare evolving radial gullies and the sheltered lake after 60\u201390 seconds."},lowland_meanders:{...B,label:"Lowland meandering river",rainRate:0,evaporationRate:.002,sourceEnabled:!0,sourceLayoutMode:0,sourceStrength:.015,sourceSedimentConcentration:.1,flowDrag:2,pipeArea:20,capacityScale:1.2,suspensionRate:.12,depositionRate:3,bankErosionRate:3,bankBendBias:80,thermalRate:.01,talusCoeff:.008,bendSedimentRate:4,straightBankErosionFactor:.005,riverBendLength:.045,openBoundaryEdges:8,erodabilityMin:.75,erodabilityMax:.9,waterOpacity:.65,renderHeightScale:.9,waterHeightScale:.9,description:"A nearly straight river develops bends through layered alluvium. Inner-bank sediment bars deflect flow, outer banks retreat, and new channels carve while abandoned reaches fill. Run continuously or compare after 360 seconds. Headwater and side boundaries are walls; the downstream edge is open."},thermal_heavy:{...B,label:"Dry talus relaxation",rainRate:0,sourceEnabled:!1,hydraulicErosionEnabled:!1,thermalRate:3,talusCoeff:.005,talusBias:.001,erodabilityMin:.55,erodabilityMax:.7,description:"Dry cliffs shed material into talus aprons without water erosion. Use the talus example; inspect the thermal view after 10\u201330 simulated seconds."},gentle_weathering:{...B,label:"Gentle weathering",rainRate:4e-4,capacityScale:1.2,suspensionRate:.3,thermalRate:.08,erodabilityMin:.12,erodabilityMax:.2,description:"Slow changes retain the starting relief. Use the alpine catchment for long runs, or as a restrained starting point for imported DEMs."}};var $={alpine:{label:"Alpine catchment",mode:0,preset:"paper_balanced",layered:!0},fan:{label:"Mountain outlet and fan",mode:1,preset:"delta_depositor",layered:!0},badlands:{label:"Warped badlands",mode:2,preset:"badlands",layered:!0},canyon:{label:"Layered canyon",mode:3,preset:"canyon_carver",layered:!0},talus:{label:"Cliffs and talus",mode:4,preset:"thermal_heavy",layered:!0},basin:{label:"Closed basin",mode:5,preset:"flash_flood",closed:!0,layered:!0},river_delta:{label:"River delta",mode:6,preset:"river_delta",layered:!0},mountain_canyon:{label:"Mountain river canyon",mode:7,preset:"mountain_canyon",layered:!0},volcano:{label:"Longonot-inspired volcano",mode:8,preset:"volcanic_cone",closed:!0,layered:!0},crater_lake:{label:"Volcanic crater lake",mode:8,preset:"volcanic_lake",closed:!0,layered:!0},lowland_river:{label:"Lowland river and floodplain",mode:9,preset:"lowland_meanders",layered:!0}},A={label:"Resistant bedrock",hardnessMin:0,hardnessMax:.02,thermalEnabled:!1},z={alpine:[{label:"Mountain soil"},{label:"Weathered rock",hardnessMin:.1,hardnessMax:.18},A],fan:[{label:"Loose fan sediment"},{label:"Older alluvium",hardnessMin:.28,hardnessMax:.45},A],badlands:[{label:"Soft surface beds"},{label:"Cemented ledges",hardnessMin:.08,hardnessMax:.16,thermalEnabled:!1},{label:"Weak sediment beds",hardnessMin:.45,hardnessMax:.65},A],canyon:[{label:"Upper sediment"},{label:"Resistant shelves",hardnessMin:.04,hardnessMax:.09,thermalEnabled:!1},{label:"Weathered lower beds",hardnessMin:.18,hardnessMax:.3},A],talus:[{label:"Loose talus"},{label:"Fractured cliff rock",hardnessMin:.25,hardnessMax:.4},A],basin:[{label:"Basin sediment"},{label:"Compacted substrate",hardnessMin:.12,hardnessMax:.22},A],river_delta:[{label:"Young delta sediment"},{label:"Older alluvium",hardnessMin:.18,hardnessMax:.3},A],lowland_river:[{label:"Loose floodplain alluvium"},{label:"Older river deposits",hardnessMin:.4,hardnessMax:.6},A],mountain_canyon:[{label:"Thin canyon soil"},{label:"Weathered rock",hardnessMin:.05,hardnessMax:.09,thermalEnabled:!1},A],volcano:[{label:"Loose volcanic ash"},{label:"Unconsolidated tephra",hardnessMin:.5,hardnessMax:.72},{label:"Resistant tuff",hardnessMin:.08,hardnessMax:.16,thermalEnabled:!1},{...A,label:"Lava core"}]};z.crater_lake=z.volcano;var lt=`
struct Options { size: u32, mode: u32, seed: u32, padding: u32 }
@group(0) @binding(0) var<uniform> options: Options;
@group(0) @binding(1) var<storage, read_write> heights: array<vec4<f32>>;
fn hash(v: u32) -> u32 {
  var h = v;
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  return h ^ (h >> 16u);
}
fn contribution(cell: vec2<i32>, offset: vec2<f32>) -> f32 {
  let h = hash(bitcast<u32>(cell.x) ^ hash(bitcast<u32>(cell.y)) ^ options.seed);
  let angle = f32(h & 65535u) * (6.2831853 / 65536.0);
  let weight = max(0.5 - dot(offset, offset), 0.0);
  return weight * weight * weight * weight * dot(vec2<f32>(cos(angle), sin(angle)), offset);
}
fn simplex(p: vec2<f32>) -> f32 {
  let ij = vec2<i32>(floor(p + (p.x + p.y) * 0.3660254));
  let a = p - vec2<f32>(ij) + f32(ij.x + ij.y) * 0.21132487;
  let corner = select(vec2<i32>(0, 1), vec2<i32>(1, 0), a.x > a.y);
  let b = a - vec2<f32>(corner) + 0.21132487;
  let c = a - 1.0 + 0.42264974;
  return 70.0 * (contribution(ij, a) + contribution(ij + corner, b) + contribution(ij + vec2<i32>(1), c));
}
fn fbm(p: vec2<f32>, ridged: bool) -> f32 {
  var q = p;
  var amplitude = 0.5;
  var value = 0.0;
  for (var octave = 0; octave < 6; octave++) {
    let n = simplex(q);
    value += amplitude * select(n, 1.0 - abs(n), ridged);
    q = vec2<f32>(q.x * 0.8 - q.y * 0.6, q.x * 0.6 + q.y * 0.8) * 2.05 + 7.31;
    amplitude *= 0.48;
  }
  return value;
}
@compute @workgroup_size(8, 8)
fn generate(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= options.size || gid.y >= options.size) { return; }
  let uv = vec2<f32>(gid.xy) / f32(options.size - 1u);
  let p = uv * 5.0;
  let warp = vec2<f32>(simplex(p * 0.6 + 13.7), simplex(p * 0.6 - 8.1));
  let q = p + warp * 0.65;
  let ridge = fbm(q, true);
  let detail = fbm(q * 1.6, false);
  var height = 0.12 + 0.78 * ridge;
  switch options.mode {
    case 0u: {
      let valley = exp(-pow((uv.x - 0.5 - 0.08 * simplex(p * 0.6)) / 0.16, 2.0));
      height = 0.13 + 0.72 * ridge * (1.0 - 0.55 * valley) + (1.0 - uv.y) * 0.12;
    }
    case 1u: {
      let mountain = 1.0 - smoothstep(0.25, 0.62, uv.y);
      let outlet = exp(-pow((uv.x - 0.5) / (0.045 + uv.y * 0.14), 2.0));
      height = 0.06 + (1.0 - uv.y) * 0.12 + mountain * (0.7 * ridge) * (1.0 - 0.8 * outlet) + detail * 0.008;
    }
    case 2u: { height = 0.15 + 0.52 * ridge + 0.14 * detail + (1.0 - uv.y) * 0.12; }
    case 3u: {
      let channel = exp(-pow((uv.x - 0.5 - 0.075 * sin(uv.y * 9.0)) / 0.065, 2.0));
      let terrace = floor((0.35 + 0.45 * ridge) * 9.0) / 9.0;
      height = 0.12 + terrace * (1.0 - 0.7 * channel) + (1.0 - uv.y) * 0.08 + detail * 0.015;
    }
    case 4u: {
      let cliff = smoothstep(0.43, 0.44, ridge);
      height = 0.1 + cliff * 0.52 + ridge * 0.22 + detail * 0.01;
    }
    case 5u: {
      let r = distance(uv, vec2<f32>(0.5));
      height = 0.08 + 0.68 * smoothstep(0.18, 0.42, r) + detail * 0.025;
    }
    case 6u: {
      let center = 0.5 + 0.035 * sin(uv.y * 8.0);
      let spread = max(uv.y - 0.52, 0.0) * 0.5;
      let channelDistance = min(abs(uv.x - center), min(abs(uv.x - center - spread), abs(uv.x - center + spread)));
      let channel = exp(-pow(channelDistance / 0.027, 2.0));
      let upland = 1.0 - smoothstep(0.18, 0.58, uv.y);
      let deltaPlain = 1.0 - smoothstep(0.65, 0.8, uv.y);
      let riverBed = 0.045 + 0.16 * (1.0 - uv.y);
      let coastalShelf = 0.085 + 0.04 * (1.0 - uv.y);
      height = mix(riverBed, coastalShelf, smoothstep(0.54, 0.76, uv.y))
        + (0.38 * ridge * upland + 0.035 * deltaPlain) * (1.0 - channel);
      height += detail * 0.006 * upland * (1.0 - channel);
    }
    case 7u: {
      let center = 0.5 + 0.08 * sin(uv.y * 7.0);
      let channel = exp(-pow((uv.x - center) / 0.037, 2.0));
      height = 0.075 + 0.23 * (1.0 - uv.y) + (0.3 + 0.48 * ridge) * (1.0 - channel);
      height += detail * 0.012 * (1.0 - channel);
    }
    case 8u: {
      let d = (uv - vec2<f32>(0.51, 0.49)) * vec2<f32>(1.06, 0.94);
      let r = length(d);
      let direction = d / max(r, 0.001);
      let angle = atan2(d.y, d.x);
      let rimRadius = 0.175 + 0.009 * simplex(direction * 2.8 + 3.0);
      let flank = pow(clamp((0.51 - r) / (0.51 - rimRadius), 0.0, 1.0), 1.15);
      let cone = 0.065 + 0.7 * flank;
      let rim = 0.035 * exp(-pow((r - rimRadius) / 0.017, 2.0));
      let crater = 0.27 + 0.012 * detail + 0.025 * pow(clamp(r / rimRadius, 0.0, 1.0), 2.0);
      height = mix(crater, cone + rim, smoothstep(rimRadius - 0.045, rimRadius, r));
      // Shallow irregular drainage seeds, with tributaries merging outward.
      // Storm runoff deepens these through ash and tephra rather than carving
      // the final canyon depth into the starting heightmap.
      let theta = angle + 0.07 * simplex(direction * 3.0 + r * 3.0);
      let trunk = exp(-pow(sin(theta * 11.0 + simplex(d * 6.0) * 0.5) / 0.35, 2.0));
      let tributary = exp(-pow(sin(theta * 29.0 + r * 21.0 + simplex(d * 18.0)) / 0.2, 2.0));
      let drainage = max(trunk, tributary * (1.0 - smoothstep(0.27, 0.44, r)) * 0.65);
      let apron = smoothstep(rimRadius + 0.01, rimRadius + 0.04, r) * (1.0 - smoothstep(0.43, 0.53, r));
      let flankTexture = simplex(d * 14.0) * 0.018 + simplex(d * 35.0) * 0.004;
      height += (flankTexture - drainage * 0.024) * apron;
    }
    case 9u: {
      // Nearly straight starter channel. Sub-cell bank irregularities seed
      // instability; the simulation, rather than this DEM, creates the bends.
      let center = 0.5 + 0.008 * sin(uv.y * 18.84956 - 0.6) + 0.002 * simplex(vec2<f32>(uv.y * 12.0, 3.0));
      let channel = exp(-pow((uv.x - center) / 0.016, 2.0));
      height = 0.065 + 0.04 * (1.0 - uv.y) + 0.025 * (1.0 - channel) + detail * 0.0015 * (1.0 - channel);
    }
    default: {}
  }
  height = clamp(height, 0.02, 0.98);
  var depth = vec3<f32>(0.07 + 0.025 * ridge, 0.1 + 0.025 * detail, 0.12);
  switch options.mode {
    case 1u: { depth = vec3<f32>(0.04 + 0.025 * uv.y, 0.06 + 0.025 * ridge, 0.0); }
    case 2u: { depth = vec3<f32>(0.055 + 0.035 * ridge, 0.026 + 0.012 * detail, 0.16 + 0.04 * ridge); }
    case 3u: { depth = vec3<f32>(0.08 + 0.03 * detail, 0.025, 0.11 + 0.03 * ridge); }
    case 4u: { depth = vec3<f32>(0.13 + 0.04 * detail, 0.3, 0.0); }
    case 5u: { depth = vec3<f32>(0.055 + 0.025 * ridge, 0.08, 0.0); }
    case 6u: { depth = vec3<f32>(0.035 + 0.012 * ridge, 0.025, 0.0); }
    case 7u: { depth = vec3<f32>(0.045, 0.025 + 0.01 * ridge, 0.0); }
    case 9u: { depth = vec3<f32>(0.035 + 0.01 * ridge, 0.025, 0.0); }
    case 8u: {
      let r = length((uv - vec2<f32>(0.51, 0.49)) * vec2<f32>(1.06, 0.94));
      let deposits = smoothstep(0.16, 0.23, r) * (1.0 - smoothstep(0.47, 0.55, r));
      depth = vec3<f32>(0.008 + deposits * (0.025 + 0.018 * ridge), 0.018 + deposits * (0.12 + 0.04 * detail), 0.055 + 0.018 * ridge);
    }
    default: {}
  }
  heights[gid.y * options.size + gid.x] = vec4<f32>(height, max(height - depth.x, 0.0), max(height - depth.x - depth.y, 0.0), max(height - depth.x - depth.y - depth.z, 0.0));
}
`;async function Le(s,e="alpine",t=512,i=7){let a=$[e];if(!a)throw new Error(`Unknown terrain example: ${e}`);t=Math.max(64,Math.min(1024,Math.floor(t)||512));let r=t*t*16,n=s.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),l=s.createBuffer({size:r,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),o=s.createBuffer({size:r,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{let c=await s.createComputePipelineAsync({layout:"auto",compute:{module:s.createShaderModule({code:lt}),entryPoint:"generate"}}),d=s.createBindGroup({layout:c.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:l}}]});s.queue.writeBuffer(n,0,new Uint32Array([t,a.mode,i>>>0,0]));let p=s.createCommandEncoder(),h=p.beginComputePass();h.setPipeline(c),h.setBindGroup(0,d),h.dispatchWorkgroups(Math.ceil(t/8),Math.ceil(t/8)),h.end(),p.copyBufferToBuffer(l,0,o,0,r),s.queue.submit([p.finish()]),await o.mapAsync(GPUMapMode.READ);let v=new Float32Array(o.getMappedRange()),m=Array.from({length:z[e].length},(S,b)=>Float32Array.from({length:t*t},(w,y)=>v[y*4+b])),f=m[0];o.unmap();let u={width:t,height:t,values:f};return a.layered&&(u.bands=m),(e==="river_delta"||e==="mountain_canyon"||e==="crater_lake"||e==="lowland_river")&&(u.water=Float32Array.from(f,(S,b)=>{let w=b%t/(t-1),y=Math.floor(b/t)/(t-1);if(e==="crater_lake")return Math.hypot((w-.51)*1.06,(y-.49)*.94)<.2?Math.max(.34-S,0):0;let P=e==="lowland_river"?.065+.04*(1-y)+.008:e==="river_delta"?Math.max(.12,.045+.16*(1-y)+.012):.075+.23*(1-y)+.012;return Math.max(P-S,0)})),u}finally{n.destroy(),l.destroy(),o.destroy()}}function Ie(s,e){return $[s]?.layered?{layerConfig:{mode:"packed_rgba",layers:z[s].map(t=>({enabled:!0,heightMin:0,heightMax:1,hardnessMin:e.erodabilityMin,hardnessMax:e.erodabilityMax,thermalEnabled:!0,...t}))}}:{}}function De(s,e,t){if(t!=="river_delta"&&t!=="mountain_canyon"&&t!=="lowland_river")return;let i=Math.round((e.height-1)*(t==="lowland_river"?.035:.08)),a=Math.round(e.width*.5);for(let r=Math.floor(e.width*.2);r<e.width*.8;r++)e.values[i*e.width+r]<e.values[i*e.width+a]&&(a=r);s.paintSpringBrush({x:a,y:i,radius:Math.max(2,4*e.width/256),strength:s.simulationParams.sourceStrength,hardness:1})}var x=s=>document.getElementById(s),L=x("terrain"),We={lowland_river:180,river_delta:90,mountain_canyon:90,volcano:90,crater_lake:90},g,ce,E,xe,Q,F,U=!1,k=!1,he=0,de=0,be=null,ve=45,ue=52,J=2.8,_,V=new Z,G={ready:!1,snapshots:0,meshIteration:null,errors:[]},N=new URLSearchParams(location.search);Object.hasOwn(We,N.get("scene"))&&(x("scene").value=N.get("scene"));["256","512","1024"].includes(N.get("size"))&&(x("size").value=N.get("size"));function C(s){let e=g?.getStats();x("status").textContent=s||`${e?.simTime.toFixed(2)??"0.00"} simulated seconds \xB7 ${e?.iterationCount??0} steps \xB7 ${U?"running":"paused"}`,x("run").textContent=U?"Pause":"Run";for(let t of["scene","size","run","reset","advance","build","mode","min-spacing","max-spacing","guides","color"])x(t).disabled=k||!G.ready;x("surface").disabled=x("wire").disabled=x("wire-opacity").disabled=k||!E,x("view-label").textContent=E?x("surface").value==="adaptive"?"Final mesh \xB7 adaptive triangles":"Final mesh comparison \xB7 full grid":"Live erosion \xB7 grid surface",L.dataset.running=String(U)}function ee(s){G.errors.push(s.message),x("error").textContent=s.message,C()}function te(){U=!1,de++,cancelAnimationFrame(he),he=0,C()}function we(){E?.destroy(),E=null,Q=null,F=null,G.meshIteration=null,be=null,x("mesh-stats").textContent="Build a mesh to compare triangle counts."}function ie(){g?.ready&&(E?E.render({adaptive:x("surface").value==="adaptive",wireframe:x("wire").checked,wireOpacity:Number(x("wire-opacity").value)}):g.render())}function Me(){if(!g?.ready)return;let s=ve*Math.PI/180,e=ue*Math.PI/180;g.setSimulationParams({cameraAzimuthDeg:ve,cameraElevationDeg:ue,cameraPosX:Math.cos(e)*Math.cos(s)*J,cameraPosY:.12+Math.sin(e)*J,cameraPosZ:Math.cos(e)*Math.sin(s)*J}),ie()}async function ae(s){if(!k){te(),k=!0,x("error").textContent="",C();try{await s()}catch(e){ee(e)}finally{k=!1,C()}}}async function Se(){we(),await g.device.queue.onSubmittedWorkDone(),C("Preparing layered terrain\u2026");let s=x("scene").value,e=Ge[$[s].preset];xe=await Le(g.device,s,Number(x("size").value),7),g.setSimulationParams({...e,closedBoundaries:!!$[s].closed,talusSlopeCoeff:e.talusCoeff,talusSlopeBias:e.talusBias,hardnessBase:e.erodabilityMin,hardnessVariation:e.erodabilityMax-e.erodabilityMin,renderGridStep:1,renderMode:Number(x("color").value),sourceTimeOffset:0}),await g.setDEM(xe,Ie(s,e)),De(g,xe,s),x("layers").textContent="Material layers \xB7 "+z[s].map(t=>t.label).join(" / "),G.ready=!0,Me()}async function ze(s){if(!(!U||k||s!==de))try{let e=V.steps,t=performance.now();if(g.stepAndRender(e),await g.device.queue.onSubmittedWorkDone(),!U||s!==de)return;V.observe(e,performance.now()-t),C(),he=requestAnimationFrame(()=>ze(s))}catch(e){te(),ee(e)}}function ct(){if(!G.ready||k||U)return;we(),U=!0;let s=++de;V.reset(),C(),he=requestAnimationFrame(()=>ze(s))}async function Pe(s=We[x("scene").value]){we();let e=g.simulationParams.timeStep,t=Math.round(s/e);V.reset(32);let i=0;for(;g.getStats().iterationCount<t;){let a=Math.min(V.steps,t-g.getStats().iterationCount),r=performance.now();g.step(a),await g.device.queue.onSubmittedWorkDone(),V.observe(a,performance.now()-r),(performance.now()-i>=100||g.getStats().iterationCount>=t)&&(g.render(),C(`Evolving terrain \xB7 ${g.getStats().simTime.toFixed(1)} / ${s} seconds`),i=performance.now()),await new Promise(requestAnimationFrame)}}async function ht(){if(Q)return Q;let s=g.device.createBuffer({size:g.cellCount*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{let e=g.device.createCommandEncoder();e.copyBufferToBuffer(g.stateBuffers[0],0,s,0,s.size),g.device.queue.submit([e.finish()]),await s.mapAsync(GPUMapMode.READ);let t=new Float32Array(s.getMappedRange()),i=new Float32Array(g.cellCount),a=new Uint8Array(g.cellCount);for(let r=0;r<g.cellCount;r++)i[r]=t[r*8],a[r]=t[r*8+4]>.5?1:0;return s.unmap(),G.snapshots++,Q={width:g.width,height:g.height,values:i,mask:a,iteration:g.getStats().iterationCount},Q}finally{s.destroy()}}async function _e(){C("Capturing paused terrain\u2026");let s=await ht(),e=performance.now();ce??=new K({debugLogging:!1,onstatus:c=>C(c)}),await ce.setRaster({width:s.width,height:s.height,values:s.values.slice(),mask:s.mask.slice()});let t=Math.max(1,Math.min(128,Number(x("min-spacing").value)||3)),i=Math.max(t,Math.min(256,Number(x("max-spacing").value)||28));x("min-spacing").value=t,x("max-spacing").value=i;let a=1/0,r=-1/0;for(let c=0;c<s.values.length;c++)s.mask[c]&&(a=Math.min(a,s.values[c]),r=Math.max(r,s.values[c]));let n=Math.max(r-a,.001),l=await ce.build({topologyBuildOptions:{mode:x("mode").value,cellSize:1,minSpacing:t,maxSpacing:i,borderSpacing:i,wetnessBins:21,channelThreshold:256,candidateStride:1,relaxIterations:0,preserveBorder:!0,preserveChannels:!1,preserveMaskBoundary:!0,preserveInnerBorder:!0},riverOptions:{maxNeighborDistance:1/0,boundaryEpsilon:.5,minDrop:1e-5,flatTolerance:1e-4,sinkMaxRise:n*.015,sinkClimbPenalty:2.5},guideOptions:{minWeight:3,step:Math.max(t*1.75,Math.min(i*.6,t*5)),maxPoints:Math.min(16e3,Math.floor(g.cellCount*.45))},floodplainOptions:{maxRise:n*.05,maxClimbStep:n*.015,maxEdgeSlope:.75,maxWorldDistance:Math.max(t*6,i*1.5),minTriangleFloodplainVerts:2},minRiverWeight:6,guideEnabled:x("guides").checked});C("Uploading final triangles\u2026");let o=new j(g);try{await o.initialize(l.topology)}catch(c){throw o.destroy(),c}E?.destroy(),E=o,G.meshIteration=s.iteration,be=l.topology,F={vertices:l.topology.positionsGrid.length/2,triangles:l.topology.indices.length/3,gridTriangles:(g.width-1)*(g.height-1)*2,buildMs:performance.now()-e},x("mesh-stats").textContent=`${F.vertices.toLocaleString()} vertices \xB7 ${F.triangles.toLocaleString()} triangles
Full grid: ${F.gridTriangles.toLocaleString()} triangles
${(100*(1-F.triangles/F.gridTriangles)).toFixed(1)}% fewer \xB7 ${(F.buildMs/1e3).toFixed(2)} s build
Frozen at ${g.getStats().simTime.toFixed(2)} simulated seconds`,x("surface").value="adaptive",ie(),await g.device.queue.onSubmittedWorkDone()}x("run").onclick=()=>U?te():ct();x("reset").onclick=()=>ae(Se);x("advance").onclick=()=>ae(()=>Pe());x("build").onclick=()=>ae(_e);for(let s of["scene","size"])x(s).onchange=()=>ae(Se);for(let s of["surface","wire"])x(s).onchange=()=>{ie(),C()};x("wire-opacity").oninput=ie;x("color").onchange=()=>{g.setSimulationParams({renderMode:Number(x("color").value)}),ie()};L.onpointerdown=s=>{_=[s.clientX,s.clientY],L.setPointerCapture(s.pointerId)};L.onpointermove=s=>{!_||k||(ve+=(s.clientX-_[0])*.35,ue=Math.max(10,Math.min(85,ue+(s.clientY-_[1])*.25)),_=[s.clientX,s.clientY],Me())};L.onpointerup=L.onpointercancel=()=>{_=null};L.addEventListener("wheel",s=>{s.preventDefault(),!k&&(J=Math.max(1.1,Math.min(8,J*Math.exp(s.deltaY*.001))),Me())},{passive:!1});window.addEventListener("pagehide",()=>{te(),ce?.dispose(),E?.destroy(),g?.destroy()});window.erosionMeshStudio={state:G,get sim(){return g},get mesh(){return F},get renderer(){return E},get topology(){return be},get busy(){return k},advance:s=>ae(()=>Pe(s))};async function dt(){k=!0,C(),g=new Y({canvas:L}),await g.initialize(),g.device.addEventListener("uncapturederror",e=>ee(e.error)),g.device.lost.then(e=>{e.reason!=="destroyed"&&(te(),G.ready=!1,ee(Error(e.message||"GPU device lost. Reload to reconnect.")))}),await Se();let s=Number(N.get("seconds"));Number.isFinite(s)&&s>0&&s<=720&&await Pe(s),N.get("mesh")==="1"&&await _e(),k=!1,C()}dt().catch(s=>{k=!1,ee(s)});})();
