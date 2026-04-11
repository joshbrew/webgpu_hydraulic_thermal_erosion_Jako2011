(()=>{var Ja=`(()=>{var STATE_FLOATS_PER_CELL=8;var VEC4_BYTES_PER_CELL=16;var WORKGROUP_SIZE_X=8;var WORKGROUP_SIZE_Y=8;function clamp(value,min,max){return value<min?min:value>max?max:value}function alignTo(value,alignment){return Math.ceil(value/alignment)*alignment}function degToRad(value){return value*Math.PI/180}function normalizeVec3(v){const len=Math.hypot(v[0],v[1],v[2])||1;return[v[0]/len,v[1]/len,v[2]/len]}function subtractVec3(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]]}function crossVec3(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}function multiplyMat4(a,b){const out=new Float32Array(16);for(let c=0;c<4;c++){for(let r=0;r<4;r++){out[c*4+r]=a[0*4+r]*b[c*4+0]+a[1*4+r]*b[c*4+1]+a[2*4+r]*b[c*4+2]+a[3*4+r]*b[c*4+3]}}return out}function perspectiveMat4(fovy,aspect,near,far){const f=1/Math.tan(fovy*.5);const nf=1/(near-far);const out=new Float32Array(16);out[0]=f/Math.max(aspect,1e-6);out[5]=f;out[10]=(far+near)*nf;out[11]=-1;out[14]=2*far*near*nf;return out}function lookAtMat4(eye,target,up){const z=normalizeVec3(subtractVec3(eye,target));const x=normalizeVec3(crossVec3(up,z));const y=crossVec3(z,x);const out=new Float32Array(16);out[0]=x[0];out[1]=y[0];out[2]=z[0];out[3]=0;out[4]=x[1];out[5]=y[1];out[6]=z[1];out[7]=0;out[8]=x[2];out[9]=y[2];out[10]=z[2];out[11]=0;out[12]=-(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]);out[13]=-(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]);out[14]=-(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]);out[15]=1;return out}var WebGPUTerrainErosion=class{constructor(options={}){this.canvas=options.canvas??null;this.adapter=null;this.device=null;this.context=null;this.presentationFormat=null;this.width=0;this.height=0;this.cellCount=0;this.stateByteLength=0;this.vec4ByteLength=0;this.iterationCount=0;this.allocatedStateByteLength=0;this.allocatedVec4ByteLength=0;this.dispatchX=0;this.dispatchY=0;this.drawCount=0;this.stateBuffers=[];this.fluxBuffer=null;this.velocityBuffer=null;this.thermalPipeBufferA=null;this.thermalPipeBufferB=null;this.readbackBuffer=null;this.customSpringBuffer=null;this.layerTopBuffer=null;this.layerSurfaceValues=new Float32Array(0);this.layerMaterialValues=new Float32Array(16);this.layerMaterialBuffer=null;this.paintedSpringMap=new Float32Array(0);this.allocatedPaintedSpringByteLength=0;this.allocatedReadbackByteLength=0;this.renderUniformValues=new Float32Array(40);this.renderUniformBufferTerrain=null;this.renderUniformBufferWater=null;this.depthTexture=null;this.depthTextureView=null;this.depthTextureSize={width:0,height:0};this.sceneColorTexture=null;this.sceneColorView=null;this.sceneColorSize={width:0,height:0};this.sceneSampler=null;this.dummySceneTexture=null;this.dummySceneView=null;this.paramValues=new Float32Array(36);this.paramBuffer=null;this.stepParamBuffer=null;this.stepParamStride=256;this.stepParamCapacity=0;this.stepParamValuesCpu=null;this.stepParamValuesCpuCapacity=0;this.stepParamTemplateDirty=true;this.stepParamPreparedIterations=0;this.computeDynamicOffset=new Uint32Array(1);this.terrainRenderBundle=null;this.finalRenderBundle=null;this.terrainRenderBundleList=[];this.finalRenderBundleList=[];this.computeBindGroupLayout=null;this.renderBindGroupLayout=null;this.compositeBindGroupLayout=null;this.computeBindGroup01=null;this.computeBindGroup10=null;this.renderBindGroupTerrain=null;this.renderBindGroupWater=null;this.compositeBindGroup=null;this.fluxPipeline=null;this.flowPipeline=null;this.erosionPipeline=null;this.transportPipeline=null;this.thermalOutflowPipeline=null;this.thermalApplyPipeline=null;this.renderPipeline=null;this.waterRenderPipeline=null;this.compositePipeline=null;this.initialized=false;this.ready=false;this.latestReadbackStats=null;this.lastErosionPassMs=0;this.lastRenderPassMs=0;this.avgErosionPassMs=0;this.avgRenderPassMs=0;this.lastCpuErosionPassMs=0;this.lastCpuRenderPassMs=0;this.avgCpuErosionPassMs=0;this.avgCpuRenderPassMs=0;this.lastGpuErosionPassMs=0;this.lastGpuRenderPassMs=0;this.avgGpuErosionPassMs=0;this.avgGpuRenderPassMs=0;this.layeredDemEnabled=false;this.layerConfigMode="single";this.simulationParams={cellSize:1,timeStep:.02,rainRate:.001,evaporationRate:.015,pipeArea:20,gravity:9.81,capacityScale:.82,suspensionRate:.32,depositionRate:1.22,softeningRate:2.8,maxErosionDepth:.09,thermalRate:.24,talusSlopeCoeff:.92,talusSlopeBias:.12,renderHeightScale:.08,waterOpacity:.28,sedimentTint:.35,hardnessBase:.16,hardnessVariation:.03,sourceCenterX:.5,sourceCenterY:.5,sourceRadius:4,sourceStrength:.06,sourceEnabled:false,rainDuration:0,pulse2Duration:0,sourceLayoutMode:0,randomSpringCount:4,sourceSeed:1,metersPerPixel:100,sourceTimeOffset:0,historyDecay:.9997,edgeDrainStrength:.08,edgeWaterFloor:0,renderMode:0,cameraAzimuthDeg:45,cameraElevationDeg:42,cameraDistance:2.9,cameraPosX:1.5238999619464006,cameraPosY:1.9404787584406888,cameraPosZ:1.5238999619464006,waterHeightScale:.12,thermalVisualizationScale:160,hydraulicErosionEnabled:true};this.setSimulationParams(options.simulationParams??{})}async initialize(canvas=this.canvas){if(!canvas){throw new Error("A canvas is required to initialize WebGPU erosion.")}this.canvas=canvas;this.#syncCanvasPixelSize();if(!navigator.gpu){throw new Error("WebGPU is not available in this browser.")}this.adapter=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!this.adapter){throw new Error("Failed to acquire a WebGPU adapter.")}this.device=await this.adapter.requestDevice();this.stepParamStride=Math.max(256,this.device.limits?.minUniformBufferOffsetAlignment||256);this.context=canvas.getContext("webgpu");if(!this.context){throw new Error("Failed to acquire a WebGPU canvas context.")}this.presentationFormat=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:this.presentationFormat,alphaMode:"opaque"});this.paramBuffer=this.device.createBuffer({size:alignTo(this.paramValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-erosion-params"});this.renderUniformBufferTerrain=this.device.createBuffer({size:alignTo(this.renderUniformValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-render-params-terrain"});this.renderUniformBufferWater=this.device.createBuffer({size:alignTo(this.renderUniformValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-render-params-water"});this.layerMaterialBuffer=this.device.createBuffer({size:alignTo(this.layerMaterialValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-layer-material-params"});this.sceneSampler=this.device.createSampler({addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge",magFilter:"linear",minFilter:"linear",mipmapFilter:"linear"});this.#ensureCustomSpringBuffer();this.#ensureDummySceneTexture();await this.#createPipelines();this.#writeParams();this.initialized=true}destroy(){for(const buffer of this.stateBuffers){buffer?.destroy?.()}this.stateBuffers=[];this.fluxBuffer?.destroy?.();this.velocityBuffer?.destroy?.();this.thermalPipeBufferA?.destroy?.();this.thermalPipeBufferB?.destroy?.();this.readbackBuffer?.destroy?.();this.customSpringBuffer?.destroy?.();this.layerTopBuffer?.destroy?.();this.depthTexture?.destroy?.();this.sceneColorTexture?.destroy?.();this.dummySceneTexture?.destroy?.();this.renderUniformBufferTerrain?.destroy?.();this.renderUniformBufferWater?.destroy?.();this.layerMaterialBuffer?.destroy?.();this.paramBuffer?.destroy?.();this.stepParamBuffer?.destroy?.();this.fluxBuffer=null;this.velocityBuffer=null;this.thermalPipeBufferA=null;this.thermalPipeBufferB=null;this.readbackBuffer=null;this.customSpringBuffer=null;this.layerTopBuffer=null;this.layerSurfaceValues=new Float32Array(0);this.layerMaterialBuffer=null;this.layerMaterialValues=new Float32Array(16);this.paintedSpringMap=new Float32Array(0);this.allocatedPaintedSpringByteLength=0;this.allocatedReadbackByteLength=0;this.depthTexture=null;this.depthTextureView=null;this.depthTextureSize={width:0,height:0};this.sceneColorTexture=null;this.sceneColorView=null;this.sceneColorSize={width:0,height:0};this.dummySceneTexture=null;this.dummySceneView=null;this.renderUniformBufferTerrain=null;this.renderUniformBufferWater=null;this.layerMaterialBuffer=null;this.paramBuffer=null;this.stepParamBuffer=null;this.stepParamCapacity=0;this.stepParamValuesCpu=null;this.stepParamValuesCpuCapacity=0;this.#invalidateRenderBundles();this.computeBindGroup01=null;this.computeBindGroup10=null;this.renderBindGroupTerrain=null;this.renderBindGroupWater=null;this.compositeBindGroup=null;this.ready=false;this.width=0;this.height=0;this.cellCount=0;this.iterationCount=0;this.dispatchX=0;this.dispatchY=0;this.drawCount=0;this.latestReadbackStats=null;this.lastErosionPassMs=0;this.lastRenderPassMs=0;this.avgErosionPassMs=0;this.avgRenderPassMs=0;this.lastCpuErosionPassMs=0;this.lastCpuRenderPassMs=0;this.avgCpuErosionPassMs=0;this.avgCpuRenderPassMs=0;this.lastGpuErosionPassMs=0;this.lastGpuRenderPassMs=0;this.avgGpuErosionPassMs=0;this.avgGpuRenderPassMs=0;this.paintedSpringMap=new Float32Array(0)}setSimulationParams(next={}){const normalized={...next};if(Number.isFinite(normalized.erodabilityBase)&&!Number.isFinite(normalized.hardnessBase))normalized.hardnessBase=normalized.erodabilityBase;if(Number.isFinite(normalized.erodabilityVariation)&&!Number.isFinite(normalized.hardnessVariation))normalized.hardnessVariation=normalized.erodabilityVariation;Object.assign(this.simulationParams,normalized);if(this.paramBuffer&&this.device){this.#writeParams()}if((this.renderUniformBufferTerrain||this.renderUniformBufferWater)&&this.device){this.#writeRenderParams()}}resetRainTimer(){this.simulationParams.sourceTimeOffset=this.iterationCount*this.simulationParams.timeStep;if(this.paramBuffer&&this.device){this.#writeParams()}}restartSources(){this.resetRainTimer()}async applyTerrainBrush(brush={}){if(!this.ready||this.width<=0||this.height<=0){throw new Error("Terrain painting is only available after the simulation is initialized.")}const centerX=clamp(Number(brush.x)||0,0,Math.max(this.width-1,0));const centerY=clamp(Number(brush.y)||0,0,Math.max(this.height-1,0));const radius=Math.max(.5,Number(brush.radius)||1);const amount=Math.max(1e-4,Number(brush.amount)||.01);const hardness=clamp(Number(brush.hardness)||.5,.01,1);const subtract=!!brush.subtract;const data=await this.#readbackStateData();const minX=Math.max(0,Math.floor(centerX-radius-1));const maxX=Math.min(this.width-1,Math.ceil(centerX+radius+1));const minY=Math.max(0,Math.floor(centerY-radius-1));const maxY=Math.min(this.height-1,Math.ceil(centerY+radius+1));const falloffPower=2.65-hardness*2.2;const direction=subtract?-1:1;for(let y=minY;y<=maxY;y++){for(let x=minX;x<=maxX;x++){const dx=x-centerX;const dy=y-centerY;const dist=Math.hypot(dx,dy);if(dist>radius)continue;const t=1-dist/radius;const falloff=Math.pow(clamp(t,0,1),falloffPower);const i=y*this.width+x;const base=i*STATE_FLOATS_PER_CELL;if(data[base+4]<.5)continue;const terrain=clamp(data[base]+direction*amount*falloff,0,2);data[base]=terrain;const fallbackHardness=Number.isFinite(data[base+3])?data[base+3]:this.simulationParams.hardnessBase;data[base+3]=this.layerSurfaceValues.length===this.cellCount*4?this.#resolveLayerHardnessForTerrain(this.layerSurfaceValues,i*4,terrain,fallbackHardness):fallbackHardness}}this.device.queue.writeBuffer(this.stateBuffers[0],0,data);this.device.queue.writeBuffer(this.stateBuffers[1],0,data);this.latestReadbackStats=null;this.render();return this.getStats()}paintSpringBrush(brush={}){if(!this.ready||this.width<=0||this.height<=0){throw new Error("Spring painting is only available after the simulation is initialized.")}const centerX=clamp(Number(brush.x)||0,0,Math.max(this.width-1,0));const centerY=clamp(Number(brush.y)||0,0,Math.max(this.height-1,0));const radius=Math.max(.5,Number(brush.radius)||1);const amount=Math.max(0,Number(brush.strength)||0);const hardness=clamp(Number(brush.hardness)||.5,.01,1);const erase=!!brush.erase;this.#ensurePaintedSpringMap();const minX=Math.max(0,Math.floor(centerX-radius-1));const maxX=Math.min(this.width-1,Math.ceil(centerX+radius+1));const minY=Math.max(0,Math.floor(centerY-radius-1));const maxY=Math.min(this.height-1,Math.ceil(centerY+radius+1));const falloffPower=2.65-hardness*2.2;for(let y=minY;y<=maxY;y++){for(let x=minX;x<=maxX;x++){const dx=x-centerX;const dy=y-centerY;const dist=Math.hypot(dx,dy);if(dist>radius)continue;const t=1-dist/radius;const falloff=Math.pow(clamp(t,0,1),falloffPower);const i=y*this.width+x;if(erase){this.paintedSpringMap[i]=Math.max(0,this.paintedSpringMap[i]-Math.max(amount,1)*falloff)}else{this.paintedSpringMap[i]=clamp(this.paintedSpringMap[i]+amount*falloff,0,1)}}}this.#writeCustomSpringBuffer();this.#writeParams();this.render();return this.getSourcePoints()}clearPaintedSprings(){if(this.paintedSpringMap instanceof Float32Array&&this.paintedSpringMap.length===this.cellCount){this.paintedSpringMap.fill(0)}this.#writeCustomSpringBuffer();this.#writeParams();this.render();return this.getSourcePoints()}#normalizeLayerConfigArray(options={}){const baseHardMin=clamp(this.simulationParams.hardnessBase??.16,0,1);const baseHardMax=clamp(baseHardMin+Math.max(0,this.simulationParams.hardnessVariation??.03),0,1);const defaults=[{enabled:true,label:"Base",hardnessMin:baseHardMin,hardnessMax:baseHardMax,thermalEnabled:true},{enabled:false,label:"Silt",hardnessMin:.08,hardnessMax:.16,thermalEnabled:true},{enabled:false,label:"Clay",hardnessMin:.16,hardnessMax:.28,thermalEnabled:true},{enabled:false,label:"Bedrock",hardnessMin:.58,hardnessMax:.9,thermalEnabled:true}];const inputLayers=Array.isArray(options.layerConfig?.layers)?options.layerConfig.layers:[];return defaults.map((fallback,index)=>{const src=inputLayers[index]||{};const heightMin=Number.isFinite(src.heightMin)?src.heightMin:0;const heightMax=Number.isFinite(src.heightMax)?src.heightMax:1;const hardMin=Number.isFinite(src.hardnessMin)?src.hardnessMin:Number.isFinite(src.erodabilityMin)?src.erodabilityMin:fallback.hardnessMin;const hardMax=Number.isFinite(src.hardnessMax)?src.hardnessMax:Number.isFinite(src.erodabilityMax)?src.erodabilityMax:fallback.hardnessMax;return{enabled:!!src.enabled,label:src.label||fallback.label,preset:src.preset||"custom",heightMin,heightMax,hardnessMin:clamp(Math.min(hardMin,hardMax),0,1),hardnessMax:clamp(Math.max(hardMin,hardMax),0,1),thermalEnabled:src.thermalEnabled!==false}})}#resolveLayerHardnessForTerrain(surfaceArray,offset,terrain,fallbackHardness){let chosenIndex=-1;let chosenTop=Infinity;let highestTop=-Infinity;let highestIndex=-1;for(let layerIndex=0;layerIndex<4;layerIndex++){const matBase=layerIndex*4;if(this.layerMaterialValues[matBase+2]<.5)continue;const top=surfaceArray[offset+layerIndex];if(!Number.isFinite(top)||top<-1e8)continue;if(top>highestTop){highestTop=top;highestIndex=layerIndex}if(top>=terrain&&top<chosenTop){chosenTop=top;chosenIndex=layerIndex}}if(chosenIndex<0){chosenIndex=highestIndex;chosenTop=highestTop}if(chosenIndex<0||!Number.isFinite(chosenTop)){return clamp(fallbackHardness,0,1)}let lowerTop=-Infinity;for(let layerIndex=0;layerIndex<4;layerIndex++){const matBase=layerIndex*4;if(this.layerMaterialValues[matBase+2]<.5)continue;const top=surfaceArray[offset+layerIndex];if(!Number.isFinite(top)||top<-1e8||top>=chosenTop)continue;if(top>lowerTop)lowerTop=top}let depthT=0;if(Number.isFinite(lowerTop)&&lowerTop>-1e8){depthT=clamp((chosenTop-terrain)/Math.max(chosenTop-lowerTop,1e-6),0,1)}else if(terrain<chosenTop){depthT=1}const materialBase=chosenIndex*4;const hardMin=this.layerMaterialValues[materialBase];const hardMax=this.layerMaterialValues[materialBase+1];return clamp(hardMin+(hardMax-hardMin)*depthT,0,1)}#writeLayerMaterialParams(){if(this.layerMaterialBuffer&&this.device){this.device.queue.writeBuffer(this.layerMaterialBuffer,0,this.layerMaterialValues)}}getSourcePoints(){if(this.width<=0||this.height<=0){return[]}const layoutMode=this.simulationParams.sourceLayoutMode|0;const springsEnabled=!!this.simulationParams.sourceEnabled;if(layoutMode===0){const values=this.paintedSpringMap;if(!(values instanceof Float32Array)||values.length!==this.cellCount){return[]}const points2=[];const threshold=.0025;let activeCount=0;for(let i=0;i<values.length;i++){if(values[i]>threshold)activeCount++}const maxPoints=8192;const stride=Math.max(1,Math.ceil(activeCount/maxPoints));let seen=0;for(let y=0;y<this.height;y++){for(let x=0;x<this.width;x++){const i=y*this.width+x;const strength=values[i];if(strength<=threshold)continue;if(seen++%stride!==0)continue;points2.push({x,y,radius:.7,strength,painted:true,active:springsEnabled})}}return points2}if(!springsEnabled){return[]}const points=[];const count=Math.min(16,Math.max(1,this.simulationParams.randomSpringCount|0));for(let i=0;i<count;i++){const seed=(this.simulationParams.sourceSeed??1)*37+i*17+1;const x=Math.sin(seed*127.1+11.7)*43758.5453;const y=Math.sin(seed*311.7+73.1)*24634.6345;points.push({x:(x-Math.floor(x))*Math.max(this.width-1,0),y:(y-Math.floor(y))*Math.max(this.height-1,0),radius:this.simulationParams.sourceRadius,strength:this.simulationParams.sourceStrength,painted:false,active:true})}return points}async setDEM(raster,options={}){if(!this.initialized){await this.initialize(this.canvas)}if(!raster||!(raster.values instanceof Float32Array)){throw new Error("setDEM expects a raster with Float32Array values.")}const width=raster.width|0;const height=raster.height|0;if(width<=0||height<=0){throw new Error("Raster width and height must be positive.")}if(raster.values.length!==width*height){throw new Error("Raster value count does not match width * height.")}const minHeight=Number.isFinite(options.minHeight)?options.minHeight:0;const maxHeight=Number.isFinite(options.maxHeight)?options.maxHeight:1;const range=Math.max(1e-6,maxHeight-minHeight);const mask=raster.mask instanceof Uint8Array&&raster.mask.length===width*height?raster.mask:null;const rasterBands=Array.isArray(raster.bands)?raster.bands:null;const layerConfigs=this.#normalizeLayerConfigArray(options);const demSourceMode=options.demSourceMode||options.layerConfig?.mode||"single";const hasLayerBands=Array.isArray(rasterBands)&&rasterBands.some(band=>band instanceof Float32Array&&band.length===width*height);const useLayeredDem=hasLayerBands||demSourceMode!=="single";this.width=width;this.height=height;this.cellCount=width*height;this.stateByteLength=this.cellCount*STATE_FLOATS_PER_CELL*4;this.vec4ByteLength=this.cellCount*VEC4_BYTES_PER_CELL;this.iterationCount=0;this.dispatchX=Math.ceil(this.width/WORKGROUP_SIZE_X);this.dispatchY=Math.ceil(this.height/WORKGROUP_SIZE_Y);this.drawCount=Math.max(1,(this.width-1)*(this.height-1));this.#invalidateRenderBundles();this.#ensureDepthTexture();this.#ensureSceneTargets();this.#ensureBuffers();this.#writeParams();const initialState=new Float32Array(this.cellCount*STATE_FLOATS_PER_CELL);const layerSurfaceValues=useLayeredDem?new Float32Array(this.cellCount*4):null;if(layerSurfaceValues)layerSurfaceValues.fill(-1e9);const hardnessBase=this.simulationParams.hardnessBase;const hardnessVariation=this.simulationParams.hardnessVariation;this.layerMaterialValues.fill(0);for(let layerIndex=0;layerIndex<4;layerIndex++){const config=layerConfigs[layerIndex];const base=layerIndex*4;this.layerMaterialValues[base]=config.hardnessMin;this.layerMaterialValues[base+1]=config.hardnessMax;const hasBand=!!(rasterBands&&rasterBands[layerIndex]instanceof Float32Array&&rasterBands[layerIndex].length===this.cellCount);const fallbackSingleLayer=useLayeredDem&&!hasLayerBands&&layerIndex===0;this.layerMaterialValues[base+2]=config.enabled&&(hasBand||fallbackSingleLayer)?1:0;this.layerMaterialValues[base+3]=config.thermalEnabled?1:0}for(let i=0;i<this.cellCount;i++){const base=i*STATE_FLOATS_PER_CELL;const layerBase=i*4;const valid=mask?mask[i]?1:0:1;const fallbackTerrain=minHeight+raster.values[i]*range;const normalizedTerrain=clamp(raster.values[i],0,1);const noise=Math.sin(i*12.9898+width*.031+height*.017)*43758.5453;const frac=noise-Math.floor(noise);const noiseTerm=(frac*2-1)*hardnessVariation*.25;const fallbackHardness=clamp(hardnessBase+(1-normalizedTerrain)*hardnessVariation+noiseTerm,0,1);let terrain=fallbackTerrain;let hardness=fallbackHardness;if(useLayeredDem){let highestLayerTop=-Infinity;for(let layerIndex=0;layerIndex<4;layerIndex++){const matBase=layerIndex*4;if(this.layerMaterialValues[matBase+2]<.5)continue;const band=rasterBands&&rasterBands[layerIndex]instanceof Float32Array&&rasterBands[layerIndex].length===this.cellCount?rasterBands[layerIndex]:layerIndex===0?raster.values:null;if(!band)continue;const config=layerConfigs[layerIndex];const normalizedBand=clamp(band[i],0,1);const topHeight=config.heightMin+normalizedBand*(config.heightMax-config.heightMin);layerSurfaceValues[layerBase+layerIndex]=topHeight;if(topHeight>highestLayerTop)highestLayerTop=topHeight}if(Number.isFinite(highestLayerTop)&&highestLayerTop>-1e8){terrain=highestLayerTop}else{layerSurfaceValues[layerBase]=terrain;this.layerMaterialValues[2]=1}hardness=this.#resolveLayerHardnessForTerrain(layerSurfaceValues,layerBase,terrain,fallbackHardness)}initialState[base]=terrain;initialState[base+1]=0;initialState[base+2]=0;initialState[base+3]=hardness;initialState[base+4]=valid;initialState[base+5]=0;initialState[base+6]=0;initialState[base+7]=0}const clearEncoder=this.device.createCommandEncoder({label:"paper-reset-sim-scratch"});clearEncoder.clearBuffer(this.fluxBuffer);clearEncoder.clearBuffer(this.velocityBuffer);clearEncoder.clearBuffer(this.thermalPipeBufferA);clearEncoder.clearBuffer(this.thermalPipeBufferB);if(!useLayeredDem){clearEncoder.clearBuffer(this.layerTopBuffer)}this.device.queue.submit([clearEncoder.finish()]);this.device.queue.writeBuffer(this.stateBuffers[0],0,initialState);this.device.queue.writeBuffer(this.stateBuffers[1],0,initialState);if(useLayeredDem&&layerSurfaceValues){this.device.queue.writeBuffer(this.layerTopBuffer,0,layerSurfaceValues)}this.layeredDemEnabled=useLayeredDem;this.layerSurfaceValues=this.layeredDemEnabled&&layerSurfaceValues?layerSurfaceValues:new Float32Array(0);this.#writeLayerMaterialParams();this.layerConfigMode=demSourceMode;this.ready=true;this.latestReadbackStats=null;this.paintedSpringMap=new Float32Array(0);this.#writeCustomSpringBuffer();this.#writeRenderParams();this.#ensureDepthTexture();this.#ensureSceneTargets();this.render()}resize(){if(!this.context||!this.device||!this.canvas)return;this.#syncCanvasPixelSize();this.context.configure({device:this.device,format:this.presentationFormat,alphaMode:"opaque"});this.#ensureDepthTexture();this.#ensureSceneTargets();this.#writeRenderParams()}#createStepCommandBuffer(iterations=1,label="paper-erosion-step"){const encoder=this.device.createCommandEncoder({label});this.#encodeStepCommands(encoder,Math.max(1,iterations|0));return encoder.finish()}#createRenderCommandBuffer(label="paper-erosion-render"){this.#ensureDepthTexture();this.#ensureSceneTargets();this.#writeRenderParams();const view=this.context.getCurrentTexture().createView();const encoder=this.device.createCommandEncoder({label});this.#encodeRenderCommands(encoder,view);return encoder.finish()}async#submitForGpuTiming(commandBuffer){const submitAt=performance.now();this.device.queue.submit([commandBuffer]);await this.device.queue.onSubmittedWorkDone();return Math.max(0,performance.now()-submitAt)}async stepBench(iterations=1){if(!this.ready)return null;const cpuStartAt=performance.now();const commandBuffer=this.#createStepCommandBuffer(Math.max(1,iterations|0),"paper-erosion-step-bench");const cpuMs=Math.max(0,performance.now()-cpuStartAt);const gpuMs=await this.#submitForGpuTiming(commandBuffer);this.#recordErosionPassMs(cpuMs);this.#recordGpuErosionPassMs(gpuMs);return{cpuErosionMs:cpuMs,gpuErosionMs:gpuMs,cpuRenderMs:0,gpuRenderMs:0}}async stepAndRenderBench(iterations=1){if(!this.ready)return null;const canRender=!!(this.context&&this.width>=2&&this.height>=2);const erosionCpuStartAt=performance.now();const erosionCommandBuffer=this.#createStepCommandBuffer(Math.max(1,iterations|0),canRender?"paper-erosion-step-bench":"paper-erosion-step-render-bench");const cpuErosionMs=Math.max(0,performance.now()-erosionCpuStartAt);const gpuErosionMs=await this.#submitForGpuTiming(erosionCommandBuffer);this.#recordErosionPassMs(cpuErosionMs);this.#recordGpuErosionPassMs(gpuErosionMs);let cpuRenderMs=0;let gpuRenderMs=0;if(canRender){const renderCpuStartAt=performance.now();const renderCommandBuffer=this.#createRenderCommandBuffer("paper-erosion-render-bench");cpuRenderMs=Math.max(0,performance.now()-renderCpuStartAt);gpuRenderMs=await this.#submitForGpuTiming(renderCommandBuffer);this.#recordRenderPassMs(cpuRenderMs);this.#recordGpuRenderPassMs(gpuRenderMs)}else{this.#recordRenderPassMs(0)}return{cpuErosionMs,gpuErosionMs,cpuRenderMs,gpuRenderMs}}step(iterations=1){if(!this.ready)return;const startAt=performance.now();const commandBuffer=this.#createStepCommandBuffer(Math.max(1,iterations|0),"paper-erosion-step");this.device.queue.submit([commandBuffer]);this.#recordErosionPassMs(Math.max(0,performance.now()-startAt))}stepAndRender(iterations=1){if(!this.ready)return;const canRender=!!(this.context&&this.width>=2&&this.height>=2);const encoder=this.device.createCommandEncoder({label:canRender?"paper-erosion-step-render":"paper-erosion-step"});const erosionStartAt=performance.now();this.#encodeStepCommands(encoder,Math.max(1,iterations|0));const erosionEndAt=performance.now();let renderStartAt=erosionEndAt;let renderEndAt=erosionEndAt;if(canRender){renderStartAt=performance.now();this.#ensureDepthTexture();this.#ensureSceneTargets();this.#writeRenderParams();const view=this.context.getCurrentTexture().createView();this.#encodeRenderCommands(encoder,view);renderEndAt=performance.now()}const submitStartAt=performance.now();this.device.queue.submit([encoder.finish()]);const submitMs=performance.now()-submitStartAt;let erosionMs=Math.max(0,erosionEndAt-erosionStartAt);let renderMs=Math.max(0,renderEndAt-renderStartAt);if(canRender&&erosionMs+renderMs>1e-6){const totalMs=erosionMs+renderMs;erosionMs+=submitMs*(erosionMs/totalMs);renderMs+=submitMs*(renderMs/totalMs)}else{erosionMs+=submitMs;renderMs=0}this.#recordErosionPassMs(erosionMs);this.#recordRenderPassMs(renderMs)}render(){if(!this.ready||!this.context||this.width<2||this.height<2)return;const startAt=performance.now();const commandBuffer=this.#createRenderCommandBuffer("paper-erosion-render");this.device.queue.submit([commandBuffer]);this.#recordRenderPassMs(Math.max(0,performance.now()-startAt))}async readbackStats(){if(!this.ready){return this.getStats()}const data=await this.#readbackStateData();let validCells=0;let minTerrain=Infinity;let maxTerrain=-Infinity;let minWater=Infinity;let maxWater=-Infinity;let minSediment=Infinity;let maxSediment=-Infinity;let totalWater=0;let totalSediment=0;let hardnessSum=0;let minHistory=Infinity;let maxHistory=-Infinity;for(let i=0;i<this.cellCount;i++){const base=i*STATE_FLOATS_PER_CELL;const mask=data[base+4];if(mask<.5)continue;validCells++;const terrain=data[base];const water=data[base+1];const sediment=data[base+2];const hardness=data[base+3];const history=data[base+7];if(terrain<minTerrain)minTerrain=terrain;if(terrain>maxTerrain)maxTerrain=terrain;if(water<minWater)minWater=water;if(water>maxWater)maxWater=water;if(sediment<minSediment)minSediment=sediment;if(sediment>maxSediment)maxSediment=sediment;totalWater+=water;totalSediment+=sediment;hardnessSum+=hardness;if(history<minHistory)minHistory=history;if(history>maxHistory)maxHistory=history}this.latestReadbackStats={validCells,terrainRange:Number.isFinite(minTerrain)?{min:minTerrain,max:maxTerrain}:null,waterRange:Number.isFinite(minWater)?{min:minWater,max:maxWater}:null,sedimentRange:Number.isFinite(minSediment)?{min:minSediment,max:maxSediment}:null,historyRange:Number.isFinite(minHistory)?{min:minHistory,max:maxHistory}:null,totalWater,totalSediment,averageHardness:validCells>0?hardnessSum/validCells:0};return this.getStats()}async exportTerrainPng(){if(!this.ready||this.width<=0||this.height<=0){throw new Error("Terrain export is only available after the simulation is initialized.")}if(typeof OffscreenCanvas!=="function"){throw new Error("OffscreenCanvas is not available for terrain export.")}const data=await this.#readbackStateData();let minTerrain=Infinity;let maxTerrain=-Infinity;for(let i=0;i<this.cellCount;i++){const base=i*STATE_FLOATS_PER_CELL;if(data[base+4]<.5)continue;const terrain=data[base];if(terrain<minTerrain)minTerrain=terrain;if(terrain>maxTerrain)maxTerrain=terrain}if(!Number.isFinite(minTerrain)||!Number.isFinite(maxTerrain)){minTerrain=0;maxTerrain=1}const range=Math.max(maxTerrain-minTerrain,1e-6);const rgba=new Uint8ClampedArray(this.width*this.height*4);for(let y=0;y<this.height;y++){const outY=this.height-1-y;for(let x=0;x<this.width;x++){const srcIndex=y*this.width+x;const base=srcIndex*STATE_FLOATS_PER_CELL;const dstIndex=(outY*this.width+x)*4;const mask=data[base+4]>=.5;const terrain=data[base];const normalized=mask?clamp((terrain-minTerrain)/range,0,1):0;const value=Math.max(0,Math.min(255,Math.round(normalized*255)));rgba[dstIndex]=value;rgba[dstIndex+1]=value;rgba[dstIndex+2]=value;rgba[dstIndex+3]=mask?255:0}}const canvas=new OffscreenCanvas(this.width,this.height);const ctx=canvas.getContext("2d");if(!ctx){throw new Error("Failed to create a terrain export canvas.")}ctx.putImageData(new ImageData(rgba,this.width,this.height),0,0);const blob=await canvas.convertToBlob({type:"image/png"});const arrayBuffer=await blob.arrayBuffer();return{width:this.width,height:this.height,minTerrain,maxTerrain,data:arrayBuffer}}#blendAverage(prev,next){return prev>0?prev*.88+next*.12:next}#recordErosionPassMs(value){const ms=Math.max(0,Number(value)||0);this.lastErosionPassMs=ms;this.lastCpuErosionPassMs=ms;this.avgErosionPassMs=this.#blendAverage(this.avgErosionPassMs,ms);this.avgCpuErosionPassMs=this.#blendAverage(this.avgCpuErosionPassMs,ms)}#recordRenderPassMs(value){const ms=Math.max(0,Number(value)||0);this.lastRenderPassMs=ms;this.lastCpuRenderPassMs=ms;this.avgRenderPassMs=this.#blendAverage(this.avgRenderPassMs,ms);this.avgCpuRenderPassMs=this.#blendAverage(this.avgCpuRenderPassMs,ms)}#recordGpuErosionPassMs(value){const ms=Math.max(0,Number(value)||0);this.lastGpuErosionPassMs=ms;this.avgGpuErosionPassMs=this.#blendAverage(this.avgGpuErosionPassMs,ms)}#recordGpuRenderPassMs(value){const ms=Math.max(0,Number(value)||0);this.lastGpuRenderPassMs=ms;this.avgGpuRenderPassMs=this.#blendAverage(this.avgGpuRenderPassMs,ms)}getStats(){const simTime=this.iterationCount*this.simulationParams.timeStep;const rainDuration=Math.max(0,this.simulationParams.rainDuration??0);const rainActive=rainDuration<=0?true:simTime<rainDuration;return{width:this.width,height:this.height,cellCount:this.cellCount,ready:this.ready,iterationCount:this.iterationCount,simTime,rainActive,rainDuration,layeredDemEnabled:this.layeredDemEnabled,layerConfigMode:this.layerConfigMode,activeLayerCount:[0,1,2,3].reduce((sum,layerIndex)=>sum+(this.layerMaterialValues[layerIndex*4+2]>.5?1:0),0),lastErosionPassMs:this.lastErosionPassMs,lastRenderPassMs:this.lastRenderPassMs,avgErosionPassMs:this.avgErosionPassMs,avgRenderPassMs:this.avgRenderPassMs,lastCpuErosionPassMs:this.lastCpuErosionPassMs,lastCpuRenderPassMs:this.lastCpuRenderPassMs,avgCpuErosionPassMs:this.avgCpuErosionPassMs,avgCpuRenderPassMs:this.avgCpuRenderPassMs,lastGpuErosionPassMs:this.lastGpuErosionPassMs,lastGpuRenderPassMs:this.lastGpuRenderPassMs,avgGpuErosionPassMs:this.avgGpuErosionPassMs,avgGpuRenderPassMs:this.avgGpuRenderPassMs,...this.latestReadbackStats??{}}}#syncCanvasPixelSize(){if(!this.canvas)return;if(typeof this.canvas.getBoundingClientRect!=="function"){const pixelWidth2=Math.max(1,Math.round(this.canvas.width||1));const pixelHeight2=Math.max(1,Math.round(this.canvas.height||1));if(this.canvas.width!==pixelWidth2)this.canvas.width=pixelWidth2;if(this.canvas.height!==pixelHeight2)this.canvas.height=pixelHeight2;return}const rect=this.canvas.getBoundingClientRect();const cssWidth=Math.max(1,Math.round(rect.width||this.canvas.width||1));const cssHeight=Math.max(1,Math.round(rect.height||this.canvas.height||1));const dpr=Math.max(1,Math.min(globalThis.devicePixelRatio||1,2));const pixelWidth=Math.max(1,Math.round(cssWidth*dpr));const pixelHeight=Math.max(1,Math.round(cssHeight*dpr));if(this.canvas.width!==pixelWidth)this.canvas.width=pixelWidth;if(this.canvas.height!==pixelHeight)this.canvas.height=pixelHeight}#ensureDepthTexture(){if(!this.device||!this.canvas)return;const width=Math.max(1,this.canvas.width||this.canvas.clientWidth||1);const height=Math.max(1,this.canvas.height||this.canvas.clientHeight||1);if(this.depthTexture&&this.depthTextureSize.width===width&&this.depthTextureSize.height===height){return}this.depthTexture?.destroy?.();this.depthTexture=this.device.createTexture({size:{width,height},format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT,label:"paper-depth"});this.depthTextureView=this.depthTexture.createView();this.depthTextureSize={width,height}}#ensureSceneTargets(){if(!this.device||!this.canvas)return;const width=Math.max(1,this.canvas.width||this.canvas.clientWidth||1);const height=Math.max(1,this.canvas.height||this.canvas.clientHeight||1);if(this.sceneColorTexture&&this.sceneColorSize.width===width&&this.sceneColorSize.height===height){return}this.sceneColorTexture?.destroy?.();this.sceneColorTexture=this.device.createTexture({size:{width,height},format:this.presentationFormat,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,label:"paper-scene-color"});this.sceneColorView=this.sceneColorTexture.createView();this.sceneColorSize={width,height};if(this.stateBuffers.length===2&&this.sceneSampler){this.#rebuildBindGroups()}}async#readbackStateData(){this.#ensureReadbackBuffer();const encoder=this.device.createCommandEncoder({label:"paper-erosion-readback"});encoder.copyBufferToBuffer(this.stateBuffers[0],0,this.readbackBuffer,0,this.stateByteLength);this.device.queue.submit([encoder.finish()]);await this.readbackBuffer.mapAsync(GPUMapMode.READ);const mapped=this.readbackBuffer.getMappedRange();const copy=mapped.slice(0);this.readbackBuffer.unmap();return new Float32Array(copy)}#ensureCustomSpringBuffer(){if(!this.device)return;const size=Math.max(4,this.cellCount*4);if(this.customSpringBuffer&&this.allocatedPaintedSpringByteLength===size)return;this.customSpringBuffer?.destroy?.();this.customSpringBuffer=this.device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-painted-springs"});this.allocatedPaintedSpringByteLength=size;this.#clearCustomSpringBuffer()}#writeCustomSpringBuffer(){if(!this.device)return;this.#ensureCustomSpringBuffer();if(!this.customSpringBuffer)return;if(!(this.paintedSpringMap instanceof Float32Array)||this.paintedSpringMap.length!==this.cellCount){this.#clearCustomSpringBuffer();return}this.device.queue.writeBuffer(this.customSpringBuffer,0,this.paintedSpringMap)}#ensureReadbackBuffer(){if(!this.device)return;if(this.readbackBuffer&&this.allocatedReadbackByteLength===this.stateByteLength)return;this.readbackBuffer?.destroy?.();this.readbackBuffer=this.device.createBuffer({size:this.stateByteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,label:"paper-readback"});this.allocatedReadbackByteLength=this.stateByteLength}#ensurePaintedSpringMap(){if(!(this.paintedSpringMap instanceof Float32Array)||this.paintedSpringMap.length!==this.cellCount){this.paintedSpringMap=new Float32Array(this.cellCount)}}#clearCustomSpringBuffer(){if(!this.device)return;this.#ensureCustomSpringBuffer();if(!this.customSpringBuffer)return;const encoder=this.device.createCommandEncoder({label:"paper-clear-painted-springs"});encoder.clearBuffer(this.customSpringBuffer);this.device.queue.submit([encoder.finish()])}#ensureDummySceneTexture(){if(!this.device||!this.presentationFormat||this.dummySceneTexture)return;this.dummySceneTexture=this.device.createTexture({size:{width:1,height:1},format:this.presentationFormat,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST,label:"paper-dummy-scene-color"});this.dummySceneView=this.dummySceneTexture.createView();this.device.queue.writeTexture({texture:this.dummySceneTexture},new Uint8Array([14,17,20,255]),{bytesPerRow:256},{width:1,height:1,depthOrArrayLayers:1})}#invalidateRenderBundles(){this.terrainRenderBundle=null;this.finalRenderBundle=null;this.terrainRenderBundleList.length=0;this.finalRenderBundleList.length=0}#ensureRenderBundles(){if(!this.device||!this.renderPipeline||!this.compositePipeline||!this.waterRenderPipeline)return;if(!this.renderBindGroupTerrain||!this.renderBindGroupWater||!this.compositeBindGroup)return;if(this.terrainRenderBundle&&this.finalRenderBundle)return;const drawCount=this.drawCount||Math.max(1,(this.width-1)*(this.height-1));const terrainEncoder=this.device.createRenderBundleEncoder({colorFormats:[this.presentationFormat],depthStencilFormat:"depth24plus"});terrainEncoder.setPipeline(this.renderPipeline);terrainEncoder.setBindGroup(0,this.renderBindGroupTerrain);terrainEncoder.draw(6,drawCount,0,0);this.terrainRenderBundle=terrainEncoder.finish({label:"paper-render-terrain-bundle"});this.terrainRenderBundleList[0]=this.terrainRenderBundle;const finalEncoder=this.device.createRenderBundleEncoder({colorFormats:[this.presentationFormat],depthStencilFormat:"depth24plus"});finalEncoder.setPipeline(this.compositePipeline);finalEncoder.setBindGroup(0,this.compositeBindGroup);finalEncoder.draw(3,1,0,0);finalEncoder.setPipeline(this.waterRenderPipeline);finalEncoder.setBindGroup(0,this.renderBindGroupWater);finalEncoder.draw(6,drawCount,0,0);this.finalRenderBundle=finalEncoder.finish({label:"paper-render-final-bundle"});this.finalRenderBundleList[0]=this.finalRenderBundle}#encodeStepCommands(encoder,iterCount){if(!this.ready)return;const dispatchX=this.dispatchX||Math.ceil(this.width/WORKGROUP_SIZE_X);const dispatchY=this.dispatchY||Math.ceil(this.height/WORKGROUP_SIZE_Y);this.#ensureStepParamTemplate(iterCount);const strideFloats=this.stepParamStride>>>2;const stepParamValues=this.stepParamValuesCpu;const dt=this.paramValues[3];const baseTime=this.iterationCount*dt;for(let i=0;i<iterCount;i++){const offset=i*strideFloats;stepParamValues[offset+25]=baseTime+i*dt}this.device.queue.writeBuffer(this.stepParamBuffer,0,stepParamValues.buffer,0,this.stepParamStride*iterCount);const fluxPipeline=this.fluxPipeline;const flowPipeline=this.flowPipeline;const thermalOutflowPipeline=this.thermalOutflowPipeline;const erosionPipeline=this.erosionPipeline;const transportPipeline=this.transportPipeline;const thermalApplyPipeline=this.thermalApplyPipeline;const computeBindGroup01=this.computeBindGroup01;const computeBindGroup10=this.computeBindGroup10;const dynamicOffset=this.computeDynamicOffset;const pass=encoder.beginComputePass({label:"paper-erosion-compute"});for(let i=0;i<iterCount;i++){dynamicOffset[0]=i*this.stepParamStride;pass.setBindGroup(0,computeBindGroup01,dynamicOffset);pass.setPipeline(fluxPipeline);pass.dispatchWorkgroups(dispatchX,dispatchY);pass.setPipeline(flowPipeline);pass.dispatchWorkgroups(dispatchX,dispatchY);pass.setBindGroup(0,computeBindGroup10,dynamicOffset);pass.setPipeline(thermalOutflowPipeline);pass.dispatchWorkgroups(dispatchX,dispatchY);pass.setPipeline(erosionPipeline);pass.dispatchWorkgroups(dispatchX,dispatchY);pass.setBindGroup(0,computeBindGroup01,dynamicOffset);pass.setPipeline(transportPipeline);pass.dispatchWorkgroups(dispatchX,dispatchY);pass.setBindGroup(0,computeBindGroup10,dynamicOffset);pass.setPipeline(thermalApplyPipeline);pass.dispatchWorkgroups(dispatchX,dispatchY)}pass.end();this.iterationCount+=iterCount;this.#fillParamValues(this.iterationCount,this.paramValues)}#encodeRenderCommands(encoder,view){this.#ensureRenderBundles();{const pass=encoder.beginRenderPass({colorAttachments:[{view:this.sceneColorView,clearValue:{r:.055,g:.065,b:.08,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:this.depthTextureView?{view:this.depthTextureView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}:void 0});if(this.terrainRenderBundle){pass.executeBundles(this.terrainRenderBundleList)}else{const drawCount=this.drawCount||Math.max(1,(this.width-1)*(this.height-1));pass.setPipeline(this.renderPipeline);pass.setBindGroup(0,this.renderBindGroupTerrain);pass.draw(6,drawCount,0,0)}pass.end()}{const pass=encoder.beginRenderPass({colorAttachments:[{view,clearValue:{r:.055,g:.065,b:.08,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:this.depthTextureView?{view:this.depthTextureView,depthLoadOp:"load",depthStoreOp:"discard"}:void 0});if(this.finalRenderBundle){pass.executeBundles(this.finalRenderBundleList)}else{const drawCount=this.drawCount||Math.max(1,(this.width-1)*(this.height-1));pass.setPipeline(this.compositePipeline);pass.setBindGroup(0,this.compositeBindGroup);pass.draw(3,1,0,0);pass.setPipeline(this.waterRenderPipeline);pass.setBindGroup(0,this.renderBindGroupWater);pass.draw(6,drawCount,0,0)}pass.end()}}#writeRenderParams(){if(!this.renderUniformBufferTerrain&&!this.renderUniformBufferWater||!this.device)return;const aspect=this.canvas?Math.max((this.canvas.width||1)/Math.max(this.canvas.height||1,1),1e-6):1;const azimuth=degToRad(this.simulationParams.cameraAzimuthDeg);const elevation=degToRad(this.simulationParams.cameraElevationDeg);const worldScale=this.width>1||this.height>1?2/Math.max(this.width-1,this.height-1,1):1;const eye=[this.simulationParams.cameraPosX??1.5238999619464006,this.simulationParams.cameraPosY??1.9404787584406888,this.simulationParams.cameraPosZ??1.5238999619464006];const forward=normalizeVec3([-Math.cos(elevation)*Math.cos(azimuth),-Math.sin(elevation),-Math.cos(elevation)*Math.sin(azimuth)]);const target=[eye[0]+forward[0],eye[1]+forward[1],eye[2]+forward[2]];const view=lookAtMat4(eye,target,[0,1,0]);const proj=perspectiveMat4(degToRad(50),aspect,.01,32);const viewProj=multiplyMat4(proj,view);const effectiveCellSize=Math.max(.25,(this.simulationParams.metersPerPixel??100)/100);this.renderUniformValues.set(viewProj,0);this.renderUniformValues[16]=this.width;this.renderUniformValues[17]=this.height;this.renderUniformValues[18]=effectiveCellSize;this.renderUniformValues[19]=this.simulationParams.renderHeightScale;this.renderUniformValues[20]=this.simulationParams.waterOpacity;this.renderUniformValues[21]=this.simulationParams.sedimentTint;this.renderUniformValues[22]=this.simulationParams.renderMode;this.renderUniformValues[23]=this.simulationParams.thermalVisualizationScale;this.renderUniformValues[24]=worldScale;this.renderUniformValues[25]=this.simulationParams.waterHeightScale;this.renderUniformValues[27]=5e-4;const light=normalizeVec3([.45,.82,.36]);this.renderUniformValues[28]=light[0];this.renderUniformValues[29]=light[1];this.renderUniformValues[30]=light[2];this.renderUniformValues[31]=0;this.renderUniformValues[32]=eye[0];this.renderUniformValues[33]=eye[1];this.renderUniformValues[34]=eye[2];this.renderUniformValues[35]=0;this.renderUniformValues[36]=this.iterationCount*this.simulationParams.timeStep;this.renderUniformValues[37]=this.iterationCount;this.renderUniformValues[38]=this.canvas?.width||1;this.renderUniformValues[39]=this.canvas?.height||1;this.renderUniformValues[26]=0;this.device.queue.writeBuffer(this.renderUniformBufferTerrain,0,this.renderUniformValues);this.renderUniformValues[26]=1;this.device.queue.writeBuffer(this.renderUniformBufferWater,0,this.renderUniformValues)}#ensureBuffers(){const needsRebuild=this.stateBuffers.length!==2||!this.fluxBuffer||!this.velocityBuffer||!this.thermalPipeBufferA||!this.thermalPipeBufferB||!this.layerTopBuffer||this.allocatedStateByteLength!==this.stateByteLength||this.allocatedVec4ByteLength!==this.vec4ByteLength;if(needsRebuild){for(const buffer of this.stateBuffers)buffer?.destroy?.();this.fluxBuffer?.destroy?.();this.velocityBuffer?.destroy?.();this.thermalPipeBufferA?.destroy?.();this.thermalPipeBufferB?.destroy?.();this.readbackBuffer?.destroy?.();this.customSpringBuffer?.destroy?.();this.layerTopBuffer?.destroy?.();this.customSpringBuffer=null;this.layerTopBuffer=null;this.stateBuffers=[0,1].map(index=>this.device.createBuffer({size:this.stateByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:\`paper-state-\${index}\`}));this.fluxBuffer=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-flux"});this.velocityBuffer=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-velocity"});this.thermalPipeBufferA=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-thermal-pipes-a"});this.thermalPipeBufferB=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-thermal-pipes-b"});this.layerTopBuffer=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-layer-top-heights"});this.readbackBuffer=null;this.allocatedReadbackByteLength=0;this.allocatedStateByteLength=this.stateByteLength;this.allocatedVec4ByteLength=this.vec4ByteLength;this.#rebuildBindGroups()}}#fillParamValues(iterationCount,target=this.paramValues){const effectiveCellSize=Math.max(.25,(this.simulationParams.metersPerPixel??100)/100);target[0]=this.width;target[1]=this.height;target[2]=effectiveCellSize;target[3]=this.simulationParams.timeStep;target[4]=this.simulationParams.rainRate;target[5]=this.simulationParams.evaporationRate;target[6]=this.simulationParams.pipeArea;target[7]=this.simulationParams.gravity;target[8]=this.simulationParams.capacityScale;target[9]=this.simulationParams.suspensionRate;target[10]=this.simulationParams.depositionRate;target[11]=this.simulationParams.softeningRate;target[12]=this.simulationParams.maxErosionDepth;target[13]=this.simulationParams.thermalRate;target[14]=this.simulationParams.talusSlopeCoeff;target[15]=this.simulationParams.talusSlopeBias;target[16]=this.simulationParams.sourceLayoutMode;target[17]=this.simulationParams.randomSpringCount;target[18]=this.simulationParams.pulse2Duration;target[19]=this.simulationParams.edgeDrainStrength;target[20]=this.simulationParams.sourceCenterX;target[21]=this.simulationParams.sourceCenterY;target[22]=this.simulationParams.sourceRadius;target[23]=this.simulationParams.sourceStrength;target[24]=this.simulationParams.sourceEnabled?1:0;target[25]=iterationCount*this.simulationParams.timeStep;target[26]=Math.max(0,this.simulationParams.rainDuration??0);target[27]=this.simulationParams.sourceTimeOffset;target[28]=this.simulationParams.historyDecay;target[29]=1e-6;target[30]=0;target[31]=4;target[32]=Math.floor(this.simulationParams.sourceSeed??1);target[33]=this.simulationParams.metersPerPixel??100;target[34]=this.simulationParams.hydraulicErosionEnabled?1:0;target[35]=Math.max(0,this.simulationParams.edgeWaterFloor??0);return target}#writeParams(){this.#fillParamValues(this.iterationCount,this.paramValues);this.stepParamTemplateDirty=true;this.stepParamPreparedIterations=0;if(this.paramBuffer){this.device.queue.writeBuffer(this.paramBuffer,0,this.paramValues)}}#ensureStepParamTemplate(requiredIterations=1){this.#ensureStepParamBuffer(requiredIterations);if(!this.stepParamValuesCpu)return;if(!this.stepParamTemplateDirty&&this.stepParamPreparedIterations>=requiredIterations){return}const strideFloats=this.stepParamStride>>>2;const startIteration=this.stepParamTemplateDirty?0:this.stepParamPreparedIterations;for(let i=startIteration;i<requiredIterations;i++){this.stepParamValuesCpu.set(this.paramValues,i*strideFloats)}this.stepParamTemplateDirty=false;this.stepParamPreparedIterations=Math.max(this.stepParamPreparedIterations,requiredIterations)}#ensureStepParamBuffer(requiredIterations=1){const neededCapacity=Math.max(1,requiredIterations|0);if(this.stepParamBuffer&&this.stepParamCapacity>=neededCapacity&&this.stepParamValuesCpu&&this.stepParamValuesCpuCapacity>=neededCapacity){return}const nextCapacity=Math.max(8,neededCapacity,this.stepParamCapacity>0?this.stepParamCapacity*2:0);const strideFloats=this.stepParamStride>>>2;if(!this.stepParamValuesCpu||this.stepParamValuesCpuCapacity<neededCapacity){this.stepParamValuesCpu=new Float32Array(strideFloats*nextCapacity);this.stepParamValuesCpuCapacity=nextCapacity;this.stepParamTemplateDirty=true;this.stepParamPreparedIterations=0}if(!this.stepParamBuffer||this.stepParamCapacity<neededCapacity){this.stepParamBuffer?.destroy?.();this.stepParamBuffer=this.device.createBuffer({size:this.stepParamStride*nextCapacity,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-step-params"});this.stepParamCapacity=nextCapacity;this.stepParamTemplateDirty=true;this.stepParamPreparedIterations=0;if(this.stateBuffers.length===2){this.#rebuildBindGroups()}}}async#createPipelines(){const computeModule=this.device.createShaderModule({code:COMPUTE_WGSL_SOURCE,label:"paper-erosion-compute-wgsl"});const renderModule=this.device.createShaderModule({code:RENDER_WGSL_SOURCE,label:"paper-erosion-render-wgsl"});const compositeModule=this.device.createShaderModule({code:COMPOSITE_WGSL_SOURCE,label:"paper-erosion-composite-wgsl"});const createComputePipeline=typeof this.device.createComputePipelineAsync==="function"?descriptor=>this.device.createComputePipelineAsync(descriptor):async descriptor=>this.device.createComputePipeline(descriptor);const createRenderPipeline=typeof this.device.createRenderPipelineAsync==="function"?descriptor=>this.device.createRenderPipelineAsync(descriptor):async descriptor=>this.device.createRenderPipeline(descriptor);this.computeBindGroupLayout=this.device.createBindGroupLayout({label:"paper-compute-bgl",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"uniform",hasDynamicOffset:true}},{binding:1,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});this.renderBindGroupLayout=this.device.createBindGroupLayout({label:"paper-render-bgl",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});this.compositeBindGroupLayout=this.device.createBindGroupLayout({label:"paper-composite-bgl",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});const computeLayout=this.device.createPipelineLayout({label:"paper-compute-layout",bindGroupLayouts:[this.computeBindGroupLayout]});const renderLayout=this.device.createPipelineLayout({label:"paper-render-layout",bindGroupLayouts:[this.renderBindGroupLayout]});const compositeLayout=this.device.createPipelineLayout({label:"paper-composite-layout",bindGroupLayouts:[this.compositeBindGroupLayout]});[this.fluxPipeline,this.flowPipeline,this.erosionPipeline,this.transportPipeline,this.thermalOutflowPipeline,this.thermalApplyPipeline,this.renderPipeline,this.waterRenderPipeline,this.compositePipeline]=await Promise.all([createComputePipeline({label:"paper-flux-pipeline",layout:computeLayout,compute:{module:computeModule,entryPoint:"fluxMain"}}),createComputePipeline({label:"paper-flow-pipeline",layout:computeLayout,compute:{module:computeModule,entryPoint:"flowMain"}}),createComputePipeline({label:"paper-erosion-pipeline",layout:computeLayout,compute:{module:computeModule,entryPoint:"erosionMain"}}),createComputePipeline({label:"paper-transport-pipeline",layout:computeLayout,compute:{module:computeModule,entryPoint:"transportMain"}}),createComputePipeline({label:"paper-thermal-outflow-pipeline",layout:computeLayout,compute:{module:computeModule,entryPoint:"thermalOutflowMain"}}),createComputePipeline({label:"paper-thermal-apply-pipeline",layout:computeLayout,compute:{module:computeModule,entryPoint:"thermalApplyMain"}}),createRenderPipeline({label:"paper-render-pipeline",layout:renderLayout,vertex:{module:renderModule,entryPoint:"vsMesh"},fragment:{module:renderModule,entryPoint:"fsMesh",targets:[{format:this.presentationFormat}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:true,depthCompare:"less"}}),createRenderPipeline({label:"paper-water-render-pipeline",layout:renderLayout,vertex:{module:renderModule,entryPoint:"vsMesh"},fragment:{module:renderModule,entryPoint:"fsMesh",targets:[{format:this.presentationFormat,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:false,depthCompare:"less-equal"}}),createRenderPipeline({label:"paper-composite-pipeline",layout:compositeLayout,vertex:{module:compositeModule,entryPoint:"vsComposite"},fragment:{module:compositeModule,entryPoint:"fsComposite",targets:[{format:this.presentationFormat}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:false,depthCompare:"always"}})])}#rebuildBindGroups(){if(!this.sceneColorView||!this.sceneSampler){return}this.#invalidateRenderBundles();this.#ensureDummySceneTexture();this.#ensureCustomSpringBuffer();this.#ensureStepParamBuffer(1);if(!this.dummySceneView||!this.stepParamBuffer||!this.customSpringBuffer||!this.layerTopBuffer||!this.layerMaterialBuffer){return}this.computeBindGroup01=this.device.createBindGroup({label:"paper-compute-0to1",layout:this.computeBindGroupLayout,entries:[{binding:0,resource:{buffer:this.stepParamBuffer,size:this.paramValues.byteLength}},{binding:1,resource:{buffer:this.stateBuffers[0]}},{binding:2,resource:{buffer:this.stateBuffers[1]}},{binding:3,resource:{buffer:this.fluxBuffer}},{binding:4,resource:{buffer:this.velocityBuffer}},{binding:5,resource:{buffer:this.thermalPipeBufferA}},{binding:6,resource:{buffer:this.thermalPipeBufferB}},{binding:7,resource:{buffer:this.customSpringBuffer}},{binding:8,resource:{buffer:this.layerTopBuffer}},{binding:9,resource:{buffer:this.layerMaterialBuffer}}]});this.computeBindGroup10=this.device.createBindGroup({label:"paper-compute-1to0",layout:this.computeBindGroupLayout,entries:[{binding:0,resource:{buffer:this.stepParamBuffer,size:this.paramValues.byteLength}},{binding:1,resource:{buffer:this.stateBuffers[1]}},{binding:2,resource:{buffer:this.stateBuffers[0]}},{binding:3,resource:{buffer:this.fluxBuffer}},{binding:4,resource:{buffer:this.velocityBuffer}},{binding:5,resource:{buffer:this.thermalPipeBufferA}},{binding:6,resource:{buffer:this.thermalPipeBufferB}},{binding:7,resource:{buffer:this.customSpringBuffer}},{binding:8,resource:{buffer:this.layerTopBuffer}},{binding:9,resource:{buffer:this.layerMaterialBuffer}}]});this.renderBindGroupTerrain=this.device.createBindGroup({label:"paper-render-terrain",layout:this.renderBindGroupLayout,entries:[{binding:0,resource:{buffer:this.renderUniformBufferTerrain}},{binding:1,resource:{buffer:this.stateBuffers[0]}},{binding:2,resource:this.sceneSampler},{binding:3,resource:this.dummySceneView}]});this.renderBindGroupWater=this.device.createBindGroup({label:"paper-render-water",layout:this.renderBindGroupLayout,entries:[{binding:0,resource:{buffer:this.renderUniformBufferWater}},{binding:1,resource:{buffer:this.stateBuffers[0]}},{binding:2,resource:this.sceneSampler},{binding:3,resource:this.sceneColorView}]});this.compositeBindGroup=this.device.createBindGroup({label:"paper-composite",layout:this.compositeBindGroupLayout,entries:[{binding:0,resource:this.sceneSampler},{binding:1,resource:this.sceneColorView}]})}};var COMPUTE_WGSL_SOURCE=\`

struct SimParams {
  dims: vec4<f32>,
  hydro0: vec4<f32>,
  hydro1: vec4<f32>,
  thermal0: vec4<f32>,
  render0: vec4<f32>,
  source0: vec4<f32>,
  source1: vec4<f32>,
  misc0: vec4<f32>,
  source2: vec4<f32>,
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
@group(0) @binding(7) var<storage, read> paintedSourceState: FloatBuffer;
@group(0) @binding(8) var<storage, read> layerTopState: Vec4Buffer;
@group(0) @binding(9) var<uniform> layerMaterials: LayerMaterialParams;

fn gridWidth() -> u32 { return u32(params.dims.x); }
fn gridHeight() -> u32 { return u32(params.dims.y); }
fn timeStep() -> f32 { return params.dims.w; }
fn cellSize() -> f32 { return max(params.dims.z, 1e-6); }
fn cellArea() -> f32 { let c = cellSize(); return c * c; }
fn idx(x: u32, y: u32) -> u32 { return y * gridWidth() + x; }
fn hydraulicErosionEnabled() -> bool { return params.source2.z > 0.5; }

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
    if (top > highestTop) {
      highestTop = top;
      highestIndex = layerIndex;
    }
    if (top >= terrain && top < bestAbove) {
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
    depthT = 1.0;
  }

  let selectedMaterial = layerMaterials.layers[chosenIndex];
  return vec4<f32>(clamp(mix(selectedMaterial.x, selectedMaterial.y, depthT), 0.0, 1.0), selectedMaterial.w, f32(chosenIndex), depthT);
}

fn resolveMaterialHardness(i: u32, terrain: f32, fallbackHardness: f32) -> f32 {
  let info = resolveMaterialInfo(i, terrain);
  if (info.z < 0.0) {
    return clamp(fallbackHardness, 0.0, 1.0);
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

fn terrainNormal(x: i32, y: i32) -> vec3<f32> {
  let tl = finiteOr(readStateClamped(x - 1, y - 1).terrain, 0.0);
  let tc = finiteOr(readStateClamped(x, y - 1).terrain, 0.0);
  let tr = finiteOr(readStateClamped(x + 1, y - 1).terrain, 0.0);
  let ml = finiteOr(readStateClamped(x - 1, y).terrain, 0.0);
  let mr = finiteOr(readStateClamped(x + 1, y).terrain, 0.0);
  let bl = finiteOr(readStateClamped(x - 1, y + 1).terrain, 0.0);
  let bc = finiteOr(readStateClamped(x, y + 1).terrain, 0.0);
  let br = finiteOr(readStateClamped(x + 1, y + 1).terrain, 0.0);
  let inv = 1.0 / max(16.0 * cellSize(), 1e-6);
  let dzdx = ((tr + 10.0 * mr + br) - (tl + 10.0 * ml + bl)) * inv;
  let dzdy = ((bl + 10.0 * bc + br) - (tl + 10.0 * tc + tr)) * inv;
  return normalize(vec3<f32>(-dzdx, 1.0, -dzdy));
}

fn terrainSinAlpha(x: i32, y: i32) -> f32 {
  let n = terrainNormal(x, y);
  return sqrt(max(0.0, 1.0 - n.y * n.y));
}

fn depthLimiter(water: f32) -> f32 {
  let limit = max(params.thermal0.x, 1e-6);
  if (water <= 0.0) { return 0.0; }
  if (water >= limit) { return 1.0; }
  return 1.0 - (limit - water) / limit;
}

fn springHash(seed: f32) -> vec2<f32> {
  let hx = fract(sin(seed * 127.1 + 11.7) * 43758.5453);
  let hy = fract(sin(seed * 311.7 + 73.1) * 24634.6345);
  return vec2<f32>(hx, hy);
}

fn rainAmountAt(x: u32, y: u32) -> f32 {
  let simTime = max(0.0, params.source1.y - params.source1.w);
  let rainDuration = params.source1.z;
  let pulse2Duration = max(params.render0.z, 0.0);
  let pulse1Active = !(rainDuration > 0.0 && simTime >= rainDuration);
  let pulse2Active = pulse2Duration > 0.0 && simTime >= rainDuration && simTime < (rainDuration + pulse2Duration);
  let pulseActive = pulse1Active || pulse2Active;
  var rain = select(params.hydro0.x, 0.0, !pulseActive);
  let pos = vec2<f32>(f32(x), f32(y));
  if (params.source1.x > 0.5 && pulseActive && params.render0.x >= 0.5) {
    let radius = max(params.source0.z, 1.0);
    let count = clamp(u32(max(params.render0.y, 1.0)), 1u, 16u);
    let perSpring = params.source0.w / max(f32(count), 1.0);
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      if (j >= count) { break; }
      let uv = springHash(params.source2.x * 37.0 + f32(j) * 17.0 + 1.0);
      let center = vec2<f32>(uv.x * max(f32(gridWidth()) - 1.0, 0.0), uv.y * max(f32(gridHeight()) - 1.0, 0.0));
      let dist = distance(pos, center);
      if (dist < radius) {
        let falloff = 1.0 - dist / radius;
        rain += perSpring * falloff * falloff;
      }
    }
  }
  if (pulseActive && params.source1.x > 0.5 && params.render0.x < 0.5) {
    rain += max(paintedSourceState.values[idx(x, y)], 0.0);
  }
  return rain;
}

fn bilinearSedimentOrZero(pos: vec2<f32>) -> f32 {
  if (pos.x < 0.0 || pos.y < 0.0 || pos.x > f32(gridWidth()) - 1.0 || pos.y > f32(gridHeight()) - 1.0) {
    return 0.0;
  }
  let px = clamp(pos.x, 0.0, max(f32(gridWidth()) - 1.001, 0.0));
  let py = clamp(pos.y, 0.0, max(f32(gridHeight()) - 1.001, 0.0));

  let x0 = i32(floor(px));
  let y0 = i32(floor(py));
  let x1 = x0 + 1;
  let y1 = y0 + 1;

  let tx = fract(px);
  let ty = fract(py);

  let s00 = readStateClamped(x0, y0).sediment;
  let s10 = readStateClamped(x1, y0).sediment;
  let s01 = readStateClamped(x0, y1).sediment;
  let s11 = readStateClamped(x1, y1).sediment;

  let a = mix(s00, s10, tx);
  let b = mix(s01, s11, tx);
  return finiteOr(mix(a, b, ty), 0.0);
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
  let rainWater = cell.water + dt * rainAmountAt(gid.x, gid.y);
  let centerTotal = cell.terrain + rainWater;

  let leftCell = readStateClamped(x - 1, y);
  let rightCell = readStateClamped(x + 1, y);
  let topCell = readStateClamped(x, y - 1);
  let bottomCell = readStateClamped(x, y + 1);

  let edgeDrain = max(params.render0.w, 0.0);
  let retainedEdgeWater = max(params.source2.w, 0.0);
  let outsideTotal = select(centerTotal, cell.terrain + retainedEdgeWater - edgeDrain, rainWater > retainedEdgeWater + 1e-6);
  let leftTotal = select(outsideTotal, leftCell.terrain + leftCell.water + dt * rainAmountAt(u32(max(x - 1, 0)), gid.y), inBounds(x - 1, y));
  let rightTotal = select(outsideTotal, rightCell.terrain + rightCell.water + dt * rainAmountAt(u32(min(x + 1, i32(gridWidth()) - 1)), gid.y), inBounds(x + 1, y));
  let topTotal = select(outsideTotal, topCell.terrain + topCell.water + dt * rainAmountAt(gid.x, u32(max(y - 1, 0))), inBounds(x, y - 1));
  let bottomTotal = select(outsideTotal, bottomCell.terrain + bottomCell.water + dt * rainAmountAt(gid.x, u32(min(y + 1, i32(gridHeight()) - 1))), inBounds(x, y + 1));

  let oldFlux = fluxState.values[i];
  let flowScale = dt * params.hydro0.z * params.hydro0.w / cellSize();
  var nextFlux = vec4<f32>(
    max(0.0, oldFlux.x + flowScale * (centerTotal - leftTotal)),
    max(0.0, oldFlux.y + flowScale * (centerTotal - rightTotal)),
    max(0.0, oldFlux.z + flowScale * (centerTotal - topTotal)),
    max(0.0, oldFlux.w + flowScale * (centerTotal - bottomTotal))
  );

  nextFlux = vec4<f32>(
    finiteOr(max(0.0, nextFlux.x), 0.0),
    finiteOr(max(0.0, nextFlux.y), 0.0),
    finiteOr(max(0.0, nextFlux.z), 0.0),
    finiteOr(max(0.0, nextFlux.w), 0.0)
  );
  let sumOut = finiteOr(nextFlux.x + nextFlux.y + nextFlux.z + nextFlux.w, 0.0);
  let maxOut = finiteOr(rainWater * cellArea() / max(dt, 1e-6), 0.0);
  if (sumOut > maxOut && sumOut > 1e-6) {
    nextFlux *= maxOut / sumOut;
  }

  let maxOutsideOut = max(rainWater - retainedEdgeWater, 0.0) * cellArea() / max(dt, 1e-6);
  let outsideFluxX = select(0.0, nextFlux.x, !inBounds(x - 1, y));
  let outsideFluxY = select(0.0, nextFlux.y, !inBounds(x + 1, y));
  let outsideFluxZ = select(0.0, nextFlux.z, !inBounds(x, y - 1));
  let outsideFluxW = select(0.0, nextFlux.w, !inBounds(x, y + 1));
  let sumOutsideOut = outsideFluxX + outsideFluxY + outsideFluxZ + outsideFluxW;
  if (sumOutsideOut > maxOutsideOut && sumOutsideOut > 1e-6) {
    let outsideScale = maxOutsideOut / sumOutsideOut;
    if (!inBounds(x - 1, y)) { nextFlux.x *= outsideScale; }
    if (!inBounds(x + 1, y)) { nextFlux.y *= outsideScale; }
    if (!inBounds(x, y - 1)) { nextFlux.z *= outsideScale; }
    if (!inBounds(x, y + 1)) { nextFlux.w *= outsideScale; }
  }
  nextFlux *= max(0.0, 1.0 - params.misc0.z * dt);
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
  let rainWater = cell.water + dt * rainAmountAt(gid.x, gid.y);

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
  velocityState.values[i] = vec4<f32>(finiteOr(vel.x, 0.0), finiteOr(vel.y, 0.0), finiteOr(speed, 0.0), 0.0);

  let activeHardness = resolveMaterialHardness(i, finiteOr(cell.terrain, 0.0), finiteOr(cell.hardness, 0.1));
  dstState.cells[i] = CellState(finiteOr(cell.terrain, 0.0), finiteOr(water, 0.0), finiteOr(cell.sediment, 0.0), activeHardness, cell.mask, 0.0, 0.0, finiteOr(cell.aux2, 0.0));
}

fn terrainNeighborMean(x: i32, y: i32, center: f32) -> f32 {
  let tl = readStateClamped(x - 1, y - 1);
  let tc = readStateClamped(x, y - 1);
  let tr = readStateClamped(x + 1, y - 1);
  let ml = readStateClamped(x - 1, y);
  let mr = readStateClamped(x + 1, y);
  let bl = readStateClamped(x - 1, y + 1);
  let bc = readStateClamped(x, y + 1);
  let br = readStateClamped(x + 1, y + 1);
  var sum = 0.0;
  var count = 0.0;
  if (tl.mask > 0.5) { sum += tl.terrain; count += 1.0; }
  if (tc.mask > 0.5) { sum += tc.terrain; count += 1.0; }
  if (tr.mask > 0.5) { sum += tr.terrain; count += 1.0; }
  if (ml.mask > 0.5) { sum += ml.terrain; count += 1.0; }
  if (mr.mask > 0.5) { sum += mr.terrain; count += 1.0; }
  if (bl.mask > 0.5) { sum += bl.terrain; count += 1.0; }
  if (bc.mask > 0.5) { sum += bc.terrain; count += 1.0; }
  if (br.mask > 0.5) { sum += br.terrain; count += 1.0; }
  if (count <= 0.0) { return center; }
  return sum / count;
}

@compute @workgroup_size(8, 8)
fn erosionMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }

  let i = idx(gid.x, gid.y);
  let cell = srcState.cells[i];
  if (cell.mask < 0.5) {
    dstState.cells[i] = CellState(0.0, 0.0, 0.0, cell.hardness, 0.0, 0.0, 0.0, 0.0);
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let velocity = readVelocityClamped(x, y);
  let speed = finiteOr(velocity.z, 0.0);
  let sinAlpha = finiteOr(terrainSinAlpha(x, y), 0.0);
  let n = terrainNormal(x, y);
  let flow3 = vec3<f32>(
    velocity.x,
    -(velocity.x * n.x + velocity.y * n.z) / max(n.y, 0.15),
    velocity.y
  );
  let flow3Dir = flow3 / max(length(flow3), 1e-6);
  let collisionTerm = select(0.0, max(dot(-n, flow3Dir), 0.0), speed > 1e-6);
  let capacityTerm = max(max(collisionTerm * 0.80, sinAlpha * 0.12), 0.03);
  let capacity = finiteOr(params.hydro1.x * capacityTerm * speed * depthLimiter(cell.water), 0.0);

  var terrain = cell.terrain;
  var water = cell.water;
  var sediment = cell.sediment;
  var hardness = resolveMaterialHardness(i, finiteOr(cell.terrain, 0.0), finiteOr(cell.hardness, params.misc0.y));

  var history = finiteOr(cell.aux2, 0.0) * params.misc0.x;

  if (!hydraulicErosionEnabled()) {
    let preservedHardness = resolveMaterialHardness(i, clamp(finiteOr(cell.terrain, 0.0), 0.0, 2.0), finiteOr(cell.hardness, params.misc0.y));
    dstState.cells[i] = CellState(clamp(finiteOr(cell.terrain, 0.0), 0.0, 2.0), clamp(finiteOr(cell.water, 0.0), 0.0, 2.0), clamp(finiteOr(cell.sediment, 0.0), 0.0, 2.0), preservedHardness, cell.mask, finiteOr(capacity, 0.0), finiteOr(speed, 0.0), finiteOr(cell.aux2, 0.0));
    return;
  }

  if (capacity > sediment && water > 1e-6) {
    let erodeAmount = timeStep() * max(hardness, 1e-6) * params.hydro1.y * (capacity - sediment);
    let clampedErode = min(min(erodeAmount, max(water, 0.0)), max(terrain, 0.0));
    terrain = max(0.0, terrain - clampedErode);
    sediment += clampedErode;
    water += clampedErode;
    history -= clampedErode * 240.0;
  } else if (sediment > capacity) {
    let sedimentExcess = sediment - capacity;
    let depositAmount = timeStep() * params.hydro1.z * sedimentExcess * 1.12;
    let neighborMean = terrainNeighborMean(x, y, terrain);
    let localCeiling = max(terrain + 0.001, neighborMean + params.thermal0.w * 0.9 + max(cell.water, 0.0) * 0.08);
    let spikeGuard = max(0.0, localCeiling - terrain);
    let clampedDeposit = min(min(depositAmount, sediment), spikeGuard + max(cell.water * 0.04, 0.0006));
    terrain += clampedDeposit;
    sediment -= clampedDeposit;
    water = max(0.0, water - clampedDeposit);
    hardness = max(0.0, hardness - timeStep() * params.hydro1.w * params.hydro1.y * sedimentExcess * 0.75);
    history += clampedDeposit * 180.0;
  }

  history = clamp(history, -1.0, 1.0);
  terrain = clamp(terrain, 0.0, 2.0);
  water = clamp(water, 0.0, 2.0);
  sediment = clamp(sediment, 0.0, 2.0);
  hardness = resolveMaterialHardness(i, terrain, finiteOr(hardness, params.misc0.y));
  dstState.cells[i] = CellState(finiteOr(terrain, 0.0), finiteOr(water, 0.0), finiteOr(sediment, 0.0), finiteOr(hardness, params.misc0.y), cell.mask, finiteOr(capacity, 0.0), finiteOr(speed, 0.0), finiteOr(history, 0.0));
}

@compute @workgroup_size(8, 8)
fn transportMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }

  let i = idx(gid.x, gid.y);
  let cell = srcState.cells[i];
  if (cell.mask < 0.5) {
    dstState.cells[i] = CellState(0.0, 0.0, 0.0, cell.hardness, 0.0, 0.0, 0.0, 0.0);
    return;
  }

  let x = f32(gid.x);
  let y = f32(gid.y);
  let velocity = velocityState.values[i].xy;
  let origin = vec2<f32>(x, y) - velocity * timeStep();
  let transportedSediment = clamp(finiteOr(bilinearSedimentOrZero(origin), 0.0), 0.0, 2.0);
  let preservedSediment = clamp(finiteOr(cell.sediment, 0.0), 0.0, 2.0);
  let water = clamp(max(0.0, finiteOr(cell.water, 0.0) * (1.0 - params.hydro0.y * timeStep())), 0.0, 2.0);
  let nextSediment = select(transportedSediment, preservedSediment, !hydraulicErosionEnabled());
  let activeHardness = resolveMaterialHardness(i, clamp(finiteOr(cell.terrain, 0.0), 0.0, 2.0), finiteOr(cell.hardness, params.misc0.y));
  dstState.cells[i] = CellState(clamp(finiteOr(cell.terrain, 0.0), 0.0, 2.0), water, nextSediment, activeHardness, cell.mask, cell.aux0, cell.aux1, finiteOr(cell.aux2, 0.0));
}

@compute @workgroup_size(8, 8)
fn thermalOutflowMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }

  let i = idx(gid.x, gid.y);
  let cell = srcState.cells[i];
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
  let softness = clamp(0.20 + max(activeHardness, 0.0) * 0.85, 0.08, 0.85);
  let thresholdBase = max(params.thermal0.w * 0.7, params.thermal0.z * max(activeHardness, 0.0) * 0.75 + params.thermal0.w * 0.55);

  var rawCard = vec4<f32>(0.0);
  var rawDiag = vec4<f32>(0.0);
  var weightSum = 0.0;
  var maxExcess = 0.0;

  let left = readStateClamped(x - 1, y);
  let dropL = cell.terrain - left.terrain;
  let thrL = thresholdBase * thermalDistanceScale(-1, 0);
  let excessL = max(0.0, dropL - thrL);
  if (left.mask > 0.5 && excessL > 0.0) { rawCard.x = excessL; weightSum += excessL; maxExcess = max(maxExcess, excessL); }

  let right = readStateClamped(x + 1, y);
  let dropR = cell.terrain - right.terrain;
  let thrR = thresholdBase * thermalDistanceScale(1, 0);
  let excessR = max(0.0, dropR - thrR);
  if (right.mask > 0.5 && excessR > 0.0) { rawCard.y = excessR; weightSum += excessR; maxExcess = max(maxExcess, excessR); }

  let top = readStateClamped(x, y - 1);
  let dropT = cell.terrain - top.terrain;
  let thrT = thresholdBase * thermalDistanceScale(0, -1);
  let excessT = max(0.0, dropT - thrT);
  if (top.mask > 0.5 && excessT > 0.0) { rawCard.z = excessT; weightSum += excessT; maxExcess = max(maxExcess, excessT); }

  let bottom = readStateClamped(x, y + 1);
  let dropB = cell.terrain - bottom.terrain;
  let thrB = thresholdBase * thermalDistanceScale(0, 1);
  let excessB = max(0.0, dropB - thrB);
  if (bottom.mask > 0.5 && excessB > 0.0) { rawCard.w = excessB; weightSum += excessB; maxExcess = max(maxExcess, excessB); }

  let tl = readStateClamped(x - 1, y - 1);
  let dropTL = cell.terrain - tl.terrain;
  let thrTL = thresholdBase * thermalDistanceScale(-1, -1);
  let excessTL = max(0.0, dropTL - thrTL);
  if (tl.mask > 0.5 && excessTL > 0.0) { rawDiag.x = excessTL; weightSum += excessTL; maxExcess = max(maxExcess, excessTL); }

  let tr = readStateClamped(x + 1, y - 1);
  let dropTR = cell.terrain - tr.terrain;
  let thrTR = thresholdBase * thermalDistanceScale(1, -1);
  let excessTR = max(0.0, dropTR - thrTR);
  if (tr.mask > 0.5 && excessTR > 0.0) { rawDiag.y = excessTR; weightSum += excessTR; maxExcess = max(maxExcess, excessTR); }

  let bl = readStateClamped(x - 1, y + 1);
  let dropBL = cell.terrain - bl.terrain;
  let thrBL = thresholdBase * thermalDistanceScale(-1, 1);
  let excessBL = max(0.0, dropBL - thrBL);
  if (bl.mask > 0.5 && excessBL > 0.0) { rawDiag.z = excessBL; weightSum += excessBL; maxExcess = max(maxExcess, excessBL); }

  let br = readStateClamped(x + 1, y + 1);
  let dropBR = cell.terrain - br.terrain;
  let thrBR = thresholdBase * thermalDistanceScale(1, 1);
  let excessBR = max(0.0, dropBR - thrBR);
  if (br.mask > 0.5 && excessBR > 0.0) { rawDiag.w = excessBR; weightSum += excessBR; maxExcess = max(maxExcess, excessBR); }

  if (weightSum <= 1e-6 || maxExcess <= 1e-6) {
    thermalPipeA.values[i] = vec4<f32>(0.0);
    thermalPipeB.values[i] = vec4<f32>(0.0);
    return;
  }

  let totalOut = min(cell.terrain, cellArea() * timeStep() * params.thermal0.y * softness * maxExcess * 0.72);
  thermalPipeA.values[i] = rawCard * (totalOut / weightSum);
  thermalPipeB.values[i] = rawDiag * (totalOut / weightSum);
}

@compute @workgroup_size(8, 8)
fn thermalApplyMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= gridWidth() || gid.y >= gridHeight()) { return; }

  let i = idx(gid.x, gid.y);
  let cell = srcState.cells[i];
  if (cell.mask < 0.5) {
    dstState.cells[i] = CellState(0.0, 0.0, 0.0, cell.hardness, 0.0, 0.0, 0.0, 0.0);
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let selfA = readThermalAOrZero(x, y);
  let selfB = readThermalBOrZero(x, y);
  let selfOut = selfA.x + selfA.y + selfA.z + selfA.w + selfB.x + selfB.y + selfB.z + selfB.w;

  let incoming =
    readThermalAOrZero(x - 1, y).y +
    readThermalAOrZero(x + 1, y).x +
    readThermalAOrZero(x, y - 1).w +
    readThermalAOrZero(x, y + 1).z +
    readThermalBOrZero(x - 1, y - 1).w +
    readThermalBOrZero(x + 1, y - 1).z +
    readThermalBOrZero(x - 1, y + 1).y +
    readThermalBOrZero(x + 1, y + 1).x;

  let terrain = clamp(max(0.0, finiteOr(cell.terrain, 0.0) - finiteOr(selfOut, 0.0) + finiteOr(incoming, 0.0)), 0.0, 2.0);
  let activeHardness = resolveMaterialHardness(i, terrain, finiteOr(cell.hardness, params.misc0.y));
  dstState.cells[i] = CellState(terrain, clamp(finiteOr(cell.water, 0.0), 0.0, 2.0), clamp(finiteOr(cell.sediment, 0.0), 0.0, 2.0), activeHardness, cell.mask, finiteOr(selfOut, 0.0), finiteOr(incoming, 0.0), finiteOr(cell.aux2, 0.0));
}

\`;var COMPOSITE_WGSL_SOURCE=\`

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
\`;var RENDER_WGSL_SOURCE=\`

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
}

@group(0) @binding(0) var<uniform> renderParams: RenderParams;
@group(0) @binding(1) var<storage, read> renderState: StateBuffer;
@group(0) @binding(2) var sceneSampler: sampler;
@group(0) @binding(3) var sceneTexture: texture_2d<f32>;

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
  return clamp(finiteOr(renderStateAt(x, y).terrain, 0.0), 0.0, 1.2) * renderParams.dims.w;
}

fn renderWaterDepthAt(x: i32, y: i32) -> f32 {
  return clamp(finiteOr(renderStateAt(x, y).water, 0.0), 0.0, 1.2) * renderParams.misc.y;
}

fn renderNormal(x: i32, y: i32) -> vec3<f32> {
  let worldScale = max(renderParams.misc.x, 1e-5);
  let tl = renderTerrainHeightAt(x - 1, y - 1);
  let tc = renderTerrainHeightAt(x, y - 1);
  let tr = renderTerrainHeightAt(x + 1, y - 1);
  let ml = renderTerrainHeightAt(x - 1, y);
  let mr = renderTerrainHeightAt(x + 1, y);
  let bl = renderTerrainHeightAt(x - 1, y + 1);
  let bc = renderTerrainHeightAt(x, y + 1);
  let br = renderTerrainHeightAt(x + 1, y + 1);
  let inv = 1.0 / max(16.0 * worldScale, 1e-5);
  let dzdx = ((tr + 10.0 * mr + br) - (tl + 10.0 * ml + bl)) * inv;
  let dzdz = ((tl + 10.0 * tc + tr) - (bl + 10.0 * bc + br)) * inv;
  return normalize(vec3<f32>(-dzdx, 1.0, -dzdz));
}

fn renderWaterLevelAt(x: i32, y: i32) -> f32 {
  let centerWater = renderWaterDepthAt(x, y);
  let leftWater = renderWaterDepthAt(x - 1, y);
  let rightWater = renderWaterDepthAt(x + 1, y);
  let upWater = renderWaterDepthAt(x, y - 1);
  let downWater = renderWaterDepthAt(x, y + 1);
  let smoothedWater = centerWater * 0.58 + (leftWater + rightWater + upWater + downWater) * 0.105;
  let cappedWater = min(max(smoothedWater, 0.0), centerWater * 1.16 + 0.0060);
  return max(cappedWater, 0.0);
}

fn renderRawWaterSurfaceAt(x: i32, y: i32) -> f32 {
  return renderTerrainHeightAt(x, y) + renderWaterLevelAt(x, y);
}

fn renderWaterSurfaceAt(x: i32, y: i32) -> f32 {
  let centerTerrain = renderTerrainHeightAt(x, y);
  let centerDepth = renderWaterLevelAt(x, y);
  if (centerDepth <= 1e-6) {
    return centerTerrain;
  }

  let leftDepth = renderWaterLevelAt(x - 1, y);
  let rightDepth = renderWaterLevelAt(x + 1, y);
  let upDepth = renderWaterLevelAt(x, y - 1);
  let downDepth = renderWaterLevelAt(x, y + 1);

  let leftWeight = smoothstep(0.0003, 0.0060, leftDepth) * 0.70;
  let rightWeight = smoothstep(0.0003, 0.0060, rightDepth) * 0.70;
  let upWeight = smoothstep(0.0003, 0.0060, upDepth) * 0.70;
  let downWeight = smoothstep(0.0003, 0.0060, downDepth) * 0.70;

  let centerSurface = renderRawWaterSurfaceAt(x, y);
  let sum = centerSurface * 2.4 +
    renderRawWaterSurfaceAt(x - 1, y) * leftWeight +
    renderRawWaterSurfaceAt(x + 1, y) * rightWeight +
    renderRawWaterSurfaceAt(x, y - 1) * upWeight +
    renderRawWaterSurfaceAt(x, y + 1) * downWeight;
  let weight = 2.4 + leftWeight + rightWeight + upWeight + downWeight;
  let smoothedSurface = sum / max(weight, 1e-6);
  let minSurface = centerTerrain + centerDepth * 0.22;
  let maxSurface = centerSurface + 0.0030;
  return clamp(smoothedSurface, minSurface, maxSurface);
}

fn renderDisplayedWaterDepthAt(x: i32, y: i32) -> f32 {
  return max(renderWaterSurfaceAt(x, y) - renderTerrainHeightAt(x, y), 0.0);
}

fn renderWaterDepthBilinear(pos: vec2<f32>) -> f32 {
  let maxX = max(renderParams.dims.x - 1.001, 0.0);
  let maxY = max(renderParams.dims.y - 1.001, 0.0);
  let px = clamp(pos.x, 0.0, maxX);
  let py = clamp(pos.y, 0.0, maxY);
  let x0 = i32(floor(px));
  let y0 = i32(floor(py));
  let x1 = x0 + 1;
  let y1 = y0 + 1;
  let tx = fract(px);
  let ty = fract(py);
  let h00 = renderDisplayedWaterDepthAt(x0, y0);
  let h10 = renderDisplayedWaterDepthAt(x1, y0);
  let h01 = renderDisplayedWaterDepthAt(x0, y1);
  let h11 = renderDisplayedWaterDepthAt(x1, y1);
  let a = mix(h00, h10, tx);
  let b = mix(h01, h11, tx);
  return mix(a, b, ty);
}

fn renderWaterSurfaceBilinear(pos: vec2<f32>) -> f32 {
  let maxX = max(renderParams.dims.x - 1.001, 0.0);
  let maxY = max(renderParams.dims.y - 1.001, 0.0);
  let px = clamp(pos.x, 0.0, maxX);
  let py = clamp(pos.y, 0.0, maxY);
  let x0 = i32(floor(px));
  let y0 = i32(floor(py));
  let x1 = x0 + 1;
  let y1 = y0 + 1;
  let tx = fract(px);
  let ty = fract(py);
  let h00 = renderWaterSurfaceAt(x0, y0);
  let h10 = renderWaterSurfaceAt(x1, y0);
  let h01 = renderWaterSurfaceAt(x0, y1);
  let h11 = renderWaterSurfaceAt(x1, y1);
  let a = mix(h00, h10, tx);
  let b = mix(h01, h11, tx);
  return mix(a, b, ty);
}

fn renderWaterNormalAtPos(pos: vec2<f32>) -> vec3<f32> {
  let worldScale = max(renderParams.misc.x, 1e-5);
  let dx = vec2<f32>(1.0, 0.0);
  let dy = vec2<f32>(0.0, 1.0);
  let hL = renderWaterSurfaceBilinear(pos - dx);
  let hR = renderWaterSurfaceBilinear(pos + dx);
  let hU = renderWaterSurfaceBilinear(pos - dy);
  let hD = renderWaterSurfaceBilinear(pos + dy);
  let dzdx = (hR - hL) / max(2.0 * worldScale, 1e-5);
  let dzdz = (hU - hD) / max(2.0 * worldScale, 1e-5);
  return normalize(vec3<f32>(-dzdx * 20.0, 1.0, -dzdz * 20.0));
}

fn renderWaterMicroNormalAtPos(pos: vec2<f32>) -> vec3<f32> {
  let worldScale = max(renderParams.misc.x, 1e-5);
  let dx = vec2<f32>(0.30, 0.0);
  let dy = vec2<f32>(0.0, 0.30);
  let hL = renderWaterSurfaceBilinear(pos - dx);
  let hR = renderWaterSurfaceBilinear(pos + dx);
  let hU = renderWaterSurfaceBilinear(pos - dy);
  let hD = renderWaterSurfaceBilinear(pos + dy);
  let dzdx = (hR - hL) / max(0.60 * worldScale, 1e-5);
  let dzdz = (hU - hD) / max(0.60 * worldScale, 1e-5);
  return normalize(vec3<f32>(-dzdx * 34.0, 1.0, -dzdz * 34.0));
}

fn renderWaterNormal(x: i32, y: i32) -> vec3<f32> {
  let worldScale = max(renderParams.misc.x, 1e-5);
  let tl = renderWaterSurfaceAt(x - 1, y - 1);
  let tc = renderWaterSurfaceAt(x, y - 1);
  let tr = renderWaterSurfaceAt(x + 1, y - 1);
  let ml = renderWaterSurfaceAt(x - 1, y);
  let mr = renderWaterSurfaceAt(x + 1, y);
  let bl = renderWaterSurfaceAt(x - 1, y + 1);
  let bc = renderWaterSurfaceAt(x, y + 1);
  let br = renderWaterSurfaceAt(x + 1, y + 1);
  let inv = 1.0 / max(16.0 * worldScale, 1e-5);
  let dzdx = ((tr + 10.0 * mr + br) - (tl + 10.0 * ml + bl)) * inv;
  let dzdz = ((tl + 10.0 * tc + tr) - (bl + 10.0 * bc + br)) * inv;
  return normalize(vec3<f32>(-dzdx, 1.0, -dzdz));
}

fn sampleSceneColor(uv: vec2<f32>) -> vec3<f32> {
  return textureSampleLevel(sceneTexture, sceneSampler, clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999)), 0.0).rgb;
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
  let cellsWide = max(width - 1u, 1u);
  let cellX = instanceIndex % cellsWide;
  let cellY = instanceIndex / cellsWide;
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
    return out;
  }

  let corner = cornerCoord(vertexIndex % 6u);
  let gx = cellX + corner.x;
  let gy = cellY + corner.y;
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
  let waterLift = max(waterY - 0.0002, 0.0);
  let worldY = select(terrainY, terrainY + waterLift, waterPass && waterY > 1e-6);

  out.position = renderParams.viewProj * vec4<f32>(worldX, worldY, worldZ, 1.0);
  out.normal = renderNormal(i32(gx), i32(gy));
  out.terrain = finiteOr(cell.terrain, 0.0);
  out.water = finiteOr(waterY / max(renderParams.misc.y, 1e-6), 0.0);
  out.sediment = finiteOr(cell.sediment, 0.0);
  out.hardness = finiteOr(cell.hardness, 0.1);
  out.thermal = clamp(pow(max(finiteOr((cell.aux0 + cell.aux1) * renderParams.shading.w, 0.0), 0.0), 0.45), 0.0, 1.0);
  out.history = clamp(finiteOr(cell.aux2, 0.0), -1.0, 1.0);
  out.mask = cell.mask;
  out.worldPos = vec3<f32>(worldX, worldY, worldZ);
  out.gridPos = vec2<f32>(f32(gx), f32(gy));
  return out;
}

@fragment
fn fsMesh(in: RenderVertexOut) -> @location(0) vec4<f32> {
  if (in.mask < 0.5) {
    return vec4<f32>(0.03, 0.04, 0.07, 1.0);
  }

  let waterPass = renderParams.misc.z > 0.5;
  let terrainNormal = normalize(vec3<f32>(in.normal.x, max(in.normal.y, 0.35), in.normal.z));
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

  let shorelineFade = smoothstep(0.010, 0.080, in.water);
  let waterAlpha = clamp(in.water * shorelineFade * max(renderParams.shading.x * 2.6, 0.0), 0.0, 0.97);
  let mode = i32(round(renderParams.shading.z));

  if (waterPass) {
    if (waterAlpha <= max(renderParams.misc.w, 0.01)) {
      return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let w = clamp(in.water * 2.2, 0.0, 1.0);
    let s = clamp(in.sediment * renderParams.shading.y * 2.7, 0.0, 1.0);
    let viewDir = normalize(renderParams.cameraPos.xyz - in.worldPos);
    let lightDir = normalize(renderParams.lightDir.xyz);
    let macroNormal = renderWaterNormalAtPos(in.gridPos);
    let microNormal = renderWaterMicroNormalAtPos(in.gridPos);
    let combinedNormal = normalize(vec3<f32>(
      macroNormal.x * 1.10 + microNormal.x * 1.90,
      max(macroNormal.y * 0.65 + microNormal.y, 0.12),
      macroNormal.z * 1.10 + microNormal.z * 1.90
    ));
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
  if (mode == 7) {
    let signedHistory = clamp(in.history, -1.0, 1.0);
    let erosion = smoothstep(0.02, 0.72, max(-signedHistory, 0.0));
    let deposition = smoothstep(0.02, 0.72, max(signedHistory, 0.0));
    let thermal = clamp(in.thermal, 0.0, 1.0);
    let steepness = pow(1.0 - clamp(terrainNormal.y, 0.0, 1.0), 0.68);
    let lowland = 1.0 - smoothstep(0.14, 0.60, h);
    let waterPresence = smoothstep(0.0, 0.05, in.water);
    let lowElevationMarine = (1.0 - smoothstep(0.05, 0.30, h)) * mix(0.40, 1.0, waterPresence);
    let shoalBand = smoothstep(0.03, 0.15, h) * (1.0 - smoothstep(0.15, 0.30, h));
    let beachBand = smoothstep(0.10, 0.23, h) * (1.0 - smoothstep(0.23, 0.40, h));
    let settledActivity = clamp(deposition * (0.70 + 0.35 * lowland) + thermal * (0.34 + 0.18 * lowland), 0.0, 1.0);

    var natural = mix(vec3<f32>(0.18, 0.24, 0.15), vec3<f32>(0.42, 0.38, 0.22), smoothstep(0.10, 0.52, h));
    natural = mix(natural, vec3<f32>(0.68, 0.61, 0.50), smoothstep(0.56, 0.96, h));

    let soilColor = mix(vec3<f32>(0.22, 0.50, 0.20), vec3<f32>(0.52, 0.46, 0.24), smoothstep(0.18, 0.82, h));
    let fertileLowlandColor = vec3<f32>(0.18, 0.56, 0.20);
    let riparianColor = vec3<f32>(0.12, 0.48, 0.22);
    let alluviumColor = mix(vec3<f32>(0.54, 0.58, 0.20), vec3<f32>(0.82, 0.66, 0.28), smoothstep(0.22, 0.72, h));
    let colluviumColor = vec3<f32>(0.58, 0.46, 0.28);
    let rockColor = mix(vec3<f32>(0.30, 0.28, 0.25), vec3<f32>(0.78, 0.74, 0.68), smoothstep(0.24, 0.92, steepness));
    let deepMarineColor = vec3<f32>(0.01, 0.12, 0.56);
    let shallowMarineColor = vec3<f32>(0.04, 0.90, 0.96);
    let shoalOrangeColor = vec3<f32>(1.00, 0.62, 0.06);
    let marineColor = mix(deepMarineColor, shallowMarineColor, smoothstep(0.03, 0.18, h));

    let soilBuild = clamp(deposition * (0.68 + 0.30 * lowland) + thermal * 0.20 - steepness * 0.24, 0.0, 1.0);
    let alluvium = clamp(deposition * (0.54 + 0.42 * lowland) + thermal * 0.14, 0.0, 1.0);
    let colluvium = clamp(thermal * (0.52 + 0.60 * steepness) + deposition * 0.18, 0.0, 1.0);
    let exposedRock = clamp(erosion * 0.94 + steepness * 0.54 - deposition * 0.26, 0.0, 1.0);
    let fertilePlain = clamp(settledActivity * lowland * (1.0 - steepness) * (1.0 - 0.45 * waterPresence), 0.0, 1.0);
    let riparian = clamp((0.24 + 0.76 * settledActivity) * lowland * (1.0 - steepness) * smoothstep(0.0, 0.10, in.water), 0.0, 1.0);
    let foothillGreen = clamp(settledActivity * (1.0 - lowland) * (1.0 - steepness) * 0.34, 0.0, 1.0);

    natural = mix(natural, marineColor, lowElevationMarine * 0.82);
    natural = mix(natural, shoalOrangeColor, shoalBand * (0.18 + 0.16 * lowland + 0.10 * deposition));
    natural = mix(natural, shoalOrangeColor * vec3<f32>(1.0, 0.90, 0.68), beachBand * (0.10 + 0.08 * deposition));
    natural = mix(natural, soilColor, soilBuild * 0.58);
    natural = mix(natural, alluviumColor, alluvium * 0.44);
    natural = mix(natural, colluviumColor, colluvium * 0.38);
    natural = mix(natural, fertileLowlandColor, fertilePlain * 0.74);
    natural = mix(natural, riparianColor, riparian * 0.62);
    natural = mix(natural, vec3<f32>(0.38, 0.54, 0.26), foothillGreen * 0.24);
    natural = mix(natural, rockColor, exposedRock);

    let matteLambert = 0.60 + 0.28 * lambert + 0.22 * hemi;
    let microContour = 0.95 + 0.05 * abs(sin(h * 30.0 + signedHistory * 8.0 + thermal * 6.0));
    var naturalColor = clamp(natural * matteLambert * microContour, vec3<f32>(0.0), vec3<f32>(1.0));
    let luma = dot(naturalColor, vec3<f32>(0.299, 0.587, 0.114));
    naturalColor = mix(vec3<f32>(luma), naturalColor, 1.18);
    naturalColor = pow(clamp(naturalColor, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.94));
    return vec4<f32>(clamp(naturalColor, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
  }

  return vec4<f32>(terrainColor, 1.0);
}
\`;var gpu=null;var offscreenCanvas=null;var canvasMetrics={width:1,height:1,dpr:1};var running=false;var loopHandle=0;var lastFrameAt=0;var lastFrameMs=0;var lastStatusPostAt=0;var lastRenderAt=0;var lastGpuBenchAt=0;var loopIterationsPerFrame=5;var simulationParams={};var sourceRaster=null;var sourceImageInfo=null;var gpuInitPromise=null;var hasWorkerRAF=typeof self.requestAnimationFrame==="function";var workerRAF=hasWorkerRAF?cb=>self.requestAnimationFrame(cb):cb=>self.setTimeout(()=>cb(performance.now()),0);var workerCAF=hasWorkerRAF?id=>self.cancelAnimationFrame(id):id=>self.clearTimeout(id);function cloneStats(stats){if(!stats)return null;return{...stats,terrainRange:stats.terrainRange?{...stats.terrainRange}:null,waterRange:stats.waterRange?{...stats.waterRange}:null,sedimentRange:stats.sedimentRange?{...stats.sedimentRange}:null,historyRange:stats.historyRange?{...stats.historyRange}:null}}function getSourcePoints(){return gpu?.getSourcePoints?.()||[]}function postStatus(force=false){const stats=gpu?cloneStats(gpu.getStats()):null;self.postMessage({type:"status",stats,sourcePoints:force?getSourcePoints():void 0,lastFrameMs,running,sourceImageInfo:force?sourceImageInfo:void 0})}function applyCanvasMetrics(){if(!offscreenCanvas)return;const width=Math.max(1,Math.round((canvasMetrics.width||1)*(canvasMetrics.dpr||1)));const height=Math.max(1,Math.round((canvasMetrics.height||1)*(canvasMetrics.dpr||1)));offscreenCanvas.width=width;offscreenCanvas.height=height;if(gpu){gpu.resize()}}async function ensureGpu(){if(!offscreenCanvas){throw new Error("Worker canvas has not been initialized.")}if(gpu){return gpu}if(!gpuInitPromise){gpuInitPromise=(async()=>{const instance=new WebGPUTerrainErosion({canvas:offscreenCanvas});await instance.initialize();if(Object.keys(simulationParams).length>0){instance.setSimulationParams(simulationParams)}gpu=instance;applyCanvasMetrics();return instance})().catch(error=>{gpu=null;gpuInitPromise=null;throw error})}try{return await gpuInitPromise}finally{if(gpu){gpuInitPromise=null}}}function warmGpuInBackground(){if(gpu||gpuInitPromise||!offscreenCanvas)return;void ensureGpu().then(()=>{postStatus(true)}).catch(error=>{self.postMessage({type:"workerError",message:error instanceof Error?error.message:String(error)})})}function stopLoopInternal(){if(!running)return;running=false;if(loopHandle){workerCAF(loopHandle);loopHandle=0}}function scheduleNextFrame(){loopHandle=workerRAF(now=>{void frame(now)})}function getLoopRenderIntervalMs(){const iterations=Math.max(1,loopIterationsPerFrame|0);if(iterations<=4)return 0;if(iterations<=8)return 16;if(iterations<=16)return 33;return 50}async function frame(now){if(!running)return;const currentGpu=gpu;if(!currentGpu?.ready){scheduleNextFrame();return}lastFrameMs=lastFrameAt>0?Math.max(0,now-lastFrameAt):0;lastFrameAt=now;try{const iterations=Math.max(1,loopIterationsPerFrame|0);const renderIntervalMs=getLoopRenderIntervalMs();const shouldRender=renderIntervalMs<=0||lastRenderAt<=0||now-lastRenderAt>=renderIntervalMs;const shouldBenchmarkGpu=lastGpuBenchAt<=0||now-lastGpuBenchAt>=1e3;if(shouldRender){if(shouldBenchmarkGpu){await currentGpu.stepAndRenderBench(iterations);lastGpuBenchAt=now}else{currentGpu.stepAndRender(iterations)}lastRenderAt=now}else if(shouldBenchmarkGpu){await currentGpu.stepBench(iterations);lastGpuBenchAt=now}else{currentGpu.step(iterations)}if(now-lastStatusPostAt>=1e3){lastStatusPostAt=now;postStatus()}}catch(error){stopLoopInternal();self.postMessage({type:"workerError",message:error instanceof Error?error.message:String(error)});return}scheduleNextFrame()}function reply(requestId,payload={},transfer=[]){if(!requestId)return;self.postMessage({type:"response",requestId,ok:true,...payload},transfer)}function replyError(requestId,error){if(!requestId)return;self.postMessage({type:"response",requestId,ok:false,error:error instanceof Error?error.message:String(error)})}async function buildRasterFromBlob(blob,mode="single"){if(!(blob instanceof Blob)){throw new Error("DEM image payload must be a Blob or File.")}if(typeof createImageBitmap!=="function"){throw new Error("createImageBitmap is not available in this worker.")}const bitmap=await createImageBitmap(blob,{imageOrientation:"none"});try{return await buildRasterFromBitmap(bitmap,mode)}finally{bitmap.close?.()}}async function buildRasterFromBitmap(bitmap,mode="single"){const width=bitmap.width|0;const height=bitmap.height|0;const canvas=new OffscreenCanvas(width,height);const ctx=canvas.getContext("2d",{willReadFrequently:true});if(!ctx){throw new Error("Failed to acquire worker raster canvas context.")}ctx.setTransform(1,0,0,-1,0,height);ctx.clearRect(0,0,width,height);ctx.drawImage(bitmap,0,0,width,height);ctx.setTransform(1,0,0,1,0,0);const imageData=ctx.getImageData(0,0,width,height);const rgba=imageData.data;const count=width*height;const packedRgba=mode==="packed_rgba";const mask=new Uint8Array(count);if(packedRgba){const bands=[new Float32Array(count),new Float32Array(count),new Float32Array(count),new Float32Array(count)];const values2=bands[0];for(let i=0,j=0;i<count;i++,j+=4){values2[i]=rgba[j]/255;bands[1][i]=rgba[j+1]/255;bands[2][i]=rgba[j+2]/255;bands[3][i]=rgba[j+3]/255;mask[i]=1}return{width,height,values:values2,mask,bands}}const values=new Float32Array(count);for(let i=0,j=0;i<count;i++,j+=4){values[i]=(rgba[j]+rgba[j+1]+rgba[j+2])/(3*255);mask[i]=rgba[j+3]===0?0:1}return{width,height,values,mask,bands:null}}async function buildStackedLayerRasterFromBlobs(layerBlobs=[]){const firstIndex=layerBlobs.findIndex(Boolean);if(firstIndex<0){throw new Error("Layer stack mode requires at least one layer PNG.")}const first=await buildRasterFromBlob(layerBlobs[firstIndex],"single");const width=first.width|0;const height=first.height|0;const count=width*height;const bands=[new Float32Array(count),new Float32Array(count),new Float32Array(count),new Float32Array(count)];const values=bands[0];const mask=first.mask instanceof Uint8Array&&first.mask.length===count?first.mask.slice(0):new Uint8Array(count).fill(1);bands[firstIndex].set(first.values);releaseRaster(first);for(let layerIndex=0;layerIndex<4;layerIndex++){if(layerIndex===firstIndex||!layerBlobs[layerIndex])continue;const layer=await buildRasterFromBlob(layerBlobs[layerIndex],"single");if(layer.width!==width||layer.height!==height){throw new Error("All layer PNGs in stack mode must have the same dimensions.")}bands[layerIndex].set(layer.values);if(layer.mask instanceof Uint8Array&&layer.mask.length===count){for(let i=0;i<count;i++){mask[i]=mask[i]&&layer.mask[i]?1:0}}releaseRaster(layer)}return{width,height,values,mask,bands}}function resampleRasterBilinear(raster,scale){const safeScale=Math.max(1,Math.floor(Number(scale)||1));if(!raster||safeScale===1)return raster;const srcWidth=raster.width|0;const srcHeight=raster.height|0;const dstWidth=Math.max(1,(srcWidth-1)*safeScale+1);const dstHeight=Math.max(1,(srcHeight-1)*safeScale+1);const dstMask=new Uint8Array(dstWidth*dstHeight);const srcValues=raster.values;const srcMask=raster.mask instanceof Uint8Array?raster.mask:null;const srcBands=Array.isArray(raster.bands)?raster.bands:null;const dstBands=srcBands?srcBands.map(()=>new Float32Array(dstWidth*dstHeight)):null;const dstValues=dstBands?dstBands[0]:new Float32Array(dstWidth*dstHeight);function sampleArray(srcArray,px,py){const x0=Math.floor(px);const y0=Math.floor(py);const x1=Math.min(x0+1,srcWidth-1);const y1=Math.min(y0+1,srcHeight-1);const tx=px-x0;const ty=py-y0;const i00=y0*srcWidth+x0;const i10=y0*srcWidth+x1;const i01=y1*srcWidth+x0;const i11=y1*srcWidth+x1;const v00=srcArray[i00];const v10=srcArray[i10];const v01=srcArray[i01];const v11=srcArray[i11];const a=v00+(v10-v00)*tx;const b=v01+(v11-v01)*tx;return a+(b-a)*ty}function sampleMask(px,py){if(!srcMask)return 1;const x0=Math.floor(px);const y0=Math.floor(py);const x1=Math.min(x0+1,srcWidth-1);const y1=Math.min(y0+1,srcHeight-1);const tx=px-x0;const ty=py-y0;const i00=y0*srcWidth+x0;const i10=y0*srcWidth+x1;const i01=y1*srcWidth+x0;const i11=y1*srcWidth+x1;const m00=srcMask[i00];const m10=srcMask[i10];const m01=srcMask[i01];const m11=srcMask[i11];const a=m00+(m10-m00)*tx;const b=m01+(m11-m01)*tx;return a+(b-a)*ty>=.5?1:0}for(let y=0;y<dstHeight;y++){const srcY=y/safeScale;for(let x=0;x<dstWidth;x++){const srcX=x/safeScale;const dstIndex=y*dstWidth+x;dstValues[dstIndex]=sampleArray(srcValues,srcX,srcY);dstMask[dstIndex]=sampleMask(srcX,srcY);if(dstBands&&srcBands){for(let bandIndex=1;bandIndex<dstBands.length;bandIndex++){dstBands[bandIndex][dstIndex]=sampleArray(srcBands[bandIndex],srcX,srcY)}}}}return{width:dstWidth,height:dstHeight,values:dstValues,mask:dstMask,bands:dstBands}}function releaseRaster(raster){if(!raster)return;raster.values=null;raster.mask=null;raster.bands=null}async function loadRasterIntoGpu(message){const currentGpu=await ensureGpu();if(Object.keys(simulationParams).length>0){currentGpu.setSimulationParams(simulationParams)}const demSourceMode=message.demSourceMode||message.options?.demSourceMode||"single";if(Array.isArray(message.layerBlobs)&&message.layerBlobs.some(Boolean)){sourceRaster=await buildStackedLayerRasterFromBlobs(message.layerBlobs);sourceImageInfo={width:sourceRaster.width,height:sourceRaster.height,mode:"stack4"}}else if(message.blob){sourceRaster=await buildRasterFromBlob(message.blob,demSourceMode);sourceImageInfo={width:sourceRaster.width,height:sourceRaster.height,mode:demSourceMode}}if(!sourceRaster){throw new Error("No DEM image has been loaded into the worker yet.")}const tessellation=Math.max(1,Math.floor(Number(message.tessellation)||1));const raster=resampleRasterBilinear(sourceRaster,tessellation);await currentGpu.setDEM(raster,{...message.options||{},demSourceMode});currentGpu.render();const stats=cloneStats(currentGpu.getStats());if(raster!==sourceRaster){releaseRaster(raster)}return{stats,sourcePoints:getSourcePoints(),ready:true,sourceImageInfo,simSize:{width:raster.width,height:raster.height}}}self.onmessage=async event=>{const message=event.data||{};const{type,requestId}=message;try{switch(type){case"init":{offscreenCanvas=message.canvas;canvasMetrics={width:Math.max(1,Number(message.width)||1),height:Math.max(1,Number(message.height)||1),dpr:Math.max(1,Number(message.dpr)||1)};applyCanvasMetrics();warmGpuInBackground();reply(requestId,{initialized:true,warming:true});postStatus(true);break}case"resize":{canvasMetrics={width:Math.max(1,Number(message.width)||canvasMetrics.width||1),height:Math.max(1,Number(message.height)||canvasMetrics.height||1),dpr:Math.max(1,Number(message.dpr)||canvasMetrics.dpr||1)};applyCanvasMetrics();gpu?.render?.();reply(requestId,{resized:true});break}case"setParams":{simulationParams={...simulationParams,...message.params||{}};if(gpu){gpu.setSimulationParams(simulationParams)}reply(requestId,{applied:true});break}case"loadDEMImage":{const payload=await loadRasterIntoGpu(message);reply(requestId,payload);postStatus(true);break}case"setDEM":{const currentGpu=await ensureGpu();if(Object.keys(simulationParams).length>0){currentGpu.setSimulationParams(simulationParams)}const raster=message.raster||{};await currentGpu.setDEM({width:raster.width|0,height:raster.height|0,values:raster.values instanceof Float32Array?raster.values:new Float32Array(raster.values||[]),mask:raster.mask?raster.mask instanceof Uint8Array?raster.mask:new Uint8Array(raster.mask):null,bands:Array.isArray(raster.bands)?raster.bands.map(band=>band instanceof Float32Array?band:new Float32Array(band||[])):null},{...message.options||{},demSourceMode:message.options?.demSourceMode||"single"});currentGpu.render();const stats=cloneStats(currentGpu.getStats());reply(requestId,{stats,sourcePoints:getSourcePoints(),ready:true,sourceImageInfo});postStatus(true);break}case"render":{gpu?.render?.();reply(requestId,{rendered:true});break}case"step":{if(gpu?.ready){if(message.render!==false){await gpu.stepAndRenderBench(Math.max(1,Number(message.iterations)||1))}else{await gpu.stepBench(Math.max(1,Number(message.iterations)||1))}}reply(requestId,{stepped:true,stats:cloneStats(gpu?.getStats?.()||null),sourcePoints:getSourcePoints()});break}case"readbackStats":{const stats=cloneStats(gpu?.getStats?.()||null);reply(requestId,{stats,sourcePoints:getSourcePoints(),lastFrameMs,running,sourceImageInfo});break}case"exportTerrainPng":{if(!gpu?.ready){throw new Error("Terrain export is only available after the GPU sim is ready.")}const exportResult=await gpu.exportTerrainPng();reply(requestId,{width:exportResult.width,height:exportResult.height,minTerrain:exportResult.minTerrain,maxTerrain:exportResult.maxTerrain,data:exportResult.data},[exportResult.data]);break}case"paintTerrainBrush":{if(!gpu?.ready){throw new Error("Terrain painting is only available after the GPU sim is ready.")}const stats=cloneStats(await gpu.applyTerrainBrush(message.brush||{}));gpu.render();reply(requestId,{stats,sourcePoints:getSourcePoints()});postStatus(true);break}case"paintSpringBrush":{if(!gpu?.ready){throw new Error("Spring painting is only available after the GPU sim is ready.")}gpu.paintSpringBrush(message.brush||{});const stats=cloneStats(gpu.getStats());reply(requestId,{stats,sourcePoints:getSourcePoints()});postStatus(true);break}case"clearPaintedSprings":{if(gpu?.ready){gpu.clearPaintedSprings()}const stats=cloneStats(gpu?.getStats?.()||null);reply(requestId,{stats,sourcePoints:getSourcePoints()});postStatus(true);break}case"resetRainTimer":{if(gpu?.ready){gpu.resetRainTimer();gpu.render()}const stats=cloneStats(gpu?.getStats?.()||null);reply(requestId,{stats,sourcePoints:getSourcePoints()});break}case"restartSources":{if(gpu?.ready){gpu.resetRainTimer();gpu.render()}const stats=cloneStats(gpu?.getStats?.()||null);reply(requestId,{stats,sourcePoints:getSourcePoints()});break}case"startLoop":{if(Number.isFinite(message.iterationsPerFrame)){loopIterationsPerFrame=Math.max(1,Number(message.iterationsPerFrame)|0)}if(!running){running=true;lastFrameAt=0;lastStatusPostAt=0;lastRenderAt=0;lastGpuBenchAt=0;scheduleNextFrame()}reply(requestId,{running:true,iterationsPerFrame:loopIterationsPerFrame});break}case"stopLoop":{stopLoopInternal();if(gpu?.ready){gpu.render()}reply(requestId,{running:false});postStatus(true);break}case"clear":{stopLoopInternal();if(gpu){gpu.destroy();gpu=null}gpuInitPromise=null;sourceRaster=null;sourceImageInfo=null;reply(requestId,{cleared:true});postStatus(true);break}case"getStatus":{reply(requestId,{stats:cloneStats(gpu?.getStats?.()||null),sourcePoints:getSourcePoints(),lastFrameMs,running,sourceImageInfo});break}case"destroy":{stopLoopInternal();if(gpu){gpu.destroy();gpu=null}gpuInitPromise=null;sourceRaster=null;sourceImageInfo=null;reply(requestId,{destroyed:true});self.close();break}default:throw new Error(\`Unknown worker message type: \${String(type)}\`)}}catch(error){replyError(requestId,error)}};;if(typeof import_meta !== 'undefined')import_meta.url=location.origin+"/dist/";})();
`,Qa=URL.createObjectURL(new globalThis.Blob([Ja],{type:"text/javascript"})),Pa=Qa;var It=0,Dt=1,se=5,it=128,Ot=.02,Ut=.001,Gt=.015,Wt=20,_t=9.81,zt=.82,Nt=.32,Ht=1.28,Vt=2.8,Yt=.09,$t=.24,qt=.92,Xt=.12,Kt=.045,st=.06,Zt=0,jt=.92,Jt=.35,re=.16,ot=.03,Qt=35,ea=18,ta=4,aa=.06,lt=0,dt=0,Ma=0,ut=4,ra=1,ct=100,ht=45,pt=42;var Ba=1.5238999619464006,Ea=1.9404787584406888,Ca=1.5238999619464006,er=1.35,tr=.14,Ra=1,ar=!0,rr=!0,nr=!0,ir=!0,na=12,ia=.03,sa=.7,Ta="single",fe={sand:{label:"sand",hardnessMin:.42,hardnessMax:.7},silt:{label:"silt",hardnessMin:.24,hardnessMax:.42},clay:{label:"clay",hardnessMin:.1,hardnessMax:.22},soft_rock:{label:"soft rock",hardnessMin:.03,hardnessMax:.1},bedrock:{label:"bedrock",hardnessMin:0,hardnessMax:.02},custom:{label:"custom",hardnessMin:.2,hardnessMax:.4},default_base:{label:"base",hardnessMin:re,hardnessMax:re+ot}},At=[{enabled:!0,preset:"default_base",heightMin:0,heightMax:1,thermalEnabled:!0},{enabled:!0,preset:"silt",heightMin:.46,heightMax:.82,thermalEnabled:!0},{enabled:!0,preset:"clay",heightMin:.2,heightMax:.58,thermalEnabled:!0},{enabled:!0,preset:"bedrock",heightMin:0,heightMax:.3,thermalEnabled:!0}],sr=!1;function mt(...e){sr&&console.log("[WebGPU Erosion UI]",...e)}function J(e){let t=document.createElement("button");return t.textContent=e,t.style.padding="8px 12px",t.style.border="1px solid #444",t.style.borderRadius="8px",t.style.background="#1f1f1f",t.style.color="#f0f0f0",t.style.cursor="pointer",t.addEventListener("mouseenter",()=>{t.style.background="#2a2a2a"}),t.addEventListener("mouseleave",()=>{t.style.background="#1f1f1f"}),t}function d(e,t,r="88px",i={}){let n=document.createElement("label");n.textContent=e,n.style.display="inline-flex",n.style.alignItems="center",n.style.gap="8px";let o=document.createElement("input");return o.type="number",o.value=String(t),o.min=String(i.min??-1e9),o.max=String(i.max??1e9),o.step=String(i.step??.01),o.style.width=r,o.style.padding="6px 8px",o.style.border="1px solid #444",o.style.borderRadius="6px",o.style.background="#1a1a1a",o.style.color="#e8e8e8",n.input=o,n.appendChild(o),n}function ne(e,t){let r=document.createElement("label");r.style.display="inline-flex",r.style.alignItems="center",r.style.gap="6px";let i=document.createElement("input");i.type="checkbox",i.checked=t;let n=document.createElement("span");return n.textContent=e,r.input=i,r.append(i,n),r}function or(e){let t=document.createElement("label");t.style.display="inline-flex",t.style.alignItems="center",t.style.gap="8px",t.style.flexWrap="wrap";let r=document.createElement("span");r.textContent=e,r.style.fontSize="12px",r.style.opacity="0.9";let i=document.createElement("input");return i.type="file",i.accept=".png,image/png",i.style.padding="6px 8px",i.style.border="1px solid #444",i.style.borderRadius="6px",i.style.background="#1a1a1a",i.style.color="#e8e8e8",t.input=i,t.caption=r,t.append(r,i),t}function oe(e,t,r){let i=document.createElement("label");i.textContent=e,i.style.display="inline-flex",i.style.alignItems="center",i.style.gap="8px";let n=document.createElement("select");n.style.padding="6px 8px",n.style.border="1px solid #444",n.style.borderRadius="6px",n.style.background="#1a1a1a",n.style.color="#e8e8e8";for(let{value:o,label:l}of t){let c=document.createElement("option");c.value=String(o),c.textContent=l,o===r&&(c.selected=!0),n.appendChild(c)}return i.input=n,i.appendChild(n),i}function s(e,t){let r=Number(e.input.value);return Number.isFinite(r)?r:t}function V(e){let t=document.createElement("section");t.style.border="1px solid #2d2d2d",t.style.borderRadius="10px",t.style.padding="10px",t.style.background="#141414";let r=document.createElement("div");r.textContent=e,r.style.fontSize="13px",r.style.fontWeight="600",r.style.marginBottom="8px",r.style.letterSpacing="0.02em";let i=document.createElement("div");return i.style.display="flex",i.style.flexWrap="wrap",i.style.gap="8px",t.append(r,i),t.body=i,t.heading=r,t}function Y(e){let t=document.createElement("div");return t.textContent=e,t.style.fontSize="12px",t.style.lineHeight="1.35",t.style.color="rgba(255,255,255,0.68)",t.style.margin="-2px 0 8px 0",t}function Aa(){return Math.max(1,Number(ca.input.value)||Ra)}function at(e,t=4){return!e||!Number.isFinite(e.min)||!Number.isFinite(e.max)?"n/a":`${e.min.toFixed(t)} to ${e.max.toFixed(t)}`}function R(e,t,r){return e<t?t:e>r?r:e}function rt(e){return e*Math.PI/180}function me(e){let t=Math.hypot(e[0],e[1],e[2])||1;return[e[0]/t,e[1]/t,e[2]/t]}function La(e,t){return[e[0]-t[0],e[1]-t[1],e[2]-t[2]]}function Lt(e,t){return[e[1]*t[2]-e[2]*t[1],e[2]*t[0]-e[0]*t[2],e[0]*t[1]-e[1]*t[0]]}function Fa(e,t){let r=new Float32Array(16);for(let i=0;i<4;i++)for(let n=0;n<4;n++)r[i*4+n]=e[0+n]*t[i*4+0]+e[4+n]*t[i*4+1]+e[8+n]*t[i*4+2]+e[12+n]*t[i*4+3];return r}function ka(e,t,r,i){let n=1/Math.tan(e*.5),o=1/(r-i),l=new Float32Array(16);return l[0]=n/Math.max(t,1e-6),l[5]=n,l[10]=(i+r)*o,l[11]=-1,l[14]=2*i*r*o,l}function Ia(e,t,r){let i=me(La(e,t)),n=me(Lt(r,i)),o=Lt(i,n),l=new Float32Array(16);return l[0]=n[0],l[1]=o[0],l[2]=i[0],l[3]=0,l[4]=n[1],l[5]=o[1],l[6]=i[1],l[7]=0,l[8]=n[2],l[9]=o[2],l[10]=i[2],l[11]=0,l[12]=-(n[0]*e[0]+n[1]*e[1]+n[2]*e[2]),l[13]=-(o[0]*e[0]+o[1]*e[1]+o[2]*e[2]),l[14]=-(i[0]*e[0]+i[1]*e[1]+i[2]*e[2]),l[15]=1,l}function lr(e){let t=new Float32Array(16),r=e[0],i=e[1],n=e[2],o=e[3],l=e[4],c=e[5],E=e[6],v=e[7],M=e[8],p=e[9],m=e[10],S=e[11],L=e[12],w=e[13],B=e[14],C=e[15],F=r*c-i*l,G=r*E-n*l,W=r*v-o*l,I=i*E-n*c,_=i*v-o*c,z=n*v-o*E,b=M*w-p*L,ae=M*B-m*L,We=M*C-S*L,_e=p*B-m*w,ze=p*C-S*w,Ne=m*C-S*B,f=F*Ne-G*ze+W*_e+I*We-_*ae+z*b;return f?(f=1/f,t[0]=(c*Ne-E*ze+v*_e)*f,t[1]=(n*ze-i*Ne-o*_e)*f,t[2]=(w*z-B*_+C*I)*f,t[3]=(m*_-p*z-S*I)*f,t[4]=(E*We-l*Ne-v*ae)*f,t[5]=(r*Ne-n*We+o*ae)*f,t[6]=(B*W-L*z-C*G)*f,t[7]=(M*z-m*W+S*G)*f,t[8]=(l*ze-c*We+v*b)*f,t[9]=(i*We-r*ze-o*b)*f,t[10]=(L*_-w*W+C*F)*f,t[11]=(p*W-M*_-S*F)*f,t[12]=(c*ae-l*_e-E*b)*f,t[13]=(r*_e-i*ae+n*b)*f,t[14]=(w*G-L*I-B*F)*f,t[15]=(M*I-p*G+m*F)*f,t):null}function Ft(e,t){return[e[0]*t[0]+e[4]*t[1]+e[8]*t[2]+e[12]*t[3],e[1]*t[0]+e[5]*t[1]+e[9]*t[2]+e[13]*t[3],e[2]*t[0]+e[6]*t[1]+e[10]*t[2]+e[14]*t[3],e[3]*t[0]+e[7]*t[1]+e[11]*t[2]+e[15]*t[3]]}var a={sourceImageInfo:null,sourceFile:null,sourceFileName:"",sourceUploadedToWorker:!1,layerSourceFiles:[null,null,null,null],worker:null,workerReady:!1,workerCanvasTransferred:!1,running:!1,lastFrameMs:0,buildCount:0,readbackPending:!1,lastReadbackAt:0,gpuStats:null,sourcePoints:[],isPainting:!1,lastPaintAt:0,cameraPosX:Ba,cameraPosY:Ea,cameraPosZ:Ca,pointerLookActive:!1,navLoopHandle:0,navLastAt:0,navKeys:{KeyW:!1,KeyA:!1,KeyS:!1,KeyD:!1,Space:!1,KeyC:!1},statsFrameHandle:0},dr=1,kt=new Map;function Da(){let e=U.getBoundingClientRect(),t=Math.max(1,Math.round(e.width||960)),r=Math.max(1,Math.round(e.height||Math.round(t*2/3))),i=Math.max(1,Math.min(window.devicePixelRatio||1,2));return{width:t,height:r,dpr:i}}function ur(e){e.addEventListener("message",t=>{let r=t.data||{};if(r.type==="response"&&r.requestId){let i=kt.get(r.requestId);if(!i)return;kt.delete(r.requestId),r.ok?i.resolve(r):i.reject(new Error(r.error||"Worker request failed."));return}if(r.type==="status"){r.stats&&(a.gpuStats=r.stats),Array.isArray(r.sourcePoints)&&(a.sourcePoints=r.sourcePoints),r.sourceImageInfo&&(a.sourceImageInfo=r.sourceImageInfo),Number.isFinite(r.lastFrameMs)&&(a.lastFrameMs=r.lastFrameMs),typeof r.running=="boolean"&&(a.running=r.running),T();return}r.type==="workerError"&&(mt("workerError",r.message),g(r.message||"Worker error"))}),e.addEventListener("error",t=>{console.error("[WebGPU Erosion UI] worker error",t),g(t.message||"Worker crashed")})}async function cr(){if(a.worker)return a.worker;if(!("transferControlToOffscreen"in u))throw new Error("OffscreenCanvas worker rendering is not available in this browser.");let e=new Worker(Pa,{type:"module"});ur(e),a.worker=e;let t=u.transferControlToOffscreen();a.workerCanvasTransferred=!0;let r=Da();return await D("init",{canvas:t,width:r.width,height:r.height,dpr:r.dpr},[t]),a.workerReady=!0,e}function x(e,t={},r=[]){a.worker&&a.worker.postMessage({type:e,...t},r)}function D(e,t={},r=[]){return new Promise((i,n)=>{if(!a.worker){n(new Error("Worker is not initialized."));return}let o=dr++;kt.set(o,{resolve:i,reject:n}),a.worker.postMessage({type:e,requestId:o,...t},r)})}function Oa(){if(!a.workerReady)return;let e=Da();x("resize",e)}function hr(){let e=ee[0],t=R(s(e.hardnessMinLabel,re),0,1),r=R(s(e.hardnessMaxLabel,re+ot),0,1),i=Math.min(t,r),n=Math.max(t,r);return{timeStep:s(He,Ot),rainRate:Fe.input.checked?s(ve,Ut):0,evaporationRate:s(Ve,Gt),pipeArea:s(Ye,Wt),gravity:s($e,_t),capacityScale:s(Se,zt),suspensionRate:s(we,Nt),depositionRate:s(Pe,Ht),softeningRate:s(Me,Vt),maxErosionDepth:s(Be,Yt),thermalRate:ke.input.checked?s(Ee,$t):0,hydraulicErosionEnabled:!!Ke.input.checked,talusSlopeCoeff:s(Ce,qt),talusSlopeBias:s(Re,Xt),renderHeightScale:s(yt,Kt),waterHeightScale:s(qe,st),edgeWaterFloor:Math.max(0,s(la,Zt)),waterOpacity:s(xt,jt),sedimentTint:s(bt,Jt),hardnessBase:i,hardnessVariation:Math.max(0,n-i),sourceCenterX:s(da,Qt)/100,sourceCenterY:s(ua,ea)/100,sourceRadius:s(ue,ta),sourceStrength:s(ce,aa),rainDuration:s(Te,lt),pulse2Duration:s(Xe,dt),sourceLayoutMode:Number(Ae.input.value)||0,randomSpringCount:Math.max(1,Math.floor(s(he,ut))),sourceSeed:Math.floor(s(Le,ra)),metersPerPixel:Math.max(1,s(vt,ct)),hydraulic8Way:!!St.input.checked,sourceEnabled:!!ie.input.checked,renderMode:Number(ge.input.value)||0,cameraAzimuthDeg:s(ye,ht),cameraElevationDeg:s(xe,pt),cameraPosX:a.cameraPosX,cameraPosY:a.cameraPosY,cameraPosZ:a.cameraPosZ}}function be(){return oa.input.value||Ta}function pr(){return{mode:be(),layers:ee.map((e,t)=>({enabled:!!e.enableLabel.input.checked,preset:e.materialPresetLabel.input.value,label:`Layer ${t+1}`,heightMin:s(e.heightMinLabel,At[t].heightMin),heightMax:s(e.heightMaxLabel,At[t].heightMax),hardnessMin:R(s(e.hardnessMinLabel,fe.custom.hardnessMin),0,1),hardnessMax:R(s(e.hardnessMaxLabel,fe.custom.hardnessMax),0,1),thermalEnabled:!!e.thermalEnableLabel.input.checked}))}}function q(){return be()==="stack4"?a.layerSourceFiles.some(Boolean):!!a.sourceFile}function Ua(){return be()==="stack4"?a.layerSourceFiles.map((e,t)=>e?`L${t+1}:${e.name}`:null).filter(Boolean).join(" | "):a.sourceFile?.name||""}function mr(e){let t=fe[e.materialPresetLabel.input.value];!t||e.materialPresetLabel.input.value==="custom"||(e.hardnessMinLabel.input.value=String(t.hardnessMin),e.hardnessMaxLabel.input.value=String(t.hardnessMax))}function K(){let e=ee[0],t=s(e.heightMinLabel,It),r=s(e.heightMaxLabel,Dt),i=R(s(e.hardnessMinLabel,re),0,1),n=R(s(e.hardnessMaxLabel,re+ot),0,1),o=Math.min(i,n),l=Math.max(i,n);Wa.input.value=String(t),_a.input.value=String(r),fr.input.value=String(o),gr.input.value=String(Math.max(0,l-o))}function Ga(){let e=be(),t=e==="stack4",r=e==="single";de.style.display=t?"none":"inline-flex",Va.textContent=r?"Single grayscale uses only Layer 1. Use that row to set the height range and erodability for the base DEM.":"Use a packed RGBA DEM where each channel is a layer, or a stack of up to four grayscale PNGs. Height min/max defines each layer top surface, erodability min/max defines how easily each layer cuts, and each layer can opt in or out of thermal collapse.";for(let[i,n]of ee.entries()){let o=!r||i===0;n.row.style.display=o?"grid":"none",n.filePickerLabel.style.display=o&&t?"inline-flex":"none",n.filePickerLabel.input.disabled=!(o&&t),n.enableLabel.style.display=r&&i===0?"none":"inline-flex",r&&i===0?(n.enableLabel.input.checked=!0,n.heading.textContent="Layer 1 / base material"):n.heading.textContent=`Layer ${i+1}`}K()}var Q=document.createElement("div");Q.style.boxSizing="border-box";Q.style.padding="16px";Q.style.fontFamily="system-ui, sans-serif";Q.style.color="#e8e8e8";Q.style.background="#111";Q.style.minHeight="100vh";var ft=document.createElement("h2");ft.textContent="WebGPU Fast Hydraulic + Thermal Erosion (Jako)";ft.style.margin="0 0 12px 0";ft.style.fontWeight="600";var le=document.createElement("div");le.style.display="flex";le.style.flexWrap="wrap";le.style.alignItems="center";le.style.gap="8px";le.style.marginBottom="14px";var de=document.createElement("label");de.textContent="DEM PNG";de.style.display="inline-flex";de.style.alignItems="center";de.style.gap="8px";var O=document.createElement("input");O.type="file";O.accept=".png,image/png";O.style.padding="6px 8px";O.style.border="1px solid #444";O.style.borderRadius="6px";O.style.background="#1a1a1a";O.style.color="#e8e8e8";de.appendChild(O);var oa=oe("DEM source",[{value:"single",label:"single grayscale"},{value:"packed_rgba",label:"packed RGBA layers"},{value:"stack4",label:"4 grayscale stack"}],Ta),Wa=d("Height floor",It,"88px"),_a=d("Height ceiling",Dt,"88px"),Z=d("Sim steps/frame",se,"112px",{min:1,step:1}),gt=d("Single-step size",it,"118px",{min:1,step:1}),He=d("Time step",Ot,"88px",{min:.001,step:.001}),ve=d("Background rain",Ut,"108px",{min:0,step:1e-4}),Ve=d("Evaporation",Gt,"96px",{min:0,step:.001}),Ye=d("Water flow width",Wt,"112px",{min:.1,step:.1}),$e=d("Gravity",_t,"84px",{min:.1,step:.1}),Se=d("Sediment carry",zt,"104px",{min:0,step:.01}),we=d("Erode into water",Nt,"112px",{min:0,step:.01}),Pe=d("Deposit from water",Ht,"118px",{min:0,step:.01}),Me=d("Bank softening",Vt,"108px",{min:0,step:.01}),Be=d("Max cut depth",Yt,"104px",{min:.001,step:.01}),Ee=d("Slope collapse",$t,"104px",{min:0,step:.01}),Ce=d("Slope limit scale",qt,"118px",{min:0,step:.01}),Re=d("Slope limit bias",Xt,"110px",{min:0,step:.01}),yt=d("Terrain exaggeration",Kt,"130px",{min:0,step:.05}),qe=d("Water height scale",st,"122px",{min:0,step:.01}),la=d("Min edge water",Zt,"110px",{min:0,step:.005}),xt=d("Water visibility",jt,"112px",{min:0,max:1,step:.05}),bt=d("Sediment boost",Jt,"104px",{min:0,max:2,step:.05}),fr=d("Base erodability",re,"108px",{min:0,max:1,step:.01}),gr=d("Erodability variation",ot,"122px",{min:0,max:1,step:.01}),ie=ne("Enable springs",!1),da=d("Spring X %",Qt,"92px",{min:0,max:100,step:1}),ua=d("Spring Y %",ea,"92px",{min:0,max:100,step:1}),ue=d("Spring radius px",ta,"108px",{min:1,step:1}),ce=d("Spring flow rate",aa,"104px",{min:0,step:.005}),Te=d("Rain duration s",lt,"104px",{min:0,step:.5}),Xe=d("Pulse 2 sec",dt,"92px",{min:0,step:.5}),Ae=oe("Spring mode",[{value:0,label:"painted springs"},{value:1,label:"fixed random springs"}],Ma),he=d("Spring count",ut,"96px",{min:1,max:16,step:1}),Le=d("Spring seed",ra,"96px",{min:0,step:1}),vt=d("Meters / pixel",ct,"104px",{min:1,step:1}),ca=oe("Sim tess",[{value:1,label:"1x"},{value:2,label:"2x"},{value:4,label:"4x"}],Ra),St=ne("8-way hydraulic pipes",ar),Fe=ne("Enable precipitation",rr),ke=ne("Enable thermal erosion",nr),Ke=ne("Enable hydraulic erosion",ir),ge=oe("View",[{value:0,label:"shaded"},{value:2,label:"water"},{value:3,label:"sediment"},{value:4,label:"erodability"},{value:5,label:"thermal"},{value:6,label:"erosion/deposition history"},{value:7,label:"natural geomorph"}],0),za=oe("Preset",[{value:"paper_balanced",label:"paper balanced"},{value:"river_cut",label:"river cut"},{value:"delta_depositor",label:"delta depositor"},{value:"meander_builder",label:"meander builder"},{value:"thermal_heavy",label:"thermal heavy"},{value:"gentle_weathering",label:"gentle weathering"},{value:"rapid_incision",label:"rapid incision"},{value:"flash_flood",label:"flash flood"},{value:"badlands",label:"badlands"},{value:"canyon_carver",label:"canyon carver"}],"paper_balanced"),ye=d("Azimuth",ht,"84px",{min:-180,max:180,step:1}),xe=d("Elevation",pt,"84px",{min:5,max:89,step:1}),k=oe("Paint",[{value:"none",label:"off"},{value:"raise",label:"terrain +"},{value:"lower",label:"terrain -"},{value:"spring_add",label:"spring +"},{value:"spring_erase",label:"spring erase"}],"none"),wt=d("Brush radius px",na,"112px",{min:1,step:1}),Pt=d("Brush amount",ia,"104px",{min:.001,step:.005}),Mt=d("Brush hardness",sa,"116px",{min:.05,max:1,step:.05}),yr=Object.entries(fe).map(([e,t])=>({value:e,label:t.label})),ee=At.map((e,t)=>{let r=fe[e.preset]||fe.custom,i=document.createElement("div");i.style.display="grid",i.style.gridTemplateColumns="repeat(2, minmax(0, 1fr))",i.style.gap="8px",i.style.padding="8px",i.style.border="1px solid #2c2c2c",i.style.borderRadius="8px",i.style.background="#121212";let n=document.createElement("div");n.textContent=`Layer ${t+1}`,n.style.gridColumn="1 / -1",n.style.fontSize="12px",n.style.fontWeight="600",n.style.opacity="0.95";let o=ne("Enable layer",e.enabled),l=oe("Preset",yr,e.preset),c=d("Height min",e.heightMin,"86px",{min:-10,max:10,step:.01}),E=d("Height max",e.heightMax,"86px",{min:-10,max:10,step:.01}),v=d("Erod min",r.hardnessMin,"86px",{min:0,max:1,step:.01}),M=d("Erod max",r.hardnessMax,"86px",{min:0,max:1,step:.01}),p=ne("Thermal on this layer",e.thermalEnabled!==!1),m=or(`Layer ${t+1} PNG`);return m.style.gridColumn="1 / -1",i.append(n,o,l,c,E,v,M,p,m),{row:i,heading:n,enableLabel:o,materialPresetLabel:l,heightMinLabel:c,heightMaxLabel:E,hardnessMinLabel:v,hardnessMaxLabel:M,thermalEnableLabel:p,filePickerLabel:m}}),pe=J("Run");pe.disabled=!0;var Ze=J("Step");Ze.disabled=!0;var Na=J("Apply preset"),Bt=J("Reset rain timer");Bt.disabled=!0;var Ie=J("Reset DEM");Ie.disabled=!0;var Ha=J("Clear"),N=J("Export DEM PNG");N.disabled=!0;var De=J("Clear painted springs");De.disabled=!0;var je=document.createElement("div");je.style.marginLeft="12px";je.style.opacity="0.9";je.style.fontSize="14px";le.append(de,za,Na,pe,Ze,N,Ie,Ha,je);var Oe=document.createElement("div");Oe.style.display="grid";Oe.style.gridTemplateColumns="minmax(320px, 1fr) minmax(280px, 380px)";Oe.style.gap="16px";Oe.style.alignItems="start";var te=document.createElement("div");te.style.background="#181818";te.style.border="1px solid #2d2d2d";te.style.borderRadius="12px";te.style.padding="12px";te.style.boxSizing="border-box";te.style.minWidth="0";var Je=document.createElement("div");Je.textContent="Render view:";Je.style.fontSize="15px";Je.style.fontWeight="600";Je.style.marginBottom="8px";var U=document.createElement("div");U.style.width="100%";U.style.overflow="hidden";U.style.border="1px solid #333";U.style.borderRadius="10px";U.style.background="#0d0d0d";U.style.position="relative";U.style.aspectRatio="3 / 1.5";var y=document.createElement("canvas");y.width=960;y.height=640;y.style.display="block";y.style.width="100%";y.style.height="100%";y.style.background="#0d0d0d";y.style.position="absolute";y.style.inset="0";y.style.zIndex="0";var u=document.createElement("canvas");u.width=960;u.height=640;u.style.display="block";u.style.width="100%";u.style.height="100%";u.style.background="#0d0d0d";u.style.position="absolute";u.style.inset="0";u.style.zIndex="1";u.style.visibility="hidden";u.style.touchAction="none";u.tabIndex=0;var h=document.createElement("canvas");h.width=960;h.height=640;h.style.display="block";h.style.width="100%";h.style.height="100%";h.style.position="absolute";h.style.inset="0";h.style.zIndex="2";h.style.pointerEvents="none";h.style.visibility="hidden";U.append(y,u,h);te.append(Je,U);var $=document.createElement("div");$.style.background="#181818";$.style.border="1px solid #2d2d2d";$.style.borderRadius="12px";$.style.padding="12px";$.style.boxSizing="border-box";$.style.maxHeight="85vh";$.style.overflowY="scroll";var Qe=document.createElement("div");Qe.textContent="Stats";Qe.style.fontSize="15px";Qe.style.fontWeight="600";Qe.style.marginBottom="8px";var j=document.createElement("div");j.style.whiteSpace="pre-line";j.style.fontSize="13px";j.style.opacity="0.95";j.style.lineHeight="1.45";j.textContent="Load a DEM PNG to begin.";var et=document.createElement("div");et.style.display="grid";et.style.gap="10px";et.style.marginBottom="12px";var ha=V("Process toggles");ha.append(Y("Master on/off switches for precipitation, springs, hydraulic erosion, and thermal erosion."));ha.body.append(Fe,ie,ke,Ke,Te,Bt);var Et=V("DEM source and layered materials"),Va=Y("Use a single grayscale DEM, a packed RGBA DEM where each channel is a layer, or a stack of up to four grayscale PNGs. Height min/max defines each layer top surface, erodability min/max defines how easily each layer cuts, and each layer can opt in or out of thermal collapse.");Et.append(Va);Et.body.append(oa);for(let e of ee)Et.body.append(e.row);var pa=V("Shared water flow and simulation");pa.append(Y("These affect precipitation, springs, water flow, edge runoff retention, and general simulation pacing. They are not thermal-only."));pa.body.append(Z,gt,He,ve,Ve,Ye,$e,ca,St,vt,la);var ma=V("Hydraulic erosion only");ma.append(Y("These only change hydraulic erosion and deposition behavior. Water can still animate with hydraulic erosion disabled."));ma.body.append(Se,we,Pe,Me,Be);var fa=V("Thermal erosion only");fa.append(Y("These only affect slope collapse and talus behavior."));fa.body.append(Ee,Ce,Re);var ga=V("Springs");ga.append(Y("Spring mode and spring-source controls. Random-spring controls only apply in fixed random springs mode."));ga.body.append(Ae,ue,ce,Xe,he,Le,De);var ya=V("View and shading");ya.append(Y("Render-only controls. These do not change erosion behavior. Natural geomorph uses erosion/deposition history plus thermal activity to tint rock, soil, alluvium, and colluvium."));ya.body.append(ge,yt,qe,xt,bt);var xa=V("Camera and navigation");xa.append(Y("Orbit inputs below. Click the canvas with paint off for pointer look, then use WASD to move, Space to go up, C to go down, and Esc to release."));xa.body.append(ye,xe);var ba=V("Painting");ba.append(Y("Terrain and spring painting tools. Pointer look is disabled while a paint mode is active."));ba.body.append(k,wt,Pt,Mt);et.append(xa,ya,Et,ha,pa,ma,fa,ga,ba);$.append(et,Qe,j);Oe.append(te,$);Q.append(ft,le,Oe);document.body.style.margin="0";document.body.style.background="#111";document.body.appendChild(Q);window.addEventListener("beforeunload",()=>{Ge(),a.worker&&(x("destroy"),a.worker.terminate(),a.worker=null,a.workerReady=!1)});window.addEventListener("resize",()=>{Ct(),Oa(),x("render")});O.addEventListener("change",()=>{let e=O.files?.[0];e&&Mr(e)});oa.input.addEventListener("change",()=>{a.sourceUploadedToWorker=!1,Ga(),K(),q()?X():T()});for(let[e,t]of ee.entries()){t.materialPresetLabel.input.addEventListener("change",()=>{mr(t),e===0&&K(),q()?X():T()});for(let r of[t.enableLabel,t.heightMinLabel,t.heightMaxLabel,t.hardnessMinLabel,t.hardnessMaxLabel,t.thermalEnableLabel])r.input.addEventListener("input",()=>{e===0&&K(),q()?X():P()}),r.input.addEventListener("change",()=>{e===0&&K(),q()?X():P()});t.filePickerLabel.input.addEventListener("change",()=>{let r=t.filePickerLabel.input.files?.[0]||null;a.layerSourceFiles[e]=r,a.sourceUploadedToWorker=!1,a.sourceFileName=Ua(),q()?X():T()})}Na.addEventListener("click",()=>{ja(za.input.value)});pe.addEventListener("click",()=>{a.running?Ge():Rr()});Bt.addEventListener("click",async()=>{if(!a.workerReady||!a.gpuStats?.ready)return;A();let e=await D("resetRainTimer");e.stats&&(a.gpuStats=e.stats),Array.isArray(e.sourcePoints)&&(a.sourcePoints=e.sourcePoints),g("Rain timer reset"),P()});Ze.addEventListener("click",async()=>{if(!a.workerReady||!a.gpuStats?.ready)return;A();let e=await D("step",{iterations:Math.max(1,Math.floor(s(gt,it))),render:!0});e.stats&&(a.gpuStats=e.stats),Array.isArray(e.sourcePoints)&&(a.sourcePoints=e.sourcePoints),await Tt(!0),P()});Z.input.addEventListener("input",async()=>{let e=Math.max(1,Math.floor(s(Z,se)));P(),a.workerReady&&a.running&&await D("startLoop",{iterationsPerFrame:e})});N.addEventListener("click",async()=>{if(!(!a.workerReady||!a.gpuStats?.ready)){N.disabled=!0,g("Exporting DEM PNG\u2026");try{let e=await D("exportTerrainPng"),t=e.data instanceof ArrayBuffer?e.data:e.data?.buffer;if(!(t instanceof ArrayBuffer))throw new Error("Terrain export did not return PNG data.");let r=new Blob([t],{type:"image/png"}),i=(a.sourceFileName||"terrain").replace(/\.[^.]+$/,""),n=document.createElement("a"),o=URL.createObjectURL(r);n.href=o,n.download=`${i}_dem_export.png`,document.body.appendChild(n),n.click(),n.remove(),setTimeout(()=>URL.revokeObjectURL(o),0);let l=Number.isFinite(e.minTerrain)?e.minTerrain.toFixed(4):"n/a",c=Number.isFinite(e.maxTerrain)?e.maxTerrain.toFixed(4):"n/a";g(`Exported DEM PNG (${e.width}x${e.height}, range ${l} to ${c})`)}catch(e){g(e instanceof Error?e.message:String(e))}finally{N.disabled=!a.gpuStats?.ready}}});De.addEventListener("click",async()=>{if(!a.workerReady)return;let e=await D("clearPaintedSprings");e.stats&&(a.gpuStats=e.stats),Array.isArray(e.sourcePoints)&&(a.sourcePoints=e.sourcePoints),g("Cleared painted springs"),P()});k.input.addEventListener("change",()=>{tt(),P()});for(let e of[Fe,ke,Ke])e.input.addEventListener("change",()=>{Ue(),A(),x("render"),P()});for(let e of[wt,Pt,Mt])e.input.addEventListener("input",()=>{P()});Ie.addEventListener("click",()=>{X()});Ha.addEventListener("click",wa);var xr=new Set([ue,ce,he,Le]);for(let e of[He,ve,Ve,Ye,$e,Se,we,Pe,Me,Be,Ee,Ce,Re,yt,qe,xt,bt,ye,xe,da,ua,ue,ce,Te,Xe,he,Le])e.input.addEventListener("input",()=>{A(),a.running?x("startLoop",{iterationsPerFrame:Math.max(1,Math.floor(s(Z,se)))}):x("render"),xr.has(e)&&Rt(),T()});ie.input.addEventListener("change",()=>{Ue(),K(),A(),x("render"),Rt(),P()});Ae.input.addEventListener("change",()=>{Ue(),A(),x("render"),Rt(),P()});St.input.addEventListener("change",()=>{A(),a.running?x("startLoop",{iterationsPerFrame:Math.max(1,Math.floor(s(Z,se)))}):x("render"),P()});ge.input.addEventListener("change",()=>{A(),x("render"),P()});ca.input.addEventListener("change",()=>{T(),q()&&X()});ja("paper_balanced");Ga();K();Ct();nt(!1);g();Ue();tt();H();function Ya(){return Number(Ae.input.value)||0}function br(){return Ya()===1?"fixed random springs":"painted springs"}function $a(){return k.input.value==="none"}function tt(){let e=k.input.value!=="none";document.pointerLockElement===u&&e&&document.exitPointerLock?.(),u.style.cursor=e?"crosshair":document.pointerLockElement===u?"none":"grab"}function vr(){return[a.cameraPosX,a.cameraPosY,a.cameraPosZ]}function va(){let e=rt(s(ye,ht)),t=rt(s(xe,pt)),r=vr(),i=me([-Math.cos(t)*Math.cos(e),-Math.sin(t),-Math.cos(t)*Math.sin(e)]),n=me([i[0],0,i[2]]),o=me(Lt(n,[0,1,0])),l=[r[0]+i[0],r[1]+i[1],r[2]+i[2]];return{azimuth:e,elevation:t,eye:r,target:l,forward:i,forwardFlat:n,right:o}}function Sr(){return!!(a.navKeys.KeyW||a.navKeys.KeyA||a.navKeys.KeyS||a.navKeys.KeyD||a.navKeys.Space||a.navKeys.KeyC)}function qa(e=performance.now()){if(!a.pointerLookActive){a.navLoopHandle=0,a.navLastAt=0;return}let t=a.navLastAt>0?Math.min(.05,Math.max(.001,(e-a.navLastAt)/1e3)):1/60;if(a.navLastAt=e,Sr()){let r=va(),i=er*t,n=0,o=0,l=0;a.navKeys.KeyW&&(n+=r.forwardFlat[0]*i,l+=r.forwardFlat[2]*i),a.navKeys.KeyS&&(n-=r.forwardFlat[0]*i,l-=r.forwardFlat[2]*i),a.navKeys.KeyD&&(n+=r.right[0]*i,l+=r.right[2]*i),a.navKeys.KeyA&&(n-=r.right[0]*i,l-=r.right[2]*i),a.navKeys.Space&&(o+=i),a.navKeys.KeyC&&(o-=i),a.cameraPosX+=n,a.cameraPosY+=o,a.cameraPosZ+=l,A(),x("render"),H(),P()}a.pointerLookActive?a.navLoopHandle=requestAnimationFrame(qa):a.navLoopHandle=0}function Xa(){!a.pointerLookActive||a.navLoopHandle||(a.navLastAt=0,a.navLoopHandle=requestAnimationFrame(qa))}function wr(){if(a.pointerLookActive=document.pointerLockElement===u,a.pointerLookActive)Xa();else{for(let e of Object.keys(a.navKeys))a.navKeys[e]=!1;a.navLoopHandle&&cancelAnimationFrame(a.navLoopHandle),a.navLoopHandle=0,a.navLastAt=0}tt()}function Pr(e){if(document.pointerLockElement!==u||!$a())return;e.preventDefault();let t=tr,r=s(ye,ht)+e.movementX*t,i=R(s(xe,pt)+e.movementY*t,5,89);ye.input.value=String(r),xe.input.value=String(i),A(),x("render"),H(),T()}function Ka(e,t){document.pointerLockElement===u&&e.code in a.navKeys&&(e.preventDefault(),a.navKeys[e.code]=t,t&&Xa())}document.addEventListener("pointerlockchange",wr);document.addEventListener("pointermove",Pr,{passive:!1});document.addEventListener("keydown",e=>Ka(e,!0),{passive:!1});document.addEventListener("keyup",e=>Ka(e,!1),{passive:!1});u.addEventListener("wheel",e=>{document.pointerLockElement===u&&e.preventDefault()},{passive:!1});function Ue(){let e=!!ie.input.checked,t=!!Fe.input.checked,r=!!ke.input.checked,i=!!Ke.input.checked,n=Ya()===1;ve.input.disabled=!t,Se.input.disabled=!i,we.input.disabled=!i,Pe.input.disabled=!i,Me.input.disabled=!i,Be.input.disabled=!i,Ee.input.disabled=!r,Ce.input.disabled=!r,Re.input.disabled=!r,Te.input.disabled=!t,Ae.input.disabled=!e,ue.input.disabled=!e||!n,ce.input.disabled=!e||!n,he.input.disabled=!e||!n,Le.input.disabled=!e||!n,Bt.disabled=!a.gpuStats?.ready,tt()}u.addEventListener("contextmenu",e=>{(k.input.value!=="none"||document.pointerLockElement===u)&&e.preventDefault()});u.addEventListener("click",e=>{!a.gpuStats?.ready||!$a()||(e.preventDefault(),u.focus(),document.pointerLockElement!==u&&u.requestPointerLock?.())});u.addEventListener("pointerdown",e=>{(k.input.value!=="none"||document.pointerLockElement===u)&&a.gpuStats?.ready&&e.preventDefault(),!(k.input.value==="none"||!a.gpuStats?.ready)&&(a.isPainting=!0,u.setPointerCapture?.(e.pointerId),Za(e))});u.addEventListener("pointermove",e=>{a.isPainting&&(e.preventDefault(),Za(e))});u.addEventListener("pointerup",Sa);u.addEventListener("pointerleave",Sa);u.addEventListener("pointercancel",Sa);function nt(e){u.style.visibility=e?"visible":"hidden",h.style.visibility=e?"visible":"hidden",y.style.visibility=e?"hidden":"visible",H(),tt()}function Ct(){let e=U.getBoundingClientRect(),t=Math.max(1,Math.round(e.width||960)),r=Math.max(1,Math.round(e.height||Math.round(t*2/3))),i=Math.max(1,Math.min(window.devicePixelRatio||1,2)),n=Math.max(1,Math.round(t*i)),o=Math.max(1,Math.round(r*i));(y.width!==n||y.height!==o)&&(y.width=n,y.height=o),(h.width!==n||h.height!==o)&&(h.width=n,h.height=o),H()}async function Mr(e){mt("loadSourceImage begin",e.name),Ge(),nt(!1),Ct(),a.sourceFile=e,a.sourceFileName=e.name,a.sourceImageInfo=null,a.sourceUploadedToWorker=!1,a.gpuStats=null,a.sourcePoints=[],a.lastFrameMs=0,Ie.disabled=!1,N.disabled=!0,De.disabled=!0,g("PNG selected, initializing GPU\u2026"),T();try{await X()}catch(t){wa(),g(t instanceof Error?t.message:String(t))}}async function X(){if(q()){Ge(),N.disabled=!0,g("Initializing WebGPU\u2026");try{Ct(),await cr(),Oa(),K(),A();let e=be(),t={demSourceMode:e,tessellation:Aa(),options:{minHeight:s(Wa,It),maxHeight:s(_a,Dt),demSourceMode:e,layerConfig:pr()}};a.sourceUploadedToWorker||(e==="stack4"?t.layerBlobs=a.layerSourceFiles.slice():t.blob=a.sourceFile);let r=await D("loadDEMImage",t);a.sourceUploadedToWorker=!0,a.buildCount++,a.sourceFileName=Ua(),a.sourceImageInfo=r.sourceImageInfo??a.sourceImageInfo,a.gpuStats=r.stats??a.gpuStats,a.sourcePoints=r.sourcePoints??a.sourcePoints,pe.disabled=!1,Ze.disabled=!1,N.disabled=!1,De.disabled=!1,Ie.disabled=!1,Ue(),nt(!0),g("GPU sim ready"),T(),Tt(!0)}catch(e){console.error("[WebGPU Erosion UI] initializeSimulation failed",e),nt(!1),g(e instanceof Error?e.message:String(e))}}}function P(){a.statsFrameHandle||(a.statsFrameHandle=requestAnimationFrame(()=>{a.statsFrameHandle=0,T()}))}function A(){let e=hr();a.workerReady&&x("setParams",{params:e})}async function Rt(){if(a.workerReady)try{let e=await D("getStatus");e.stats&&(a.gpuStats=e.stats),Array.isArray(e.sourcePoints)&&(a.sourcePoints=e.sourcePoints),typeof e.running=="boolean"&&(a.running=e.running),Number.isFinite(e.lastFrameMs)&&(a.lastFrameMs=e.lastFrameMs),H(),P()}catch(e){mt("refreshWorkerStatusLight failed",e)}}function Br(){return k.input.value==="spring_add"||k.input.value==="spring_erase"}function Er(){return{mode:k.input.value,radius:Math.max(1,s(wt,na)),amount:Math.max(.001,s(Pt,ia)),hardness:R(s(Mt,sa),.05,1)}}function Cr(e){let t=a.gpuStats;if(!t?.ready||!t.width||!t.height)return null;let r=u.getBoundingClientRect(),i=(e.clientX-r.left)/Math.max(r.width,1),n=(e.clientY-r.top)/Math.max(r.height,1);if(i<0||i>1||n<0||n>1)return null;let o=va(),l=t.width>1||t.height>1?2/Math.max(t.width-1,t.height-1,1):1,c=Ia(o.eye,o.target,[0,1,0]),E=ka(rt(50),Math.max(r.width/Math.max(r.height,1),1e-6),.01,32),v=lr(Fa(E,c));if(!v)return null;let M=i*2-1,p=1-n*2,m=Ft(v,[M,p,-1,1]),S=Ft(v,[M,p,1,1]),L=[m[0]/m[3],m[1]/m[3],m[2]/m[3]],w=[S[0]/S[3],S[1]/S[3],S[2]/S[3]],B=me(La(w,L)),C,F,G=0;if(Math.abs(B[1])>1e-5){let b=(G-L[1])/B[1];b>0&&(C=L[0]+B[0]*b,F=L[2]+B[2]*b)}if(!Number.isFinite(C)||!Number.isFinite(F)){let b=i*Math.max(t.width-1,0),ae=n*Math.max(t.height-1,0);return{x:R(b,0,Math.max(t.width-1,0)),y:R(ae,0,Math.max(t.height-1,0))}}let W=(t.width-1)*.5,I=(t.height-1)*.5,_=C/Math.max(l,1e-6)+W,z=I-F/Math.max(l,1e-6);return{x:R(_,0,Math.max(t.width-1,0)),y:R(z,0,Math.max(t.height-1,0))}}async function Za(e){let t=Er();if(t.mode==="none"||!a.workerReady||!a.gpuStats?.ready)return;let r=performance.now();if(r-a.lastPaintAt<20)return;a.lastPaintAt=r;let i=Cr(e);if(i){if(a.running&&Ge(),t.mode==="raise"||t.mode==="lower"){let n=await D("paintTerrainBrush",{brush:{x:i.x,y:i.y,radius:t.radius,amount:t.amount,hardness:t.hardness,subtract:t.mode==="lower"}});n.stats&&(a.gpuStats=n.stats),Array.isArray(n.sourcePoints)&&(a.sourcePoints=n.sourcePoints),g(`${t.mode==="raise"?"Raised":"Lowered"} terrain at ${i.x.toFixed(1)}, ${i.y.toFixed(1)}`)}else{let n=await D("paintSpringBrush",{brush:{x:i.x,y:i.y,radius:t.radius,strength:t.amount,hardness:t.hardness,erase:t.mode==="spring_erase"}});n.stats&&(a.gpuStats=n.stats),Array.isArray(n.sourcePoints)&&(a.sourcePoints=n.sourcePoints),g(`${t.mode==="spring_erase"?"Erased":"Painted"} spring at ${i.x.toFixed(1)}, ${i.y.toFixed(1)}`)}T()}}function Sa(){a.isPainting&&(a.isPainting=!1,Tt(!0))}function ja(e){let t={paper_balanced:{iterationsPerFrame:5,stepIterations:128,timeStep:.02,rainRate:.001,evaporationRate:.015,pipeArea:20,gravity:9.81,capacityScale:.82,suspensionRate:.32,depositionRate:1.22,softeningRate:2.8,maxErosionDepth:.09,thermalRate:.24,talusCoeff:.92,talusBias:.12,sourceStrength:.06,sourceRadius:4,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.05,edgeWaterFloor:0,renderMode:0},river_cut:{iterationsPerFrame:5,stepIterations:160,timeStep:.02,rainRate:.0015,evaporationRate:.012,pipeArea:22,gravity:9.81,capacityScale:.95,suspensionRate:.48,depositionRate:1.02,softeningRate:2.8,maxErosionDepth:.08,thermalRate:.22,talusCoeff:.88,talusBias:.11,sourceStrength:.09,sourceRadius:4,sourceEnabled:!0,sourceLayoutMode:1,randomSpringCount:4,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.05,edgeWaterFloor:0,renderMode:0},delta_depositor:{iterationsPerFrame:5,stepIterations:192,timeStep:.02,rainRate:.016,evaporationRate:.004,pipeArea:30,gravity:9.81,capacityScale:.38,suspensionRate:.12,depositionRate:2.2,softeningRate:1.4,maxErosionDepth:.04,thermalRate:.08,talusCoeff:.82,talusBias:.1,sourceStrength:0,sourceRadius:5,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:85,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.07,renderMode:0},meander_builder:{iterationsPerFrame:5,stepIterations:160,timeStep:.02,rainRate:.003,evaporationRate:.004,pipeArea:32,gravity:9.81,capacityScale:.3,suspensionRate:.1,depositionRate:2.45,softeningRate:1.1,maxErosionDepth:.035,thermalRate:.18,talusCoeff:.86,talusBias:.1,sourceStrength:.11,sourceRadius:5,sourceEnabled:!0,sourceLayoutMode:1,randomSpringCount:3,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.07,renderMode:0},thermal_heavy:{iterationsPerFrame:5,stepIterations:128,timeStep:.02,rainRate:0,evaporationRate:.008,pipeArea:8,gravity:9.81,capacityScale:.1,suspensionRate:.05,depositionRate:.85,softeningRate:1,maxErosionDepth:.06,thermalRate:1.1,talusCoeff:.55,talusBias:.05,sourceStrength:0,sourceRadius:4,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.05,renderMode:5},gentle_weathering:{iterationsPerFrame:5,stepIterations:96,timeStep:.02,rainRate:.004,evaporationRate:.015,pipeArea:12,gravity:9.81,capacityScale:.45,suspensionRate:.18,depositionRate:1,softeningRate:2,maxErosionDepth:.1,thermalRate:.12,talusCoeff:.8,talusBias:.1,sourceStrength:0,sourceRadius:4,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.05,edgeWaterFloor:0,renderMode:0},rapid_incision:{iterationsPerFrame:5,stepIterations:192,timeStep:.022,rainRate:.018,evaporationRate:.01,pipeArea:26,gravity:9.81,capacityScale:1.4,suspensionRate:.9,depositionRate:.75,softeningRate:6,maxErosionDepth:.1,thermalRate:.12,talusCoeff:.72,talusBias:.08,sourceStrength:0,sourceRadius:4,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.05,edgeWaterFloor:0,renderMode:0},flash_flood:{iterationsPerFrame:5,stepIterations:192,timeStep:.02,rainRate:.03,evaporationRate:.008,pipeArea:24,gravity:9.81,capacityScale:1.3,suspensionRate:.85,depositionRate:.8,softeningRate:5,maxErosionDepth:.08,thermalRate:.1,talusCoeff:.75,talusBias:.08,sourceStrength:0,sourceRadius:4,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:50,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.05,renderMode:0},badlands:{iterationsPerFrame:5,stepIterations:160,timeStep:.02,rainRate:.01,evaporationRate:.012,pipeArea:18,gravity:9.81,capacityScale:1,suspensionRate:.55,depositionRate:.95,softeningRate:4,maxErosionDepth:.09,thermalRate:.22,talusCoeff:.68,talusBias:.08,sourceStrength:0,sourceRadius:4,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:25,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.05,renderMode:0},canyon_carver:{iterationsPerFrame:5,stepIterations:224,timeStep:.022,rainRate:.014,evaporationRate:.01,pipeArea:28,gravity:9.81,capacityScale:1.55,suspensionRate:1,depositionRate:.7,softeningRate:6,maxErosionDepth:.08,thermalRate:.16,talusCoeff:.72,talusBias:.08,sourceStrength:.08,sourceRadius:3,sourceEnabled:!0,sourceLayoutMode:1,randomSpringCount:4,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.05,edgeWaterFloor:0,renderMode:0}},r=t[e]||t.paper_balanced;Z.input.value=String(r.iterationsPerFrame??se),gt.input.value=String(r.stepIterations??it),He.input.value=String(r.timeStep),ve.input.value=String(r.rainRate),Ve.input.value=String(r.evaporationRate),Ye.input.value=String(r.pipeArea),$e.input.value=String(r.gravity),Se.input.value=String(r.capacityScale),we.input.value=String(r.suspensionRate),Pe.input.value=String(r.depositionRate),Me.input.value=String(r.softeningRate),Be.input.value=String(r.maxErosionDepth),Ee.input.value=String(r.thermalRate),Ce.input.value=String(r.talusCoeff),Re.input.value=String(r.talusBias),ce.input.value=String(r.sourceStrength),Te.input.value=String(r.rainDuration??lt),Xe.input.value=String(r.pulse2Duration??dt),ue.input.value=String(r.sourceRadius),Fe.input.checked=(r.rainRate??0)>0,ke.input.checked=(r.thermalRate??0)>0,Ke.input.checked=!0,ie.input.checked=!!r.sourceEnabled,Ae.input.value=String(r.sourceLayoutMode??Ma),he.input.value=String(r.randomSpringCount??ut),vt.input.value=String(r.metersPerPixel??ct),qe.input.value=String(r.waterHeightScale??st),la.input.value=String(r.edgeWaterFloor??Zt),ge.input.value=String(r.renderMode),Ue(),A(),x("render"),Rt(),T()}async function Tt(e=!1){if(!a.workerReady||a.readbackPending)return;let t=performance.now();if(!(!e&&t-a.lastReadbackAt<500)){a.readbackPending=!0;try{let r=await D("readbackStats");a.gpuStats=r.stats??a.gpuStats,a.sourcePoints=r.sourcePoints??a.sourcePoints,a.sourceImageInfo=r.sourceImageInfo??a.sourceImageInfo,Number.isFinite(r.lastFrameMs)&&(a.lastFrameMs=r.lastFrameMs),a.lastReadbackAt=performance.now()}catch(r){mt("readback failed",r)}finally{a.readbackPending=!1}}}function Rr(){!a.workerReady||a.running||(a.running=!0,pe.textContent="Pause",A(),x("startLoop",{iterationsPerFrame:Math.max(1,Math.floor(s(Z,se)))}),Tt(!0),g("Running"))}function Ge(){a.running&&(a.running=!1,pe.textContent="Run",x("stopLoop"),g("Paused"))}function wa(){Ge(),document.exitPointerLock?.(),x("clear"),a.sourceImageInfo=null,a.sourceFile=null,a.layerSourceFiles=[null,null,null,null],a.sourceUploadedToWorker=!1,a.buildCount=0,a.lastFrameMs=0,a.readbackPending=!1,a.lastReadbackAt=0,a.gpuStats=null,a.sourcePoints=[],a.sourceFileName="",a.cameraPosX=Ba,a.cameraPosY=Ea,a.cameraPosZ=Ca,O.value="";for(let t of ee)t.filePickerLabel.input.value="";pe.disabled=!0,Ze.disabled=!0,N.disabled=!0,De.disabled=!0,Ie.disabled=!0,g(),T(),H();let e=y.getContext("2d");e&&(e.fillStyle="#0d0d0d",e.fillRect(0,0,y.width,y.height),e.fillStyle="rgba(255,255,255,0.55)",e.font="16px system-ui, sans-serif",e.fillText("Load a DEM PNG, packed RGBA DEM, or 4-layer grayscale stack to initialize the worker-owned WebGPU erosion sim.",24,36))}function H(){let e=h.getContext("2d");if(!e||(e.clearRect(0,0,h.width,h.height),h.style.visibility==="hidden"))return;let t=a.gpuStats,r=a.sourcePoints||[];if(!t?.ready||!t.width||!t.height||r.length===0)return;let i=va(),n=Ia(i.eye,i.target,[0,1,0]),o=ka(rt(50),Math.max(h.width/Math.max(h.height,1),1e-6),.01,32),l=Fa(o,n),c=t.width>1||t.height>1?2/Math.max(t.width-1,t.height-1,1):1,E=(t.width-1)*.5,v=(t.height-1)*.5,M=Math.max(2,Math.min(6,Math.min(h.width,h.height)/Math.max(t.width,t.height,1)*.85));for(let p of r){let m=(p.x-E)*c,S=(v-p.y)*c,w=Ft(l,[m,0,S,1]);if(!w[3]||w[3]<=0)continue;let B=w[0]/w[3],C=w[1]/w[3];if(Math.abs(B)>1.2||Math.abs(C)>1.2)continue;let F=(B*.5+.5)*h.width,G=(1-(C*.5+.5))*h.height,W=!!p.painted,I=p.active!==!1;if(!W&&!Br())continue;if(W){let z=R(Number(p.strength)||0,0,1),b=M*(.8+.9*z);e.fillStyle=I?"rgba(176, 96, 255, 0.98)":"rgba(120, 96, 150, 0.85)",e.fillRect(Math.round(F-b),Math.round(G-b),Math.max(1,Math.round(b*2)),Math.max(1,Math.round(b*2)));continue}let _=Math.max(5,(p.radius||2)*Math.min(h.width,h.height)/Math.max(t.width,t.height,1)*1.2);e.beginPath(),e.arc(F,G,_,0,Math.PI*2),e.fillStyle=I?"rgba(60, 220, 255, 0.30)":"rgba(90, 110, 120, 0.18)",e.fill(),e.lineWidth=2,e.strokeStyle=I?"rgba(138, 240, 255, 0.95)":"rgba(140, 165, 175, 0.55)",e.stroke()}}function g(e=""){let t=[];e&&t.push(e),t.push(a.sourceFile?`image: ${a.sourceFileName||"loaded"}`:"image: none"),t.push(`webgpu: ${a.workerReady?"ready":"idle"}`),a.running&&t.push("sim: running"),je.textContent=t.join("  |  ")}function T(){if(!q()&&!a.sourceImageInfo){j.textContent="Load a DEM PNG or layered DEM stack to begin.",H();return}let e=a.sourceImageInfo?.width||a.gpuStats?.width||0,t=a.sourceImageInfo?.height||a.gpuStats?.height||0,r=a.gpuStats?.width||e,i=a.gpuStats?.height||t,n=a.gpuStats??{width:0,height:0,cellCount:0,ready:!1,iterationCount:0},o=a.lastFrameMs>0?1e3/Math.max(a.lastFrameMs,1e-6):0,l=[`fps: ${Number.isFinite(o)&&o>0?o.toFixed(1):"n/a"}`,`last frame: ${a.lastFrameMs.toFixed(3)} ms`,`erosion cpu: ${Number.isFinite(n.lastCpuErosionPassMs)?n.lastCpuErosionPassMs.toFixed(3):Number.isFinite(n.lastErosionPassMs)?n.lastErosionPassMs.toFixed(3):"n/a"} ms`,`erosion gpu: ${Number.isFinite(n.lastGpuErosionPassMs)?n.lastGpuErosionPassMs.toFixed(3):"n/a"} ms`,`render cpu: ${Number.isFinite(n.lastCpuRenderPassMs)?n.lastCpuRenderPassMs.toFixed(3):Number.isFinite(n.lastRenderPassMs)?n.lastRenderPassMs.toFixed(3):"n/a"} ms`,`render gpu: ${Number.isFinite(n.lastGpuRenderPassMs)?n.lastGpuRenderPassMs.toFixed(3):"n/a"} ms`,`erosion cpu avg: ${Number.isFinite(n.avgCpuErosionPassMs)?n.avgCpuErosionPassMs.toFixed(3):Number.isFinite(n.avgErosionPassMs)?n.avgErosionPassMs.toFixed(3):"n/a"} ms`,`erosion gpu avg: ${Number.isFinite(n.avgGpuErosionPassMs)?n.avgGpuErosionPassMs.toFixed(3):"n/a"} ms`,`render cpu avg: ${Number.isFinite(n.avgCpuRenderPassMs)?n.avgCpuRenderPassMs.toFixed(3):Number.isFinite(n.avgRenderPassMs)?n.avgRenderPassMs.toFixed(3):"n/a"} ms`,`render gpu avg: ${Number.isFinite(n.avgGpuRenderPassMs)?n.avgGpuRenderPassMs.toFixed(3):"n/a"} ms`,"",`file: ${a.sourceFileName||"(loaded image)"}`,`DEM source: ${be()}`,`layered DEM active: ${n.layeredDemEnabled?"yes":"no"}`,`active layers: ${Number.isFinite(n.activeLayerCount)?n.activeLayerCount:ee.filter(c=>c.enableLabel.input.checked).length}`,`image size: ${e} x ${t}`,`sim tess: ${Aa()}x`,`sim grid: ${n.width||r} x ${n.height||i}`,`cells: ${n.cellCount||r*i}`,`gpu ready: ${n.ready?"yes":"no"}`,`build count: ${a.buildCount}`,`iterations: ${n.iterationCount||0}`,`sim time: ${Number.isFinite(n.simTime)?n.simTime.toFixed(2):"0.00"} s`,`rain active: ${n.rainActive?"yes":"no"}`,`precipitation: ${Fe.input.checked?"on":"off"}`,`spring toggle: ${ie.input.checked?"on":"off"}`,`thermal erosion: ${ke.input.checked?"on":"off"}`,`spring mode: ${br()}`,`painted springs: ${(a.sourcePoints||[]).filter(c=>c.painted).length}`,`random spring count: ${Math.max(1,Math.floor(s(he,ut)))}`,`spring seed: ${Math.floor(s(Le,ra))}`,`spring centers: ${(a.sourcePoints||[]).slice(0,6).map(c=>`${c.painted?"p:":"r:"}${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" | ")||"n/a"}`,`meters / pixel: ${s(vt,ct).toFixed(0)}`,`running: ${a.running?"yes":"no"}`,`pointer look: ${a.pointerLookActive?"locked":"off"}`,`camera pos: ${a.cameraPosX.toFixed(2)}, ${a.cameraPosY.toFixed(2)}, ${a.cameraPosZ.toFixed(2)}`,`paint mode: ${k.input.options[k.input.selectedIndex]?.textContent||"off"}`,`brush radius: ${s(wt,na).toFixed(1)}`,`brush amount: ${s(Pt,ia).toFixed(3)}`,`brush hardness: ${s(Mt,sa).toFixed(2)}`,`readback pending: ${a.readbackPending?"yes":"no"}`,"",`terrain range: ${at(n.terrainRange)}`,`water range: ${at(n.waterRange)}`,`sediment range: ${at(n.sedimentRange)}`,`history range: ${at(n.historyRange)}`,`total water: ${Number.isFinite(n.totalWater)?n.totalWater.toFixed(4):"n/a"}`,`total sediment: ${Number.isFinite(n.totalSediment)?n.totalSediment.toFixed(4):"n/a"}`,`avg erodability: ${Number.isFinite(n.averageHardness)?n.averageHardness.toFixed(4):"n/a"}`,"",`\u0394t: ${s(He,Ot).toFixed(3)}`,`Kr rain: ${s(ve,Ut).toFixed(4)}`,`Ke evap: ${s(Ve,Gt).toFixed(3)}`,`A pipe: ${s(Ye,Wt).toFixed(2)}`,`hydraulic pipes: ${St.input.checked?"8-way":"4-way"}`,`g gravity: ${s($e,_t).toFixed(2)}`,`Kc capacity: ${s(Se,zt).toFixed(3)}`,`Ks suspend: ${s(we,Nt).toFixed(3)}`,`Kd deposit: ${s(Pe,Ht).toFixed(3)}`,`Kh soften: ${s(Me,Vt).toFixed(3)}`,`depth cap: ${s(Be,Yt).toFixed(3)}`,`Kt thermal: ${s(Ee,$t).toFixed(3)}`,`Ka talus: ${s(Ce,qt).toFixed(3)}`,`Ki talus bias: ${s(Re,Xt).toFixed(3)}`,`render h: ${s(yt,Kt).toFixed(3)}`,`water h: ${s(qe,st).toFixed(3)}`,`water \u03B1: ${s(xt,jt).toFixed(3)}`,`sediment tint: ${s(bt,Jt).toFixed(3)}`,`springs: ${ie.input.checked?"on":"off"}`,`rain duration s: ${s(Te,lt).toFixed(2)}`,`pulse 2 s: ${s(Xe,dt).toFixed(2)}`,`source center: ${s(da,Qt).toFixed(0)}%, ${s(ua,ea).toFixed(0)}%`,`source radius: ${s(ue,ta).toFixed(1)}`,`source strength: ${s(ce,aa).toFixed(3)}`,`view mode: ${ge.input.options[ge.input.selectedIndex]?.textContent||"shaded"}`,`iters/frame: ${Math.max(1,Math.floor(s(Z,se)))}`,`step iters: ${Math.max(1,Math.floor(s(gt,it)))}`];j.textContent=l.join(`
`),g(),H()}wa();})();
