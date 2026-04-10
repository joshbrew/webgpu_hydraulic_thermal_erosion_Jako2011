(()=>{var Wr=`(()=>{var STATE_FLOATS_PER_CELL=8;var VEC4_BYTES_PER_CELL=16;var WORKGROUP_SIZE_X=8;var WORKGROUP_SIZE_Y=8;function clamp(value,min,max){return value<min?min:value>max?max:value}function alignTo(value,alignment){return Math.ceil(value/alignment)*alignment}function degToRad(value){return value*Math.PI/180}function normalizeVec3(v){const len=Math.hypot(v[0],v[1],v[2])||1;return[v[0]/len,v[1]/len,v[2]/len]}function subtractVec3(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]]}function crossVec3(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}function multiplyMat4(a,b){const out=new Float32Array(16);for(let c=0;c<4;c++){for(let r=0;r<4;r++){out[c*4+r]=a[0*4+r]*b[c*4+0]+a[1*4+r]*b[c*4+1]+a[2*4+r]*b[c*4+2]+a[3*4+r]*b[c*4+3]}}return out}function perspectiveMat4(fovy,aspect,near,far){const f=1/Math.tan(fovy*.5);const nf=1/(near-far);const out=new Float32Array(16);out[0]=f/Math.max(aspect,1e-6);out[5]=f;out[10]=(far+near)*nf;out[11]=-1;out[14]=2*far*near*nf;return out}function lookAtMat4(eye,target,up){const z=normalizeVec3(subtractVec3(eye,target));const x=normalizeVec3(crossVec3(up,z));const y=crossVec3(z,x);const out=new Float32Array(16);out[0]=x[0];out[1]=y[0];out[2]=z[0];out[3]=0;out[4]=x[1];out[5]=y[1];out[6]=z[1];out[7]=0;out[8]=x[2];out[9]=y[2];out[10]=z[2];out[11]=0;out[12]=-(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]);out[13]=-(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]);out[14]=-(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]);out[15]=1;return out}var WebGPUTerrainErosion=class{constructor(options={}){this.canvas=options.canvas??null;this.adapter=null;this.device=null;this.context=null;this.presentationFormat=null;this.width=0;this.height=0;this.cellCount=0;this.stateByteLength=0;this.vec4ByteLength=0;this.iterationCount=0;this.allocatedStateByteLength=0;this.allocatedVec4ByteLength=0;this.dispatchX=0;this.dispatchY=0;this.drawCount=0;this.stateBuffers=[];this.fluxBuffer=null;this.velocityBuffer=null;this.thermalPipeBufferA=null;this.thermalPipeBufferB=null;this.readbackBuffer=null;this.customSpringBuffer=null;this.paintedSpringMap=new Float32Array(0);this.allocatedPaintedSpringByteLength=0;this.renderUniformValues=new Float32Array(40);this.renderUniformBufferTerrain=null;this.renderUniformBufferWater=null;this.depthTexture=null;this.depthTextureView=null;this.depthTextureSize={width:0,height:0};this.sceneColorTexture=null;this.sceneColorView=null;this.sceneColorSize={width:0,height:0};this.sceneSampler=null;this.dummySceneTexture=null;this.dummySceneView=null;this.paramValues=new Float32Array(36);this.paramBuffer=null;this.stepParamBuffer=null;this.stepParamStride=256;this.stepParamCapacity=0;this.stepParamValuesCpu=null;this.stepParamValuesCpuCapacity=0;this.stepParamTemplateDirty=true;this.stepParamPreparedIterations=0;this.computeDynamicOffset=new Uint32Array(1);this.terrainRenderBundle=null;this.finalRenderBundle=null;this.terrainRenderBundleList=[];this.finalRenderBundleList=[];this.computeBindGroupLayout=null;this.renderBindGroupLayout=null;this.compositeBindGroupLayout=null;this.computeBindGroup01=null;this.computeBindGroup10=null;this.renderBindGroupTerrain=null;this.renderBindGroupWater=null;this.compositeBindGroup=null;this.fluxPipeline=null;this.flowPipeline=null;this.erosionPipeline=null;this.transportPipeline=null;this.thermalOutflowPipeline=null;this.thermalApplyPipeline=null;this.renderPipeline=null;this.waterRenderPipeline=null;this.compositePipeline=null;this.initialized=false;this.ready=false;this.latestReadbackStats=null;this.simulationParams={cellSize:1,timeStep:.02,rainRate:.001,evaporationRate:.015,pipeArea:20,gravity:9.81,capacityScale:1,suspensionRate:.5,depositionRate:.92,softeningRate:5,maxErosionDepth:.12,thermalRate:.45,talusSlopeCoeff:.8,talusSlopeBias:.1,renderHeightScale:.08,waterOpacity:.28,sedimentTint:.35,hardnessBase:.16,hardnessVariation:.03,sourceCenterX:.5,sourceCenterY:.5,sourceRadius:4,sourceStrength:.06,sourceEnabled:false,rainDuration:0,pulse2Duration:0,sourceLayoutMode:0,randomSpringCount:4,sourceSeed:1,metersPerPixel:100,sourceTimeOffset:0,historyDecay:.9997,edgeDrainStrength:.08,renderMode:0,cameraAzimuthDeg:45,cameraElevationDeg:42,cameraDistance:2.9,cameraPosX:1.5238999619464006,cameraPosY:1.9404787584406888,cameraPosZ:1.5238999619464006,waterHeightScale:.12,thermalVisualizationScale:160,hydraulicErosionEnabled:true};this.setSimulationParams(options.simulationParams??{})}async initialize(canvas=this.canvas){if(!canvas){throw new Error("A canvas is required to initialize WebGPU erosion.")}this.canvas=canvas;this.#syncCanvasPixelSize();if(!navigator.gpu){throw new Error("WebGPU is not available in this browser.")}this.adapter=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!this.adapter){throw new Error("Failed to acquire a WebGPU adapter.")}this.device=await this.adapter.requestDevice();this.stepParamStride=Math.max(256,this.device.limits?.minUniformBufferOffsetAlignment||256);this.context=canvas.getContext("webgpu");if(!this.context){throw new Error("Failed to acquire a WebGPU canvas context.")}this.presentationFormat=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:this.presentationFormat,alphaMode:"opaque"});this.paramBuffer=this.device.createBuffer({size:alignTo(this.paramValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-erosion-params"});this.renderUniformBufferTerrain=this.device.createBuffer({size:alignTo(this.renderUniformValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-render-params-terrain"});this.renderUniformBufferWater=this.device.createBuffer({size:alignTo(this.renderUniformValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-render-params-water"});this.sceneSampler=this.device.createSampler({addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge",magFilter:"linear",minFilter:"linear",mipmapFilter:"linear"});this.#ensureCustomSpringBuffer();this.#ensureDummySceneTexture();this.#createPipelines();this.#writeParams();this.initialized=true}destroy(){for(const buffer of this.stateBuffers){buffer?.destroy?.()}this.stateBuffers=[];this.fluxBuffer?.destroy?.();this.velocityBuffer?.destroy?.();this.thermalPipeBufferA?.destroy?.();this.thermalPipeBufferB?.destroy?.();this.readbackBuffer?.destroy?.();this.customSpringBuffer?.destroy?.();this.depthTexture?.destroy?.();this.sceneColorTexture?.destroy?.();this.dummySceneTexture?.destroy?.();this.renderUniformBufferTerrain?.destroy?.();this.renderUniformBufferWater?.destroy?.();this.paramBuffer?.destroy?.();this.stepParamBuffer?.destroy?.();this.fluxBuffer=null;this.velocityBuffer=null;this.thermalPipeBufferA=null;this.thermalPipeBufferB=null;this.readbackBuffer=null;this.customSpringBuffer=null;this.paintedSpringMap=new Float32Array(0);this.allocatedPaintedSpringByteLength=0;this.depthTexture=null;this.depthTextureView=null;this.depthTextureSize={width:0,height:0};this.sceneColorTexture=null;this.sceneColorView=null;this.sceneColorSize={width:0,height:0};this.dummySceneTexture=null;this.dummySceneView=null;this.renderUniformBufferTerrain=null;this.renderUniformBufferWater=null;this.paramBuffer=null;this.stepParamBuffer=null;this.stepParamCapacity=0;this.stepParamValuesCpu=null;this.stepParamValuesCpuCapacity=0;this.#invalidateRenderBundles();this.computeBindGroup01=null;this.computeBindGroup10=null;this.renderBindGroupTerrain=null;this.renderBindGroupWater=null;this.compositeBindGroup=null;this.ready=false;this.width=0;this.height=0;this.cellCount=0;this.iterationCount=0;this.dispatchX=0;this.dispatchY=0;this.drawCount=0;this.latestReadbackStats=null;this.paintedSpringMap=new Float32Array(0)}setSimulationParams(next={}){Object.assign(this.simulationParams,next);if(this.paramBuffer&&this.device){this.#writeParams()}if((this.renderUniformBufferTerrain||this.renderUniformBufferWater)&&this.device){this.#writeRenderParams()}}resetRainTimer(){this.simulationParams.sourceTimeOffset=this.iterationCount*this.simulationParams.timeStep;if(this.paramBuffer&&this.device){this.#writeParams()}}restartSources(){this.resetRainTimer()}async applyTerrainBrush(brush={}){if(!this.ready||this.width<=0||this.height<=0){throw new Error("Terrain painting is only available after the simulation is initialized.")}const centerX=clamp(Number(brush.x)||0,0,Math.max(this.width-1,0));const centerY=clamp(Number(brush.y)||0,0,Math.max(this.height-1,0));const radius=Math.max(.5,Number(brush.radius)||1);const amount=Math.max(1e-4,Number(brush.amount)||.01);const hardness=clamp(Number(brush.hardness)||.5,.01,1);const subtract=!!brush.subtract;const data=await this.#readbackStateData();const minX=Math.max(0,Math.floor(centerX-radius-1));const maxX=Math.min(this.width-1,Math.ceil(centerX+radius+1));const minY=Math.max(0,Math.floor(centerY-radius-1));const maxY=Math.min(this.height-1,Math.ceil(centerY+radius+1));const falloffPower=2.65-hardness*2.2;const direction=subtract?-1:1;for(let y=minY;y<=maxY;y++){for(let x=minX;x<=maxX;x++){const dx=x-centerX;const dy=y-centerY;const dist=Math.hypot(dx,dy);if(dist>radius)continue;const t=1-dist/radius;const falloff=Math.pow(clamp(t,0,1),falloffPower);const i=y*this.width+x;const base=i*STATE_FLOATS_PER_CELL;if(data[base+4]<.5)continue;const terrain=clamp(data[base]+direction*amount*falloff,0,2);data[base]=terrain}}this.device.queue.writeBuffer(this.stateBuffers[0],0,data);this.device.queue.writeBuffer(this.stateBuffers[1],0,data);this.latestReadbackStats=null;this.render();return this.getStats()}paintSpringBrush(brush={}){if(!this.ready||this.width<=0||this.height<=0){throw new Error("Spring painting is only available after the simulation is initialized.")}const centerX=clamp(Number(brush.x)||0,0,Math.max(this.width-1,0));const centerY=clamp(Number(brush.y)||0,0,Math.max(this.height-1,0));const radius=Math.max(.5,Number(brush.radius)||1);const amount=Math.max(0,Number(brush.strength)||0);const hardness=clamp(Number(brush.hardness)||.5,.01,1);const erase=!!brush.erase;if(!(this.paintedSpringMap instanceof Float32Array)||this.paintedSpringMap.length!==this.cellCount){this.paintedSpringMap=new Float32Array(this.cellCount)}const minX=Math.max(0,Math.floor(centerX-radius-1));const maxX=Math.min(this.width-1,Math.ceil(centerX+radius+1));const minY=Math.max(0,Math.floor(centerY-radius-1));const maxY=Math.min(this.height-1,Math.ceil(centerY+radius+1));const falloffPower=2.65-hardness*2.2;for(let y=minY;y<=maxY;y++){for(let x=minX;x<=maxX;x++){const dx=x-centerX;const dy=y-centerY;const dist=Math.hypot(dx,dy);if(dist>radius)continue;const t=1-dist/radius;const falloff=Math.pow(clamp(t,0,1),falloffPower);const i=y*this.width+x;if(erase){this.paintedSpringMap[i]=Math.max(0,this.paintedSpringMap[i]-Math.max(amount,1)*falloff)}else{this.paintedSpringMap[i]=clamp(this.paintedSpringMap[i]+amount*falloff,0,1)}}}this.#writeCustomSpringBuffer();this.#writeParams();this.render();return this.getSourcePoints()}clearPaintedSprings(){if(!(this.paintedSpringMap instanceof Float32Array)||this.paintedSpringMap.length!==this.cellCount){this.paintedSpringMap=new Float32Array(this.cellCount)}else{this.paintedSpringMap.fill(0)}this.#writeCustomSpringBuffer();this.#writeParams();this.render();return this.getSourcePoints()}getSourcePoints(){if(this.width<=0||this.height<=0){return[]}const layoutMode=this.simulationParams.sourceLayoutMode|0;const springsEnabled=!!this.simulationParams.sourceEnabled;if(layoutMode===0){const points2=[];const values=this.paintedSpringMap;const threshold=.0025;let activeCount=0;for(let i=0;i<values.length;i++){if(values[i]>threshold)activeCount++}const maxPoints=8192;const stride=Math.max(1,Math.ceil(activeCount/maxPoints));let seen=0;for(let y=0;y<this.height;y++){for(let x=0;x<this.width;x++){const i=y*this.width+x;const strength=values[i];if(strength<=threshold)continue;if(seen++%stride!==0)continue;points2.push({x,y,radius:.7,strength,painted:true,active:springsEnabled})}}return points2}if(!springsEnabled){return[]}const points=[];const count=Math.min(16,Math.max(1,this.simulationParams.randomSpringCount|0));for(let i=0;i<count;i++){const seed=(this.simulationParams.sourceSeed??1)*37+i*17+1;const x=Math.sin(seed*127.1+11.7)*43758.5453;const y=Math.sin(seed*311.7+73.1)*24634.6345;points.push({x:(x-Math.floor(x))*Math.max(this.width-1,0),y:(y-Math.floor(y))*Math.max(this.height-1,0),radius:this.simulationParams.sourceRadius,strength:this.simulationParams.sourceStrength,painted:false,active:true})}return points}async setDEM(raster,options={}){if(!this.initialized){await this.initialize(this.canvas)}if(!raster||!(raster.values instanceof Float32Array)){throw new Error("setDEM expects a raster with Float32Array values.")}const width=raster.width|0;const height=raster.height|0;if(width<=0||height<=0){throw new Error("Raster width and height must be positive.")}if(raster.values.length!==width*height){throw new Error("Raster value count does not match width * height.")}const minHeight=Number.isFinite(options.minHeight)?options.minHeight:0;const maxHeight=Number.isFinite(options.maxHeight)?options.maxHeight:1;const range=Math.max(1e-6,maxHeight-minHeight);const mask=raster.mask instanceof Uint8Array&&raster.mask.length===width*height?raster.mask:null;this.width=width;this.height=height;this.cellCount=width*height;this.stateByteLength=this.cellCount*STATE_FLOATS_PER_CELL*4;this.vec4ByteLength=this.cellCount*VEC4_BYTES_PER_CELL;this.iterationCount=0;this.dispatchX=Math.ceil(this.width/WORKGROUP_SIZE_X);this.dispatchY=Math.ceil(this.height/WORKGROUP_SIZE_Y);this.drawCount=Math.max(1,(this.width-1)*(this.height-1));this.#invalidateRenderBundles();this.#ensureDepthTexture();this.#ensureSceneTargets();this.#ensureBuffers();this.#writeParams();const initialState=new Float32Array(this.cellCount*STATE_FLOATS_PER_CELL);const hardnessBase=this.simulationParams.hardnessBase;const hardnessVariation=this.simulationParams.hardnessVariation;for(let i=0;i<this.cellCount;i++){const base=i*STATE_FLOATS_PER_CELL;const valid=mask?mask[i]?1:0:1;const terrain=minHeight+raster.values[i]*range;const normalizedTerrain=clamp(raster.values[i],0,1);const noise=Math.sin(i*12.9898+width*.031+height*.017)*43758.5453;const frac=noise-Math.floor(noise);const noiseTerm=(frac*2-1)*hardnessVariation*.25;const hardness=clamp(hardnessBase+(1-normalizedTerrain)*hardnessVariation+noiseTerm,.02,1);initialState[base]=terrain;initialState[base+1]=0;initialState[base+2]=0;initialState[base+3]=hardness;initialState[base+4]=valid;initialState[base+5]=0;initialState[base+6]=0;initialState[base+7]=0}const zeros=new Float32Array(this.cellCount*4);this.device.queue.writeBuffer(this.stateBuffers[0],0,initialState);this.device.queue.writeBuffer(this.stateBuffers[1],0,initialState);this.device.queue.writeBuffer(this.fluxBuffer,0,zeros);this.device.queue.writeBuffer(this.velocityBuffer,0,zeros);this.device.queue.writeBuffer(this.thermalPipeBufferA,0,zeros);this.device.queue.writeBuffer(this.thermalPipeBufferB,0,zeros);this.ready=true;this.latestReadbackStats=null;this.paintedSpringMap=new Float32Array(this.cellCount);this.#writeCustomSpringBuffer();this.#writeRenderParams();this.#ensureDepthTexture();this.#ensureSceneTargets();this.render()}resize(){if(!this.context||!this.device||!this.canvas)return;this.#syncCanvasPixelSize();this.context.configure({device:this.device,format:this.presentationFormat,alphaMode:"opaque"});this.#ensureDepthTexture();this.#ensureSceneTargets();this.#writeRenderParams()}step(iterations=1){if(!this.ready)return;const encoder=this.device.createCommandEncoder({label:"paper-erosion-step"});this.#encodeStepCommands(encoder,Math.max(1,iterations|0));this.device.queue.submit([encoder.finish()])}stepAndRender(iterations=1){if(!this.ready)return;const canRender=!!(this.context&&this.width>=2&&this.height>=2);const encoder=this.device.createCommandEncoder({label:canRender?"paper-erosion-step-render":"paper-erosion-step"});this.#encodeStepCommands(encoder,Math.max(1,iterations|0));if(canRender){this.#ensureDepthTexture();this.#ensureSceneTargets();this.#writeRenderParams();const view=this.context.getCurrentTexture().createView();this.#encodeRenderCommands(encoder,view)}this.device.queue.submit([encoder.finish()])}render(){if(!this.ready||!this.context||this.width<2||this.height<2)return;this.#ensureDepthTexture();this.#ensureSceneTargets();this.#writeRenderParams();const view=this.context.getCurrentTexture().createView();const encoder=this.device.createCommandEncoder({label:"paper-erosion-render"});this.#encodeRenderCommands(encoder,view);this.device.queue.submit([encoder.finish()])}async readbackStats(){if(!this.ready||!this.readbackBuffer){return this.getStats()}const data=await this.#readbackStateData();let validCells=0;let minTerrain=Infinity;let maxTerrain=-Infinity;let minWater=Infinity;let maxWater=-Infinity;let minSediment=Infinity;let maxSediment=-Infinity;let totalWater=0;let totalSediment=0;let hardnessSum=0;let minHistory=Infinity;let maxHistory=-Infinity;for(let i=0;i<this.cellCount;i++){const base=i*STATE_FLOATS_PER_CELL;const mask=data[base+4];if(mask<.5)continue;validCells++;const terrain=data[base];const water=data[base+1];const sediment=data[base+2];const hardness=data[base+3];const history=data[base+7];if(terrain<minTerrain)minTerrain=terrain;if(terrain>maxTerrain)maxTerrain=terrain;if(water<minWater)minWater=water;if(water>maxWater)maxWater=water;if(sediment<minSediment)minSediment=sediment;if(sediment>maxSediment)maxSediment=sediment;totalWater+=water;totalSediment+=sediment;hardnessSum+=hardness;if(history<minHistory)minHistory=history;if(history>maxHistory)maxHistory=history}this.latestReadbackStats={validCells,terrainRange:Number.isFinite(minTerrain)?{min:minTerrain,max:maxTerrain}:null,waterRange:Number.isFinite(minWater)?{min:minWater,max:maxWater}:null,sedimentRange:Number.isFinite(minSediment)?{min:minSediment,max:maxSediment}:null,historyRange:Number.isFinite(minHistory)?{min:minHistory,max:maxHistory}:null,totalWater,totalSediment,averageHardness:validCells>0?hardnessSum/validCells:0};return this.getStats()}async exportTerrainPng(){if(!this.ready||!this.readbackBuffer||this.width<=0||this.height<=0){throw new Error("Terrain export is only available after the simulation is initialized.")}if(typeof OffscreenCanvas!=="function"){throw new Error("OffscreenCanvas is not available for terrain export.")}const data=await this.#readbackStateData();let minTerrain=Infinity;let maxTerrain=-Infinity;for(let i=0;i<this.cellCount;i++){const base=i*STATE_FLOATS_PER_CELL;if(data[base+4]<.5)continue;const terrain=data[base];if(terrain<minTerrain)minTerrain=terrain;if(terrain>maxTerrain)maxTerrain=terrain}if(!Number.isFinite(minTerrain)||!Number.isFinite(maxTerrain)){minTerrain=0;maxTerrain=1}const range=Math.max(maxTerrain-minTerrain,1e-6);const rgba=new Uint8ClampedArray(this.width*this.height*4);for(let y=0;y<this.height;y++){const outY=this.height-1-y;for(let x=0;x<this.width;x++){const srcIndex=y*this.width+x;const base=srcIndex*STATE_FLOATS_PER_CELL;const dstIndex=(outY*this.width+x)*4;const mask=data[base+4]>=.5;const terrain=data[base];const normalized=mask?clamp((terrain-minTerrain)/range,0,1):0;const value=Math.max(0,Math.min(255,Math.round(normalized*255)));rgba[dstIndex]=value;rgba[dstIndex+1]=value;rgba[dstIndex+2]=value;rgba[dstIndex+3]=mask?255:0}}const canvas=new OffscreenCanvas(this.width,this.height);const ctx=canvas.getContext("2d");if(!ctx){throw new Error("Failed to create a terrain export canvas.")}ctx.putImageData(new ImageData(rgba,this.width,this.height),0,0);const blob=await canvas.convertToBlob({type:"image/png"});const arrayBuffer=await blob.arrayBuffer();return{width:this.width,height:this.height,minTerrain,maxTerrain,data:arrayBuffer}}getStats(){const simTime=this.iterationCount*this.simulationParams.timeStep;const rainDuration=Math.max(0,this.simulationParams.rainDuration??0);const rainActive=rainDuration<=0?true:simTime<rainDuration;return{width:this.width,height:this.height,cellCount:this.cellCount,ready:this.ready,iterationCount:this.iterationCount,simTime,rainActive,rainDuration,...this.latestReadbackStats??{}}}#syncCanvasPixelSize(){if(!this.canvas)return;if(typeof this.canvas.getBoundingClientRect!=="function"){const pixelWidth2=Math.max(1,Math.round(this.canvas.width||1));const pixelHeight2=Math.max(1,Math.round(this.canvas.height||1));if(this.canvas.width!==pixelWidth2)this.canvas.width=pixelWidth2;if(this.canvas.height!==pixelHeight2)this.canvas.height=pixelHeight2;return}const rect=this.canvas.getBoundingClientRect();const cssWidth=Math.max(1,Math.round(rect.width||this.canvas.width||1));const cssHeight=Math.max(1,Math.round(rect.height||this.canvas.height||1));const dpr=Math.max(1,Math.min(globalThis.devicePixelRatio||1,2));const pixelWidth=Math.max(1,Math.round(cssWidth*dpr));const pixelHeight=Math.max(1,Math.round(cssHeight*dpr));if(this.canvas.width!==pixelWidth)this.canvas.width=pixelWidth;if(this.canvas.height!==pixelHeight)this.canvas.height=pixelHeight}#ensureDepthTexture(){if(!this.device||!this.canvas)return;const width=Math.max(1,this.canvas.width||this.canvas.clientWidth||1);const height=Math.max(1,this.canvas.height||this.canvas.clientHeight||1);if(this.depthTexture&&this.depthTextureSize.width===width&&this.depthTextureSize.height===height){return}this.depthTexture?.destroy?.();this.depthTexture=this.device.createTexture({size:{width,height},format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT,label:"paper-depth"});this.depthTextureView=this.depthTexture.createView();this.depthTextureSize={width,height}}#ensureSceneTargets(){if(!this.device||!this.canvas)return;const width=Math.max(1,this.canvas.width||this.canvas.clientWidth||1);const height=Math.max(1,this.canvas.height||this.canvas.clientHeight||1);if(this.sceneColorTexture&&this.sceneColorSize.width===width&&this.sceneColorSize.height===height){return}this.sceneColorTexture?.destroy?.();this.sceneColorTexture=this.device.createTexture({size:{width,height},format:this.presentationFormat,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,label:"paper-scene-color"});this.sceneColorView=this.sceneColorTexture.createView();this.sceneColorSize={width,height};if(this.stateBuffers.length===2&&this.sceneSampler){this.#rebuildBindGroups()}}async#readbackStateData(){const encoder=this.device.createCommandEncoder({label:"paper-erosion-readback"});encoder.copyBufferToBuffer(this.stateBuffers[0],0,this.readbackBuffer,0,this.stateByteLength);this.device.queue.submit([encoder.finish()]);await this.readbackBuffer.mapAsync(GPUMapMode.READ);const mapped=this.readbackBuffer.getMappedRange();const copy=mapped.slice(0);this.readbackBuffer.unmap();return new Float32Array(copy)}#ensureCustomSpringBuffer(){if(!this.device||this.customSpringBuffer)return;const size=Math.max(4,this.cellCount*4);this.customSpringBuffer=this.device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-painted-springs"});this.allocatedPaintedSpringByteLength=size;this.#writeCustomSpringBuffer()}#writeCustomSpringBuffer(){if(!this.device)return;this.#ensureCustomSpringBuffer();if(!this.customSpringBuffer)return;if(!(this.paintedSpringMap instanceof Float32Array)||this.paintedSpringMap.length!==this.cellCount){this.paintedSpringMap=new Float32Array(this.cellCount)}this.device.queue.writeBuffer(this.customSpringBuffer,0,this.paintedSpringMap)}#ensureDummySceneTexture(){if(!this.device||!this.presentationFormat||this.dummySceneTexture)return;this.dummySceneTexture=this.device.createTexture({size:{width:1,height:1},format:this.presentationFormat,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST,label:"paper-dummy-scene-color"});this.dummySceneView=this.dummySceneTexture.createView();this.device.queue.writeTexture({texture:this.dummySceneTexture},new Uint8Array([14,17,20,255]),{bytesPerRow:256},{width:1,height:1,depthOrArrayLayers:1})}#invalidateRenderBundles(){this.terrainRenderBundle=null;this.finalRenderBundle=null;this.terrainRenderBundleList.length=0;this.finalRenderBundleList.length=0}#ensureRenderBundles(){if(!this.device||!this.renderPipeline||!this.compositePipeline||!this.waterRenderPipeline)return;if(!this.renderBindGroupTerrain||!this.renderBindGroupWater||!this.compositeBindGroup)return;if(this.terrainRenderBundle&&this.finalRenderBundle)return;const drawCount=this.drawCount||Math.max(1,(this.width-1)*(this.height-1));const terrainEncoder=this.device.createRenderBundleEncoder({colorFormats:[this.presentationFormat],depthStencilFormat:"depth24plus"});terrainEncoder.setPipeline(this.renderPipeline);terrainEncoder.setBindGroup(0,this.renderBindGroupTerrain);terrainEncoder.draw(6,drawCount,0,0);this.terrainRenderBundle=terrainEncoder.finish({label:"paper-render-terrain-bundle"});this.terrainRenderBundleList[0]=this.terrainRenderBundle;const finalEncoder=this.device.createRenderBundleEncoder({colorFormats:[this.presentationFormat],depthStencilFormat:"depth24plus"});finalEncoder.setPipeline(this.compositePipeline);finalEncoder.setBindGroup(0,this.compositeBindGroup);finalEncoder.draw(3,1,0,0);finalEncoder.setPipeline(this.waterRenderPipeline);finalEncoder.setBindGroup(0,this.renderBindGroupWater);finalEncoder.draw(6,drawCount,0,0);this.finalRenderBundle=finalEncoder.finish({label:"paper-render-final-bundle"});this.finalRenderBundleList[0]=this.finalRenderBundle}#encodeStepCommands(encoder,iterCount){if(!this.ready)return;const dispatchX=this.dispatchX||Math.ceil(this.width/WORKGROUP_SIZE_X);const dispatchY=this.dispatchY||Math.ceil(this.height/WORKGROUP_SIZE_Y);this.#ensureStepParamTemplate(iterCount);const strideFloats=this.stepParamStride>>>2;const stepParamValues=this.stepParamValuesCpu;const dt=this.paramValues[3];const baseTime=this.iterationCount*dt;for(let i=0;i<iterCount;i++){const offset=i*strideFloats;stepParamValues[offset+25]=baseTime+i*dt}this.device.queue.writeBuffer(this.stepParamBuffer,0,stepParamValues.buffer,0,this.stepParamStride*iterCount);const fluxPipeline=this.fluxPipeline;const flowPipeline=this.flowPipeline;const thermalOutflowPipeline=this.thermalOutflowPipeline;const erosionPipeline=this.erosionPipeline;const transportPipeline=this.transportPipeline;const thermalApplyPipeline=this.thermalApplyPipeline;const computeBindGroup01=this.computeBindGroup01;const computeBindGroup10=this.computeBindGroup10;const dynamicOffset=this.computeDynamicOffset;const pass=encoder.beginComputePass({label:"paper-erosion-compute"});for(let i=0;i<iterCount;i++){dynamicOffset[0]=i*this.stepParamStride;pass.setBindGroup(0,computeBindGroup01,dynamicOffset);pass.setPipeline(fluxPipeline);pass.dispatchWorkgroups(dispatchX,dispatchY);pass.setPipeline(flowPipeline);pass.dispatchWorkgroups(dispatchX,dispatchY);pass.setBindGroup(0,computeBindGroup10,dynamicOffset);pass.setPipeline(thermalOutflowPipeline);pass.dispatchWorkgroups(dispatchX,dispatchY);pass.setPipeline(erosionPipeline);pass.dispatchWorkgroups(dispatchX,dispatchY);pass.setBindGroup(0,computeBindGroup01,dynamicOffset);pass.setPipeline(transportPipeline);pass.dispatchWorkgroups(dispatchX,dispatchY);pass.setBindGroup(0,computeBindGroup10,dynamicOffset);pass.setPipeline(thermalApplyPipeline);pass.dispatchWorkgroups(dispatchX,dispatchY)}pass.end();this.iterationCount+=iterCount;this.#fillParamValues(this.iterationCount,this.paramValues)}#encodeRenderCommands(encoder,view){this.#ensureRenderBundles();{const pass=encoder.beginRenderPass({colorAttachments:[{view:this.sceneColorView,clearValue:{r:.055,g:.065,b:.08,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:this.depthTextureView?{view:this.depthTextureView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}:void 0});if(this.terrainRenderBundle){pass.executeBundles(this.terrainRenderBundleList)}else{const drawCount=this.drawCount||Math.max(1,(this.width-1)*(this.height-1));pass.setPipeline(this.renderPipeline);pass.setBindGroup(0,this.renderBindGroupTerrain);pass.draw(6,drawCount,0,0)}pass.end()}{const pass=encoder.beginRenderPass({colorAttachments:[{view,clearValue:{r:.055,g:.065,b:.08,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:this.depthTextureView?{view:this.depthTextureView,depthLoadOp:"load",depthStoreOp:"discard"}:void 0});if(this.finalRenderBundle){pass.executeBundles(this.finalRenderBundleList)}else{const drawCount=this.drawCount||Math.max(1,(this.width-1)*(this.height-1));pass.setPipeline(this.compositePipeline);pass.setBindGroup(0,this.compositeBindGroup);pass.draw(3,1,0,0);pass.setPipeline(this.waterRenderPipeline);pass.setBindGroup(0,this.renderBindGroupWater);pass.draw(6,drawCount,0,0)}pass.end()}}#writeRenderParams(){if(!this.renderUniformBufferTerrain&&!this.renderUniformBufferWater||!this.device)return;const aspect=this.canvas?Math.max((this.canvas.width||1)/Math.max(this.canvas.height||1,1),1e-6):1;const azimuth=degToRad(this.simulationParams.cameraAzimuthDeg);const elevation=degToRad(this.simulationParams.cameraElevationDeg);const worldScale=this.width>1||this.height>1?2/Math.max(this.width-1,this.height-1,1):1;const eye=[this.simulationParams.cameraPosX??1.5238999619464006,this.simulationParams.cameraPosY??1.9404787584406888,this.simulationParams.cameraPosZ??1.5238999619464006];const forward=normalizeVec3([-Math.cos(elevation)*Math.cos(azimuth),-Math.sin(elevation),-Math.cos(elevation)*Math.sin(azimuth)]);const target=[eye[0]+forward[0],eye[1]+forward[1],eye[2]+forward[2]];const view=lookAtMat4(eye,target,[0,1,0]);const proj=perspectiveMat4(degToRad(50),aspect,.01,32);const viewProj=multiplyMat4(proj,view);const effectiveCellSize=Math.max(.25,(this.simulationParams.metersPerPixel??100)/100);this.renderUniformValues.set(viewProj,0);this.renderUniformValues[16]=this.width;this.renderUniformValues[17]=this.height;this.renderUniformValues[18]=effectiveCellSize;this.renderUniformValues[19]=this.simulationParams.renderHeightScale;this.renderUniformValues[20]=this.simulationParams.waterOpacity;this.renderUniformValues[21]=this.simulationParams.sedimentTint;this.renderUniformValues[22]=this.simulationParams.renderMode;this.renderUniformValues[23]=this.simulationParams.thermalVisualizationScale;this.renderUniformValues[24]=worldScale;this.renderUniformValues[25]=this.simulationParams.waterHeightScale;this.renderUniformValues[27]=5e-4;const light=normalizeVec3([.45,.82,.36]);this.renderUniformValues[28]=light[0];this.renderUniformValues[29]=light[1];this.renderUniformValues[30]=light[2];this.renderUniformValues[31]=0;this.renderUniformValues[32]=eye[0];this.renderUniformValues[33]=eye[1];this.renderUniformValues[34]=eye[2];this.renderUniformValues[35]=0;this.renderUniformValues[36]=this.iterationCount*this.simulationParams.timeStep;this.renderUniformValues[37]=this.iterationCount;this.renderUniformValues[38]=this.canvas?.width||1;this.renderUniformValues[39]=this.canvas?.height||1;this.renderUniformValues[26]=0;this.device.queue.writeBuffer(this.renderUniformBufferTerrain,0,this.renderUniformValues);this.renderUniformValues[26]=1;this.device.queue.writeBuffer(this.renderUniformBufferWater,0,this.renderUniformValues)}#ensureBuffers(){const needsRebuild=this.stateBuffers.length!==2||!this.fluxBuffer||!this.velocityBuffer||!this.thermalPipeBufferA||!this.thermalPipeBufferB||!this.readbackBuffer||this.allocatedStateByteLength!==this.stateByteLength||this.allocatedVec4ByteLength!==this.vec4ByteLength;if(needsRebuild){for(const buffer of this.stateBuffers)buffer?.destroy?.();this.fluxBuffer?.destroy?.();this.velocityBuffer?.destroy?.();this.thermalPipeBufferA?.destroy?.();this.thermalPipeBufferB?.destroy?.();this.readbackBuffer?.destroy?.();this.customSpringBuffer?.destroy?.();this.customSpringBuffer=null;this.stateBuffers=[0,1].map(index=>this.device.createBuffer({size:this.stateByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:\`paper-state-\${index}\`}));this.fluxBuffer=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-flux"});this.velocityBuffer=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-velocity"});this.thermalPipeBufferA=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-thermal-pipes-a"});this.thermalPipeBufferB=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-thermal-pipes-b"});this.readbackBuffer=this.device.createBuffer({size:this.stateByteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,label:"paper-readback"});this.allocatedStateByteLength=this.stateByteLength;this.allocatedVec4ByteLength=this.vec4ByteLength;this.#rebuildBindGroups()}}#fillParamValues(iterationCount,target=this.paramValues){const effectiveCellSize=Math.max(.25,(this.simulationParams.metersPerPixel??100)/100);target[0]=this.width;target[1]=this.height;target[2]=effectiveCellSize;target[3]=this.simulationParams.timeStep;target[4]=this.simulationParams.rainRate;target[5]=this.simulationParams.evaporationRate;target[6]=this.simulationParams.pipeArea;target[7]=this.simulationParams.gravity;target[8]=this.simulationParams.capacityScale;target[9]=this.simulationParams.suspensionRate;target[10]=this.simulationParams.depositionRate;target[11]=this.simulationParams.softeningRate;target[12]=this.simulationParams.maxErosionDepth;target[13]=this.simulationParams.thermalRate;target[14]=this.simulationParams.talusSlopeCoeff;target[15]=this.simulationParams.talusSlopeBias;target[16]=this.simulationParams.sourceLayoutMode;target[17]=this.simulationParams.randomSpringCount;target[18]=this.simulationParams.pulse2Duration;target[19]=this.simulationParams.edgeDrainStrength;target[20]=this.simulationParams.sourceCenterX;target[21]=this.simulationParams.sourceCenterY;target[22]=this.simulationParams.sourceRadius;target[23]=this.simulationParams.sourceStrength;target[24]=this.simulationParams.sourceEnabled?1:0;target[25]=iterationCount*this.simulationParams.timeStep;target[26]=Math.max(0,this.simulationParams.rainDuration??0);target[27]=this.simulationParams.sourceTimeOffset;target[28]=this.simulationParams.historyDecay;target[29]=.02;target[30]=0;target[31]=4;target[32]=Math.floor(this.simulationParams.sourceSeed??1);target[33]=this.simulationParams.metersPerPixel??100;target[34]=this.simulationParams.hydraulicErosionEnabled?1:0;target[35]=0;return target}#writeParams(){this.#fillParamValues(this.iterationCount,this.paramValues);this.stepParamTemplateDirty=true;this.stepParamPreparedIterations=0;if(this.paramBuffer){this.device.queue.writeBuffer(this.paramBuffer,0,this.paramValues)}}#ensureStepParamTemplate(requiredIterations=1){this.#ensureStepParamBuffer(requiredIterations);if(!this.stepParamValuesCpu)return;if(!this.stepParamTemplateDirty&&this.stepParamPreparedIterations>=requiredIterations){return}const strideFloats=this.stepParamStride>>>2;const startIteration=this.stepParamTemplateDirty?0:this.stepParamPreparedIterations;for(let i=startIteration;i<requiredIterations;i++){this.stepParamValuesCpu.set(this.paramValues,i*strideFloats)}this.stepParamTemplateDirty=false;this.stepParamPreparedIterations=Math.max(this.stepParamPreparedIterations,requiredIterations)}#ensureStepParamBuffer(requiredIterations=1){const neededCapacity=Math.max(1,requiredIterations|0);if(this.stepParamBuffer&&this.stepParamCapacity>=neededCapacity&&this.stepParamValuesCpu&&this.stepParamValuesCpuCapacity>=neededCapacity){return}const nextCapacity=Math.max(8,neededCapacity,this.stepParamCapacity>0?this.stepParamCapacity*2:0);const strideFloats=this.stepParamStride>>>2;if(!this.stepParamValuesCpu||this.stepParamValuesCpuCapacity<neededCapacity){this.stepParamValuesCpu=new Float32Array(strideFloats*nextCapacity);this.stepParamValuesCpuCapacity=nextCapacity;this.stepParamTemplateDirty=true;this.stepParamPreparedIterations=0}if(!this.stepParamBuffer||this.stepParamCapacity<neededCapacity){this.stepParamBuffer?.destroy?.();this.stepParamBuffer=this.device.createBuffer({size:this.stepParamStride*nextCapacity,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-step-params"});this.stepParamCapacity=nextCapacity;this.stepParamTemplateDirty=true;this.stepParamPreparedIterations=0;if(this.stateBuffers.length===2){this.#rebuildBindGroups()}}}#createPipelines(){const computeModule=this.device.createShaderModule({code:COMPUTE_WGSL_SOURCE,label:"paper-erosion-compute-wgsl"});const renderModule=this.device.createShaderModule({code:RENDER_WGSL_SOURCE,label:"paper-erosion-render-wgsl"});const compositeModule=this.device.createShaderModule({code:COMPOSITE_WGSL_SOURCE,label:"paper-erosion-composite-wgsl"});this.computeBindGroupLayout=this.device.createBindGroupLayout({label:"paper-compute-bgl",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"uniform",hasDynamicOffset:true}},{binding:1,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});this.renderBindGroupLayout=this.device.createBindGroupLayout({label:"paper-render-bgl",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});this.compositeBindGroupLayout=this.device.createBindGroupLayout({label:"paper-composite-bgl",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});const computeLayout=this.device.createPipelineLayout({label:"paper-compute-layout",bindGroupLayouts:[this.computeBindGroupLayout]});const renderLayout=this.device.createPipelineLayout({label:"paper-render-layout",bindGroupLayouts:[this.renderBindGroupLayout]});const compositeLayout=this.device.createPipelineLayout({label:"paper-composite-layout",bindGroupLayouts:[this.compositeBindGroupLayout]});this.fluxPipeline=this.device.createComputePipeline({label:"paper-flux-pipeline",layout:computeLayout,compute:{module:computeModule,entryPoint:"fluxMain"}});this.flowPipeline=this.device.createComputePipeline({label:"paper-flow-pipeline",layout:computeLayout,compute:{module:computeModule,entryPoint:"flowMain"}});this.erosionPipeline=this.device.createComputePipeline({label:"paper-erosion-pipeline",layout:computeLayout,compute:{module:computeModule,entryPoint:"erosionMain"}});this.transportPipeline=this.device.createComputePipeline({label:"paper-transport-pipeline",layout:computeLayout,compute:{module:computeModule,entryPoint:"transportMain"}});this.thermalOutflowPipeline=this.device.createComputePipeline({label:"paper-thermal-outflow-pipeline",layout:computeLayout,compute:{module:computeModule,entryPoint:"thermalOutflowMain"}});this.thermalApplyPipeline=this.device.createComputePipeline({label:"paper-thermal-apply-pipeline",layout:computeLayout,compute:{module:computeModule,entryPoint:"thermalApplyMain"}});this.renderPipeline=this.device.createRenderPipeline({label:"paper-render-pipeline",layout:renderLayout,vertex:{module:renderModule,entryPoint:"vsMesh"},fragment:{module:renderModule,entryPoint:"fsMesh",targets:[{format:this.presentationFormat}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:true,depthCompare:"less"}});this.waterRenderPipeline=this.device.createRenderPipeline({label:"paper-water-render-pipeline",layout:renderLayout,vertex:{module:renderModule,entryPoint:"vsMesh"},fragment:{module:renderModule,entryPoint:"fsMesh",targets:[{format:this.presentationFormat,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:false,depthCompare:"less-equal"}});this.compositePipeline=this.device.createRenderPipeline({label:"paper-composite-pipeline",layout:compositeLayout,vertex:{module:compositeModule,entryPoint:"vsComposite"},fragment:{module:compositeModule,entryPoint:"fsComposite",targets:[{format:this.presentationFormat}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:false,depthCompare:"always"}})}#rebuildBindGroups(){if(!this.sceneColorView||!this.sceneSampler){return}this.#invalidateRenderBundles();this.#ensureDummySceneTexture();this.#ensureCustomSpringBuffer();this.#ensureStepParamBuffer(1);if(!this.dummySceneView||!this.stepParamBuffer||!this.customSpringBuffer){return}this.computeBindGroup01=this.device.createBindGroup({label:"paper-compute-0to1",layout:this.computeBindGroupLayout,entries:[{binding:0,resource:{buffer:this.stepParamBuffer,size:this.paramValues.byteLength}},{binding:1,resource:{buffer:this.stateBuffers[0]}},{binding:2,resource:{buffer:this.stateBuffers[1]}},{binding:3,resource:{buffer:this.fluxBuffer}},{binding:4,resource:{buffer:this.velocityBuffer}},{binding:5,resource:{buffer:this.thermalPipeBufferA}},{binding:6,resource:{buffer:this.thermalPipeBufferB}},{binding:7,resource:{buffer:this.customSpringBuffer}}]});this.computeBindGroup10=this.device.createBindGroup({label:"paper-compute-1to0",layout:this.computeBindGroupLayout,entries:[{binding:0,resource:{buffer:this.stepParamBuffer,size:this.paramValues.byteLength}},{binding:1,resource:{buffer:this.stateBuffers[1]}},{binding:2,resource:{buffer:this.stateBuffers[0]}},{binding:3,resource:{buffer:this.fluxBuffer}},{binding:4,resource:{buffer:this.velocityBuffer}},{binding:5,resource:{buffer:this.thermalPipeBufferA}},{binding:6,resource:{buffer:this.thermalPipeBufferB}},{binding:7,resource:{buffer:this.customSpringBuffer}}]});this.renderBindGroupTerrain=this.device.createBindGroup({label:"paper-render-terrain",layout:this.renderBindGroupLayout,entries:[{binding:0,resource:{buffer:this.renderUniformBufferTerrain}},{binding:1,resource:{buffer:this.stateBuffers[0]}},{binding:2,resource:this.sceneSampler},{binding:3,resource:this.dummySceneView}]});this.renderBindGroupWater=this.device.createBindGroup({label:"paper-render-water",layout:this.renderBindGroupLayout,entries:[{binding:0,resource:{buffer:this.renderUniformBufferWater}},{binding:1,resource:{buffer:this.stateBuffers[0]}},{binding:2,resource:this.sceneSampler},{binding:3,resource:this.sceneColorView}]});this.compositeBindGroup=this.device.createBindGroup({label:"paper-composite",layout:this.compositeBindGroupLayout,entries:[{binding:0,resource:this.sceneSampler},{binding:1,resource:this.sceneColorView}]})}};var COMPUTE_WGSL_SOURCE=\`

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

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> srcState: StateBuffer;
@group(0) @binding(2) var<storage, read_write> dstState: StateBuffer;
@group(0) @binding(3) var<storage, read_write> fluxState: Vec4Buffer;
@group(0) @binding(4) var<storage, read_write> velocityState: Vec4Buffer;
@group(0) @binding(5) var<storage, read_write> thermalPipeA: Vec4Buffer;
@group(0) @binding(6) var<storage, read_write> thermalPipeB: Vec4Buffer;
@group(0) @binding(7) var<storage, read> paintedSourceState: FloatBuffer;

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
  let outsideTotal = -edgeDrain;
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

  dstState.cells[i] = CellState(finiteOr(cell.terrain, 0.0), finiteOr(water, 0.0), finiteOr(cell.sediment, 0.0), finiteOr(cell.hardness, 0.1), cell.mask, 0.0, 0.0, finiteOr(cell.aux2, 0.0));
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
  let capacityTerm = max(max(collisionTerm, sinAlpha * 0.2), 0.05);
  let capacity = finiteOr(params.hydro1.x * capacityTerm * speed * depthLimiter(cell.water), 0.0);

  var terrain = cell.terrain;
  var water = cell.water;
  var sediment = cell.sediment;
  var hardness = cell.hardness;

  var history = finiteOr(cell.aux2, 0.0) * params.misc0.x;

  if (!hydraulicErosionEnabled()) {
    dstState.cells[i] = CellState(clamp(finiteOr(cell.terrain, 0.0), 0.0, 2.0), clamp(finiteOr(cell.water, 0.0), 0.0, 2.0), clamp(finiteOr(cell.sediment, 0.0), 0.0, 2.0), finiteOr(cell.hardness, params.misc0.y), cell.mask, finiteOr(capacity, 0.0), finiteOr(speed, 0.0), finiteOr(cell.aux2, 0.0));
    return;
  }

  if (capacity > sediment && water > 1e-6) {
    let erodeAmount = timeStep() * max(hardness, 0.02) * params.hydro1.y * (capacity - sediment);
    let clampedErode = min(min(erodeAmount, max(water, 0.0)), max(terrain, 0.0));
    terrain = max(0.0, terrain - clampedErode);
    sediment += clampedErode;
    water += clampedErode;
    history -= clampedErode * 240.0;
  } else if (sediment > capacity) {
    let sedimentExcess = sediment - capacity;
    let depositAmount = timeStep() * params.hydro1.z * sedimentExcess;
    let neighborMean = terrainNeighborMean(x, y, terrain);
    let localCeiling = max(terrain + 0.001, neighborMean + params.thermal0.w * 0.9 + max(cell.water, 0.0) * 0.08);
    let spikeGuard = max(0.0, localCeiling - terrain);
    let clampedDeposit = min(min(depositAmount, sediment), spikeGuard + max(cell.water * 0.04, 0.0006));
    terrain += clampedDeposit;
    sediment -= clampedDeposit;
    water = max(0.0, water - clampedDeposit);
    hardness = max(params.misc0.y, hardness - timeStep() * params.hydro1.w * params.hydro1.y * sedimentExcess);
    history += clampedDeposit * 180.0;
  }

  history = clamp(history, -1.0, 1.0);
  terrain = clamp(terrain, 0.0, 2.0);
  water = clamp(water, 0.0, 2.0);
  sediment = clamp(sediment, 0.0, 2.0);
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
  dstState.cells[i] = CellState(clamp(finiteOr(cell.terrain, 0.0), 0.0, 2.0), water, nextSediment, finiteOr(cell.hardness, params.misc0.y), cell.mask, cell.aux0, cell.aux1, finiteOr(cell.aux2, 0.0));
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
  let softness = clamp(0.30 + max(cell.hardness, 0.02) * 1.35, 0.20, 1.20);
  let thresholdBase = max(params.thermal0.w * 0.5, params.thermal0.z * max(cell.hardness, 0.02) * 0.55 + params.thermal0.w * 0.35);

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

  let totalOut = min(cell.terrain, cellArea() * timeStep() * params.thermal0.y * softness * maxExcess * 1.35);
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
  dstState.cells[i] = CellState(terrain, clamp(finiteOr(cell.water, 0.0), 0.0, 2.0), clamp(finiteOr(cell.sediment, 0.0), 0.0, 2.0), finiteOr(cell.hardness, params.misc0.y), cell.mask, finiteOr(selfOut, 0.0), finiteOr(incoming, 0.0), finiteOr(cell.aux2, 0.0));
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
  if (waterPass) {
    if (waterAlpha <= max(renderParams.misc.w, 0.01)) {
      return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let w = clamp(in.water * 2.2, 0.0, 1.0);
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
    let fresnel = pow(1.0 - ndv, 4.5);
    let halfVec = normalize(viewDir + lightDir);
    let sunSpecTight = pow(max(dot(combinedNormal, halfVec), 0.0), 120.0);
    let sunSpecBroad = pow(max(dot(combinedNormal, halfVec), 0.0), 32.0);
    let sunFacing = clamp(dot(combinedNormal, lightDir), 0.0, 1.0);
    let deepTint = mix(vec3<f32>(0.004, 0.018, 0.052), vec3<f32>(0.010, 0.038, 0.100), w);
    let shallowTint = mix(vec3<f32>(0.012, 0.060, 0.145), vec3<f32>(0.030, 0.115, 0.220), w);
    let waterBase = mix(deepTint, shallowTint, clamp(0.16 + 0.56 * w, 0.0, 1.0));
    var waterColorOnly = waterBase * (0.72 + 0.22 * sunFacing);
    waterColorOnly += vec3<f32>(1.80, 1.65, 1.40) * sunSpecTight;
    waterColorOnly += vec3<f32>(0.16, 0.14, 0.12) * (sunSpecBroad * 0.35);
    waterColorOnly += vec3<f32>(0.02, 0.02, 0.018) * fresnel;
    let finalAlpha = clamp(waterAlpha * (0.94 + 0.06 * fresnel), 0.0, 0.97);
    return vec4<f32>(clamp(waterColorOnly, vec3<f32>(0.0), vec3<f32>(1.0)), finalAlpha);
  }

  let mode = i32(round(renderParams.shading.z));

  if (mode == 0) {
    return vec4<f32>(terrainColor, 1.0);
  }
  if (mode == 2) {
    let w = clamp(in.water * 1.6, 0.0, 1.0);
    return vec4<f32>(mix(vec3<f32>(0.02, 0.16, 0.46), vec3<f32>(0.40, 0.96, 1.00), w), 1.0);
  }
  if (mode == 3) {
    let s = clamp(in.sediment * renderParams.shading.y * 2.1, 0.0, 1.0);
    let matteLambert = 0.72 + 0.18 * lambert + 0.10 * hemi;
    let sedimentBase = mix(vec3<f32>(0.14, 0.11, 0.09), vec3<f32>(0.96, 0.78, 0.44), s);
    var sedimentColor = sedimentBase * matteLambert;
    sedimentColor = mix(sedimentColor, vec3<f32>(0.98, 0.86, 0.62), s * 0.30);
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

  return vec4<f32>(terrainColor, 1.0);
}
\`;var gpu=null;var offscreenCanvas=null;var canvasMetrics={width:1,height:1,dpr:1};var running=false;var loopHandle=0;var lastFrameAt=0;var lastFrameMs=0;var lastStatusPostAt=0;var lastRenderAt=0;var loopIterationsPerFrame=5;var simulationParams={};var sourceRaster=null;var sourceImageInfo=null;var hasWorkerRAF=typeof self.requestAnimationFrame==="function";var workerRAF=hasWorkerRAF?cb=>self.requestAnimationFrame(cb):cb=>self.setTimeout(()=>cb(performance.now()),0);var workerCAF=hasWorkerRAF?id=>self.cancelAnimationFrame(id):id=>self.clearTimeout(id);function cloneStats(stats){if(!stats)return null;return{...stats,terrainRange:stats.terrainRange?{...stats.terrainRange}:null,waterRange:stats.waterRange?{...stats.waterRange}:null,sedimentRange:stats.sedimentRange?{...stats.sedimentRange}:null,historyRange:stats.historyRange?{...stats.historyRange}:null}}function getSourcePoints(){return gpu?.getSourcePoints?.()||[]}function postStatus(force=false){const stats=gpu?cloneStats(gpu.getStats()):null;self.postMessage({type:"status",stats,sourcePoints:force?getSourcePoints():void 0,lastFrameMs,running,sourceImageInfo:force?sourceImageInfo:void 0})}function applyCanvasMetrics(){if(!offscreenCanvas)return;const width=Math.max(1,Math.round((canvasMetrics.width||1)*(canvasMetrics.dpr||1)));const height=Math.max(1,Math.round((canvasMetrics.height||1)*(canvasMetrics.dpr||1)));offscreenCanvas.width=width;offscreenCanvas.height=height;if(gpu){gpu.resize()}}async function ensureGpu(){if(!offscreenCanvas){throw new Error("Worker canvas has not been initialized.")}if(!gpu){gpu=new WebGPUTerrainErosion({canvas:offscreenCanvas});await gpu.initialize();if(Object.keys(simulationParams).length>0){gpu.setSimulationParams(simulationParams)}applyCanvasMetrics()}return gpu}function stopLoopInternal(){if(!running)return;running=false;if(loopHandle){workerCAF(loopHandle);loopHandle=0}}function scheduleNextFrame(){loopHandle=workerRAF(now=>{void frame(now)})}function getLoopRenderIntervalMs(){const iterations=Math.max(1,loopIterationsPerFrame|0);if(iterations<=4)return 0;if(iterations<=8)return 16;if(iterations<=16)return 33;return 50}async function frame(now){if(!running)return;const currentGpu=gpu;if(!currentGpu?.ready){scheduleNextFrame();return}lastFrameMs=lastFrameAt>0?Math.max(0,now-lastFrameAt):0;lastFrameAt=now;try{const iterations=Math.max(1,loopIterationsPerFrame|0);const renderIntervalMs=getLoopRenderIntervalMs();const shouldRender=renderIntervalMs<=0||lastRenderAt<=0||now-lastRenderAt>=renderIntervalMs;if(shouldRender){currentGpu.stepAndRender(iterations);lastRenderAt=now}else{currentGpu.step(iterations)}if(now-lastStatusPostAt>=1e3){lastStatusPostAt=now;postStatus()}}catch(error){stopLoopInternal();self.postMessage({type:"workerError",message:error instanceof Error?error.message:String(error)});return}scheduleNextFrame()}function reply(requestId,payload={},transfer=[]){if(!requestId)return;self.postMessage({type:"response",requestId,ok:true,...payload},transfer)}function replyError(requestId,error){if(!requestId)return;self.postMessage({type:"response",requestId,ok:false,error:error instanceof Error?error.message:String(error)})}async function buildNormalizedRasterFromBlob(blob){if(!(blob instanceof Blob)){throw new Error("DEM image payload must be a Blob or File.")}if(typeof createImageBitmap!=="function"){throw new Error("createImageBitmap is not available in this worker.")}const bitmap=await createImageBitmap(blob,{imageOrientation:"none"});try{return await buildNormalizedRasterFromBitmap(bitmap)}finally{bitmap.close?.()}}async function buildNormalizedRasterFromBitmap(bitmap){const width=bitmap.width|0;const height=bitmap.height|0;const canvas=new OffscreenCanvas(width,height);const ctx=canvas.getContext("2d",{willReadFrequently:true});if(!ctx){throw new Error("Failed to acquire worker raster canvas context.")}ctx.setTransform(1,0,0,-1,0,height);ctx.clearRect(0,0,width,height);ctx.drawImage(bitmap,0,0,width,height);ctx.setTransform(1,0,0,1,0,0);const imageData=ctx.getImageData(0,0,width,height);const rgba=imageData.data;const values=new Float32Array(width*height);const mask=new Uint8Array(width*height);for(let i=0,j=0;i<values.length;i++,j+=4){values[i]=(rgba[j]+rgba[j+1]+rgba[j+2])/(3*255);mask[i]=rgba[j+3]===0?0:1}return{width,height,values,mask}}function resampleRasterBilinear(raster,scale){const safeScale=Math.max(1,Math.floor(Number(scale)||1));if(!raster||safeScale===1)return raster;const srcWidth=raster.width|0;const srcHeight=raster.height|0;const dstWidth=Math.max(1,(srcWidth-1)*safeScale+1);const dstHeight=Math.max(1,(srcHeight-1)*safeScale+1);const dstValues=new Float32Array(dstWidth*dstHeight);const dstMask=new Uint8Array(dstWidth*dstHeight);const srcValues=raster.values;const srcMask=raster.mask instanceof Uint8Array?raster.mask:null;function sampleValue(px,py){const x0=Math.floor(px);const y0=Math.floor(py);const x1=Math.min(x0+1,srcWidth-1);const y1=Math.min(y0+1,srcHeight-1);const tx=px-x0;const ty=py-y0;const i00=y0*srcWidth+x0;const i10=y0*srcWidth+x1;const i01=y1*srcWidth+x0;const i11=y1*srcWidth+x1;const v00=srcValues[i00];const v10=srcValues[i10];const v01=srcValues[i01];const v11=srcValues[i11];const a=v00+(v10-v00)*tx;const b=v01+(v11-v01)*tx;return a+(b-a)*ty}function sampleMask(px,py){if(!srcMask)return 1;const x0=Math.floor(px);const y0=Math.floor(py);const x1=Math.min(x0+1,srcWidth-1);const y1=Math.min(y0+1,srcHeight-1);const tx=px-x0;const ty=py-y0;const i00=y0*srcWidth+x0;const i10=y0*srcWidth+x1;const i01=y1*srcWidth+x0;const i11=y1*srcWidth+x1;const m00=srcMask[i00];const m10=srcMask[i10];const m01=srcMask[i01];const m11=srcMask[i11];const a=m00+(m10-m00)*tx;const b=m01+(m11-m01)*tx;return a+(b-a)*ty>=.5?1:0}for(let y=0;y<dstHeight;y++){const srcY=y/safeScale;for(let x=0;x<dstWidth;x++){const srcX=x/safeScale;const dstIndex=y*dstWidth+x;dstValues[dstIndex]=sampleValue(srcX,srcY);dstMask[dstIndex]=sampleMask(srcX,srcY)}}return{width:dstWidth,height:dstHeight,values:dstValues,mask:dstMask}}async function loadRasterIntoGpu(message){const currentGpu=await ensureGpu();if(Object.keys(simulationParams).length>0){currentGpu.setSimulationParams(simulationParams)}if(message.blob){sourceRaster=await buildNormalizedRasterFromBlob(message.blob);sourceImageInfo={width:sourceRaster.width,height:sourceRaster.height}}if(!sourceRaster){throw new Error("No DEM image has been loaded into the worker yet.")}const tessellation=Math.max(1,Math.floor(Number(message.tessellation)||1));const raster=resampleRasterBilinear(sourceRaster,tessellation);await currentGpu.setDEM(raster,message.options||{});currentGpu.render();const stats=cloneStats(await currentGpu.readbackStats());return{stats,sourcePoints:getSourcePoints(),ready:true,sourceImageInfo,simSize:{width:raster.width,height:raster.height}}}self.onmessage=async event=>{const message=event.data||{};const{type,requestId}=message;try{switch(type){case"init":{offscreenCanvas=message.canvas;canvasMetrics={width:Math.max(1,Number(message.width)||1),height:Math.max(1,Number(message.height)||1),dpr:Math.max(1,Number(message.dpr)||1)};await ensureGpu();reply(requestId,{initialized:true});postStatus(true);break}case"resize":{canvasMetrics={width:Math.max(1,Number(message.width)||canvasMetrics.width||1),height:Math.max(1,Number(message.height)||canvasMetrics.height||1),dpr:Math.max(1,Number(message.dpr)||canvasMetrics.dpr||1)};applyCanvasMetrics();gpu?.render?.();reply(requestId,{resized:true});break}case"setParams":{simulationParams={...simulationParams,...message.params||{}};if(gpu){gpu.setSimulationParams(simulationParams)}reply(requestId,{applied:true});break}case"loadDEMImage":{const payload=await loadRasterIntoGpu(message);reply(requestId,payload);postStatus(true);break}case"setDEM":{const currentGpu=await ensureGpu();if(Object.keys(simulationParams).length>0){currentGpu.setSimulationParams(simulationParams)}const raster=message.raster||{};await currentGpu.setDEM({width:raster.width|0,height:raster.height|0,values:raster.values instanceof Float32Array?raster.values:new Float32Array(raster.values||[]),mask:raster.mask?raster.mask instanceof Uint8Array?raster.mask:new Uint8Array(raster.mask):null},message.options||{});currentGpu.render();const stats=cloneStats(await currentGpu.readbackStats());reply(requestId,{stats,sourcePoints:getSourcePoints(),ready:true,sourceImageInfo});postStatus(true);break}case"render":{gpu?.render?.();reply(requestId,{rendered:true});break}case"step":{if(gpu?.ready){if(message.render!==false){gpu.stepAndRender(Math.max(1,Number(message.iterations)||1))}else{gpu.step(Math.max(1,Number(message.iterations)||1))}}reply(requestId,{stepped:true,stats:cloneStats(gpu?.getStats?.()||null),sourcePoints:getSourcePoints()});break}case"readbackStats":{const stats=gpu?.ready?cloneStats(await gpu.readbackStats()):cloneStats(gpu?.getStats?.()||null);reply(requestId,{stats,sourcePoints:getSourcePoints(),lastFrameMs,running,sourceImageInfo});break}case"exportTerrainPng":{if(!gpu?.ready){throw new Error("Terrain export is only available after the GPU sim is ready.")}const exportResult=await gpu.exportTerrainPng();reply(requestId,{width:exportResult.width,height:exportResult.height,minTerrain:exportResult.minTerrain,maxTerrain:exportResult.maxTerrain,data:exportResult.data},[exportResult.data]);break}case"paintTerrainBrush":{if(!gpu?.ready){throw new Error("Terrain painting is only available after the GPU sim is ready.")}const stats=cloneStats(await gpu.applyTerrainBrush(message.brush||{}));gpu.render();reply(requestId,{stats,sourcePoints:getSourcePoints()});postStatus(true);break}case"paintSpringBrush":{if(!gpu?.ready){throw new Error("Spring painting is only available after the GPU sim is ready.")}gpu.paintSpringBrush(message.brush||{});const stats=cloneStats(gpu.getStats());reply(requestId,{stats,sourcePoints:getSourcePoints()});postStatus(true);break}case"clearPaintedSprings":{if(gpu?.ready){gpu.clearPaintedSprings()}const stats=cloneStats(gpu?.getStats?.()||null);reply(requestId,{stats,sourcePoints:getSourcePoints()});postStatus(true);break}case"resetRainTimer":{if(gpu?.ready){gpu.resetRainTimer();gpu.render()}const stats=gpu?.ready?cloneStats(await gpu.readbackStats()):cloneStats(gpu?.getStats?.()||null);reply(requestId,{stats,sourcePoints:getSourcePoints()});break}case"restartSources":{if(gpu?.ready){gpu.resetRainTimer();gpu.render()}const stats=gpu?.ready?cloneStats(await gpu.readbackStats()):cloneStats(gpu?.getStats?.()||null);reply(requestId,{stats,sourcePoints:getSourcePoints()});break}case"startLoop":{if(Number.isFinite(message.iterationsPerFrame)){loopIterationsPerFrame=Math.max(1,Number(message.iterationsPerFrame)|0)}if(!running){running=true;lastFrameAt=0;lastStatusPostAt=0;lastRenderAt=0;scheduleNextFrame()}reply(requestId,{running:true,iterationsPerFrame:loopIterationsPerFrame});break}case"stopLoop":{stopLoopInternal();if(gpu?.ready){gpu.render()}reply(requestId,{running:false});postStatus(true);break}case"clear":{stopLoopInternal();if(gpu){gpu.destroy();gpu=null}sourceRaster=null;sourceImageInfo=null;reply(requestId,{cleared:true});postStatus(true);break}case"getStatus":{reply(requestId,{stats:cloneStats(gpu?.getStats?.()||null),sourcePoints:getSourcePoints(),lastFrameMs,running,sourceImageInfo});break}case"destroy":{stopLoopInternal();if(gpu){gpu.destroy();gpu=null}sourceRaster=null;sourceImageInfo=null;reply(requestId,{destroyed:true});self.close();break}default:throw new Error(\`Unknown worker message type: \${String(type)}\`)}}catch(error){replyError(requestId,error)}};;if(typeof import_meta !== 'undefined')import_meta.url=location.origin+"/dist/";})();
`,_r=URL.createObjectURL(new globalThis.Blob([Wr],{type:"text/javascript"})),ur=_r;var cr=0,dr=1,Q=5,je=128,bt=.02,St=.001,wt=.015,Pt=20,Bt=9.81,Et=1,Tt=.5,Ct=1,Mt=5,Rt=.12,At=.45,Lt=.8,Ft=.1,kt=.045,Je=.06,Dt=.92,Ot=.35,hr=.16,pr=.03,Ut=35,It=18,Wt=4,_t=.06,Qe=0,et=0,mr=0,tt=4,Gt=1,rt=100,at=45,nt=42;var fr=1.5238999619464006,gr=1.9404787584406888,xr=1.5238999619464006,Gr=1.35,zr=.14,yr=1,Nr=!0,Vr=!0,Hr=!0,Yr=!0,zt=12,Nt=.03,Vt=.7,$r=!1;function Ht(...e){$r&&console.log("[WebGPU Erosion UI]",...e)}function Y(e){let t=document.createElement("button");return t.textContent=e,t.style.padding="8px 12px",t.style.border="1px solid #444",t.style.borderRadius="8px",t.style.background="#1f1f1f",t.style.color="#f0f0f0",t.style.cursor="pointer",t.addEventListener("mouseenter",()=>{t.style.background="#2a2a2a"}),t.addEventListener("mouseleave",()=>{t.style.background="#1f1f1f"}),t}function c(e,t,a="88px",n={}){let i=document.createElement("label");i.textContent=e,i.style.display="inline-flex",i.style.alignItems="center",i.style.gap="8px";let o=document.createElement("input");return o.type="number",o.value=String(t),o.min=String(n.min??-1e9),o.max=String(n.max??1e9),o.step=String(n.step??.01),o.style.width=a,o.style.padding="6px 8px",o.style.border="1px solid #444",o.style.borderRadius="6px",o.style.background="#1a1a1a",o.style.color="#e8e8e8",i.input=o,i.appendChild(o),i}function Le(e,t){let a=document.createElement("label");a.style.display="inline-flex",a.style.alignItems="center",a.style.gap="6px";let n=document.createElement("input");n.type="checkbox",n.checked=t;let i=document.createElement("span");return i.textContent=e,a.input=n,a.append(n,i),a}function Fe(e,t,a){let n=document.createElement("label");n.textContent=e,n.style.display="inline-flex",n.style.alignItems="center",n.style.gap="8px";let i=document.createElement("select");i.style.padding="6px 8px",i.style.border="1px solid #444",i.style.borderRadius="6px",i.style.background="#1a1a1a",i.style.color="#e8e8e8";for(let{value:o,label:l}of t){let f=document.createElement("option");f.value=String(o),f.textContent=l,o===a&&(f.selected=!0),i.appendChild(f)}return n.input=i,n.appendChild(i),n}function s(e,t){let a=Number(e.input.value);return Number.isFinite(a)?a:t}function $(e){let t=document.createElement("section");t.style.border="1px solid #2d2d2d",t.style.borderRadius="10px",t.style.padding="10px",t.style.background="#141414";let a=document.createElement("div");a.textContent=e,a.style.fontSize="13px",a.style.fontWeight="600",a.style.marginBottom="8px",a.style.letterSpacing="0.02em";let n=document.createElement("div");return n.style.display="flex",n.style.flexWrap="wrap",n.style.gap="8px",t.append(a,n),t.body=n,t.heading=a,t}function q(e){let t=document.createElement("div");return t.textContent=e,t.style.fontSize="12px",t.style.lineHeight="1.35",t.style.color="rgba(255,255,255,0.68)",t.style.margin="-2px 0 8px 0",t}function vr(){return Math.max(1,Number(Kt.input.value)||yr)}function Xe(e,t=4){return!e||!Number.isFinite(e.min)||!Number.isFinite(e.max)?"n/a":`${e.min.toFixed(t)} to ${e.max.toFixed(t)}`}function j(e,t,a){return e<t?t:e>a?a:e}function Ke(e){return e*Math.PI/180}function re(e){let t=Math.hypot(e[0],e[1],e[2])||1;return[e[0]/t,e[1]/t,e[2]/t]}function br(e,t){return[e[0]-t[0],e[1]-t[1],e[2]-t[2]]}function xt(e,t){return[e[1]*t[2]-e[2]*t[1],e[2]*t[0]-e[0]*t[2],e[0]*t[1]-e[1]*t[0]]}function Sr(e,t){let a=new Float32Array(16);for(let n=0;n<4;n++)for(let i=0;i<4;i++)a[n*4+i]=e[0+i]*t[n*4+0]+e[4+i]*t[n*4+1]+e[8+i]*t[n*4+2]+e[12+i]*t[n*4+3];return a}function wr(e,t,a,n){let i=1/Math.tan(e*.5),o=1/(a-n),l=new Float32Array(16);return l[0]=i/Math.max(t,1e-6),l[5]=i,l[10]=(n+a)*o,l[11]=-1,l[14]=2*n*a*o,l}function Pr(e,t,a){let n=re(br(e,t)),i=re(xt(a,n)),o=xt(n,i),l=new Float32Array(16);return l[0]=i[0],l[1]=o[0],l[2]=n[0],l[3]=0,l[4]=i[1],l[5]=o[1],l[6]=n[1],l[7]=0,l[8]=i[2],l[9]=o[2],l[10]=n[2],l[11]=0,l[12]=-(i[0]*e[0]+i[1]*e[1]+i[2]*e[2]),l[13]=-(o[0]*e[0]+o[1]*e[1]+o[2]*e[2]),l[14]=-(n[0]*e[0]+n[1]*e[1]+n[2]*e[2]),l[15]=1,l}function qr(e){let t=new Float32Array(16),a=e[0],n=e[1],i=e[2],o=e[3],l=e[4],f=e[5],k=e[6],T=e[7],R=e[8],y=e[9],b=e[10],S=e[11],C=e[12],w=e[13],P=e[14],B=e[15],M=a*f-n*l,D=a*k-i*l,I=a*T-o*l,A=n*k-i*f,W=n*T-o*f,_=i*T-o*k,v=R*w-y*C,Z=R*P-b*C,Ce=R*B-S*C,Me=y*P-b*w,Re=y*B-S*w,Ae=b*B-S*P,p=M*Ae-D*Re+I*Me+A*Ce-W*Z+_*v;return p?(p=1/p,t[0]=(f*Ae-k*Re+T*Me)*p,t[1]=(i*Re-n*Ae-o*Me)*p,t[2]=(w*_-P*W+B*A)*p,t[3]=(b*W-y*_-S*A)*p,t[4]=(k*Ce-l*Ae-T*Z)*p,t[5]=(a*Ae-i*Ce+o*Z)*p,t[6]=(P*I-C*_-B*D)*p,t[7]=(R*_-b*I+S*D)*p,t[8]=(l*Re-f*Ce+T*v)*p,t[9]=(n*Ce-a*Re-o*v)*p,t[10]=(C*W-w*I+B*M)*p,t[11]=(y*I-R*W-S*M)*p,t[12]=(f*Z-l*Me-k*v)*p,t[13]=(a*Me-n*Z+i*v)*p,t[14]=(w*D-C*A-P*M)*p,t[15]=(R*A-y*D+b*M)*p,t):null}function yt(e,t){return[e[0]*t[0]+e[4]*t[1]+e[8]*t[2]+e[12]*t[3],e[1]*t[0]+e[5]*t[1]+e[9]*t[2]+e[13]*t[3],e[2]*t[0]+e[6]*t[1]+e[10]*t[2]+e[14]*t[3],e[3]*t[0]+e[7]*t[1]+e[11]*t[2]+e[15]*t[3]]}var r={sourceImageInfo:null,sourceFile:null,sourceFileName:"",sourceUploadedToWorker:!1,worker:null,workerReady:!1,workerCanvasTransferred:!1,running:!1,lastFrameMs:0,buildCount:0,readbackPending:!1,lastReadbackAt:0,gpuStats:null,sourcePoints:[],isPainting:!1,lastPaintAt:0,cameraPosX:fr,cameraPosY:gr,cameraPosZ:xr,pointerLookActive:!1,navLoopHandle:0,navLastAt:0,navKeys:{KeyW:!1,KeyA:!1,KeyS:!1,KeyD:!1,Space:!1,KeyC:!1}},Xr=1,vt=new Map;function Br(){let e=F.getBoundingClientRect(),t=Math.max(1,Math.round(e.width||960)),a=Math.max(1,Math.round(e.height||Math.round(t*2/3))),n=Math.max(1,Math.min(window.devicePixelRatio||1,2));return{width:t,height:a,dpr:n}}function Kr(e){e.addEventListener("message",t=>{let a=t.data||{};if(a.type==="response"&&a.requestId){let n=vt.get(a.requestId);if(!n)return;vt.delete(a.requestId),a.ok?n.resolve(a):n.reject(new Error(a.error||"Worker request failed."));return}if(a.type==="status"){a.stats&&(r.gpuStats=a.stats),Array.isArray(a.sourcePoints)&&(r.sourcePoints=a.sourcePoints),a.sourceImageInfo&&(r.sourceImageInfo=a.sourceImageInfo),Number.isFinite(a.lastFrameMs)&&(r.lastFrameMs=a.lastFrameMs),typeof a.running=="boolean"&&(r.running=a.running),h();return}a.type==="workerError"&&(Ht("workerError",a.message),m(a.message||"Worker error"))}),e.addEventListener("error",t=>{console.error("[WebGPU Erosion UI] worker error",t),m(t.message||"Worker crashed")})}async function Zr(){if(r.worker)return r.worker;if(!("transferControlToOffscreen"in u))throw new Error("OffscreenCanvas worker rendering is not available in this browser.");let e=new Worker(ur,{type:"module"});Kr(e),r.worker=e;let t=u.transferControlToOffscreen();r.workerCanvasTransferred=!0;let a=Br();return await O("init",{canvas:t,width:a.width,height:a.height,dpr:a.dpr},[t]),r.workerReady=!0,e}function x(e,t={},a=[]){r.worker&&r.worker.postMessage({type:e,...t},a)}function O(e,t={},a=[]){return new Promise((n,i)=>{if(!r.worker){i(new Error("Worker is not initialized."));return}let o=Xr++;vt.set(o,{resolve:n,reject:i}),r.worker.postMessage({type:e,requestId:o,...t},a)})}function Er(){if(!r.workerReady)return;let e=Br();x("resize",e)}function jr(){return{timeStep:s(ke,bt),rainRate:Se.input.checked?s(oe,St):0,evaporationRate:s(De,wt),pipeArea:s(Oe,Pt),gravity:s(Ue,Bt),capacityScale:s(le,Et),suspensionRate:s(ue,Tt),depositionRate:s(ce,Ct),softeningRate:s(de,Mt),maxErosionDepth:s(he,Rt),thermalRate:we.input.checked?s(pe,At):0,hydraulicErosionEnabled:!!Ge.input.checked,talusSlopeCoeff:s(me,Lt),talusSlopeBias:s(fe,Ft),renderHeightScale:s(ot,kt),waterHeightScale:s(Ie,Je),waterOpacity:s(lt,Dt),sedimentTint:s(ut,Ot),hardnessBase:s(Yt,hr),hardnessVariation:s($t,pr),sourceCenterX:s(qt,Ut)/100,sourceCenterY:s(Xt,It)/100,sourceRadius:s(ge,Wt),sourceStrength:s(xe,_t),rainDuration:s(ye,Qe),pulse2Duration:s(We,et),sourceLayoutMode:Number(ve.input.value)||0,randomSpringCount:Math.max(1,Math.floor(s(be,tt))),sourceSeed:Math.floor(s(_e,Gt)),metersPerPixel:Math.max(1,s(ct,rt)),hydraulic8Way:!!dt.input.checked,sourceEnabled:!!J.input.checked,renderMode:Number(ae.input.value)||0,cameraAzimuthDeg:s(ne,at),cameraElevationDeg:s(ie,nt),cameraPosX:r.cameraPosX,cameraPosY:r.cameraPosY,cameraPosZ:r.cameraPosZ}}var X=document.createElement("div");X.style.boxSizing="border-box";X.style.padding="16px";X.style.fontFamily="system-ui, sans-serif";X.style.color="#e8e8e8";X.style.background="#111";X.style.minHeight="100vh";var it=document.createElement("h2");it.textContent="WebGPU Fast Hydraulic + Thermal Erosion (Jako)";it.style.margin="0 0 12px 0";it.style.fontWeight="600";var ee=document.createElement("div");ee.style.display="flex";ee.style.flexWrap="wrap";ee.style.alignItems="center";ee.style.gap="8px";ee.style.marginBottom="14px";var se=document.createElement("label");se.textContent="DEM PNG";se.style.display="inline-flex";se.style.alignItems="center";se.style.gap="8px";var L=document.createElement("input");L.type="file";L.accept=".png,image/png";L.style.padding="6px 8px";L.style.border="1px solid #444";L.style.borderRadius="6px";L.style.background="#1a1a1a";L.style.color="#e8e8e8";se.appendChild(L);var Tr=c("Height floor",cr,"88px"),Cr=c("Height ceiling",dr,"88px"),N=c("Sim steps/frame",Q,"112px",{min:1,step:1}),st=c("Single-step size",je,"118px",{min:1,step:1}),ke=c("Time step",bt,"88px",{min:.001,step:.001}),oe=c("Background rain",St,"108px",{min:0,step:1e-4}),De=c("Evaporation",wt,"96px",{min:0,step:.001}),Oe=c("Water flow width",Pt,"112px",{min:.1,step:.1}),Ue=c("Gravity",Bt,"84px",{min:.1,step:.1}),le=c("Sediment carry",Et,"104px",{min:0,step:.01}),ue=c("Erode into water",Tt,"112px",{min:0,step:.01}),ce=c("Deposit from water",Ct,"118px",{min:0,step:.01}),de=c("Bank softening",Mt,"108px",{min:0,step:.01}),he=c("Max cut depth",Rt,"104px",{min:.001,step:.01}),pe=c("Slope collapse",At,"104px",{min:0,step:.01}),me=c("Slope limit scale",Lt,"118px",{min:0,step:.01}),fe=c("Slope limit bias",Ft,"110px",{min:0,step:.01}),ot=c("Terrain exaggeration",kt,"130px",{min:0,step:.05}),Ie=c("Water height scale",Je,"122px",{min:0,step:.01}),lt=c("Water visibility",Dt,"112px",{min:0,max:1,step:.05}),ut=c("Sediment boost",Ot,"104px",{min:0,max:2,step:.05}),Yt=c("Base resistance",hr,"108px",{min:.05,max:1,step:.01}),$t=c("Resistance variation",pr,"122px",{min:0,max:1,step:.01}),J=Le("Enable springs",!1),qt=c("Spring X %",Ut,"92px",{min:0,max:100,step:1}),Xt=c("Spring Y %",It,"92px",{min:0,max:100,step:1}),ge=c("Spring radius px",Wt,"108px",{min:1,step:1}),xe=c("Spring flow rate",_t,"104px",{min:0,step:.005}),ye=c("Rain duration s",Qe,"104px",{min:0,step:.5}),We=c("Pulse 2 sec",et,"92px",{min:0,step:.5}),ve=Fe("Spring mode",[{value:0,label:"painted springs"},{value:1,label:"fixed random springs"}],mr),be=c("Spring count",tt,"96px",{min:1,max:16,step:1}),_e=c("Spring seed",Gt,"96px",{min:0,step:1}),ct=c("Meters / pixel",rt,"104px",{min:1,step:1}),Kt=Fe("Sim tess",[{value:1,label:"1x"},{value:2,label:"2x"},{value:4,label:"4x"}],yr),dt=Le("8-way hydraulic pipes",Nr),Se=Le("Enable precipitation",Vr),we=Le("Enable thermal erosion",Hr),Ge=Le("Enable hydraulic erosion",Yr),ae=Fe("View",[{value:0,label:"shaded"},{value:2,label:"water"},{value:3,label:"sediment"},{value:4,label:"hardness"},{value:5,label:"thermal"},{value:6,label:"erosion/deposition history"}],0),Mr=Fe("Preset",[{value:"paper_balanced",label:"paper balanced"},{value:"river_cut",label:"river cut"},{value:"thermal_heavy",label:"thermal heavy"},{value:"gentle_weathering",label:"gentle weathering"},{value:"rapid_incision",label:"rapid incision"},{value:"flash_flood",label:"flash flood"},{value:"badlands",label:"badlands"},{value:"canyon_carver",label:"canyon carver"}],"paper_balanced"),ne=c("Azimuth",at,"84px",{min:-180,max:180,step:1}),ie=c("Elevation",nt,"84px",{min:5,max:89,step:1}),U=Fe("Paint",[{value:"none",label:"off"},{value:"raise",label:"terrain +"},{value:"lower",label:"terrain -"},{value:"spring_add",label:"spring +"},{value:"spring_erase",label:"spring erase"}],"none"),ht=c("Brush radius px",zt,"112px",{min:1,step:1}),pt=c("Brush amount",Nt,"104px",{min:.001,step:.005}),mt=c("Brush hardness",Vt,"116px",{min:.05,max:1,step:.05}),te=Y("Run");te.disabled=!0;var ze=Y("Step");ze.disabled=!0;var Rr=Y("Apply preset"),ft=Y("Reset rain timer");ft.disabled=!0;var Ne=Y("Reset DEM");Ne.disabled=!0;var Ar=Y("Clear"),G=Y("Export DEM PNG");G.disabled=!0;var Pe=Y("Clear painted springs");Pe.disabled=!0;var Ve=document.createElement("div");Ve.style.marginLeft="12px";Ve.style.opacity="0.9";Ve.style.fontSize="14px";ee.append(se,Mr,Rr,te,ze,G,Ne,Ar,Ve);var Be=document.createElement("div");Be.style.display="grid";Be.style.gridTemplateColumns="minmax(320px, 1fr) minmax(280px, 380px)";Be.style.gap="16px";Be.style.alignItems="start";var K=document.createElement("div");K.style.background="#181818";K.style.border="1px solid #2d2d2d";K.style.borderRadius="12px";K.style.padding="12px";K.style.boxSizing="border-box";K.style.minWidth="0";var He=document.createElement("div");He.textContent="Render view:";He.style.fontSize="15px";He.style.fontWeight="600";He.style.marginBottom="8px";var F=document.createElement("div");F.style.width="100%";F.style.overflow="hidden";F.style.border="1px solid #333";F.style.borderRadius="10px";F.style.background="#0d0d0d";F.style.position="relative";F.style.aspectRatio="3 / 1.5";var g=document.createElement("canvas");g.width=960;g.height=640;g.style.display="block";g.style.width="100%";g.style.height="100%";g.style.background="#0d0d0d";g.style.position="absolute";g.style.inset="0";g.style.zIndex="0";var u=document.createElement("canvas");u.width=960;u.height=640;u.style.display="block";u.style.width="100%";u.style.height="100%";u.style.background="#0d0d0d";u.style.position="absolute";u.style.inset="0";u.style.zIndex="1";u.style.visibility="hidden";u.style.touchAction="none";u.tabIndex=0;var d=document.createElement("canvas");d.width=960;d.height=640;d.style.display="block";d.style.width="100%";d.style.height="100%";d.style.position="absolute";d.style.inset="0";d.style.zIndex="2";d.style.pointerEvents="none";d.style.visibility="hidden";F.append(g,u,d);K.append(He,F);var z=document.createElement("div");z.style.background="#181818";z.style.border="1px solid #2d2d2d";z.style.borderRadius="12px";z.style.padding="12px";z.style.boxSizing="border-box";z.style.maxHeight="85vh";z.style.overflowY="scroll";var Ye=document.createElement("div");Ye.textContent="Stats";Ye.style.fontSize="15px";Ye.style.fontWeight="600";Ye.style.marginBottom="8px";var V=document.createElement("div");V.style.whiteSpace="pre-line";V.style.fontSize="13px";V.style.opacity="0.95";V.style.lineHeight="1.45";V.textContent="Load a DEM PNG to begin.";var $e=document.createElement("div");$e.style.display="grid";$e.style.gap="10px";$e.style.marginBottom="12px";var Zt=$("Process toggles");Zt.append(q("Master on/off switches for precipitation, springs, hydraulic erosion, and thermal erosion."));Zt.body.append(Se,J,we,Ge,ye,ft);var jt=$("Shared water flow and simulation");jt.append(q("These affect precipitation, springs, water flow, and general simulation pacing. They are not thermal-only."));jt.body.append(Tr,Cr,N,st,ke,oe,De,Oe,Ue,Kt,dt,ct);var Jt=$("Hydraulic erosion only");Jt.append(q("These only change hydraulic erosion and deposition behavior. Water can still animate with hydraulic erosion disabled."));Jt.body.append(le,ue,ce,de,he,Yt,$t);var Qt=$("Thermal erosion only");Qt.append(q("These only affect slope collapse and talus behavior."));Qt.body.append(pe,me,fe);var er=$("Springs");er.append(q("Spring mode and spring-source controls. Random-spring controls only apply in fixed random springs mode."));er.body.append(ve,ge,xe,We,be,_e,Pe);var tr=$("View and shading");tr.append(q("Render-only controls. These do not change erosion behavior."));tr.body.append(ae,ot,Ie,lt,ut);var rr=$("Camera and navigation");rr.append(q("Orbit inputs below. Click the canvas with paint off for pointer look, then use WASD to move, Space to go up, C to go down, and Esc to release."));rr.body.append(ne,ie);var ar=$("Painting");ar.append(q("Terrain and spring painting tools. Pointer look is disabled while a paint mode is active."));ar.body.append(U,ht,pt,mt);$e.append(rr,tr,Zt,jt,Jt,Qt,er,ar);z.append($e,Ye,V);Be.append(K,z);X.append(it,ee,Be);document.body.style.margin="0";document.body.style.background="#111";document.body.appendChild(X);window.addEventListener("beforeunload",()=>{Te(),r.worker&&(x("destroy"),r.worker.terminate(),r.worker=null,r.workerReady=!1)});window.addEventListener("resize",()=>{gt(),Er(),x("render")});L.addEventListener("change",()=>{let e=L.files?.[0];e&&aa(e)});Rr.addEventListener("click",()=>{Ir(Mr.input.value)});te.addEventListener("click",()=>{r.running?Te():sa()});ft.addEventListener("click",async()=>{if(!r.workerReady||!r.gpuStats?.ready)return;E();let e=await O("resetRainTimer");e.stats&&(r.gpuStats=e.stats),Array.isArray(e.sourcePoints)&&(r.sourcePoints=e.sourcePoints),m("Rain timer reset"),h()});ze.addEventListener("click",async()=>{if(!r.workerReady||!r.gpuStats?.ready)return;E();let e=await O("step",{iterations:Math.max(1,Math.floor(s(st,je))),render:!0});e.stats&&(r.gpuStats=e.stats),Array.isArray(e.sourcePoints)&&(r.sourcePoints=e.sourcePoints),await or(!0),h()});N.input.addEventListener("input",async()=>{let e=Math.max(1,Math.floor(s(N,Q)));h(),r.workerReady&&r.running&&await O("startLoop",{iterationsPerFrame:e})});G.addEventListener("click",async()=>{if(!(!r.workerReady||!r.gpuStats?.ready)){G.disabled=!0,m("Exporting DEM PNG\u2026");try{let e=await O("exportTerrainPng"),t=e.data instanceof ArrayBuffer?e.data:e.data?.buffer;if(!(t instanceof ArrayBuffer))throw new Error("Terrain export did not return PNG data.");let a=new Blob([t],{type:"image/png"}),n=(r.sourceFileName||"terrain").replace(/\.[^.]+$/,""),i=document.createElement("a"),o=URL.createObjectURL(a);i.href=o,i.download=`${n}_dem_export.png`,document.body.appendChild(i),i.click(),i.remove(),setTimeout(()=>URL.revokeObjectURL(o),0);let l=Number.isFinite(e.minTerrain)?e.minTerrain.toFixed(4):"n/a",f=Number.isFinite(e.maxTerrain)?e.maxTerrain.toFixed(4):"n/a";m(`Exported DEM PNG (${e.width}x${e.height}, range ${l} to ${f})`)}catch(e){m(e instanceof Error?e.message:String(e))}finally{G.disabled=!r.gpuStats?.ready}}});Pe.addEventListener("click",async()=>{if(!r.workerReady)return;let e=await O("clearPaintedSprings");e.stats&&(r.gpuStats=e.stats),Array.isArray(e.sourcePoints)&&(r.sourcePoints=e.sourcePoints),m("Cleared painted springs"),h()});U.input.addEventListener("change",()=>{qe(),h()});for(let e of[Se,we,Ge])e.input.addEventListener("change",()=>{Ee(),E(),x("render"),h()});for(let e of[ht,pt,mt])e.input.addEventListener("input",()=>{h()});Ne.addEventListener("click",()=>{ir()});Ar.addEventListener("click",lr);for(let e of[ke,oe,De,Oe,Ue,le,ue,ce,de,he,pe,me,fe,ot,Ie,lt,ut,Yt,$t,ne,ie,qt,Xt,ge,xe,ye,We,be,_e])e.input.addEventListener("input",()=>{E(),r.running?x("startLoop",{iterationsPerFrame:Math.max(1,Math.floor(s(N,Q)))}):x("render"),h()});J.input.addEventListener("change",()=>{Ee(),E(),x("render"),h()});ve.input.addEventListener("change",()=>{Ee(),E(),x("render"),h()});dt.input.addEventListener("change",()=>{E(),r.running?x("startLoop",{iterationsPerFrame:Math.max(1,Math.floor(s(N,Q)))}):x("render"),h()});ae.input.addEventListener("change",()=>{E(),x("render"),h()});Kt.input.addEventListener("change",()=>{h(),r.sourceFile&&ir()});Ir("paper_balanced");gt();Ze(!1);m();Ee();qe();H();function Lr(){return Number(ve.input.value)||0}function Jr(){return Lr()===1?"fixed random springs":"painted springs"}function Fr(){return U.input.value==="none"}function qe(){let e=U.input.value!=="none";document.pointerLockElement===u&&e&&document.exitPointerLock?.(),u.style.cursor=e?"crosshair":document.pointerLockElement===u?"none":"grab"}function Qr(){return[r.cameraPosX,r.cameraPosY,r.cameraPosZ]}function nr(){let e=Ke(s(ne,at)),t=Ke(s(ie,nt)),a=Qr(),n=re([-Math.cos(t)*Math.cos(e),-Math.sin(t),-Math.cos(t)*Math.sin(e)]),i=re([n[0],0,n[2]]),o=re(xt(i,[0,1,0])),l=[a[0]+n[0],a[1]+n[1],a[2]+n[2]];return{azimuth:e,elevation:t,eye:a,target:l,forward:n,forwardFlat:i,right:o}}function ea(){return!!(r.navKeys.KeyW||r.navKeys.KeyA||r.navKeys.KeyS||r.navKeys.KeyD||r.navKeys.Space||r.navKeys.KeyC)}function kr(e=performance.now()){if(!r.pointerLookActive){r.navLoopHandle=0,r.navLastAt=0;return}let t=r.navLastAt>0?Math.min(.05,Math.max(.001,(e-r.navLastAt)/1e3)):1/60;if(r.navLastAt=e,ea()){let a=nr(),n=Gr*t,i=0,o=0,l=0;r.navKeys.KeyW&&(i+=a.forwardFlat[0]*n,l+=a.forwardFlat[2]*n),r.navKeys.KeyS&&(i-=a.forwardFlat[0]*n,l-=a.forwardFlat[2]*n),r.navKeys.KeyD&&(i+=a.right[0]*n,l+=a.right[2]*n),r.navKeys.KeyA&&(i-=a.right[0]*n,l-=a.right[2]*n),r.navKeys.Space&&(o+=n),r.navKeys.KeyC&&(o-=n),r.cameraPosX+=i,r.cameraPosY+=o,r.cameraPosZ+=l,E(),x("render"),H(),h()}r.pointerLookActive?r.navLoopHandle=requestAnimationFrame(kr):r.navLoopHandle=0}function Dr(){!r.pointerLookActive||r.navLoopHandle||(r.navLastAt=0,r.navLoopHandle=requestAnimationFrame(kr))}function ta(){if(r.pointerLookActive=document.pointerLockElement===u,r.pointerLookActive)Dr();else{for(let e of Object.keys(r.navKeys))r.navKeys[e]=!1;r.navLoopHandle&&cancelAnimationFrame(r.navLoopHandle),r.navLoopHandle=0,r.navLastAt=0}qe()}function ra(e){if(document.pointerLockElement!==u||!Fr())return;e.preventDefault();let t=zr,a=s(ne,at)+e.movementX*t,n=j(s(ie,nt)+e.movementY*t,5,89);ne.input.value=String(a),ie.input.value=String(n),E(),x("render"),H(),h()}function Or(e,t){document.pointerLockElement===u&&e.code in r.navKeys&&(e.preventDefault(),r.navKeys[e.code]=t,t&&Dr())}document.addEventListener("pointerlockchange",ta);document.addEventListener("pointermove",ra,{passive:!1});document.addEventListener("keydown",e=>Or(e,!0),{passive:!1});document.addEventListener("keyup",e=>Or(e,!1),{passive:!1});u.addEventListener("wheel",e=>{document.pointerLockElement===u&&e.preventDefault()},{passive:!1});function Ee(){let e=!!J.input.checked,t=!!Se.input.checked,a=!!we.input.checked,n=!!Ge.input.checked,i=Lr()===1;oe.input.disabled=!t,le.input.disabled=!n,ue.input.disabled=!n,ce.input.disabled=!n,de.input.disabled=!n,he.input.disabled=!n,pe.input.disabled=!a,me.input.disabled=!a,fe.input.disabled=!a,ye.input.disabled=!t,ve.input.disabled=!e,ge.input.disabled=!e||!i,xe.input.disabled=!e||!i,be.input.disabled=!e||!i,_e.input.disabled=!e||!i,ft.disabled=!r.gpuStats?.ready,qe()}u.addEventListener("contextmenu",e=>{(U.input.value!=="none"||document.pointerLockElement===u)&&e.preventDefault()});u.addEventListener("click",e=>{!r.gpuStats?.ready||!Fr()||(e.preventDefault(),u.focus(),document.pointerLockElement!==u&&u.requestPointerLock?.())});u.addEventListener("pointerdown",e=>{(U.input.value!=="none"||document.pointerLockElement===u)&&r.gpuStats?.ready&&e.preventDefault(),!(U.input.value==="none"||!r.gpuStats?.ready)&&(r.isPainting=!0,u.setPointerCapture?.(e.pointerId),Ur(e))});u.addEventListener("pointermove",e=>{r.isPainting&&(e.preventDefault(),Ur(e))});u.addEventListener("pointerup",sr);u.addEventListener("pointerleave",sr);u.addEventListener("pointercancel",sr);function Ze(e){u.style.visibility=e?"visible":"hidden",d.style.visibility=e?"visible":"hidden",g.style.visibility=e?"hidden":"visible",H(),qe()}function gt(){let e=F.getBoundingClientRect(),t=Math.max(1,Math.round(e.width||960)),a=Math.max(1,Math.round(e.height||Math.round(t*2/3))),n=Math.max(1,Math.min(window.devicePixelRatio||1,2)),i=Math.max(1,Math.round(t*n)),o=Math.max(1,Math.round(a*n));(g.width!==i||g.height!==o)&&(g.width=i,g.height=o),(d.width!==i||d.height!==o)&&(d.width=i,d.height=o),H()}async function aa(e){Ht("loadSourceImage begin",e.name),Te(),Ze(!1),gt(),r.sourceFile=e,r.sourceFileName=e.name,r.sourceImageInfo=null,r.sourceUploadedToWorker=!1,r.gpuStats=null,r.sourcePoints=[],r.lastFrameMs=0,Ne.disabled=!1,G.disabled=!0,Pe.disabled=!0,m("PNG selected, initializing GPU\u2026"),h();try{await ir()}catch(t){lr(),m(t instanceof Error?t.message:String(t))}}async function ir(){if(r.sourceFile){Te(),G.disabled=!0,m("Initializing WebGPU\u2026");try{gt(),await Zr(),Er(),E();let e={tessellation:vr(),options:{minHeight:s(Tr,cr),maxHeight:s(Cr,dr)}};r.sourceUploadedToWorker||(e.blob=r.sourceFile);let t=await O("loadDEMImage",e);r.sourceUploadedToWorker=!0,r.buildCount++,r.sourceImageInfo=t.sourceImageInfo??r.sourceImageInfo,r.gpuStats=t.stats??r.gpuStats,r.sourcePoints=t.sourcePoints??r.sourcePoints,te.disabled=!1,ze.disabled=!1,G.disabled=!1,Pe.disabled=!1,Ee(),Ze(!0),m("GPU sim ready"),h()}catch(e){console.error("[WebGPU Erosion UI] initializeSimulation failed",e),Ze(!1),m(e instanceof Error?e.message:String(e))}}}function E(){let e=jr();r.workerReady&&x("setParams",{params:e})}function na(){return{mode:U.input.value,radius:Math.max(1,s(ht,zt)),amount:Math.max(.001,s(pt,Nt)),hardness:j(s(mt,Vt),.05,1)}}function ia(e){let t=r.gpuStats;if(!t?.ready||!t.width||!t.height)return null;let a=u.getBoundingClientRect(),n=(e.clientX-a.left)/Math.max(a.width,1),i=(e.clientY-a.top)/Math.max(a.height,1);if(n<0||n>1||i<0||i>1)return null;let o=nr(),l=t.width>1||t.height>1?2/Math.max(t.width-1,t.height-1,1):1,f=Pr(o.eye,o.target,[0,1,0]),k=wr(Ke(50),Math.max(a.width/Math.max(a.height,1),1e-6),.01,32),T=qr(Sr(k,f));if(!T)return null;let R=n*2-1,y=1-i*2,b=yt(T,[R,y,-1,1]),S=yt(T,[R,y,1,1]),C=[b[0]/b[3],b[1]/b[3],b[2]/b[3]],w=[S[0]/S[3],S[1]/S[3],S[2]/S[3]],P=re(br(w,C)),B,M,D=0;if(Math.abs(P[1])>1e-5){let v=(D-C[1])/P[1];v>0&&(B=C[0]+P[0]*v,M=C[2]+P[2]*v)}if(!Number.isFinite(B)||!Number.isFinite(M)){let v=n*Math.max(t.width-1,0),Z=i*Math.max(t.height-1,0);return{x:j(v,0,Math.max(t.width-1,0)),y:j(Z,0,Math.max(t.height-1,0))}}let I=(t.width-1)*.5,A=(t.height-1)*.5,W=B/Math.max(l,1e-6)+I,_=A-M/Math.max(l,1e-6);return{x:j(W,0,Math.max(t.width-1,0)),y:j(_,0,Math.max(t.height-1,0))}}async function Ur(e){let t=na();if(t.mode==="none"||!r.workerReady||!r.gpuStats?.ready)return;let a=performance.now();if(a-r.lastPaintAt<20)return;r.lastPaintAt=a;let n=ia(e);if(n){if(r.running&&Te(),t.mode==="raise"||t.mode==="lower"){let i=await O("paintTerrainBrush",{brush:{x:n.x,y:n.y,radius:t.radius,amount:t.amount,hardness:t.hardness,subtract:t.mode==="lower"}});i.stats&&(r.gpuStats=i.stats),Array.isArray(i.sourcePoints)&&(r.sourcePoints=i.sourcePoints),m(`${t.mode==="raise"?"Raised":"Lowered"} terrain at ${n.x.toFixed(1)}, ${n.y.toFixed(1)}`)}else{let i=await O("paintSpringBrush",{brush:{x:n.x,y:n.y,radius:t.radius,strength:t.amount,hardness:t.hardness,erase:t.mode==="spring_erase"}});i.stats&&(r.gpuStats=i.stats),Array.isArray(i.sourcePoints)&&(r.sourcePoints=i.sourcePoints),m(`${t.mode==="spring_erase"?"Erased":"Painted"} spring at ${n.x.toFixed(1)}, ${n.y.toFixed(1)}`)}h()}}function sr(){r.isPainting&&(r.isPainting=!1,or(!0))}function Ir(e){let t={paper_balanced:{iterationsPerFrame:5,stepIterations:128,timeStep:.02,rainRate:.001,evaporationRate:.015,pipeArea:20,gravity:9.81,capacityScale:1,suspensionRate:.5,depositionRate:.92,softeningRate:5,maxErosionDepth:.12,thermalRate:.45,talusCoeff:.8,talusBias:.08,sourceStrength:.06,sourceRadius:4,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.06,renderMode:0},river_cut:{iterationsPerFrame:5,stepIterations:160,timeStep:.02,rainRate:.0015,evaporationRate:.012,pipeArea:22,gravity:9.81,capacityScale:1.15,suspensionRate:.7,depositionRate:.75,softeningRate:4,maxErosionDepth:.1,thermalRate:.3,talusCoeff:.72,talusBias:.07,sourceStrength:.09,sourceRadius:4,sourceEnabled:!0,sourceLayoutMode:1,randomSpringCount:4,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.06,renderMode:0},thermal_heavy:{iterationsPerFrame:5,stepIterations:128,timeStep:.02,rainRate:0,evaporationRate:.008,pipeArea:8,gravity:9.81,capacityScale:.1,suspensionRate:.05,depositionRate:.85,softeningRate:1,maxErosionDepth:.06,thermalRate:1.1,talusCoeff:.55,talusBias:.05,sourceStrength:0,sourceRadius:4,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.06,renderMode:5},gentle_weathering:{iterationsPerFrame:5,stepIterations:96,timeStep:.02,rainRate:.004,evaporationRate:.015,pipeArea:12,gravity:9.81,capacityScale:.45,suspensionRate:.18,depositionRate:1,softeningRate:2,maxErosionDepth:.1,thermalRate:.12,talusCoeff:.8,talusBias:.1,sourceStrength:0,sourceRadius:4,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.06,renderMode:0},rapid_incision:{iterationsPerFrame:5,stepIterations:192,timeStep:.022,rainRate:.018,evaporationRate:.01,pipeArea:26,gravity:9.81,capacityScale:1.4,suspensionRate:.9,depositionRate:.75,softeningRate:6,maxErosionDepth:.1,thermalRate:.12,talusCoeff:.72,talusBias:.08,sourceStrength:0,sourceRadius:4,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.06,renderMode:0},flash_flood:{iterationsPerFrame:5,stepIterations:192,timeStep:.02,rainRate:.03,evaporationRate:.008,pipeArea:24,gravity:9.81,capacityScale:1.3,suspensionRate:.85,depositionRate:.8,softeningRate:5,maxErosionDepth:.08,thermalRate:.1,talusCoeff:.75,talusBias:.08,sourceStrength:0,sourceRadius:4,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:50,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.06,renderMode:0},badlands:{iterationsPerFrame:5,stepIterations:160,timeStep:.02,rainRate:.01,evaporationRate:.012,pipeArea:18,gravity:9.81,capacityScale:1,suspensionRate:.55,depositionRate:.95,softeningRate:4,maxErosionDepth:.09,thermalRate:.22,talusCoeff:.68,talusBias:.08,sourceStrength:0,sourceRadius:4,sourceEnabled:!1,sourceLayoutMode:0,randomSpringCount:1,rainDuration:25,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.06,renderMode:0},canyon_carver:{iterationsPerFrame:5,stepIterations:224,timeStep:.022,rainRate:.014,evaporationRate:.01,pipeArea:28,gravity:9.81,capacityScale:1.55,suspensionRate:1,depositionRate:.7,softeningRate:6,maxErosionDepth:.08,thermalRate:.16,talusCoeff:.72,talusBias:.08,sourceStrength:.08,sourceRadius:3,sourceEnabled:!0,sourceLayoutMode:1,randomSpringCount:4,rainDuration:0,pulse2Duration:0,metersPerPixel:100,waterHeightScale:.06,renderMode:0}},a=t[e]||t.paper_balanced;N.input.value=String(a.iterationsPerFrame??Q),st.input.value=String(a.stepIterations??je),ke.input.value=String(a.timeStep),oe.input.value=String(a.rainRate),De.input.value=String(a.evaporationRate),Oe.input.value=String(a.pipeArea),Ue.input.value=String(a.gravity),le.input.value=String(a.capacityScale),ue.input.value=String(a.suspensionRate),ce.input.value=String(a.depositionRate),de.input.value=String(a.softeningRate),he.input.value=String(a.maxErosionDepth),pe.input.value=String(a.thermalRate),me.input.value=String(a.talusCoeff),fe.input.value=String(a.talusBias),xe.input.value=String(a.sourceStrength),ye.input.value=String(a.rainDuration??Qe),We.input.value=String(a.pulse2Duration??et),ge.input.value=String(a.sourceRadius),Se.input.checked=(a.rainRate??0)>0,we.input.checked=(a.thermalRate??0)>0,Ge.input.checked=!0,J.input.checked=!!a.sourceEnabled,ve.input.value=String(a.sourceLayoutMode??mr),be.input.value=String(a.randomSpringCount??tt),ct.input.value=String(a.metersPerPixel??rt),Ie.input.value=String(a.waterHeightScale??Je),ae.input.value=String(a.renderMode),Ee(),E(),x("render"),h()}async function or(e=!1){if(!r.workerReady||r.readbackPending)return;let t=performance.now();if(!(!e&&t-r.lastReadbackAt<500)){r.readbackPending=!0;try{let a=await O("readbackStats");r.gpuStats=a.stats??r.gpuStats,r.sourcePoints=a.sourcePoints??r.sourcePoints,r.sourceImageInfo=a.sourceImageInfo??r.sourceImageInfo,Number.isFinite(a.lastFrameMs)&&(r.lastFrameMs=a.lastFrameMs),r.lastReadbackAt=performance.now()}catch(a){Ht("readback failed",a)}finally{r.readbackPending=!1}}}function sa(){!r.workerReady||r.running||(r.running=!0,te.textContent="Pause",E(),x("startLoop",{iterationsPerFrame:Math.max(1,Math.floor(s(N,Q)))}),or(!0),m("Running"))}function Te(){r.running&&(r.running=!1,te.textContent="Run",x("stopLoop"),m("Paused"))}function lr(){Te(),document.exitPointerLock?.(),x("clear"),r.sourceImageInfo=null,r.sourceFile=null,r.sourceUploadedToWorker=!1,r.buildCount=0,r.lastFrameMs=0,r.readbackPending=!1,r.lastReadbackAt=0,r.gpuStats=null,r.sourcePoints=[],r.sourceFileName="",r.cameraPosX=fr,r.cameraPosY=gr,r.cameraPosZ=xr,L.value="",te.disabled=!0,ze.disabled=!0,G.disabled=!0,Pe.disabled=!0,Ne.disabled=!0,m(),h(),H();let e=g.getContext("2d");e&&(e.fillStyle="#0d0d0d",e.fillRect(0,0,g.width,g.height),e.fillStyle="rgba(255,255,255,0.55)",e.font="16px system-ui, sans-serif",e.fillText("Load a DEM PNG to initialize the worker-owned WebGPU erosion sim.",24,36))}function H(){let e=d.getContext("2d");if(!e||(e.clearRect(0,0,d.width,d.height),d.style.visibility==="hidden"))return;let t=r.gpuStats,a=r.sourcePoints||[];if(!t?.ready||!t.width||!t.height||a.length===0)return;let n=nr(),i=Pr(n.eye,n.target,[0,1,0]),o=wr(Ke(50),Math.max(d.width/Math.max(d.height,1),1e-6),.01,32),l=Sr(o,i),f=t.width>1||t.height>1?2/Math.max(t.width-1,t.height-1,1):1,k=(t.width-1)*.5,T=(t.height-1)*.5,R=Math.max(2,Math.min(6,Math.min(d.width,d.height)/Math.max(t.width,t.height,1)*.85));for(let y of a){let b=(y.x-k)*f,S=(T-y.y)*f,w=yt(l,[b,0,S,1]);if(!w[3]||w[3]<=0)continue;let P=w[0]/w[3],B=w[1]/w[3];if(Math.abs(P)>1.2||Math.abs(B)>1.2)continue;let M=(P*.5+.5)*d.width,D=(1-(B*.5+.5))*d.height,I=!!y.painted,A=y.active!==!1;if(I){let _=j(Number(y.strength)||0,0,1),v=R*(.8+.9*_);e.fillStyle=A?"rgba(176, 96, 255, 0.98)":"rgba(120, 96, 150, 0.85)",e.fillRect(Math.round(M-v),Math.round(D-v),Math.max(1,Math.round(v*2)),Math.max(1,Math.round(v*2)));continue}let W=Math.max(5,(y.radius||2)*Math.min(d.width,d.height)/Math.max(t.width,t.height,1)*1.2);e.beginPath(),e.arc(M,D,W,0,Math.PI*2),e.fillStyle=A?"rgba(60, 220, 255, 0.30)":"rgba(90, 110, 120, 0.18)",e.fill(),e.lineWidth=2,e.strokeStyle=A?"rgba(138, 240, 255, 0.95)":"rgba(140, 165, 175, 0.55)",e.stroke()}}function m(e=""){let t=[];e&&t.push(e),t.push(r.sourceFile?`image: ${r.sourceFileName||"loaded"}`:"image: none"),t.push(`webgpu: ${r.workerReady?"ready":"idle"}`),r.running&&t.push("sim: running"),Ve.textContent=t.join("  |  ")}function h(){if(!r.sourceFile&&!r.sourceImageInfo){V.textContent="Load a DEM PNG to begin.",H();return}let e=r.sourceImageInfo?.width||r.gpuStats?.width||0,t=r.sourceImageInfo?.height||r.gpuStats?.height||0,a=r.gpuStats?.width||e,n=r.gpuStats?.height||t,i=r.gpuStats??{width:0,height:0,cellCount:0,ready:!1,iterationCount:0},o=[`file: ${r.sourceFileName||"(loaded image)"}`,`image size: ${e} x ${t}`,`sim tess: ${vr()}x`,`sim grid: ${i.width||a} x ${i.height||n}`,`cells: ${i.cellCount||a*n}`,`gpu ready: ${i.ready?"yes":"no"}`,`build count: ${r.buildCount}`,`iterations: ${i.iterationCount||0}`,`sim time: ${Number.isFinite(i.simTime)?i.simTime.toFixed(2):"0.00"} s`,`rain active: ${i.rainActive?"yes":"no"}`,`precipitation: ${Se.input.checked?"on":"off"}`,`spring toggle: ${J.input.checked?"on":"off"}`,`thermal erosion: ${we.input.checked?"on":"off"}`,`spring mode: ${Jr()}`,`painted springs: ${(r.sourcePoints||[]).filter(l=>l.painted).length}`,`random spring count: ${Math.max(1,Math.floor(s(be,tt)))}`,`spring seed: ${Math.floor(s(_e,Gt))}`,`spring centers: ${(r.sourcePoints||[]).slice(0,6).map(l=>`${l.painted?"p:":"r:"}${l.x.toFixed(1)},${l.y.toFixed(1)}`).join(" | ")||"n/a"}`,`meters / pixel: ${s(ct,rt).toFixed(0)}`,`running: ${r.running?"yes":"no"}`,`pointer look: ${r.pointerLookActive?"locked":"off"}`,`camera pos: ${r.cameraPosX.toFixed(2)}, ${r.cameraPosY.toFixed(2)}, ${r.cameraPosZ.toFixed(2)}`,`paint mode: ${U.input.options[U.input.selectedIndex]?.textContent||"off"}`,`brush radius: ${s(ht,zt).toFixed(1)}`,`brush amount: ${s(pt,Nt).toFixed(3)}`,`brush hardness: ${s(mt,Vt).toFixed(2)}`,`last frame: ${r.lastFrameMs.toFixed(3)} ms`,`readback pending: ${r.readbackPending?"yes":"no"}`,"",`terrain range: ${Xe(i.terrainRange)}`,`water range: ${Xe(i.waterRange)}`,`sediment range: ${Xe(i.sedimentRange)}`,`history range: ${Xe(i.historyRange)}`,`total water: ${Number.isFinite(i.totalWater)?i.totalWater.toFixed(4):"n/a"}`,`total sediment: ${Number.isFinite(i.totalSediment)?i.totalSediment.toFixed(4):"n/a"}`,`avg hardness: ${Number.isFinite(i.averageHardness)?i.averageHardness.toFixed(4):"n/a"}`,"",`\u0394t: ${s(ke,bt).toFixed(3)}`,`Kr rain: ${s(oe,St).toFixed(4)}`,`Ke evap: ${s(De,wt).toFixed(3)}`,`A pipe: ${s(Oe,Pt).toFixed(2)}`,`hydraulic pipes: ${dt.input.checked?"8-way":"4-way"}`,`g gravity: ${s(Ue,Bt).toFixed(2)}`,`Kc capacity: ${s(le,Et).toFixed(3)}`,`Ks suspend: ${s(ue,Tt).toFixed(3)}`,`Kd deposit: ${s(ce,Ct).toFixed(3)}`,`Kh soften: ${s(de,Mt).toFixed(3)}`,`depth cap: ${s(he,Rt).toFixed(3)}`,`Kt thermal: ${s(pe,At).toFixed(3)}`,`Ka talus: ${s(me,Lt).toFixed(3)}`,`Ki talus bias: ${s(fe,Ft).toFixed(3)}`,`render h: ${s(ot,kt).toFixed(3)}`,`water h: ${s(Ie,Je).toFixed(3)}`,`water \u03B1: ${s(lt,Dt).toFixed(3)}`,`sediment tint: ${s(ut,Ot).toFixed(3)}`,`springs: ${J.input.checked?"on":"off"}`,`rain duration s: ${s(ye,Qe).toFixed(2)}`,`pulse 2 s: ${s(We,et).toFixed(2)}`,`source center: ${s(qt,Ut).toFixed(0)}%, ${s(Xt,It).toFixed(0)}%`,`source radius: ${s(ge,Wt).toFixed(1)}`,`source strength: ${s(xe,_t).toFixed(3)}`,`view mode: ${ae.input.options[ae.input.selectedIndex]?.textContent||"shaded"}`,`iters/frame: ${Math.max(1,Math.floor(s(N,Q)))}`,`step iters: ${Math.max(1,Math.floor(s(st,je)))}`];V.textContent=o.join(`
`),m(),H()}lr();})();
