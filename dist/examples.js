(()=>{var Qt=Object.defineProperty;var Rt=(c,e,t)=>()=>{if(t)throw t[0];try{return c&&(e=c(c=0)),e}catch(i){throw t=[i],i}};var Jt=(c,e)=>{for(var t in e)Qt(c,t,{get:e[t],enumerable:!0})};var Ci,Ai,st,_t=Rt(()=>{Ci='(()=>{function computeLocalRelief(heights,width,height,radius=2,mask=null){if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||heights.length!==width*height||mask&&mask.length!==heights.length||!Number.isFinite(radius))throw new Error("Invalid relief raster dimensions or radius.");radius=Math.max(0,Math.floor(radius));const low=new Float32Array(heights.length),high=new Float32Array(heights.length),result=new Float32Array(heights.length);const minQueue=new Int32Array(Math.max(width,height)),maxQueue=new Int32Array(minQueue.length);for(let y=0;y<height;y++){let a=0,b=0,c=0,d=0,next=0;for(let x=0;x<width;x++){const end=Math.min(width-1,x+radius);while(next<=end){const i=y*width+next,v=heights[i];if((!mask||mask[i])&&Number.isFinite(v)){while(b>a&&heights[y*width+minQueue[b-1]]>=v)b--;while(d>c&&heights[y*width+maxQueue[d-1]]<=v)d--;minQueue[b++]=next;maxQueue[d++]=next}next++}while(b>a&&minQueue[a]<x-radius)a++;while(d>c&&maxQueue[c]<x-radius)c++;low[y*width+x]=b>a?heights[y*width+minQueue[a]]:Infinity;high[y*width+x]=d>c?heights[y*width+maxQueue[c]]:-Infinity}}for(let x=0;x<width;x++){let a=0,b=0,c=0,d=0,next=0;for(let y=0;y<height;y++){const end=Math.min(height-1,y+radius);while(next<=end){const i2=next*width+x;while(b>a&&low[minQueue[b-1]*width+x]>=low[i2])b--;while(d>c&&high[maxQueue[d-1]*width+x]<=high[i2])d--;minQueue[b++]=next;maxQueue[d++]=next;next++}while(b>a&&minQueue[a]<y-radius)a++;while(d>c&&maxQueue[c]<y-radius)c++;const i=y*width+x;if((!mask||mask[i])&&Number.isFinite(heights[i]))result[i]=high[maxQueue[c]*width+x]-low[minQueue[a]*width+x]}}return result}var EPSILON=Math.pow(2,-52);var EDGE_STACK=new Uint32Array(512);function orient2d(ax,ay,bx,by,cx,cy){const acx=ax-cx;const acy=ay-cy;const bcx=bx-cx;const bcy=by-cy;return acy*bcx-acx*bcy}function circumradius(ax,ay,bx,by,cx,cy){const dx=bx-ax;const dy=by-ay;const ex=cx-ax;const ey=cy-ay;const bl=dx*dx+dy*dy;const cl=ex*ex+ey*ey;const determinant=dx*ey-dy*ex;if(determinant===0)return Infinity;const factor=.5/determinant;const x=(ey*bl-dy*cl)*factor;const y=(dx*cl-ex*bl)*factor;return x*x+y*y}function hashKey(dx,dy,hashSize){const absSum=Math.abs(dx)+Math.abs(dy);if(absSum===0)return 0;const p=dx/absSum;return((dy>0?3-p:1+p)*hashSize*.25|0)%hashSize}function insertionSort(ids,dists,left,right){for(let i=left+1;i<=right;i++){const id=ids[i];const d=dists[id];let j=i-1;while(j>=left&&dists[ids[j]]>d){ids[j+1]=ids[j];j--}ids[j+1]=id}}function swap(arr,i,j){const temp=arr[i];arr[i]=arr[j];arr[j]=temp}function quicksort(ids,dists,left,right){if(right-left<=20){insertionSort(ids,dists,left,right);return}const median=left+right>>1;if(dists[ids[left]]>dists[ids[right]])swap(ids,left,right);if(dists[ids[left]]>dists[ids[median]])swap(ids,left,median);if(dists[ids[median]]>dists[ids[right]])swap(ids,median,right);const pivot=ids[median];const pivotDist=dists[pivot];swap(ids,median,right);let partitionIndex=left;for(let i=left;i<right;i++){if(dists[ids[i]]<pivotDist){swap(ids,i,partitionIndex);partitionIndex++}}swap(ids,partitionIndex,right);const leftSize=partitionIndex-left;const rightSize=right-partitionIndex;if(leftSize<rightSize){quicksort(ids,dists,left,partitionIndex-1);quicksort(ids,dists,partitionIndex+1,right)}else{quicksort(ids,dists,partitionIndex+1,right);quicksort(ids,dists,left,partitionIndex-1)}}function defaultGetX(p){return p[0]}function defaultGetY(p){return p[1]}var Delaunator=class _Delaunator{coords;triangles;halfedges;hull;pointCount;capacity;_cx;_cy;_hullStart;_hullPrev;_hullNext;_hullTri;_hullHash;_ids;_dists;_triangles;_halfedges;_hashSize;trianglesLen;static from(points,getX=defaultGetX,getY=defaultGetY,capacity=points.length){const n=points.length;if(capacity<n){throw new RangeError(`capacity ${capacity} is smaller than point count ${n}`)}const coords=new Float64Array(capacity*2);for(let i=0,j=0;i<n;i++,j+=2){const p=points[i];coords[j]=getX(p);coords[j+1]=getY(p)}return new _Delaunator(coords,n)}static withCapacity(capacity){return new _Delaunator(capacity,0)}constructor(coordsOrCapacity,pointCount){if(typeof coordsOrCapacity==="number"){const capacity=Math.max(0,Math.floor(coordsOrCapacity));this.coords=new Float64Array(capacity*2);this.capacity=capacity;this.pointCount=pointCount===void 0?0:Math.max(0,Math.floor(pointCount));if(this.pointCount>this.capacity){throw new RangeError(`pointCount ${this.pointCount} exceeds capacity ${this.capacity}`)}}else{this.coords=coordsOrCapacity;this.capacity=coordsOrCapacity.length>>1;this.pointCount=pointCount===void 0?this.capacity:Math.max(0,Math.floor(pointCount));if(this.pointCount>this.capacity){throw new RangeError(`pointCount ${this.pointCount} exceeds capacity ${this.capacity}`)}}this._allocate(this.capacity);this.update()}get activeCoords(){return this.coords.subarray(0,this.pointCount<<1)}clear(){this.pointCount=0;this.update()}reserve(minCapacity){const targetCapacity=Math.max(0,Math.floor(minCapacity));if(targetCapacity<=this.capacity)return;const nextCoords=new Float64Array(targetCapacity*2);nextCoords.set(this.coords.subarray(0,this.pointCount<<1));this.coords=nextCoords;this.capacity=targetCapacity;this._allocate(targetCapacity)}ensureCapacity(minCapacity){const needed=Math.max(0,Math.floor(minCapacity));if(needed<=this.capacity)return;let next=this.capacity>0?this.capacity:1;while(next<needed)next*=2;this.reserve(next)}setPointCount(pointCount){const count=Math.max(0,Math.floor(pointCount));this.ensureCapacity(count);this.pointCount=count}setPoint(index,x,y){const i=Math.floor(index);if(i<0){throw new RangeError(`index ${index} must be >= 0`)}this.ensureCapacity(i+1);const j=i<<1;this.coords[j]=x;this.coords[j+1]=y;if(i>=this.pointCount){this.pointCount=i+1}}setPointsFlat(source,pointCount=source.length>>1){const count=Math.max(0,Math.floor(pointCount));this.setPointCount(count);const coordCount=count<<1;for(let i=0;i<coordCount;i++){this.coords[i]=source[i]}}update(pointCount=this.pointCount){this.setPointCount(pointCount);const coords=this.coords;const hullPrev=this._hullPrev;const hullNext=this._hullNext;const hullTri=this._hullTri;const hullHash=this._hullHash;const ids=this._ids;const dists=this._dists;const n=this.pointCount;if(n===0){this.hull=new Uint32Array(0);this.trianglesLen=0;this.triangles=this._triangles.subarray(0,0);this.halfedges=this._halfedges.subarray(0,0);return}if(n<3){const hull2=new Uint32Array(n);for(let i=0;i<n;i++)hull2[i]=i;this.hull=hull2;this.trianglesLen=0;this.triangles=this._triangles.subarray(0,0);this.halfedges=this._halfedges.subarray(0,0);return}let minX=Infinity;let minY=Infinity;let maxX=-Infinity;let maxY=-Infinity;for(let i=0,j=0;i<n;i++,j+=2){const x=coords[j];const y=coords[j+1];if(x<minX)minX=x;if(y<minY)minY=y;if(x>maxX)maxX=x;if(y>maxY)maxY=y;ids[i]=i}const centerX=(minX+maxX)*.5;const centerY=(minY+maxY)*.5;let i0=0;let minDist=Infinity;for(let i=0,j=0;i<n;i++,j+=2){const dx2=centerX-coords[j];const dy2=centerY-coords[j+1];const d=dx2*dx2+dy2*dy2;if(d<minDist){i0=i;minDist=d}}const i0c=i0<<1;const i0x=coords[i0c];const i0y=coords[i0c+1];let i1=-1;minDist=Infinity;for(let i=0,j=0;i<n;i++,j+=2){if(i===i0)continue;const dx2=i0x-coords[j];const dy2=i0y-coords[j+1];const d=dx2*dx2+dy2*dy2;if(d>0&&d<minDist){i1=i;minDist=d}}if(i1===-1){this.hull=new Uint32Array([i0]);this.trianglesLen=0;this.triangles=this._triangles.subarray(0,0);this.halfedges=this._halfedges.subarray(0,0);return}let i1c=i1<<1;let i1x=coords[i1c];let i1y=coords[i1c+1];let i2=-1;let minRadius=Infinity;for(let i=0,j=0;i<n;i++,j+=2){if(i===i0||i===i1)continue;const r=circumradius(i0x,i0y,i1x,i1y,coords[j],coords[j+1]);if(r<minRadius){i2=i;minRadius=r}}if(i2===-1||minRadius===Infinity){const baseX=coords[0];const baseY=coords[1];for(let i=0,j2=0;i<n;i++,j2+=2){dists[i]=coords[j2]-baseX||coords[j2+1]-baseY}quicksort(ids,dists,0,n-1);const hull2=new Uint32Array(n);let j=0;let d0=-Infinity;for(let i=0;i<n;i++){const id=ids[i];const d=dists[id];if(d>d0){hull2[j++]=id;d0=d}}this.hull=hull2.subarray(0,j);this.trianglesLen=0;this.triangles=this._triangles.subarray(0,0);this.halfedges=this._halfedges.subarray(0,0);return}let i2c=i2<<1;let i2x=coords[i2c];let i2y=coords[i2c+1];if(orient2d(i0x,i0y,i1x,i1y,i2x,i2y)<0){const tempI=i1;i1=i2;i2=tempI;const tempX=i1x;i1x=i2x;i2x=tempX;const tempY=i1y;i1y=i2y;i2y=tempY;i1c=i1<<1;i2c=i2<<1}const dx=i1x-i0x;const dy=i1y-i0y;const ex=i2x-i0x;const ey=i2y-i0y;const bl=dx*dx+dy*dy;const cl=ex*ex+ey*ey;const determinant=dx*ey-dy*ex;const factor=.5/determinant;const cx=i0x+(ey*bl-dy*cl)*factor;const cy=i0y+(dx*cl-ex*bl)*factor;this._cx=cx;this._cy=cy;for(let i=0,j=0;i<n;i++,j+=2){const ddx=coords[j]-cx;const ddy=coords[j+1]-cy;dists[i]=ddx*ddx+ddy*ddy}quicksort(ids,dists,0,n-1);this._hullStart=i0;let hullSize=3;hullNext[i0]=i1;hullPrev[i2]=i1;hullNext[i1]=i2;hullPrev[i0]=i2;hullNext[i2]=i0;hullPrev[i1]=i0;hullTri[i0]=0;hullTri[i1]=1;hullTri[i2]=2;hullHash.fill(-1);const hashSize=this._hashSize;hullHash[hashKey(i0x-cx,i0y-cy,hashSize)]=i0;hullHash[hashKey(i1x-cx,i1y-cy,hashSize)]=i1;hullHash[hashKey(i2x-cx,i2y-cy,hashSize)]=i2;this.trianglesLen=0;this._addTriangle(i0,i1,i2,-1,-1,-1);let xp=0;let yp=0;for(let k=0;k<n;k++){const i=ids[k];const ic=i<<1;const x=coords[ic];const y=coords[ic+1];if(k>0&&Math.abs(x-xp)<=EPSILON&&Math.abs(y-yp)<=EPSILON)continue;xp=x;yp=y;if(i===i0||i===i1||i===i2)continue;let start=-1;let key=hashKey(x-cx,y-cy,hashSize);for(let j=0;j<hashSize;j++){start=hullHash[key];if(start!==-1&&start!==hullNext[start])break;key++;if(key===hashSize)key=0}if(start===-1)continue;start=hullPrev[start];let e=start;let q=0;while(true){q=hullNext[e];const ec2=e<<1;const qc=q<<1;const qx=coords[qc];const qy=coords[qc+1];if((y-qy)*(coords[ec2]-qx)-(x-qx)*(coords[ec2+1]-qy)<0)break;e=q;if(e===start){e=-1;break}}if(e===-1)continue;let t=this._addTriangle(e,i,hullNext[e],-1,-1,hullTri[e]);hullTri[i]=this._legalize(t+2);hullTri[e]=t;hullSize++;let next=hullNext[e];while(true){q=hullNext[next];const nc=next<<1;const qc=q<<1;const qx=coords[qc];const qy=coords[qc+1];if((y-qy)*(coords[nc]-qx)-(x-qx)*(coords[nc+1]-qy)>=0)break;t=this._addTriangle(next,i,q,hullTri[i],-1,hullTri[next]);hullTri[i]=this._legalize(t+2);hullNext[next]=next;hullSize--;next=q}if(e===start){while(true){q=hullPrev[e];const qc=q<<1;const ec2=e<<1;const ex2=coords[ec2];const ey2=coords[ec2+1];if((y-ey2)*(coords[qc]-ex2)-(x-ex2)*(coords[qc+1]-ey2)>=0)break;t=this._addTriangle(q,i,e,-1,hullTri[e],hullTri[q]);this._legalize(t+2);hullTri[q]=t;hullNext[e]=e;hullSize--;e=q}}this._hullStart=hullPrev[i]=e;hullNext[e]=hullPrev[next]=i;hullNext[i]=next;hullHash[hashKey(x-cx,y-cy,hashSize)]=i;const ec=e<<1;hullHash[hashKey(coords[ec]-cx,coords[ec+1]-cy,hashSize)]=e}const hull=new Uint32Array(hullSize);for(let i=0,e=this._hullStart;i<hullSize;i++){hull[i]=e;e=hullNext[e]}this.hull=hull;this.triangles=this._triangles.subarray(0,this.trianglesLen);this.halfedges=this._halfedges.subarray(0,this.trianglesLen)}_allocate(capacity){const hashSize=Math.max(1,Math.ceil(Math.sqrt(capacity)));this._hashSize=hashSize;this._hullPrev=new Uint32Array(capacity);this._hullNext=new Uint32Array(capacity);this._hullTri=new Uint32Array(capacity);this._hullHash=new Int32Array(hashSize);this._ids=new Uint32Array(capacity);this._dists=new Float64Array(capacity);const maxTriangles=Math.max(2*capacity-5,0);this._triangles=new Uint32Array(maxTriangles*3);this._halfedges=new Int32Array(maxTriangles*3);this.trianglesLen=0}_legalize(a){const triangles=this._triangles;const halfedges=this._halfedges;const coords=this.coords;const hullStart=this._hullStart;const hullTri=this._hullTri;const hullPrev=this._hullPrev;let stackLen=0;let ar=0;while(true){const b=halfedges[a];if(b===-1){if(stackLen===0)break;a=EDGE_STACK[--stackLen];continue}const a0=a-a%3;const aLocal=a-a0;ar=aLocal===0?a0+2:a-1;const al=aLocal===2?a0:a+1;const b0=b-b%3;const bLocal=b-b0;const bl=bLocal===0?b0+2:b-1;const br=bLocal===2?b0:b+1;const p0=triangles[ar];const pr=triangles[a];const pl=triangles[al];const p1=triangles[bl];const p0c=p0<<1;const prc=pr<<1;const plc=pl<<1;const p1c=p1<<1;const ax=coords[p0c];const ay=coords[p0c+1];const bx=coords[prc];const by=coords[prc+1];const cx=coords[plc];const cy=coords[plc+1];const px=coords[p1c];const py=coords[p1c+1];const dx=ax-px;const dy=ay-py;const ex=bx-px;const ey=by-py;const fx=cx-px;const fy=cy-py;const ap=dx*dx+dy*dy;const bp=ex*ex+ey*ey;const cp=fx*fx+fy*fy;const illegal=dx*(ey*cp-bp*fy)-dy*(ex*cp-bp*fx)+ap*(ex*fy-ey*fx)<0;if(illegal){triangles[a]=p1;triangles[b]=p0;const hbl=halfedges[bl];if(hbl===-1){let e=hullStart;do{if(hullTri[e]===bl){hullTri[e]=a;break}e=hullPrev[e]}while(e!==hullStart)}halfedges[a]=hbl;if(hbl!==-1)halfedges[hbl]=a;const har=halfedges[ar];halfedges[b]=har;if(har!==-1)halfedges[har]=b;halfedges[ar]=bl;halfedges[bl]=ar;if(stackLen<EDGE_STACK.length)EDGE_STACK[stackLen++]=br}else{if(stackLen===0)break;a=EDGE_STACK[--stackLen]}}return ar}_addTriangle(i0,i1,i2,a,b,c){const t=this.trianglesLen;const triangles=this._triangles;const halfedges=this._halfedges;triangles[t]=i0;triangles[t+1]=i1;triangles[t+2]=i2;halfedges[t]=a;if(a!==-1)halfedges[a]=t;halfedges[t+1]=b;if(b!==-1)halfedges[b]=t+1;halfedges[t+2]=c;if(c!==-1)halfedges[c]=t+2;this.trianglesLen=t+3;return t}};var EPSILON2=1e-6;var DEFAULT_BUCKET_COUNT=256;var LARGE_DISTANCE=1e9;function clamp(value,min,max){return value<min?min:value>max?max:value}function lerp(a,b,t){return a+(b-a)*t}function smoothstep(edge0,edge1,x){const t=clamp((x-edge0)/Math.max(edge1-edge0,EPSILON2),0,1);return t*t*(3-2*t)}function quantizeRadius(radius,minRadius,maxRadius,bins=16){if(bins<=1||maxRadius<=minRadius+EPSILON2)return radius;const t=clamp((radius-minRadius)/Math.max(maxRadius-minRadius,EPSILON2),0,1);const q=Math.round(t*(bins-1))/(bins-1);return lerp(minRadius,maxRadius,q)}function isFiniteNumber(value){return Number.isFinite(value)}function indexOfCell(x,y,width){return y*width+x}function makeFullMask(width,height){const mask=new Uint8Array(width*height);mask.fill(1);return mask}function isBoundaryCell(mask,width,height,x,y){const i=indexOfCell(x,y,width);if(mask[i]===0)return false;if(x===0||y===0||x===width-1||y===height-1)return true;const up=indexOfCell(x,y-1,width);const down=indexOfCell(x,y+1,width);const left=indexOfCell(x-1,y,width);const right=indexOfCell(x+1,y,width);return mask[up]===0||mask[down]===0||mask[left]===0||mask[right]===0}function scanMinMax(values,mask){let min=Infinity;let max=-Infinity;for(let i=0;i<values.length;i++){if(mask&&mask[i]===0)continue;const v=values[i];if(!isFiniteNumber(v))continue;if(v<min)min=v;if(v>max)max=v}if(!isFiniteNumber(min)||!isFiniteNumber(max)){return{min:0,max:1}}if(Math.abs(max-min)<=EPSILON2){return{min,max:min+1}}return{min,max}}function normalizeArray(values,mask){const out=new Float32Array(values.length);const{min,max}=scanMinMax(values,mask);const inv=1/Math.max(max-min,EPSILON2);for(let i=0;i<values.length;i++){if(mask&&mask[i]===0)continue;out[i]=clamp((values[i]-min)*inv,0,1)}return out}function normalizeAbsArray(values,mask){const absValues=new Float32Array(values.length);for(let i=0;i<values.length;i++){absValues[i]=Math.abs(values[i])}return normalizeArray(absValues,mask)}function bilinearSample(values,width,height,x,y){const x0=clamp(Math.floor(x),0,width-1);const y0=clamp(Math.floor(y),0,height-1);const x1=clamp(x0+1,0,width-1);const y1=clamp(y0+1,0,height-1);const tx=clamp(x-x0,0,1);const ty=clamp(y-y0,0,1);const v00=values[indexOfCell(x0,y0,width)];const v10=values[indexOfCell(x1,y0,width)];const v01=values[indexOfCell(x0,y1,width)];const v11=values[indexOfCell(x1,y1,width)];const a=lerp(v00,v10,tx);const b=lerp(v01,v11,tx);return lerp(a,b,ty)}function nearestMaskSample(mask,width,height,x,y){const gx=clamp(Math.round(x),0,width-1);const gy=clamp(Math.round(y),0,height-1);return mask[indexOfCell(gx,gy,width)]}function ensureImageReady(image){if(image.complete&&image.naturalWidth>0&&image.naturalHeight>0){return Promise.resolve()}if(typeof image.decode==="function"){return image.decode().catch(()=>{return new Promise((resolve,reject)=>{image.onload=()=>resolve();image.onerror=()=>reject(new Error("Failed to decode image source."))})})}return new Promise((resolve,reject)=>{image.onload=()=>resolve();image.onerror=()=>reject(new Error("Failed to load image source."))})}async function rasterFromImage(source,options={}){const valueFromRGBA=options.valueFromRGBA??((r,g,b)=>{return(r+g+b)/(3*255)});let width=0;let height=0;let imageData;if(typeof ImageData!=="undefined"&&source instanceof ImageData){width=source.width;height=source.height;imageData=source}else{if(typeof HTMLImageElement!=="undefined"&&source instanceof HTMLImageElement){await ensureImageReady(source);width=source.naturalWidth||source.width;height=source.naturalHeight||source.height}else if(typeof ImageBitmap!=="undefined"&&source instanceof ImageBitmap){width=source.width;height=source.height}else if(typeof HTMLCanvasElement!=="undefined"&&source instanceof HTMLCanvasElement){width=source.width;height=source.height}else{throw new Error("Unsupported image source.")}let canvas;let ctx;if(typeof OffscreenCanvas!=="undefined"){canvas=new OffscreenCanvas(width,height);ctx=canvas.getContext("2d")}else if(typeof document!=="undefined"){canvas=document.createElement("canvas");canvas.width=width;canvas.height=height;ctx=canvas.getContext("2d")}else{throw new Error("No canvas implementation available for image rasterization.")}if(!ctx){throw new Error("Failed to acquire a 2D canvas context.")}ctx.drawImage(source,0,0,width,height);imageData=ctx.getImageData(0,0,width,height)}const values=new Float32Array(width*height);const mask=new Uint8Array(width*height);const rgba=imageData.data;const flipY=!!options.flipY;const alphaMeansInvalid=!!options.alphaMeansInvalid;for(let y=0;y<height;y++){const srcY=flipY?height-1-y:y;for(let x=0;x<width;x++){const srcIndex=(srcY*width+x)*4;const dstIndex=indexOfCell(x,y,width);const r=rgba[srcIndex];const g=rgba[srcIndex+1];const b=rgba[srcIndex+2];const a=rgba[srcIndex+3];values[dstIndex]=valueFromRGBA(r,g,b,a,dstIndex);mask[dstIndex]=alphaMeansInvalid&&a===0?0:1}}return{width,height,values,mask}}var HeightfieldTINBuilder=class{options;width=0;height=0;heights=null;mask=null;analysis=null;topology=null;constructor(options={}){const cellSize=Math.max(options.cellSize??1,EPSILON2);const minSpacing=Math.max(options.minSpacing??cellSize,EPSILON2);const maxSpacing=Math.max(options.maxSpacing??cellSize*16,minSpacing);this.options={mode:options.mode??"hybrid",cellSize,worldMinX:options.worldMinX??0,worldMinY:options.worldMinY??0,minSpacing,maxSpacing,borderSpacing:Math.max(options.borderSpacing??maxSpacing,EPSILON2),wetnessBins:Math.max(4,Math.floor(options.wetnessBins??21)),channelThreshold:Math.max(1,Math.floor(options.channelThreshold??128)),floodplainChannelDistanceCells:Math.max(1,Math.floor(options.floodplainChannelDistanceCells??6)),candidateStride:Math.max(1,Math.floor(options.candidateStride??1)),maxPoints:Math.max(3,Math.floor(options.maxPoints??Number.MAX_SAFE_INTEGER)),preserveBorder:options.preserveBorder??true,preserveChannels:options.preserveChannels??true,preserveMaskBoundary:options.preserveMaskBoundary??true,preserveInnerBorder:options.preserveInnerBorder??true,slopeWeight:Math.max(0,options.slopeWeight??.35),curvatureWeight:Math.max(0,options.curvatureWeight??.15),reliefWeight:Math.max(0,options.reliefWeight??.2),wetnessWeight:Math.max(0,options.wetnessWeight??.4),channelWeight:Math.max(0,options.channelWeight??.2),floodplainWeight:Math.max(0,options.floodplainWeight??.25),relaxIterations:Math.max(0,Math.floor(options.relaxIterations??0)),forcedWorldPoints:options.forcedWorldPoints??null}}setRaster(raster){if(!raster||raster.width<=0||raster.height<=0){throw new Error("Raster must have positive width and height.")}if(!(raster.values instanceof Float32Array)){throw new Error("Raster values must be a Float32Array.")}if(raster.values.length!==raster.width*raster.height){throw new Error("Raster value count does not match width * height.")}const mask=raster.mask&&raster.mask.length===raster.width*raster.height?raster.mask instanceof Uint8Array?raster.mask:new Uint8Array(raster.mask):makeFullMask(raster.width,raster.height);this.width=raster.width;this.height=raster.height;this.heights=raster.values;this.mask=mask;this.invalidate()}async setImage(source,options={}){const raster=await rasterFromImage(source,options);this.setRaster(raster)}setHeightValue(x,y,value){this.ensureRaster();const gx=clamp(Math.floor(x),0,this.width-1);const gy=clamp(Math.floor(y),0,this.height-1);const i=indexOfCell(gx,gy,this.width);this.heights[i]=value;this.analysis=null;this.topology=null}async buildWithForcedWorldPoints(forcedWorldPoints){this.ensureRaster();const analysis=this.analysis??this.buildAnalysis();const previousForced=this.options.forcedWorldPoints;try{this.options.forcedWorldPoints=forcedWorldPoints??previousForced??null;const topology=this.buildTopology(analysis);this.analysis=analysis;this.topology=topology;return topology}finally{this.options.forcedWorldPoints=previousForced}}getAnalysis(){return this.analysis}getTopology(){return this.topology}async build(){this.ensureRaster();const analysis=this.buildAnalysis();const topology=this.buildTopology(analysis);this.analysis=analysis;this.topology=topology;return topology}async rebuild(){return this.build()}resampleVertexHeights(){this.ensureRaster();if(!this.topology){throw new Error("Build topology before resampling heights.")}const heights=this.sampleVertexHeights(this.topology.positionsGrid);this.topology.heights=heights;return heights}invalidate(){this.analysis=null;this.topology=null}ensureRaster(){if(!this.heights||!this.mask||this.width<=0||this.height<=0){throw new Error("No heightfield raster is loaded.")}}buildAnalysis(){const width=this.width;const height=this.height;const cellSize=this.options.cellSize;const heights=this.heights;const mask=this.mask;const slopeTan=this.computeSlopeTan(heights,width,height,cellSize,mask);const slopeDegrees=new Float32Array(width*height);for(let i=0;i<slopeTan.length;i++){slopeDegrees[i]=Math.atan(slopeTan[i])*(180/Math.PI)}const curvature=this.computeCurvature(heights,width,height,cellSize,mask);const localRelief=this.computeLocalRelief(heights,width,height,mask);const flowReceiver=this.computeFlowReceiver(heights,width,height,cellSize,mask);const flowAccumulation=this.computeFlowAccumulation(flowReceiver,width,height,mask);const wetness=this.computeWetness(flowAccumulation,slopeTan,width,height,cellSize,mask);const channels=this.extractChannels(flowAccumulation,width,height,mask);const channelDistance=this.computeChannelDistance(channels,width,height,mask);const floodplain=this.extractFloodplain(wetness,slopeTan,channelDistance,width,height,mask);const terrainPriority=this.computeTerrainPriority(slopeTan,curvature,localRelief,mask);const hydroPriority=this.computeHydroPriority(wetness,channels,channelDistance,floodplain,mask);let validCellCount=0;let channelCellCount=0;for(let i=0;i<mask.length;i++){if(mask[i])validCellCount++;if(channels[i])channelCellCount++}const totalArea=validCellCount*cellSize*cellSize;const totalChannelLength=channelCellCount*cellSize;const hillslopeLength=totalChannelLength>EPSILON2?totalArea/(2*totalChannelLength):this.options.maxSpacing;const importance=this.computeImportance(terrainPriority,hydroPriority,mask);const spacing=this.computeSpacing(terrainPriority,hydroPriority,channels,floodplain,mask,hillslopeLength);return{width,height,cellSize,heights,mask,slopeTan,slopeDegrees,curvature,localRelief,flowReceiver,flowAccumulation,wetness,channels,channelDistance,floodplain,terrainPriority,hydroPriority,importance,spacing,validCellCount,totalArea,totalChannelLength,hillslopeLength}}computeSlopeTan(heights,width,height,cellSize,mask){const slope=new Float32Array(width*height);const inv2=1/(2*cellSize);for(let y=0;y<height;y++){const ym1=y>0?y-1:y;const yp1=y<height-1?y+1:y;for(let x=0;x<width;x++){const i=indexOfCell(x,y,width);if(!mask[i])continue;const xm1=x>0?x-1:x;const xp1=x<width-1?x+1:x;const left=heights[indexOfCell(xm1,y,width)];const right=heights[indexOfCell(xp1,y,width)];const up=heights[indexOfCell(x,ym1,width)];const down=heights[indexOfCell(x,yp1,width)];const dzdx=(right-left)*inv2;const dzdy=(down-up)*inv2;slope[i]=Math.hypot(dzdx,dzdy)}}return slope}computeCurvature(heights,width,height,cellSize,mask){const curvature=new Float32Array(width*height);const inv=1/(cellSize*cellSize);for(let y=0;y<height;y++){const ym1=y>0?y-1:y;const yp1=y<height-1?y+1:y;for(let x=0;x<width;x++){const i=indexOfCell(x,y,width);if(!mask[i])continue;const xm1=x>0?x-1:x;const xp1=x<width-1?x+1:x;const c=heights[i];const left=heights[indexOfCell(xm1,y,width)];const right=heights[indexOfCell(xp1,y,width)];const up=heights[indexOfCell(x,ym1,width)];const down=heights[indexOfCell(x,yp1,width)];curvature[i]=(left+right+up+down-4*c)*inv}}return curvature}computeLocalRelief(heights,width,height,mask){return computeLocalRelief(heights,width,height,1,mask)}computeFlowReceiver(heights,width,height,cellSize,mask){const receiver=new Int32Array(width*height);receiver.fill(-1);const diag=Math.SQRT2*cellSize;for(let y=0;y<height;y++){for(let x=0;x<width;x++){const i=indexOfCell(x,y,width);if(!mask[i])continue;const z=heights[i];let bestIndex=-1;let bestSlope=0;for(let oy=-1;oy<=1;oy++){for(let ox=-1;ox<=1;ox++){if(ox===0&&oy===0)continue;const nx=x+ox;const ny=y+oy;if(nx<0||ny<0||nx>=width||ny>=height)continue;const ni=indexOfCell(nx,ny,width);if(!mask[ni])continue;const dz=z-heights[ni];if(dz<=0)continue;const dist=ox!==0&&oy!==0?diag:cellSize;const localSlope=dz/dist;if(localSlope>bestSlope){bestSlope=localSlope;bestIndex=ni}}}receiver[i]=bestIndex}}return receiver}computeFlowAccumulation(receiver,width,height,mask){const count=width*height;const indegree=new Int32Array(count);const accumulation=new Float32Array(count);const queue=new Int32Array(count);for(let i=0;i<count;i++){if(!mask[i])continue;accumulation[i]=1;const r=receiver[i];if(r>=0)indegree[r]++}let head=0;let tail=0;for(let i=0;i<count;i++){if(mask[i]&&indegree[i]===0){queue[tail++]=i}}while(head<tail){const i=queue[head++];const r=receiver[i];if(r>=0){accumulation[r]+=accumulation[i];indegree[r]--;if(indegree[r]===0){queue[tail++]=r}}}return accumulation}computeWetness(accumulation,slopeTan,width,height,cellSize,mask){const wetness=new Float32Array(width*height);for(let i=0;i<wetness.length;i++){if(!mask[i])continue;const ai=accumulation[i]*cellSize;wetness[i]=Math.log(Math.max(ai,EPSILON2)/Math.max(slopeTan[i],EPSILON2))}return wetness}extractChannels(accumulation,width,height,mask){const channels=new Uint8Array(width*height);const threshold=this.options.channelThreshold;for(let i=0;i<channels.length;i++){if(!mask[i])continue;if(accumulation[i]>=threshold){channels[i]=1}}return channels}computeChannelDistance(channels,width,height,mask){const distance=new Float32Array(width*height);const queue=new Int32Array(width*height);let head=0;let tail=0;for(let i=0;i<distance.length;i++){if(!mask[i]){distance[i]=LARGE_DISTANCE;continue}if(channels[i]){distance[i]=0;queue[tail++]=i}else{distance[i]=LARGE_DISTANCE}}while(head<tail){const i=queue[head++];const x=i%width;const y=i/width|0;const base=distance[i];for(let oy=-1;oy<=1;oy++){for(let ox=-1;ox<=1;ox++){if(ox===0&&oy===0)continue;const nx=x+ox;const ny=y+oy;if(nx<0||ny<0||nx>=width||ny>=height)continue;const ni=indexOfCell(nx,ny,width);if(!mask[ni])continue;const step=ox!==0&&oy!==0?Math.SQRT2:1;const next=base+step;if(next<distance[ni]){distance[ni]=next;queue[tail++]=ni}}}}return distance}extractFloodplain(wetness,slopeTan,channelDistance,width,height,mask){const wetnessNorm=normalizeArray(wetness,mask);const slopeNorm=normalizeArray(slopeTan,mask);const floodplain=new Uint8Array(width*height);const maxDist=this.options.floodplainChannelDistanceCells;for(let i=0;i<floodplain.length;i++){if(!mask[i])continue;const nearChannel=channelDistance[i]<=maxDist;const wetEnough=wetnessNorm[i]>=.58;const flatEnough=slopeNorm[i]<=.42;if(nearChannel&&wetEnough&&flatEnough){floodplain[i]=1}}return floodplain}computeTerrainPriority(slopeTan,curvature,localRelief,mask){const slopeNorm=normalizeArray(slopeTan,mask);const curvatureNorm=normalizeAbsArray(curvature,mask);const reliefNorm=normalizeArray(localRelief,mask);const out=new Float32Array(slopeTan.length);for(let i=0;i<out.length;i++){if(!mask[i])continue;const ruggedness=slopeNorm[i]*.48+reliefNorm[i]*.34+curvatureNorm[i]*.18;const majorRelief=Math.max(slopeNorm[i],reliefNorm[i]);const sharpBreak=Math.max(curvatureNorm[i],(slopeNorm[i]+reliefNorm[i])*.5);const terrain=Math.max(ruggedness,majorRelief*.97,sharpBreak*.9);out[i]=clamp(Math.pow(terrain,.7),0,1)}return normalizeArray(out,mask)}computeHydroPriority(wetness,channels,channelDistance,floodplain,mask){const wetnessNorm=normalizeArray(wetness,mask);const out=new Float32Array(wetness.length);let distanceMax=0;for(let i=0;i<channelDistance.length;i++){if(!mask[i])continue;if(channelDistance[i]<LARGE_DISTANCE&&channelDistance[i]>distanceMax){distanceMax=channelDistance[i]}}for(let i=0;i<out.length;i++){if(!mask[i])continue;const d=channelDistance[i]>=LARGE_DISTANCE?distanceMax:channelDistance[i];const proximity=1-clamp(d/Math.max(distanceMax,1),0,1);const hydro=wetnessNorm[i]*.46+proximity*.18+channels[i]*.22+floodplain[i]*.14;out[i]=clamp(hydro,0,1)}return normalizeArray(out,mask)}computeImportance(terrainPriority,hydroPriority,mask){const out=new Float32Array(terrainPriority.length);const mode=this.options.mode;for(let i=0;i<out.length;i++){if(!mask[i])continue;const terrain=terrainPriority[i];const hydro=hydroPriority[i];if(mode==="traditional"){out[i]=terrain}else if(mode==="hydrological"){out[i]=Math.max(terrain*.8,hydro)}else if(mode==="hydrographic"){out[i]=Math.max(terrain,hydro*.55)*.96+Math.min(terrain,hydro)*.04}else{out[i]=Math.max(terrain,hydro*.5)*.97+Math.min(terrain,hydro)*.03}out[i]=clamp(out[i],0,1)}return normalizeArray(out,mask)}computeSpacing(terrainPriority,hydroPriority,channels,floodplain,mask,hillslopeLength){const out=new Float32Array(terrainPriority.length);const cellSize=this.options.cellSize;const minSpacing=Math.max(this.options.minSpacing,cellSize);const maxSpacing=Math.max(minSpacing,Math.min(this.options.maxSpacing,Math.max(hillslopeLength,minSpacing)));const mode=this.options.mode;for(let i=0;i<out.length;i++){if(!mask[i])continue;const terrain=clamp(terrainPriority[i],0,1);const hydro=clamp(hydroPriority[i],0,1);const terrainT=smoothstep(.08,.94,terrain);const terrainSpacing=lerp(maxSpacing,minSpacing,Math.pow(terrainT,.62));let spacing=terrainSpacing;if(mode==="hydrological"){const hydroT=smoothstep(.42,.97,hydro);const hydroSpacing=lerp(maxSpacing,minSpacing*1.15,Math.pow(hydroT,1.15));spacing=Math.min(terrainSpacing,hydroSpacing)}else{if(channels[i]){const channelT=Math.max(.82,smoothstep(.45,.98,hydro));const channelSpacing=lerp(maxSpacing,minSpacing*1.02,Math.pow(channelT,1.05));spacing=Math.min(spacing,channelSpacing)}else if(floodplain[i]){const floodT=Math.max(.62,smoothstep(.45,.95,hydro));const floodSpacing=lerp(maxSpacing,minSpacing*1.28,Math.pow(floodT,1.15));spacing=Math.min(spacing,floodSpacing)}else if(hydro>=.9&&terrain>=.35){const hydroSpacing=lerp(maxSpacing,minSpacing*1.6,Math.pow(hydro,1.35));spacing=Math.min(spacing,hydroSpacing)}}if(terrain>=.82){spacing=Math.min(spacing,lerp(maxSpacing,minSpacing,.92))}else if(terrain>=.68){spacing=Math.min(spacing,lerp(maxSpacing,minSpacing,.8))}else if(terrain>=.52){spacing=Math.min(spacing,lerp(maxSpacing,minSpacing,.64))}out[i]=clamp(spacing,minSpacing,maxSpacing)}return out}buildTopology(analysis){const accepted=[];const acceptedMask=new Uint8Array(analysis.width*analysis.height);const hashCellSize=Math.max(analysis.cellSize,this.options.minSpacing*.5,this.options.maxSpacing*.25);const minWorldX=this.options.worldMinX;const minWorldY=this.options.worldMinY;const maxWorldX=minWorldX+Math.max(0,analysis.width-1)*analysis.cellSize;const maxWorldY=minWorldY+Math.max(0,analysis.height-1)*analysis.cellSize;const bucketWidth=Math.max(1,Math.floor((maxWorldX-minWorldX)/hashCellSize)+1);const bucketHeight=Math.max(1,Math.floor((maxWorldY-minWorldY)/hashCellSize)+1);const spatial=Array.from({length:bucketWidth*bucketHeight},()=>[]);const tryAccept=(gx,gy,radius,fixed)=>{if(gx<0||gy<0||gx>=analysis.width||gy>=analysis.height)return false;const cellIndex=indexOfCell(gx,gy,analysis.width);if(!analysis.mask[cellIndex])return false;if(acceptedMask[cellIndex])return false;const x=this.options.worldMinX+gx*analysis.cellSize;const y=this.options.worldMinY+gy*analysis.cellSize;const hx=clamp(Math.floor((x-minWorldX)/hashCellSize),0,bucketWidth-1);const hy=clamp(Math.floor((y-minWorldY)/hashCellSize),0,bucketHeight-1);const queryRadius=Math.max(radius,hashCellSize*.75);const ring=Math.max(1,Math.ceil(queryRadius/hashCellSize));const minHX=Math.max(0,hx-ring);const maxHX=Math.min(bucketWidth-1,hx+ring);const minHY=Math.max(0,hy-ring);const maxHY=Math.min(bucketHeight-1,hy+ring);for(let by=minHY;by<=maxHY;by++){const rowBase=by*bucketWidth;for(let bx=minHX;bx<=maxHX;bx++){const bucket=spatial[rowBase+bx];for(let i=0;i<bucket.length;i++){const other=accepted[bucket[i]];const dx=x-other.x;const dy=y-other.y;const minDist=Math.max(radius,other.radius);if(dx*dx+dy*dy<minDist*minDist){return false}}}}const pointIndex=accepted.length;accepted.push({x,y,gx,gy,radius,fixed});acceptedMask[cellIndex]=1;spatial[hy*bucketWidth+hx].push(pointIndex);return true};this.emitForcedPoints(analysis,tryAccept);this.emitAdaptivePoints(analysis,tryAccept);if(accepted.length<3){throw new Error("Adaptive sampling retained fewer than 3 valid points.")}if(this.options.relaxIterations>0){this.relaxAcceptedPoints(accepted,analysis)}const pointCount=accepted.length;const positions2D=new Float32Array(pointCount*2);const positionsGrid=new Float32Array(pointCount*2);const uvs=new Float32Array(pointCount*2);const vertexImportance=new Float32Array(pointCount);const vertexWetness=new Float32Array(pointCount);const vertexBoundaryMask=new Uint8Array(pointCount);for(let i=0;i<pointCount;i++){const p=accepted[i];const j=i*2;positions2D[j]=p.x;positions2D[j+1]=p.y;positionsGrid[j]=p.gx;positionsGrid[j+1]=p.gy;uvs[j]=analysis.width>1?p.gx/(analysis.width-1):0;uvs[j+1]=analysis.height>1?p.gy/(analysis.height-1):0;const gx=clamp(Math.round(p.gx),0,analysis.width-1);const gy=clamp(Math.round(p.gy),0,analysis.height-1);const cellIndex=indexOfCell(gx,gy,analysis.width);vertexImportance[i]=analysis.importance[cellIndex];vertexWetness[i]=analysis.wetness[cellIndex];vertexBoundaryMask[i]=isBoundaryCell(analysis.mask,analysis.width,analysis.height,gx,gy)?1:0}const triangulationCoords=new Float64Array(pointCount*2);for(let i=0;i<positions2D.length;i++){triangulationCoords[i]=positions2D[i]}const delaunay=new Delaunator(triangulationCoords);const filtered=this.filterTrianglesToMask(delaunay.triangles,positionsGrid,analysis);const halfedges=this.buildHalfedges(filtered);const{centers,areas}=this.buildTriangleGeometry(filtered,positions2D);const{offsets,neighbors}=this.buildVertexAdjacency(filtered,pointCount);const heights=this.sampleVertexHeights(positionsGrid);return{positions2D,positionsGrid,uvs,heights,indices:filtered,halfedges,triangleCenters:centers,triangleAreas:areas,vertexNeighborOffsets:offsets,vertexNeighbors:neighbors,vertexBoundaryMask,vertexImportance,vertexWetness,analysis}}emitForcedPoints(analysis,tryAccept){const minSpacing=Math.max(this.options.minSpacing,analysis.cellSize);const maxSpacing=Math.max(this.options.maxSpacing,minSpacing);const borderStepCells=Math.max(1,Math.round(this.options.borderSpacing/analysis.cellSize));const boundaryStepCells=Math.max(1,Math.round(maxSpacing/analysis.cellSize));const channelStepCells=Math.max(2,Math.round(minSpacing*2.25/analysis.cellSize));const innerRingOffset=Math.max(1,Math.round(maxSpacing/analysis.cellSize));const borderRadius=Math.max(maxSpacing*.5,analysis.cellSize);const boundaryRadius=Math.max(maxSpacing*.5,analysis.cellSize);const channelRadius=Math.max(minSpacing*1,analysis.cellSize);const forcedPointRadius=Math.max(minSpacing*.35,analysis.cellSize*.5);if(this.options.preserveBorder){for(let x=0;x<analysis.width;x+=borderStepCells){tryAccept(x,0,borderRadius,true);tryAccept(x,analysis.height-1,borderRadius,true)}for(let y=0;y<analysis.height;y+=borderStepCells){tryAccept(0,y,borderRadius,true);tryAccept(analysis.width-1,y,borderRadius,true)}tryAccept(0,0,borderRadius,true);tryAccept(analysis.width-1,0,borderRadius,true);tryAccept(0,analysis.height-1,borderRadius,true);tryAccept(analysis.width-1,analysis.height-1,borderRadius,true);if(this.options.preserveInnerBorder&&analysis.width>2&&analysis.height>2){const left=clamp(innerRingOffset,1,analysis.width-2);const right=clamp(analysis.width-1-innerRingOffset,1,analysis.width-2);const top=clamp(innerRingOffset,1,analysis.height-2);const bottom=clamp(analysis.height-1-innerRingOffset,1,analysis.height-2);for(let x=left;x<=right;x+=boundaryStepCells){tryAccept(x,top,boundaryRadius,true);tryAccept(x,bottom,boundaryRadius,true)}for(let y=top;y<=bottom;y+=boundaryStepCells){tryAccept(left,y,boundaryRadius,true);tryAccept(right,y,boundaryRadius,true)}}}if(this.options.preserveMaskBoundary){for(let y=0;y<analysis.height;y++){for(let x=0;x<analysis.width;x++){if(!isBoundaryCell(analysis.mask,analysis.width,analysis.height,x,y))continue;const keep=x===0||y===0||x===analysis.width-1||y===analysis.height-1||x%boundaryStepCells===0||y%boundaryStepCells===0;if(keep){tryAccept(x,y,boundaryRadius,true)}}}}if(this.options.preserveChannels){for(let y=0;y<analysis.height;y++){for(let x=0;x<analysis.width;x++){const i=indexOfCell(x,y,analysis.width);if(!analysis.channels[i])continue;const neighborCount=this.countMarkedNeighbors(analysis.channels,analysis.width,analysis.height,x,y);const isEndpoint=neighborCount<=1;const isJunction=neighborCount>=3;const isSampledRun=x%channelStepCells===0&&y%channelStepCells===0||(x+y)%channelStepCells===0&&neighborCount===2;if(isEndpoint||isJunction||isSampledRun){tryAccept(x,y,channelRadius,true)}}}}const forcedWorldPoints=this.options.forcedWorldPoints;if(forcedWorldPoints&&forcedWorldPoints.length>=2){for(let i=0;i+1<forcedWorldPoints.length;i+=2){const x=Number(forcedWorldPoints[i]);const y=Number(forcedWorldPoints[i+1]);if(!isFiniteNumber(x)||!isFiniteNumber(y))continue;const gx=Math.round((x-this.options.worldMinX)/analysis.cellSize);const gy=Math.round((y-this.options.worldMinY)/analysis.cellSize);tryAccept(gx,gy,forcedPointRadius,true)}}}countMarkedNeighbors(field,width,height,x,y){let count=0;for(let oy=-1;oy<=1;oy++){for(let ox=-1;ox<=1;ox++){if(ox===0&&oy===0)continue;const nx=x+ox;const ny=y+oy;if(nx<0||ny<0||nx>=width||ny>=height)continue;if(field[indexOfCell(nx,ny,width)]!==0){count++}}}return count}emitAdaptivePoints(analysis,tryAccept){const stride=this.options.candidateStride;const terrainBuckets=Array.from({length:DEFAULT_BUCKET_COUNT},()=>[]);const hydroBuckets=Array.from({length:DEFAULT_BUCKET_COUNT},()=>[]);const blendBuckets=Array.from({length:DEFAULT_BUCKET_COUNT},()=>[]);for(let y=0;y<analysis.height;y+=stride){for(let x=0;x<analysis.width;x+=stride){const i=indexOfCell(x,y,analysis.width);if(!analysis.mask[i])continue;const terrain=clamp(analysis.terrainPriority[i],0,1);const hydro=clamp(analysis.hydroPriority[i],0,1);const blend=clamp(Math.max(terrain,hydro*.45),0,1);const tb=clamp(Math.floor(Math.pow(terrain,.85)*(DEFAULT_BUCKET_COUNT-1)),0,DEFAULT_BUCKET_COUNT-1);const hb=clamp(Math.floor(Math.pow(hydro,1.55)*(DEFAULT_BUCKET_COUNT-1)),0,DEFAULT_BUCKET_COUNT-1);const bb=clamp(Math.floor(Math.pow(blend,.95)*(DEFAULT_BUCKET_COUNT-1)),0,DEFAULT_BUCKET_COUNT-1);terrainBuckets[tb].push(i);hydroBuckets[hb].push(i);blendBuckets[bb].push(i)}}const acceptedFlags=new Uint8Array(analysis.width*analysis.height);let acceptedCount=0;const processBucketSet=(buckets,radiusScale,allowHydroBoost,hardLimit)=>{for(let bucketIndex=DEFAULT_BUCKET_COUNT-1;bucketIndex>=0;bucketIndex--){const bucket=buckets[bucketIndex];for(let k=0;k<bucket.length;k++){if(acceptedCount>=this.options.maxPoints||acceptedCount>=hardLimit)return;const i=bucket[k];if(acceptedFlags[i])continue;const x=i%analysis.width;const y=i/analysis.width|0;const spacing=analysis.spacing[i];const terrain=clamp(analysis.terrainPriority[i],0,1);const hydro=clamp(analysis.hydroPriority[i],0,1);let radius=Math.max(spacing*radiusScale,analysis.cellSize*.7);if(terrain>=.85){radius*=.64}else if(terrain>=.7){radius*=.74}else if(terrain>=.55){radius*=.84}if(allowHydroBoost&&(analysis.channels[i]||analysis.floodplain[i])){radius*=.95}radius=quantizeRadius(radius,Math.max(analysis.cellSize*.7,this.options.minSpacing*.32),Math.max(this.options.minSpacing,this.options.maxSpacing*.62),16);if(tryAccept(x,y,radius,false)){acceptedFlags[i]=1;acceptedCount++}}}};const maxPoints=this.options.maxPoints;const terrainLimit=Math.floor(maxPoints*.82);const hydroLimit=Math.floor(maxPoints*.92);processBucketSet(terrainBuckets,.54,false,terrainLimit);processBucketSet(blendBuckets,.56,false,hydroLimit);processBucketSet(hydroBuckets,.6,true,maxPoints)}relaxAcceptedPoints(accepted,analysis){if(accepted.length<3)return;for(let iter=0;iter<this.options.relaxIterations;iter++){const positions=new Float64Array(accepted.length*2);for(let i=0;i<accepted.length;i++){positions[i*2]=accepted[i].x;positions[i*2+1]=accepted[i].y}const delaunay=new Delaunator(positions);const neighborSumsX=new Float64Array(accepted.length);const neighborSumsY=new Float64Array(accepted.length);const neighborCounts=new Uint32Array(accepted.length);for(let i=0;i<delaunay.triangles.length;i+=3){const a=delaunay.triangles[i];const b=delaunay.triangles[i+1];const c=delaunay.triangles[i+2];this.accumulateNeighbor(a,b,accepted,neighborSumsX,neighborSumsY,neighborCounts);this.accumulateNeighbor(a,c,accepted,neighborSumsX,neighborSumsY,neighborCounts);this.accumulateNeighbor(b,a,accepted,neighborSumsX,neighborSumsY,neighborCounts);this.accumulateNeighbor(b,c,accepted,neighborSumsX,neighborSumsY,neighborCounts);this.accumulateNeighbor(c,a,accepted,neighborSumsX,neighborSumsY,neighborCounts);this.accumulateNeighbor(c,b,accepted,neighborSumsX,neighborSumsY,neighborCounts)}for(let i=0;i<accepted.length;i++){if(accepted[i].fixed)continue;if(neighborCounts[i]===0)continue;const targetX=neighborSumsX[i]/neighborCounts[i];const targetY=neighborSumsY[i]/neighborCounts[i];const gx=clamp(Math.round((targetX-this.options.worldMinX)/analysis.cellSize),0,analysis.width-1);const gy=clamp(Math.round((targetY-this.options.worldMinY)/analysis.cellSize),0,analysis.height-1);if(analysis.mask[indexOfCell(gx,gy,analysis.width)]){accepted[i].gx=gx;accepted[i].gy=gy;accepted[i].x=this.options.worldMinX+gx*analysis.cellSize;accepted[i].y=this.options.worldMinY+gy*analysis.cellSize}}}}accumulateNeighbor(a,b,accepted,sumX,sumY,counts){sumX[a]+=accepted[b].x;sumY[a]+=accepted[b].y;counts[a]++}filterTrianglesToMask(triangles,positionsGrid,analysis){const kept=[];for(let i=0;i<triangles.length;i+=3){const a=triangles[i];const b=triangles[i+1];const c=triangles[i+2];const a2=a*2;const b2=b*2;const c2=c*2;const cx=(positionsGrid[a2]+positionsGrid[b2]+positionsGrid[c2])/3;const cy=(positionsGrid[a2+1]+positionsGrid[b2+1]+positionsGrid[c2+1])/3;if(!nearestMaskSample(analysis.mask,analysis.width,analysis.height,cx,cy)){continue}kept.push(a,b,c)}return Uint32Array.from(kept)}buildHalfedges(indices){const halfedges=new Int32Array(indices.length);halfedges.fill(-1);const edgeMap=new Map;for(let i=0;i<indices.length;i+=3){for(let e=0;e<3;e++){const a=indices[i+e];const b=indices[i+(e+1)%3];const key=`${a}:${b}`;const rev=`${b}:${a}`;if(edgeMap.has(rev)){const opposite=edgeMap.get(rev);halfedges[i+e]=opposite;halfedges[opposite]=i+e}else{edgeMap.set(key,i+e)}}}return halfedges}buildTriangleGeometry(indices,positions2D){const triCount=indices.length/3;const centers=new Float32Array(triCount*2);const areas=new Float32Array(triCount);for(let t=0;t<triCount;t++){const i0=indices[t*3]*2;const i1=indices[t*3+1]*2;const i2=indices[t*3+2]*2;const ax=positions2D[i0];const ay=positions2D[i0+1];const bx=positions2D[i1];const by=positions2D[i1+1];const cx=positions2D[i2];const cy=positions2D[i2+1];centers[t*2]=(ax+bx+cx)/3;centers[t*2+1]=(ay+by+cy)/3;const twiceArea=Math.abs((bx-ax)*(cy-ay)-(by-ay)*(cx-ax));areas[t]=.5*twiceArea}return{centers,areas}}buildVertexAdjacency(indices,vertexCount){const sets=Array.from({length:vertexCount},()=>new Set);for(let i=0;i<indices.length;i+=3){const a=indices[i];const b=indices[i+1];const c=indices[i+2];sets[a].add(b);sets[a].add(c);sets[b].add(a);sets[b].add(c);sets[c].add(a);sets[c].add(b)}const offsets=new Uint32Array(vertexCount+1);let total=0;for(let i=0;i<vertexCount;i++){offsets[i]=total;total+=sets[i].size}offsets[vertexCount]=total;const neighbors=new Uint32Array(total);let write=0;for(let i=0;i<vertexCount;i++){const list=Array.from(sets[i]);list.sort((a,b)=>a-b);for(let j=0;j<list.length;j++){neighbors[write++]=list[j]}}return{offsets,neighbors}}sampleVertexHeights(positionsGrid){const heights=new Float32Array(positionsGrid.length>>1);for(let i=0,j=0;i<heights.length;i++,j+=2){heights[i]=bilinearSample(this.heights,this.width,this.height,positionsGrid[j],positionsGrid[j+1])}return heights}};var EPSILON3=1e-6;var INF=1e30;function clamp2(value,min,max){return value<min?min:value>max?max:value}function distance2D(ax,ay,bx,by){const dx=bx-ax;const dy=by-ay;return Math.hypot(dx,dy)}function sampleGridField(field,width,height,x,y){const gx=clamp2(Math.round(x),0,width-1);const gy=clamp2(Math.round(y),0,height-1);return field[gy*width+gx]}var MinHeap=class{ids;keys;size;constructor(capacity){this.ids=new Int32Array(capacity);this.keys=new Float64Array(capacity);this.size=0}push(id,key){let i=this.size++;this.ids[i]=id;this.keys[i]=key;while(i>0){const p=i-1>>1;if(this.keys[p]<=key)break;this.ids[i]=this.ids[p];this.keys[i]=this.keys[p];i=p}this.ids[i]=id;this.keys[i]=key}pop(){if(this.size===0)return null;const rootId=this.ids[0];const rootKey=this.keys[0];this.size--;if(this.size>0){const lastId=this.ids[this.size];const lastKey=this.keys[this.size];let i=0;while(true){let c=i*2+1;if(c>=this.size)break;if(c+1<this.size&&this.keys[c+1]<this.keys[c])c++;if(this.keys[c]>=lastKey)break;this.ids[i]=this.ids[c];this.keys[i]=this.keys[c];i=c}this.ids[i]=lastId;this.keys[i]=lastKey}return{id:rootId,key:rootKey}}};var RiverNetworkBuilder=class{options;constructor(options={}){this.options={maxNeighborDistance:Math.max(options.maxNeighborDistance??Number.POSITIVE_INFINITY,0),boundaryEpsilon:Math.max(options.boundaryEpsilon??.5,0),minDrop:Math.max(options.minDrop??1e-5,0),flatTolerance:Math.max(options.flatTolerance??1e-4,0),sinkMaxRise:Math.max(options.sinkMaxRise??.02,0),sinkClimbPenalty:Math.max(options.sinkClimbPenalty??2.5,0)}}buildFromTopology(topology){const vertexCount=topology.heights.length;if(vertexCount===0){return{edges:[],downstream:new Int32Array(0),edgeIndexBySource:new Int32Array(0),vertexFlowWeight:new Float32Array(0),upstreamCount:new Uint32Array(0),boundaryMask:new Uint8Array(0),maxWeight:0,minWeight:0}}const neighbors=this.buildNeighbors(topology);const boundaryMask=this.computeBoundaryMask(topology.positionsGrid,topology.vertexBoundaryMask);const downstream=this.chooseDownstream(topology,neighbors,boundaryMask);const{edges,edgeIndexBySource,vertexFlowWeight,upstreamCount,minWeight,maxWeight}=this.accumulateFlow(topology,downstream);return{edges,downstream,edgeIndexBySource,vertexFlowWeight,upstreamCount,boundaryMask,minWeight,maxWeight}}buildSegments(topology,result,minWeight){const mask=this.collectActiveEdgeMask(result,minWeight);return this.buildSegmentsFromMask(topology,result,mask)}buildSegmentsFromMask(topology,result,mask){let activeCount=0;for(let i=0;i<mask.length;i++){if(mask[i])activeCount++}const positions=topology.positionsGrid;const segments=new Float32Array(activeCount*4);let write=0;for(let i=0;i<result.edges.length;i++){if(!mask[i])continue;const edge=result.edges[i];const a2=edge.source*2;const b2=edge.target*2;segments[write++]=positions[a2];segments[write++]=positions[a2+1];segments[write++]=positions[b2];segments[write++]=positions[b2+1]}return segments}buildActiveArtifacts(topology,result,minWeight){const mask=this.collectActiveEdgeMask(result,minWeight);const vertexActiveMask=new Uint8Array(result.vertexFlowWeight.length);let activeVertexCount=0;let activeEdgeCount=0;for(let i=0;i<result.edges.length;i++){if(!mask[i])continue;activeEdgeCount++;const edge=result.edges[i];if(!vertexActiveMask[edge.source]){vertexActiveMask[edge.source]=1;activeVertexCount++}if(!vertexActiveMask[edge.target]){vertexActiveMask[edge.target]=1;activeVertexCount++}}const activeEdges=new Array(activeEdgeCount);const segments=new Float32Array(activeEdgeCount*4);const positions=topology.positionsGrid;let edgeWrite=0;let segmentWrite=0;for(let i=0;i<result.edges.length;i++){if(!mask[i])continue;const edge=result.edges[i];activeEdges[edgeWrite++]=edge;const a2=edge.source*2;const b2=edge.target*2;segments[segmentWrite++]=positions[a2];segments[segmentWrite++]=positions[a2+1];segments[segmentWrite++]=positions[b2];segments[segmentWrite++]=positions[b2+1]}const activeVertices=new Uint32Array(activeVertexCount);let vertexWrite=0;for(let i=0;i<vertexActiveMask.length;i++){if(vertexActiveMask[i]){activeVertices[vertexWrite++]=i}}return{mask,activeEdges,activeVertices,segments}}collectActiveEdges(result,minWeight){const mask=this.collectActiveEdgeMask(result,minWeight);const out=[];for(let i=0;i<result.edges.length;i++){if(mask[i])out.push(result.edges[i])}return out}collectActiveVertices(result,minWeight){const mask=this.collectActiveEdgeMask(result,minWeight);return this.collectActiveVerticesFromMask(result,mask)}collectActiveVerticesFromMask(result,mask){const activeMask=new Uint8Array(result.vertexFlowWeight.length);let count=0;for(let i=0;i<result.edges.length;i++){if(!mask[i])continue;const edge=result.edges[i];if(!activeMask[edge.source]){activeMask[edge.source]=1;count++}if(!activeMask[edge.target]){activeMask[edge.target]=1;count++}}const out=new Uint32Array(count);let write=0;for(let i=0;i<activeMask.length;i++){if(activeMask[i])out[write++]=i}return out}buildGuidePoints(topology,result,options){const minWeight=Math.max(options.minWeight,0);const step=Math.max(options.step,.5);const maxPoints=Math.max(0,Math.floor(options.maxPoints??6e4));const activeEdges=this.collectActiveEdges(result,minWeight);const positions=topology.positionsGrid;const points=[];let minX=Infinity;let minY=Infinity;let maxX=-Infinity;let maxY=-Infinity;for(let i=0;i<positions.length;i+=2){const x=positions[i];const y=positions[i+1];if(x<minX)minX=x;if(y<minY)minY=y;if(x>maxX)maxX=x;if(y>maxY)maxY=y}const qScale=4;const quantWidth=Math.max(1,Math.floor((maxX-minX)*qScale)+3);const seen=new Set;const pushPoint=(x,y)=>{const qx=Math.round((x-minX)*qScale);const qy=Math.round((y-minY)*qScale);const key=qy*quantWidth+qx;if(seen.has(key))return;seen.add(key);points.push(x,y)};for(let i=0;i<activeEdges.length;i++){if(points.length>>1>=maxPoints)break;const edge=activeEdges[i];const a2=edge.source*2;const b2=edge.target*2;const ax=positions[a2];const ay=positions[a2+1];const bx=positions[b2];const by=positions[b2+1];const len=Math.max(distance2D(ax,ay,bx,by),EPSILON3);const steps=Math.max(1,Math.ceil(len/step));for(let s=0;s<=steps;s++){if(points.length>>1>=maxPoints)break;const t=s/steps;pushPoint(ax+(bx-ax)*t,ay+(by-ay)*t)}}return Float32Array.from(points)}buildNeighbors(topology){const vertexCount=topology.heights.length;const neighbors=Array.from({length:vertexCount},()=>[]);const positions=topology.positions2D;const offsets=topology.vertexNeighborOffsets;const verts=topology.vertexNeighbors;const maxDistance=this.options.maxNeighborDistance;for(let i=0;i<vertexCount;i++){for(let p=offsets[i];p<offsets[i+1];p++){const j=verts[p];if(j===i)continue;const a2=i*2;const b2=j*2;const d=distance2D(positions[a2],positions[a2+1],positions[b2],positions[b2+1]);if(d<=maxDistance+EPSILON3){neighbors[i].push(j)}}}return neighbors.map(list=>Uint32Array.from(list))}computeBoundaryMask(positionsGrid,provided){if(provided&&provided.length===positionsGrid.length>>1){return new Uint8Array(provided)}const vertexCount=positionsGrid.length>>1;const boundaryMask=new Uint8Array(vertexCount);let minX=Infinity;let minY=Infinity;let maxX=-Infinity;let maxY=-Infinity;for(let i=0;i<positionsGrid.length;i+=2){const x=positionsGrid[i];const y=positionsGrid[i+1];if(x<minX)minX=x;if(y<minY)minY=y;if(x>maxX)maxX=x;if(y>maxY)maxY=y}const eps=this.options.boundaryEpsilon;for(let i=0;i<vertexCount;i++){const x=positionsGrid[i*2];const y=positionsGrid[i*2+1];if(x<=minX+eps||x>=maxX-eps||y<=minY+eps||y>=maxY-eps){boundaryMask[i]=1}}return boundaryMask}chooseDownstream(topology,neighbors,boundaryMask){const vertexCount=topology.heights.length;const downstream=new Int32Array(vertexCount);downstream.fill(-1);const heights=topology.heights;const positions2D=topology.positions2D;const positionsGrid=topology.positionsGrid;const wetness=topology.vertexWetness;const analysis=topology.analysis;const width=analysis.width;const height=analysis.height;const flowAccumulation=analysis.flowAccumulation;const flowReceiver=analysis.flowReceiver;for(let i=0;i<vertexCount;i++){const a2=i*2;const gx=positionsGrid[a2];const gy=positionsGrid[a2+1];const h=heights[i];const localAccum=sampleGridField(flowAccumulation,width,height,gx,gy);const receiverCell=sampleGridField(flowReceiver,width,height,gx,gy);const rx=receiverCell>=0?receiverCell%width:-1;const ry=receiverCell>=0?Math.floor(receiverCell/width):-1;const rvx=rx>=0?rx-gx:0;const rvy=ry>=0?ry-gy:0;const rvLen=Math.hypot(rvx,rvy);let best=-1;let bestScore=-INF;for(let k=0;k<neighbors[i].length;k++){const j=neighbors[i][k];const j2=j*2;const nh=heights[j];const drop=h-nh;const downhill=drop>this.options.minDrop;const flat=nh<=h+this.options.flatTolerance&&j>i;if(!downhill&&!flat)continue;const dx=positions2D[j2]-positions2D[a2];const dy=positions2D[j2+1]-positions2D[a2+1];const dist=Math.max(Math.hypot(dx,dy),EPSILON3);const jgx=positionsGrid[j2];const jgy=positionsGrid[j2+1];const nbrAccum=sampleGridField(flowAccumulation,width,height,jgx,jgy);let alignment=0;if(rvLen>EPSILON3){alignment=(dx*rvx+dy*rvy)/(dist*rvLen)}const slopeTerm=downhill?drop/dist*6:.05/dist;const accumTerm=Math.log1p(Math.max(0,nbrAccum-localAccum))*.12;const wetnessTerm=wetness[j]*.035;const alignTerm=Math.max(0,alignment)*.7;const boundaryTerm=boundaryMask[j]?.08:0;const lowerTerm=downhill?.08:0;const score=slopeTerm+accumTerm+wetnessTerm+alignTerm+boundaryTerm+lowerTerm;if(score>bestScore){bestScore=score;best=j}}downstream[i]=best}const escape=this.computeEscapeDownstream(topology,neighbors,boundaryMask);for(let i=0;i<vertexCount;i++){if(downstream[i]<0&&!boundaryMask[i]){downstream[i]=escape[i]}}return downstream}computeEscapeDownstream(topology,neighbors,boundaryMask){const vertexCount=topology.heights.length;const bestNext=new Int32Array(vertexCount);bestNext.fill(-1);const bestCost=new Float64Array(vertexCount);bestCost.fill(INF);const heights=topology.heights;const positions2D=topology.positions2D;const positionsGrid=topology.positionsGrid;const analysis=topology.analysis;const width=analysis.width;const height=analysis.height;const flowAccumulation=analysis.flowAccumulation;const flowReceiver=analysis.flowReceiver;const maxRise=Math.max(this.options.sinkMaxRise,this.options.flatTolerance*64);const climbPenalty=this.options.sinkClimbPenalty;const heap=new MinHeap(vertexCount*4+16);for(let i=0;i<vertexCount;i++){if(boundaryMask[i]){bestCost[i]=0;heap.push(i,0)}}while(heap.size>0){const item=heap.pop();if(!item)break;const u=item.id;const costU=item.key;if(costU>bestCost[u]+EPSILON3)continue;for(let p=0;p<neighbors[u].length;p++){const v=neighbors[u][p];const stepCost=this.transitionCost(v,u,topology,flowAccumulation,flowReceiver,width,height,maxRise,climbPenalty);if(!isFinite(stepCost))continue;const candidate=costU+stepCost;if(candidate+EPSILON3<bestCost[v]){bestCost[v]=candidate;bestNext[v]=u;heap.push(v,candidate)}}}return bestNext}transitionCost(from,to,topology,flowAccumulation,flowReceiver,width,height,maxRise,climbPenalty){const heights=topology.heights;const positions2D=topology.positions2D;const positionsGrid=topology.positionsGrid;const a2=from*2;const b2=to*2;const ax=positions2D[a2];const ay=positions2D[a2+1];const bx=positions2D[b2];const by=positions2D[b2+1];const dist=Math.max(distance2D(ax,ay,bx,by),EPSILON3);const h0=heights[from];const h1=heights[to];const climb=Math.max(0,h1-h0);if(climb>maxRise+EPSILON3)return INF;const gx=positionsGrid[a2];const gy=positionsGrid[a2+1];const localAccum=sampleGridField(flowAccumulation,width,height,gx,gy);const receiverCell=sampleGridField(flowReceiver,width,height,gx,gy);const rx=receiverCell>=0?receiverCell%width:-1;const ry=receiverCell>=0?Math.floor(receiverCell/width):-1;const rvx=rx>=0?rx-gx:0;const rvy=ry>=0?ry-gy:0;const rvLen=Math.hypot(rvx,rvy);let alignment=0;if(rvLen>EPSILON3){const dx=positionsGrid[b2]-gx;const dy=positionsGrid[b2+1]-gy;alignment=(dx*rvx+dy*rvy)/(Math.max(Math.hypot(dx,dy),EPSILON3)*rvLen)}const jgx=positionsGrid[b2];const jgy=positionsGrid[b2+1];const nbrAccum=sampleGridField(flowAccumulation,width,height,jgx,jgy);const accumPenalty=nbrAccum+EPSILON3>=localAccum?0:.35*(1-nbrAccum/Math.max(localAccum,1));const alignPenalty=rvLen>EPSILON3?.5*(1-Math.max(-.25,alignment)):.2;const risePenalty=climbPenalty*(climb/Math.max(dist,EPSILON3));return dist*(1+alignPenalty+accumPenalty)+risePenalty}accumulateFlow(topology,downstream){const vertexCount=topology.heights.length;const vertexFlowWeight=new Float32Array(vertexCount);vertexFlowWeight.fill(1);const upstreamCount=new Uint32Array(vertexCount);const edgeIndexBySource=new Int32Array(vertexCount);edgeIndexBySource.fill(-1);const order=new Uint32Array(vertexCount);for(let i=0;i<vertexCount;i++)order[i]=i;const heights=topology.heights;const wetness=topology.vertexWetness;order.sort((a,b)=>{const dz=heights[b]-heights[a];if(Math.abs(dz)>EPSILON3)return dz;const dw=wetness[b]-wetness[a];if(Math.abs(dw)>EPSILON3)return dw;return b-a});for(let i=0;i<order.length;i++){const v=order[i];const d=downstream[v];if(d>=0&&d!==v){vertexFlowWeight[d]+=vertexFlowWeight[v];upstreamCount[d]++}}const edges=[];const positions=topology.positions2D;let minWeight=Infinity;let maxWeight=0;for(let v=0;v<vertexCount;v++){const d=downstream[v];if(d<0||d===v)continue;const a2=v*2;const b2=d*2;const dx=positions[b2]-positions[a2];const dy=positions[b2+1]-positions[a2+1];const length=Math.max(Math.hypot(dx,dy),EPSILON3);const weight=vertexFlowWeight[v];const edge={source:v,target:d,weight,length};edgeIndexBySource[v]=edges.length;edges.push(edge);if(weight<minWeight)minWeight=weight;if(weight>maxWeight)maxWeight=weight}if(!isFinite(minWeight))minWeight=0;return{edges,edgeIndexBySource,vertexFlowWeight,upstreamCount,minWeight,maxWeight}}collectActiveEdgeMask(result,minWeight){const threshold=Math.max(minWeight,0);const mask=new Uint8Array(result.edges.length);const vertexCount=result.vertexFlowWeight.length;for(let v=0;v<vertexCount;v++){if(result.vertexFlowWeight[v]+EPSILON3<threshold)continue;let current=v;let steps=0;while(current>=0&&steps<vertexCount){const edgeIndex=result.edgeIndexBySource[current];if(edgeIndex<0)break;if(mask[edgeIndex])break;mask[edgeIndex]=1;current=result.downstream[current];steps++}}return mask}};var EPSILON4=1e-6;var LARGE_VALUE=1e30;function clamp3(value,min,max){return value<min?min:value>max?max:value}function distance2D2(ax,ay,bx,by){const dx=bx-ax;const dy=by-ay;return Math.hypot(dx,dy)}var FloodplainClassifier=class{options;bestDistance=new Float32Array(0);channelRefHeight=new Float32Array(0);channelCeiling=new Float32Array(0);vertexMaskScratch=new Uint8Array(0);triangleMaskScratch=new Uint8Array(0);queue=new Int32Array(0);constructor(options={}){this.options={maxRise:Math.max(options.maxRise??.06,0),maxClimbStep:Math.max(options.maxClimbStep??.02,0),maxEdgeSlope:Math.max(options.maxEdgeSlope??.75,0),maxWorldDistance:Math.max(options.maxWorldDistance??Number.POSITIVE_INFINITY,0),minTriangleFloodplainVerts:clamp3(Math.floor(options.minTriangleFloodplainVerts??2),1,3)}}classify(topology,seedVerticesInput){const vertexCount=topology.heights.length;const triangleCount=topology.indices.length/3;this.ensureScratch(vertexCount,triangleCount);const positions=topology.positions2D;const heights=topology.heights;const offsets=topology.vertexNeighborOffsets;const neighbors=topology.vertexNeighbors;const seedVertices=seedVerticesInput instanceof Uint32Array?seedVerticesInput:Uint32Array.from(Array.from(seedVerticesInput));const vertexMask=this.vertexMaskScratch;const triangleMask=this.triangleMaskScratch;vertexMask.fill(0,0,vertexCount);triangleMask.fill(0,0,triangleCount);if(seedVertices.length===0){return{vertexMask:vertexMask.slice(0,vertexCount),triangleMask:triangleMask.slice(0,triangleCount),seedVertices}}const bestDistance=this.bestDistance;const channelRefHeight=this.channelRefHeight;const channelCeiling=this.channelCeiling;for(let i=0;i<vertexCount;i++){bestDistance[i]=LARGE_VALUE;channelRefHeight[i]=LARGE_VALUE;channelCeiling[i]=LARGE_VALUE}let head=0;let tail=0;for(let i=0;i<seedVertices.length;i++){const v=seedVertices[i];const refHeight=heights[v];vertexMask[v]=1;bestDistance[v]=0;channelRefHeight[v]=refHeight;channelCeiling[v]=refHeight+this.options.maxRise;this.queue[tail++]=v}while(head<tail){const vertex=this.queue[head++];const vx=positions[vertex*2];const vy=positions[vertex*2+1];const vh=heights[vertex];const activeDistance=bestDistance[vertex];const activeRefHeight=channelRefHeight[vertex];const activeCeiling=channelCeiling[vertex];for(let p=offsets[vertex];p<offsets[vertex+1];p++){const nbr=neighbors[p];const nx=positions[nbr*2];const ny=positions[nbr*2+1];const nh=heights[nbr];const edgeLen=Math.max(distance2D2(vx,vy,nx,ny),EPSILON4);const nextDistance=activeDistance+edgeLen;const localSlope=Math.abs(nh-vh)/edgeLen;const climb=nh-vh;if(nextDistance>this.options.maxWorldDistance+EPSILON4)continue;if(nh>activeCeiling+EPSILON4)continue;if(nh>activeRefHeight+this.options.maxRise+EPSILON4)continue;if(climb>this.options.maxClimbStep+EPSILON4)continue;if(localSlope>this.options.maxEdgeSlope+EPSILON4)continue;const improvesDistance=nextDistance<bestDistance[nbr]-EPSILON4;const improvesRef=activeRefHeight<channelRefHeight[nbr]-EPSILON4;if(!vertexMask[nbr]||improvesDistance||improvesRef){vertexMask[nbr]=1;bestDistance[nbr]=Math.min(bestDistance[nbr],nextDistance);channelRefHeight[nbr]=Math.min(channelRefHeight[nbr],activeRefHeight);channelCeiling[nbr]=Math.min(channelCeiling[nbr],activeCeiling);this.queue[tail++]=nbr}}}for(let triIndex=0,t=0;triIndex<triangleCount;triIndex++,t+=3){const a=topology.indices[t];const b=topology.indices[t+1];const c=topology.indices[t+2];const count=vertexMask[a]+vertexMask[b]+vertexMask[c];if(count>=this.options.minTriangleFloodplainVerts){triangleMask[triIndex]=1}}return{vertexMask:vertexMask.slice(0,vertexCount),triangleMask:triangleMask.slice(0,triangleCount),seedVertices}}ensureScratch(vertexCount,triangleCount){if(this.bestDistance.length<vertexCount){this.bestDistance=new Float32Array(vertexCount);this.channelRefHeight=new Float32Array(vertexCount);this.channelCeiling=new Float32Array(vertexCount);this.vertexMaskScratch=new Uint8Array(vertexCount);this.queue=new Int32Array(vertexCount)}if(this.triangleMaskScratch.length<triangleCount){this.triangleMaskScratch=new Uint8Array(triangleCount)}}};var DEBUG_LOGGING=false;var WORKER_NAME="Topology worker";var cachedRaster=null;var cachedBuilder=null;var cachedBuildOptionsKey="";var cachedTopology=null;var cachedRiverBuilder=null;var cachedRiverResult=null;var cachedFloodplainClassifier=null;self.addEventListener("error",event=>{logError("global error",{message:event?.message||"Unknown worker error",filename:event?.filename||null,lineno:event?.lineno||null,colno:event?.colno||null})});self.addEventListener("unhandledrejection",event=>{const reason=event?.reason instanceof Error?event.reason.stack||event.reason.message:String(event?.reason);logError("unhandled rejection",{reason})});function emitWorkerLog(level,...args){if(!DEBUG_LOGGING)return;const fn=typeof console[level]==="function"?console[level]:console.log;fn.call(console,"[TopologyWorker]",...args)}function log(...args){emitWorkerLog("log",...args)}function logError(...args){emitWorkerLog("error",...args)}function summarizeTopology(topology){if(!topology)return null;return{vertices:topology.positionsGrid?topology.positionsGrid.length>>1:0,triangles:topology.indices?Math.floor(topology.indices.length/3):0,width:topology.analysis?.width??0,height:topology.analysis?.height??0}}function summarizeRiverResult(result){if(!result)return null;return{edgeCount:result.edges?result.edges.length:0,minWeight:result.minWeight??0,maxWeight:result.maxWeight??0}}function summarizeView(view){if(!view)return null;return{segmentCount:view.segments?view.segments.length>>2:0,activeVertexCount:view.activeVertexCount??0,activeEdgeCount:view.activeEdgeCount??0,floodplainSeedCount:view.floodplainResult?.seedVertices?view.floodplainResult.seedVertices.length:0}}function cloneTypedArray(array){return new array.constructor(array)}function collectUiTopologyTransferables(topology){return[topology.positionsGrid.buffer,topology.indices.buffer,topology.analysis.mask.buffer,topology.analysis.importance.buffer]}function collectHydroTransferables(view){const transferables=[];if(view.segments)transferables.push(view.segments.buffer);if(view.floodplainResult){transferables.push(view.floodplainResult.vertexMask.buffer,view.floodplainResult.triangleMask.buffer,view.floodplainResult.seedVertices.buffer)}return transferables}function buildOptionsKey(buildOptions){return JSON.stringify(buildOptions)}function resetDerivedCaches(){cachedTopology=null;cachedRiverBuilder=null;cachedRiverResult=null;cachedFloodplainClassifier=null}function makeRiverBuilder(options){cachedRiverBuilder=new RiverNetworkBuilder(options);return cachedRiverBuilder}function makeFloodplainClassifier(options){cachedFloodplainClassifier=new FloodplainClassifier(options);return cachedFloodplainClassifier}function scanRange(values,mask=null){let min=Infinity;let max=-Infinity;for(let i=0;i<values.length;i++){if(mask&&!mask[i])continue;const v=values[i];if(!Number.isFinite(v))continue;if(v<min)min=v;if(v>max)max=v}if(!Number.isFinite(min)||!Number.isFinite(max)){return{min:null,max:null}}return{min,max}}function maxVertexDegree(offsets){let max=0;for(let i=0;i+1<offsets.length;i++){const degree=offsets[i+1]-offsets[i];if(degree>max)max=degree}return max}function makeUiTopology(topology){const analysis=topology.analysis;return{positionsGrid:cloneTypedArray(topology.positionsGrid),indices:cloneTypedArray(topology.indices),analysis:{mask:cloneTypedArray(analysis.mask),importance:cloneTypedArray(analysis.importance)},statsSummary:{validCellCount:analysis.validCellCount,hillslopeLength:analysis.hillslopeLength,wetnessRange:scanRange(analysis.wetness,analysis.mask),importanceRange:scanRange(analysis.importance,analysis.mask),spacingRange:scanRange(analysis.spacing,analysis.mask),triangleAreaRange:scanRange(topology.triangleAreas),maxVertexDegree:maxVertexDegree(topology.vertexNeighborOffsets)}}}async function ensureBuilder(buildOptions){if(!cachedRaster)throw new Error("No raster in topology worker.");const key=buildOptionsKey(buildOptions);if(!cachedBuilder||cachedBuildOptionsKey!==key){log("creating builder",{keyChanged:cachedBuildOptionsKey!==key,width:cachedRaster.width,height:cachedRaster.height});cachedBuilder=new HeightfieldTINBuilder(buildOptions);cachedBuilder.setRaster(cachedRaster);cachedBuildOptionsKey=key;resetDerivedCaches()}else{log("reusing cached builder",{width:cachedRaster.width,height:cachedRaster.height})}return cachedBuilder}async function ensureFirstTopology(buildOptions){const builder=await ensureBuilder(buildOptions);if(cachedTopology){return{topology:cachedTopology,buildMs:0,fromCache:true}}const t0=performance.now();const topology=await builder.build();const buildMs=performance.now()-t0;cachedTopology=topology;return{topology,buildMs,fromCache:false}}function deriveHydroView(minRiverWeight,floodplainOptions){if(!cachedTopology||!cachedRiverBuilder||!cachedRiverResult||!cachedFloodplainClassifier){return{segments:new Float32Array(0),activeVertexCount:0,activeEdgeCount:0,riverStats:{totalEdgeCount:0,minWeight:0,maxWeight:0},floodplainResult:{vertexMask:new Uint8Array(0),triangleMask:new Uint8Array(0),seedVertices:new Uint32Array(0)},timings:{deriveMs:0}}}const t0=performance.now();const artifacts=cachedRiverBuilder.buildActiveArtifacts(cachedTopology,cachedRiverResult,minRiverWeight);const classifier=floodplainOptions?makeFloodplainClassifier(floodplainOptions):cachedFloodplainClassifier;const floodplainResult=classifier.classify(cachedTopology,artifacts.activeVertices);return{segments:artifacts.segments,activeVertexCount:artifacts.activeVertices.length,activeEdgeCount:artifacts.activeEdges.length,riverStats:{totalEdgeCount:cachedRiverResult.edges.length,minWeight:cachedRiverResult.minWeight,maxWeight:cachedRiverResult.maxWeight},floodplainResult,timings:{deriveMs:performance.now()-t0}}}log("module loaded",{href:typeof self.location!=="undefined"?self.location.href:null,worker:WORKER_NAME});self.onmessage=async event=>{const data=event&&typeof event.data==="object"&&event.data!==null?event.data:null;const id=data?.id;const type=data?.type;const payload=data?.payload;const startedAt=performance.now();if(typeof type!=="string"){log("ignoring non-request message",{dataType:typeof event?.data,hasData:!!data,keys:data?Object.keys(data):null});return}log("message received",{id,type});try{if(type==="ping"){self.postMessage({id,ok:true,payload:{ok:true,worker:"topology"}});log("ping complete",{id,elapsedMs:performance.now()-startedAt});return}if(type==="setRaster"){cachedRaster={width:payload.width,height:payload.height,values:payload.values,mask:payload.mask};cachedBuilder=null;cachedBuildOptionsKey="";resetDerivedCaches();self.postMessage({id,ok:true,payload:{ok:true}});return}if(type==="buildGuide"){const first=await ensureFirstTopology(payload.buildOptions);const topology=first.topology;const riverBuilder=makeRiverBuilder(payload.riverOptions);const t0=performance.now();const riverResult=riverBuilder.buildFromTopology(topology);cachedRiverResult=riverResult;const guidePoints=riverBuilder.buildGuidePoints(topology,riverResult,payload.guideOptions);const activeVertexCount=riverBuilder.collectActiveVertices(riverResult,payload.guideOptions.minWeight).length;const buildGuideMs=performance.now()-t0;self.postMessage({id,ok:true,payload:{firstVertexCount:topology.positionsGrid.length>>1,firstTriangleCount:Math.floor(topology.indices.length/3),activeVertexCount,guidePoints,timings:{buildMs:first.buildMs,buildGuideMs}}},[guidePoints.buffer]);return}if(type==="buildFinal"){const builder=await ensureBuilder(payload.buildOptions);let topology=null;let buildMs=0;if(payload.reuseCachedTopology&&cachedTopology){topology=cachedTopology}else if(payload.guidePoints&&payload.guidePoints.length>=2){const t0=performance.now();topology=await builder.buildWithForcedWorldPoints(payload.guidePoints);buildMs=performance.now()-t0;cachedTopology=topology;cachedRiverResult=null}else{const first=await ensureFirstTopology(payload.buildOptions);topology=first.topology;buildMs=first.buildMs}let riverResult=cachedRiverResult;if(!riverResult){const riverBuilder=makeRiverBuilder(payload.riverOptions);riverResult=riverBuilder.buildFromTopology(topology);cachedRiverResult=riverResult}else if(!cachedRiverBuilder){makeRiverBuilder(payload.riverOptions)}cachedFloodplainClassifier=makeFloodplainClassifier(payload.floodplainOptions);const hydroView=deriveHydroView(payload.minRiverWeight,payload.floodplainOptions);const uiTopology=makeUiTopology(topology);const transferables=collectUiTopologyTransferables(uiTopology).concat(collectHydroTransferables(hydroView));self.postMessage({id,ok:true,payload:{topology:uiTopology,hydroView,timings:{buildMs,deriveMs:hydroView.timings.deriveMs}}},transferables);cachedBuilder=null;cachedBuildOptionsKey="";log("buildFinal complete",{id,topology:summarizeTopology(topology),river:summarizeRiverResult(riverResult),hydroView:summarizeView(hydroView),timings:{buildMs,deriveMs:hydroView.timings.deriveMs},elapsedMs:performance.now()-startedAt});return}if(type==="updateDerived"){const hydroView=deriveHydroView(payload.minRiverWeight,payload.floodplainOptions);const transferables=collectHydroTransferables(hydroView);self.postMessage({id,ok:true,payload:hydroView},transferables);return}throw new Error(`Unknown topology worker message: ${type}`)}catch(error){const message=error instanceof Error?error.message:String(error);const stack=error instanceof Error?error.stack:null;logError("error",{id,type,message,stack,elapsedMs:performance.now()-startedAt});self.postMessage({id,ok:false,error:stack?`${message}\n${stack}`:message})}};var Topology_worker_default=self;;if(typeof import_meta !== \'undefined\')import_meta.url=location.origin+"/dist/";})();\n',Ai=URL.createObjectURL(new globalThis.Blob([Ci],{type:"text/javascript"})),st=Ai});var Dt={};Jt(Dt,{default:()=>at});function Fi(){let c=1;return()=>c++}function vt(c){return c?{vertices:c.positionsGrid?c.positionsGrid.length>>1:0,triangles:c.indices?Math.floor(c.indices.length/3):0,analysisWidth:c.analysis?.width??0,analysisHeight:c.analysis?.height??0}:null}function Xe(c){return c?{activeVertexCount:c.activeVertexCount??0,activeEdgeCount:c.activeEdgeCount??0,segmentCount:c.segments?c.segments.length>>2:0,floodplainSeedCount:c.floodplainResult?.seedVertices?c.floodplainResult.seedVertices.length:0}:null}function bt(c,e){return e?c==="setRaster"?{width:e.width??0,height:e.height??0,valueCount:e.values?e.values.length:0,maskCount:e.mask?e.mask.length:0}:c==="buildGuide"?{hasBuildOptions:!!e.buildOptions,hasRiverOptions:!!e.riverOptions,hasGuideOptions:!!e.guideOptions}:c==="buildFinal"?{hasBuildOptions:!!e.buildOptions,guidePointCount:e.guidePoints?e.guidePoints.length>>1:0,reuseCachedTopology:!!e.reuseCachedTopology,minRiverWeight:e.minRiverWeight??null}:c==="updateDerived"?{minRiverWeight:e.minRiverWeight??null}:null:null}var Bi,ki,Ti,Ri,Ei,at,Vt=Rt(()=>{_t();Bi=15e3,ki=5e3,Ti=12e4,Ri=3e4,Ei=!0;at=class{constructor(e={}){this.nextId=Fi(),this.pending=new Map,this.requestTimeoutMs=Math.max(1e3,Math.floor(e.requestTimeoutMs??Bi)),this.readyTimeoutMs=Math.max(1e3,Math.floor(e.readyTimeoutMs??ki)),this.heavyPhaseTimeoutMs=Math.max(this.requestTimeoutMs,Math.floor(e.heavyPhaseTimeoutMs??Ti)),this.hydroUpdateTimeoutMs=Math.max(this.requestTimeoutMs,Math.floor(e.hydroUpdateTimeoutMs??Ri)),this.debugLogging=e.debugLogging??Ei,this.onstatus=typeof e.onstatus=="function"?e.onstatus:null,this.topologyWorker=new Worker(st,{type:String(st).startsWith("blob:")?"classic":"module",name:"TopologyWorker"}),this.#d(this.topologyWorker,"Topology worker"),this.#e("worker instance created",{topologyWorkerScript:String(st)}),this.topologyReady=!1,this.currentRaster=null,this.currentTopology=null,this.currentHydroView=null,this.#e("constructed",{requestTimeoutMs:this.requestTimeoutMs,readyTimeoutMs:this.readyTimeoutMs,heavyPhaseTimeoutMs:this.heavyPhaseTimeoutMs,hydroUpdateTimeoutMs:this.hydroUpdateTimeoutMs})}dispose(){this.#e("dispose"),this.#o(new Error("Build coordinator disposed.")),this.topologyWorker.terminate(),this.currentRaster=null,this.currentTopology=null,this.currentHydroView=null,this.topologyReady=!1}async#u(){this.topologyReady||(this.#i("Starting topology worker\u2026"),await this.#t(this.topologyWorker,"ping",null,[],this.readyTimeoutMs),this.topologyReady=!0,this.#e("topology worker ready"))}async setRaster(e){await this.#u(),this.currentRaster={width:e.width,height:e.height};let t={width:e.width,height:e.height,values:e.values,mask:e.mask};this.#e("setRaster begin",bt("setRaster",t)),this.#i("Uploading raster\u2026"),await this.#t(this.topologyWorker,"setRaster",t,[t.values.buffer,t.mask.buffer]),this.#e("setRaster complete",{width:e.width,height:e.height})}async build(e){if(!this.currentRaster)throw new Error("No raster has been set.");this.#e("build begin",{raster:{width:this.currentRaster.width,height:this.currentRaster.height},guideEnabled:!!e.guideEnabled,minRiverWeight:e.minRiverWeight});let t=0,i=1,r=0,n={firstTopologyMs:0,firstGuideMs:0,secondTopologyMs:0,finalHydroMs:0};if(!e.guideEnabled){this.#i("Building topology\u2026");let o=await this.#t(this.topologyWorker,"buildFinal",{buildOptions:e.topologyBuildOptions,guidePoints:null,reuseCachedTopology:!1,riverOptions:e.riverOptions,floodplainOptions:e.floodplainOptions,minRiverWeight:e.minRiverWeight},[],this.heavyPhaseTimeoutMs);return n.firstTopologyMs=o.timings?.buildMs??0,n.finalHydroMs=o.timings?.deriveMs??0,this.currentTopology=o.topology,this.currentHydroView=o.hydroView,this.#i("Build complete"),this.#e("build complete",{topology:vt(o.topology),hydroView:Xe(o.hydroView),guidePointCount:t,passCount:i,phaseTimings:n}),{topology:o.topology,hydroView:o.hydroView,guidePointCount:t,passCount:i,phaseTimings:n,buildMetrics:{firstGuideActiveVertices:r}}}this.#i("Building topology pass 1\u2026");let s=await this.#t(this.topologyWorker,"buildGuide",{buildOptions:e.topologyBuildOptions,riverOptions:e.riverOptions,guideOptions:e.guideOptions},[],this.heavyPhaseTimeoutMs);this.#e("buildGuide complete",{firstVertexCount:s.firstVertexCount||0,firstTriangleCount:s.firstTriangleCount||0,activeVertexCount:s.activeVertexCount||0,guidePointCount:s.guidePoints?s.guidePoints.length>>1:0,timings:s.timings}),t=s.guidePoints.length>>1,r=s.activeVertexCount||0,n.firstTopologyMs=s.timings?.buildMs??0,n.firstGuideMs=s.timings?.buildGuideMs??0;let l=t>=Math.max(128,Math.floor((s.firstVertexCount||0)*.015))&&t>=Math.max(64,Math.floor((s.activeVertexCount||0)*.2));i=l?2:1,this.#e("second pass decision",{shouldRunSecondPass:l,guideEnabled:!!e.guideEnabled,guidePointCount:t,firstVertexCount:s.firstVertexCount||0,firstGuideActiveVertices:r}),this.#i(l?"Building topology pass 2\u2026":"Deriving hydro overlays\u2026");let a=await this.#t(this.topologyWorker,"buildFinal",{buildOptions:e.topologyBuildOptions,guidePoints:l?s.guidePoints:null,reuseCachedTopology:!l,riverOptions:e.riverOptions,floodplainOptions:e.floodplainOptions,minRiverWeight:e.minRiverWeight},l?[s.guidePoints.buffer]:[],this.heavyPhaseTimeoutMs);return n.secondTopologyMs=l?a.timings?.buildMs??0:0,n.finalHydroMs=a.timings?.deriveMs??0,this.currentTopology=a.topology,this.currentHydroView=a.hydroView,this.#i("Build complete"),this.#e("build complete",{topology:vt(a.topology),hydroView:Xe(a.hydroView),guidePointCount:t,passCount:i,phaseTimings:n}),{topology:a.topology,hydroView:a.hydroView,guidePointCount:t,passCount:i,phaseTimings:n,buildMetrics:{firstGuideActiveVertices:r}}}async updateHydroOnly({minRiverWeight:e,floodplainOptions:t}){await this.#u(),this.#i("Updating hydro overlays\u2026"),this.#e("updateHydroOnly begin",{minRiverWeight:e});let i=await this.#t(this.topologyWorker,"updateDerived",{minRiverWeight:e,floodplainOptions:t},[],this.hydroUpdateTimeoutMs);return this.currentHydroView=i,this.#i("Hydro updated"),this.#e("updateHydroOnly complete",Xe(i)),i}#d(e,t){e.onmessage=i=>this.#s(i,t),e.onerror=i=>{let r=i?.message?`${t}: ${i.message}`:`${t} failed to load or crashed.`;this.#e("worker error",{label:t,message:r,filename:i?.filename||null,lineno:i?.lineno||null,colno:i?.colno||null}),this.#a(e,new Error(r))},e.onmessageerror=i=>{this.#e("worker messageerror",{label:t,event:i}),this.#a(e,new Error(`${t}: message serialization failed.`))}}#i(e){this.#e("status",e),this.onstatus&&this.onstatus(e)}#s(e,t="Worker"){let i=e.data||{},{id:r,ok:n,payload:s,error:l}=i;if(!this.pending.has(r)){this.#e("received response for unknown request",{label:t,id:r,ok:n,error:l,dataKeys:i?Object.keys(i):null});return}let a=this.pending.get(r);this.pending.delete(r),clearTimeout(a.timer);let o=performance.now()-a.startedAt;n?(this.#e("response ok",{label:t,id:r,type:a.type,elapsedMs:o,payloadSummary:this.#r(a.type,s)}),a.resolve(s)):(this.#e("response error",{label:t,id:r,type:a.type,elapsedMs:o,error:l}),a.reject(new Error(l||"Worker request failed.")))}#a(e,t){for(let[i,r]of this.pending.entries())r.worker===e&&(clearTimeout(r.timer),this.pending.delete(i),this.#e("reject pending for worker",{id:i,type:r.type,error:t.message}),r.reject(t))}#o(e){for(let[t,i]of this.pending.entries())clearTimeout(i.timer),this.pending.delete(t),this.#e("reject all pending",{id:t,type:i.type,error:e.message}),i.reject(e)}#t(e,t,i,r=[],n=this.requestTimeoutMs){return new Promise((s,l)=>{let a=this.nextId(),o=performance.now(),h="Topology worker",u=setTimeout(()=>{if(!this.pending.has(a))return;this.pending.delete(a);let d=performance.now()-o,g=new Error(`Timed out waiting for worker response: ${t}`);this.#e("request timeout",{worker:h,id:a,type:t,timeoutMs:n,elapsedMs:d,payloadSummary:bt(t,i),hint:t==="ping"?"Worker module may still be starting, blocked on import, or failed before onmessage was installed.":null}),l(g)},n);this.pending.set(a,{worker:e,resolve:s,reject:l,timer:u,type:t,startedAt:o}),this.#e("send request",{worker:h,id:a,type:t,timeoutMs:n,transferables:r.length,payloadSummary:bt(t,i)});try{e.postMessage({id:a,type:t,payload:i},r)}catch(d){clearTimeout(u),this.pending.delete(a),this.#e("postMessage failed",{worker:h,id:a,type:t,error:d instanceof Error?d.stack||d.message:String(d)}),l(d)}})}#r(e,t){return t?e==="ping"||e==="setRaster"?t:e==="buildGuide"?{firstVertexCount:t.firstVertexCount||0,firstTriangleCount:t.firstTriangleCount||0,activeVertexCount:t.activeVertexCount||0,guidePointCount:t.guidePoints?t.guidePoints.length>>1:0,timings:t.timings??null}:e==="buildFinal"?{topology:vt(t.topology),hydroView:Xe(t.hydroView),timings:t.timings??null}:e==="updateDerived"?{hydroView:Xe(t),timings:t.timings??null}:null:null}#e(...e){this.debugLogging&&console.log("[BuildCoordinator]",...e)}}});var ei=`
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
`,Ge=class{constructor(e){this.device=e,this.pipelines=null,this.resources=[],this.count=0,this.pending=Promise.resolve()}async initialize(){if(!this.pipelines){let e=this.device.createShaderModule({code:ei,label:"erosion-mass-reduction"});this.pipelines=Promise.all(["reduceCells","reducePartials"].map(t=>this.device.createComputePipelineAsync({layout:"auto",compute:{module:e,entryPoint:t}})))}return this.pipelines}read(e,t,i=()=>({})){let r=this.pending.then(()=>this.reduce(e,t,i));return this.pending=r.catch(()=>{}),r}async reduce(e,t,i){let[r,n]=await this.initialize();if(this.count!==t){this.destroyResources(),this.count=t;let p=Math.max(1,Math.ceil(t/256));this.buffers=[0,1].map(()=>this.device.createBuffer({size:p*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,label:"erosion-mass-partials"})),this.staging=this.device.createBuffer({size:64,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),this.resources.push(...this.buffers,this.staging),this.uniforms=[];for(let f=t;;f=Math.ceil(f/256)){let x=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});if(this.device.queue.writeBuffer(x,0,new Uint32Array([f,0,0,0])),this.resources.push(x),this.uniforms.push(x),Math.ceil(f/256)<=1)break}}let s=this.device.createCommandEncoder({label:"erosion-inspect-mass"}),l=s.beginComputePass(),a=t,o=0,h=0;for(;;){let p=h===0?r:n,f=this.device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniforms[h]}},{binding:h===0?1:2,resource:{buffer:h===0?e:this.buffers[1-o]}},{binding:3,resource:{buffer:this.buffers[o]}}]});if(l.setPipeline(p),l.setBindGroup(0,f),a=Math.ceil(a/256),l.dispatchWorkgroups(a),a<=1)break;o=1-o,h++}l.end(),s.copyBufferToBuffer(this.buffers[o],0,this.staging,0,64);let u=i();this.device.queue.submit([s.finish()]),await this.staging.mapAsync(GPUMapMode.READ);let d;try{d=new Float32Array(this.staging.getMappedRange()).slice()}finally{this.staging.unmap()}let g=d[3],m=p=>g?{min:d[4+p],max:d[8+p]}:null;return{...u,validCells:g,invalidCells:d[15],totalTerrain:d[0],totalWater:d[1],totalSediment:d[2],totalSolid:d[0]+d[2],terrainRange:m(0),waterRange:m(1),sedimentRange:m(2),historyRange:g?{min:d[12],max:d[13]}:null,averageHardness:g?d[14]/g:0,statisticsReadbackBytes:64}}destroyResources(){for(let e of this.resources)e.destroy();this.resources=[],this.count=0}destroy(){this.destroyResources()}};var xe=class{constructor(e,t){this.count=t*2,this.querySet=e.createQuerySet({type:"timestamp",count:this.count}),this.resolveBuffer=e.createBuffer({size:this.count*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),this.readBuffer=e.createBuffer({size:this.count*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ})}writes(e=0){return{querySet:this.querySet,beginningOfPassWriteIndex:e*2,endOfPassWriteIndex:e*2+1}}resolve(e){e.resolveQuerySet(this.querySet,0,this.count,this.resolveBuffer,0),e.copyBufferToBuffer(this.resolveBuffer,0,this.readBuffer,0,this.count*8)}async read(){try{await this.readBuffer.mapAsync(GPUMapMode.READ);let e=new BigUint64Array(this.readBuffer.getMappedRange()),t=0n;for(let i=0;i<this.count;i+=2)e[i+1]>=e[i]&&(t+=e[i+1]-e[i]);return Number(t)/1e6}finally{this.readBuffer.unmap(),this.readBuffer.destroy(),this.resolveBuffer.destroy(),this.querySet.destroy()}}};var Ee=8,ti=16,Et=8,Ft=8;function $(c,e,t){return c<e?e:c>t?t:c}function et(c,e){return Math.ceil(c/e)*e}function It(c){let e=c>>>0;return e=Math.imul(e^e>>>16,2146121005),e=Math.imul(e^e>>>15,2221713035),(e^e>>>16)>>>0}function pt(c){return c*Math.PI/180}function tt(c){let e=Math.hypot(c[0],c[1],c[2])||1;return[c[0]/e,c[1]/e,c[2]/e]}function ii(c,e){return[c[0]-e[0],c[1]-e[1],c[2]-e[2]]}function Ot(c,e){return[c[1]*e[2]-c[2]*e[1],c[2]*e[0]-c[0]*e[2],c[0]*e[1]-c[1]*e[0]]}function ri(c,e){let t=new Float32Array(16);for(let i=0;i<4;i++)for(let r=0;r<4;r++)t[i*4+r]=c[0+r]*e[i*4+0]+c[4+r]*e[i*4+1]+c[8+r]*e[i*4+2]+c[12+r]*e[i*4+3];return t}function ni(c,e,t,i){let r=1/Math.tan(c*.5),n=1/(t-i),s=new Float32Array(16);return s[0]=r/Math.max(e,1e-6),s[5]=r,s[10]=(i+t)*n,s[11]=-1,s[14]=2*i*t*n,s}function si(c,e,t){let i=tt(ii(c,e)),r=tt(Ot(t,i)),n=Ot(i,r),s=new Float32Array(16);return s[0]=r[0],s[1]=n[0],s[2]=i[0],s[3]=0,s[4]=r[1],s[5]=n[1],s[6]=i[1],s[7]=0,s[8]=r[2],s[9]=n[2],s[10]=i[2],s[11]=0,s[12]=-(r[0]*c[0]+r[1]*c[1]+r[2]*c[2]),s[13]=-(n[0]*c[0]+n[1]*c[1]+n[2]*c[2]),s[14]=-(i[0]*c[0]+i[1]*c[1]+i[2]*c[2]),s[15]=1,s}var _e=class{constructor(e={}){this.canvas=e.canvas??null,this.adapter=null,this.device=null,this.context=null,this.presentationFormat=null,this.width=0,this.height=0,this.cellCount=0,this.stateByteLength=0,this.vec4ByteLength=0,this.iterationCount=0,this.simulationTime=0,this.allocatedStateByteLength=0,this.allocatedVec4ByteLength=0,this.dispatchX=0,this.dispatchY=0,this.drawCount=0,this.stateBuffers=[],this.fluxBuffer=null,this.velocityBuffer=null,this.thermalPipeBufferA=null,this.thermalPipeBufferB=null,this.readbackBuffer=null,this.customSpringBuffer=null,this.layerTopBuffer=null,this.layerSurfaceValues=new Float32Array(0),this.layerMaterialValues=new Float32Array(16),this.layerMaterialBuffer=null,this.paintedSpringMap=new Float32Array(0),this.sourceFieldDirty=!0,this.sourceFieldBuildCount=0,this.allocatedPaintedSpringByteLength=0,this.allocatedReadbackByteLength=0,this.renderUniformValues=new Float32Array(40),this.renderUniformBufferTerrain=null,this.renderUniformBufferWater=null,this.depthTexture=null,this.depthTextureView=null,this.depthTextureSize={width:0,height:0},this.sceneColorTexture=null,this.sceneColorView=null,this.sceneColorSize={width:0,height:0},this.sceneSampler=null,this.dummySceneTexture=null,this.dummySceneView=null,this.paramValues=new Float32Array(52),this.paramBits=new Uint32Array(this.paramValues.buffer),this.paramBuffer=null,this.stepParamBuffer=null,this.stepParamStride=256,this.stepParamCapacity=0,this.stepParamValuesCpu=null,this.stepParamValuesCpuCapacity=0,this.stepParamTemplateDirty=!0,this.stepParamPreparedIterations=0,this.computeDynamicOffset=new Uint32Array(1),this.terrainRenderBundle=null,this.finalRenderBundle=null,this.terrainRenderBundleList=[],this.finalRenderBundleList=[],this.computeBindGroupLayout=null,this.renderBindGroupLayout=null,this.compositeBindGroupLayout=null,this.computeBindGroup01=null,this.computeBindGroup10=null,this.renderBindGroupTerrain=null,this.renderBindGroupWater=null,this.compositeBindGroup=null,this.fluxPipeline=null,this.sourcePipeline=null,this.flowPipeline=null,this.erosionPipeline=null,this.thermalOutflowPipeline=null,this.channelPipeline=null,this.channelBendPipeline=null,this.renderPipeline=null,this.waterRenderPipeline=null,this.compositePipeline=null,this.initialized=!1,this.ready=!1,this.latestReadbackStats=null,this.gpuStatistics=null,this.lastErosionPassMs=0,this.lastRenderPassMs=0,this.avgErosionPassMs=0,this.avgRenderPassMs=0,this.lastCpuErosionPassMs=0,this.lastCpuRenderPassMs=0,this.avgCpuErosionPassMs=0,this.avgCpuRenderPassMs=0,this.lastGpuErosionPassMs=0,this.lastGpuRenderPassMs=0,this.avgGpuErosionPassMs=0,this.avgGpuRenderPassMs=0,this.layeredDemEnabled=!1,this.layerConfigMode="single",this.simulationParams={cellSize:1,timeStep:.02,rainRate:.001,evaporationRate:.025,pipeArea:16,gravity:9.81,capacityScale:1.2,suspensionRate:.35,depositionRate:1.4,softeningRate:1.2,maxErosionDepth:.04,thermalRate:.25,talusSlopeCoeff:.012,talusSlopeBias:.002,renderHeightScale:.08,waterOpacity:.28,sedimentTint:.35,hardnessBase:.24,hardnessVariation:.12,sourceCenterX:.5,sourceCenterY:.5,sourceRadius:4,sourceStrength:.06,sourceEnabled:!1,rainDuration:0,pulse2Duration:0,sourceLayoutMode:0,randomSpringCount:4,sourceSeed:1,metersPerPixel:100,sourceTimeOffset:0,historyDecay:.9997,edgeDrainStrength:.08,edgeWaterFloor:0,renderMode:8,cameraAzimuthDeg:45,cameraElevationDeg:42,cameraDistance:2.9,cameraPosX:1.5238999619464006,cameraPosY:1.9404787584406888,cameraPosZ:1.5238999619464006,waterHeightScale:.12,thermalVisualizationScale:160,hydraulicErosionEnabled:!0,closedBoundaries:!1,renderGridStep:0,flowDrag:2,maxOutflowFraction:.5,bankErosionRate:0,sourceSedimentConcentration:0,bankBendBias:24,bendSedimentRate:0,riverBendLength:0,openBoundaryEdges:15,straightBankErosionFactor:.08},this.setSimulationParams(e.simulationParams??{})}async initialize(e=this.canvas){if(!e)throw new Error("A canvas is required to initialize WebGPU erosion.");if(this.canvas=e,this.#x(),!navigator.gpu)throw new Error("WebGPU is not available in this browser.");if(this.adapter=await navigator.gpu.requestAdapter({powerPreference:"high-performance"}),!this.adapter)throw new Error("Failed to acquire a WebGPU adapter.");if(this.device=await this.adapter.requestDevice({requiredFeatures:this.adapter.features.has("timestamp-query")?["timestamp-query"]:[]}),this.stepParamStride=Math.max(256,this.device.limits?.minUniformBufferOffsetAlignment||256),this.context=e.getContext("webgpu"),!this.context)throw new Error("Failed to acquire a WebGPU canvas context.");this.presentationFormat=navigator.gpu.getPreferredCanvasFormat(),this.context.configure({device:this.device,format:this.presentationFormat,alphaMode:"opaque"}),this.paramBuffer=this.device.createBuffer({size:et(this.paramValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-erosion-params"}),this.renderUniformBufferTerrain=this.device.createBuffer({size:et(this.renderUniformValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-render-params-terrain"}),this.renderUniformBufferWater=this.device.createBuffer({size:et(this.renderUniformValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-render-params-water"}),this.layerMaterialBuffer=this.device.createBuffer({size:et(this.layerMaterialValues.byteLength,16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-layer-material-params"}),this.sceneSampler=this.device.createSampler({addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge",magFilter:"linear",minFilter:"linear",mipmapFilter:"linear"}),this.#m(),this.#w(),await this.#O(),this.#n(),this.initialized=!0}destroy(){this.gpuStatistics?.destroy(),this.gpuStatistics=null;for(let e of this.stateBuffers)e?.destroy?.();this.stateBuffers=[],this.fluxBuffer?.destroy?.(),this.velocityBuffer?.destroy?.(),this.thermalPipeBufferA?.destroy?.(),this.thermalPipeBufferB?.destroy?.(),this.readbackBuffer?.destroy?.(),this.customSpringBuffer?.destroy?.(),this.layerTopBuffer?.destroy?.(),this.depthTexture?.destroy?.(),this.sceneColorTexture?.destroy?.(),this.dummySceneTexture?.destroy?.(),this.renderUniformBufferTerrain?.destroy?.(),this.renderUniformBufferWater?.destroy?.(),this.layerMaterialBuffer?.destroy?.(),this.paramBuffer?.destroy?.(),this.stepParamBuffer?.destroy?.(),this.fluxBuffer=null,this.velocityBuffer=null,this.thermalPipeBufferA=null,this.thermalPipeBufferB=null,this.readbackBuffer=null,this.customSpringBuffer=null,this.layerTopBuffer=null,this.layerSurfaceValues=new Float32Array(0),this.layerMaterialBuffer=null,this.layerMaterialValues=new Float32Array(16),this.paintedSpringMap=new Float32Array(0),this.allocatedPaintedSpringByteLength=0,this.allocatedReadbackByteLength=0,this.depthTexture=null,this.depthTextureView=null,this.depthTextureSize={width:0,height:0},this.sceneColorTexture=null,this.sceneColorView=null,this.sceneColorSize={width:0,height:0},this.dummySceneTexture=null,this.dummySceneView=null,this.renderUniformBufferTerrain=null,this.renderUniformBufferWater=null,this.layerMaterialBuffer=null,this.paramBuffer=null,this.stepParamBuffer=null,this.stepParamCapacity=0,this.stepParamValuesCpu=null,this.stepParamValuesCpuCapacity=0,this.#p(),this.computeBindGroup01=null,this.computeBindGroup10=null,this.renderBindGroupTerrain=null,this.renderBindGroupWater=null,this.compositeBindGroup=null,this.ready=!1,this.width=0,this.height=0,this.cellCount=0,this.iterationCount=0,this.simulationTime=0,this.dispatchX=0,this.dispatchY=0,this.drawCount=0,this.latestReadbackStats=null,this.lastErosionPassMs=0,this.lastRenderPassMs=0,this.avgErosionPassMs=0,this.avgRenderPassMs=0,this.lastCpuErosionPassMs=0,this.lastCpuRenderPassMs=0,this.avgCpuErosionPassMs=0,this.avgCpuRenderPassMs=0,this.lastGpuErosionPassMs=0,this.lastGpuRenderPassMs=0,this.avgGpuErosionPassMs=0,this.avgGpuRenderPassMs=0,this.paintedSpringMap=new Float32Array(0)}setSimulationParams(e={}){let t={...e};Number.isFinite(t.erodabilityBase)&&!Number.isFinite(t.hardnessBase)&&(t.hardnessBase=t.erodabilityBase),Number.isFinite(t.erodabilityVariation)&&!Number.isFinite(t.hardnessVariation)&&(t.hardnessVariation=t.erodabilityVariation),["sourceRadius","sourceStrength","sourceSeed","randomSpringCount","sourceLayoutMode"].some(n=>n in t&&t[n]!==this.simulationParams[n])&&(this.sourceFieldDirty=!0);let r="sourceLayoutMode"in t&&t.sourceLayoutMode!==this.simulationParams.sourceLayoutMode;Object.assign(this.simulationParams,t),r&&this.simulationParams.sourceLayoutMode<.5&&this.device&&this.#f(),this.layeredDemEnabled&&(Number.isFinite(t.hardnessBase)||Number.isFinite(t.hardnessVariation))&&(this.layerMaterialValues[0]=this.simulationParams.hardnessBase,this.layerMaterialValues[1]=$(this.simulationParams.hardnessBase+this.simulationParams.hardnessVariation,0,1),this.#i()),this.paramBuffer&&this.device&&this.#n(),(this.renderUniformBufferTerrain||this.renderUniformBufferWater)&&this.device&&this.#h()}resetRainTimer(){this.simulationParams.sourceTimeOffset=this.simulationTime,this.paramBuffer&&this.device&&this.#n()}restartSources(){this.resetRainTimer()}async applyTerrainBrush(e={}){if(!this.ready||this.width<=0||this.height<=0)throw new Error("Terrain painting is only available after the simulation is initialized.");let t=$(Number(e.x)||0,0,Math.max(this.width-1,0)),i=$(Number(e.y)||0,0,Math.max(this.height-1,0)),r=Math.max(.5,Number(e.radius)||1),n=Math.max(1e-4,Number(e.amount)||.01),s=$(Number(e.hardness)||.5,.01,1),l=!!e.subtract,a=await this.#v(),o=Math.max(0,Math.floor(t-r-1)),h=Math.min(this.width-1,Math.ceil(t+r+1)),u=Math.max(0,Math.floor(i-r-1)),d=Math.min(this.height-1,Math.ceil(i+r+1)),g=2.65-s*2.2,m=l?-1:1;for(let p=u;p<=d;p++)for(let f=o;f<=h;f++){let x=f-t,y=p-i,v=Math.hypot(x,y);if(v>r)continue;let b=1-v/r,w=Math.pow($(b,0,1),g),S=p*this.width+f,A=S*Ee;if(a[A+4]<.5)continue;let C=$(a[A]+m*n*w,0,2);a[A]=C;let B=Number.isFinite(a[A+3])?a[A+3]:this.simulationParams.hardnessBase;a[A+3]=this.layerSurfaceValues.length===this.cellCount*4?this.#d(this.layerSurfaceValues,S*4,C,B):B}return this.device.queue.writeBuffer(this.stateBuffers[0],0,a),this.device.queue.writeBuffer(this.stateBuffers[1],0,a),this.latestReadbackStats=null,this.render(),this.getStats()}paintSpringBrush(e={}){if(!this.ready||this.width<=0||this.height<=0)throw new Error("Spring painting is only available after the simulation is initialized.");let t=$(Number(e.x)||0,0,Math.max(this.width-1,0)),i=$(Number(e.y)||0,0,Math.max(this.height-1,0)),r=Math.max(.5,Number(e.radius)||1),n=Math.max(0,Number(e.strength)||0),s=$(Number(e.hardness)||.5,.01,1),l=!!e.erase;this.#R();let a=Math.max(0,Math.floor(t-r-1)),o=Math.min(this.width-1,Math.ceil(t+r+1)),h=Math.max(0,Math.floor(i-r-1)),u=Math.min(this.height-1,Math.ceil(i+r+1)),d=2.65-s*2.2;for(let g=h;g<=u;g++)for(let m=a;m<=o;m++){let p=m-t,f=g-i,x=Math.hypot(p,f);if(x>r)continue;let y=1-x/r,v=Math.pow($(y,0,1),d),b=g*this.width+m;l?this.paintedSpringMap[b]=Math.max(0,this.paintedSpringMap[b]-Math.max(n,1)*v):this.paintedSpringMap[b]=$(this.paintedSpringMap[b]+n*v,0,1)}return this.#f(),this.#n(),this.render(),this.getSourcePoints()}clearPaintedSprings(){return this.paintedSpringMap instanceof Float32Array&&this.paintedSpringMap.length===this.cellCount&&this.paintedSpringMap.fill(0),this.#f(),this.#n(),this.render(),this.getSourcePoints()}#u(e={}){let t=$(this.simulationParams.hardnessBase??.16,0,1),i=$(t+Math.max(0,this.simulationParams.hardnessVariation??.03),0,1),r=[{enabled:!0,label:"Base",hardnessMin:t,hardnessMax:i,thermalEnabled:!0},{enabled:!1,label:"Silt",hardnessMin:.24,hardnessMax:.42,thermalEnabled:!0},{enabled:!1,label:"Clay",hardnessMin:.1,hardnessMax:.22,thermalEnabled:!0},{enabled:!1,label:"Bedrock",hardnessMin:0,hardnessMax:.02,thermalEnabled:!0}],n=Array.isArray(e.layerConfig?.layers)?e.layerConfig.layers:[];return r.map((s,l)=>{let a=n[l]||{},o=Number.isFinite(a.heightMin)?a.heightMin:0,h=Number.isFinite(a.heightMax)?a.heightMax:1,u=Number.isFinite(a.hardnessMin)?a.hardnessMin:Number.isFinite(a.erodabilityMin)?a.erodabilityMin:s.hardnessMin,d=Number.isFinite(a.hardnessMax)?a.hardnessMax:Number.isFinite(a.erodabilityMax)?a.erodabilityMax:s.hardnessMax;return{enabled:!!a.enabled,label:a.label||s.label,preset:a.preset||"custom",heightMin:o,heightMax:h,hardnessMin:$(Math.min(u,d),0,1),hardnessMax:$(Math.max(u,d),0,1),thermalEnabled:a.thermalEnabled!==!1}})}#d(e,t,i,r){let n=-1,s=1/0,l=-1/0,a=-1;for(let m=0;m<4;m++){let p=m*4;if(this.layerMaterialValues[p+2]<.5)continue;let f=e[t+m];!Number.isFinite(f)||f<-1e8||(f>=l&&(l=f,a=m),f>=i&&f<=s&&(s=f,n=m))}if(n<0&&(n=a,s=l),n<0||!Number.isFinite(s))return $(r,0,1);let o=-1/0;for(let m=0;m<4;m++){let p=m*4;if(this.layerMaterialValues[p+2]<.5)continue;let f=e[t+m];!Number.isFinite(f)||f<-1e8||f>=s||f>o&&(o=f)}let h=0;Number.isFinite(o)&&o>-1e8?h=$((s-i)/Math.max(s-o,1e-6),0,1):i<s&&(h=$((s-i)/Math.max(s,1e-6),0,1));let u=n*4,d=this.layerMaterialValues[u],g=this.layerMaterialValues[u+1];return $(d+(g-d)*h,0,1)}#i(){this.layerMaterialBuffer&&this.device&&this.device.queue.writeBuffer(this.layerMaterialBuffer,0,this.layerMaterialValues)}getSourcePoints(){if(this.width<=0||this.height<=0)return[];let e=this.simulationParams.sourceLayoutMode|0,t=!!this.simulationParams.sourceEnabled;if(e===0){let n=this.paintedSpringMap;if(!(n instanceof Float32Array)||n.length!==this.cellCount)return[];let s=[],l=.0025,a=0;for(let d=0;d<n.length;d++)n[d]>l&&a++;let h=Math.max(1,Math.ceil(a/8192)),u=0;for(let d=0;d<this.height;d++)for(let g=0;g<this.width;g++){let m=d*this.width+g,p=n[m];p<=l||u++%h===0&&s.push({x:g,y:d,radius:.7,strength:p,painted:!0,active:t})}return s}if(!t)return[];let i=[],r=Math.min(16,Math.max(1,this.simulationParams.randomSpringCount|0));for(let n=0;n<r;n++){let s=Math.imul(Math.floor(Math.max(0,this.simulationParams.sourceSeed??1))>>>0,37)+n*17+1,l=(It(s)&16777215)/16777216,a=(It(s^2654435769)&16777215)/16777216;i.push({x:l*Math.max(this.width-1,0),y:a*Math.max(this.height-1,0),radius:this.simulationParams.sourceRadius,strength:this.simulationParams.sourceStrength,painted:!1,active:!0})}return i}async setDEM(e,t={}){if(this.initialized||await this.initialize(this.canvas),!e||!(e.values instanceof Float32Array))throw new Error("setDEM expects a raster with Float32Array values.");let i=e.width|0,r=e.height|0;if(i<=0||r<=0)throw new Error("Raster width and height must be positive.");if(e.values.length!==i*r)throw new Error("Raster value count does not match width * height.");let n=Number.isFinite(t.minHeight)?t.minHeight:0,s=Number.isFinite(t.maxHeight)?t.maxHeight:1,l=Math.max(1e-6,s-n),a=e.mask instanceof Uint8Array&&e.mask.length===i*r?e.mask:null,o=Array.isArray(e.bands)?e.bands:null,h=e.water instanceof Float32Array&&e.water.length===i*r?e.water:null,u=this.#u(t),d=t.demSourceMode||t.layerConfig?.mode||"single",g=Array.isArray(o)&&o.some(b=>b instanceof Float32Array&&b.length===i*r),m=g||d!=="single";this.width=i,this.height=r,this.cellCount=i*r,this.stateByteLength=this.cellCount*Ee*4,this.vec4ByteLength=this.cellCount*ti,this.iterationCount=0,this.simulationTime=0,this.simulationParams.sourceTimeOffset=0,this.sourceFieldDirty=!0,this.sourceFieldBuildCount=0,this.dispatchX=Math.ceil(this.width/Et),this.dispatchY=Math.ceil(this.height/Ft),this.drawCount=Math.max(1,(this.width-1)*(this.height-1)),this.#p(),this.#l(),this.#c(),this.#F(),this.#n();let p=new Float32Array(this.cellCount*Ee),f=m?new Float32Array(this.cellCount*4):null;f&&f.fill(-1e9);let x=this.simulationParams.hardnessBase,y=this.simulationParams.hardnessVariation;this.layerMaterialValues.fill(0);for(let b=0;b<4;b++){let w=u[b],S=b*4;this.layerMaterialValues[S]=w.hardnessMin,this.layerMaterialValues[S+1]=w.hardnessMax;let A=!!(o&&o[b]instanceof Float32Array&&o[b].length===this.cellCount),C=m&&!g&&b===0;this.layerMaterialValues[S+2]=w.enabled&&(A||C)?1:0,this.layerMaterialValues[S+3]=w.thermalEnabled?1:0}for(let b=0;b<this.cellCount;b++){let w=b*Ee,S=b*4,A=a?a[b]?1:0:1,C=n+e.values[b]*l,B=$(e.values[b],0,1),R=Math.sin(b*12.9898+i*.031+r*.017)*43758.5453,z=((R-Math.floor(R))*2-1)*y*.25,E=$(x+(1-B)*y+z,0,1),F=C,M=E;if(m){let k=-1/0;for(let I=0;I<4;I++){let q=I*4;if(this.layerMaterialValues[q+2]<.5)continue;let D=o&&o[I]instanceof Float32Array&&o[I].length===this.cellCount?o[I]:I===0?e.values:null;if(!D)continue;let j=u[I],O=$(D[b],0,1),Y=j.heightMin+O*(j.heightMax-j.heightMin);f[S+I]=Y,Y>k&&(k=Y)}Number.isFinite(k)&&k>-1e8?F=k:(f[S]=F,this.layerMaterialValues[2]=1),M=this.#d(f,S,F,E)}p[w]=F,p[w+1]=A&&Number.isFinite(h?.[b])?Math.max(h[b],0):0,p[w+2]=0,p[w+3]=M,p[w+4]=A,p[w+5]=0,p[w+6]=0,p[w+7]=0}let v=this.device.createCommandEncoder({label:"paper-reset-sim-scratch"});v.clearBuffer(this.fluxBuffer),v.clearBuffer(this.velocityBuffer),v.clearBuffer(this.thermalPipeBufferA),v.clearBuffer(this.thermalPipeBufferB),m||v.clearBuffer(this.layerTopBuffer),this.device.queue.submit([v.finish()]),this.device.queue.writeBuffer(this.stateBuffers[0],0,p),this.device.queue.writeBuffer(this.stateBuffers[1],0,p),m&&f&&this.device.queue.writeBuffer(this.layerTopBuffer,0,f),this.layeredDemEnabled=m,this.layerSurfaceValues=this.layeredDemEnabled&&f?f:new Float32Array(0),this.#i(),this.layerConfigMode=d,this.ready=!0,this.latestReadbackStats=null,this.paintedSpringMap=new Float32Array(0),this.#f(),this.#h(),this.#l(),this.#c(),this.render()}resize(){!this.context||!this.device||!this.canvas||(this.#x(),this.context.configure({device:this.device,format:this.presentationFormat,alphaMode:"opaque"}),this.#l(),this.#c(),this.#h())}#s(e=1,t="paper-erosion-step",i=null){let r=this.device.createCommandEncoder({label:t});return this.#M(r,Math.max(1,e|0),i),i?.resolve(r),r.finish()}#a(e="paper-erosion-render",t=null){this.#l(),this.#c(),this.#h();let i=this.context.getCurrentTexture().createView(),r=this.device.createCommandEncoder({label:e});return this.#S(r,i,t),t?.resolve(r),r.finish()}async#o(e,t){let i=performance.now();return this.device.queue.submit([e]),t?t.read():(await this.device.queue.onSubmittedWorkDone(),Math.max(0,performance.now()-i))}async stepBench(e=1){if(!this.ready)return null;let t=this.device.features.has("timestamp-query")?new xe(this.device,1):null,i=performance.now(),r=this.#s(Math.max(1,e|0),"paper-erosion-step-bench",t),n=Math.max(0,performance.now()-i),s=await this.#o(r,t);return this.#r(n),this.#y(s),{cpuErosionMs:n,gpuErosionMs:s,cpuRenderMs:0,gpuRenderMs:0,gpuTimingMode:this.getStats().gpuTimingMode}}async stepAndRenderBench(e=1){if(!this.ready)return null;let t=!!(this.context&&this.width>=2&&this.height>=2),i=this.device.features.has("timestamp-query")?new xe(this.device,1):null,r=performance.now(),n=this.#s(Math.max(1,e|0),t?"paper-erosion-step-bench":"paper-erosion-step-render-bench",i),s=Math.max(0,performance.now()-r),l=await this.#o(n,i);this.#r(s),this.#y(l);let a=0,o=0;if(t){let h=this.device.features.has("timestamp-query")?new xe(this.device,2):null,u=performance.now(),d=this.#a("paper-erosion-render-bench",h);a=Math.max(0,performance.now()-u),o=await this.#o(d,h),this.#e(a),this.#k(o)}else this.#e(0);return{cpuErosionMs:s,gpuErosionMs:l,cpuRenderMs:a,gpuRenderMs:o,gpuTimingMode:this.getStats().gpuTimingMode}}step(e=1){if(!this.ready)return;let t=performance.now(),i=this.#s(Math.max(1,e|0),"paper-erosion-step");this.device.queue.submit([i]),this.#r(Math.max(0,performance.now()-t))}stepAndRender(e=1){if(!this.ready)return;let t=!!(this.context&&this.width>=2&&this.height>=2),i=this.device.createCommandEncoder({label:t?"paper-erosion-step-render":"paper-erosion-step"}),r=performance.now();this.#M(i,Math.max(1,e|0));let n=performance.now(),s=n,l=n;if(t){s=performance.now(),this.#l(),this.#c(),this.#h();let d=this.context.getCurrentTexture().createView();this.#S(i,d),l=performance.now()}let a=performance.now();this.device.queue.submit([i.finish()]);let o=performance.now()-a,h=Math.max(0,n-r),u=Math.max(0,l-s);if(t&&h+u>1e-6){let d=h+u;h+=o*(h/d),u+=o*(u/d)}else h+=o,u=0;this.#r(h),this.#e(u)}render(){if(!this.ready||!this.context||this.width<2||this.height<2)return;let e=performance.now(),t=this.#a("paper-erosion-render");this.device.queue.submit([t]),this.#e(Math.max(0,performance.now()-e))}async readbackStats(){return this.ready?(this.gpuStatistics??=new Ge(this.device),this.latestReadbackStats=await this.gpuStatistics.read(this.stateBuffers[0],this.cellCount,()=>({statisticsIteration:this.iterationCount,statisticsSimTime:this.simulationTime})),this.getStats()):this.getStats()}async exportTerrainPng(){if(!this.ready||this.width<=0||this.height<=0)throw new Error("Terrain export is only available after the simulation is initialized.");if(typeof OffscreenCanvas!="function")throw new Error("OffscreenCanvas is not available for terrain export.");let e=await this.#v(),t=1/0,i=-1/0;for(let h=0;h<this.cellCount;h++){let u=h*Ee;if(e[u+4]<.5)continue;let d=e[u];d<t&&(t=d),d>i&&(i=d)}(!Number.isFinite(t)||!Number.isFinite(i))&&(t=0,i=1);let r=Math.max(i-t,1e-6),n=new Uint8ClampedArray(this.width*this.height*4);for(let h=0;h<this.height;h++){let u=this.height-1-h;for(let d=0;d<this.width;d++){let m=(h*this.width+d)*Ee,p=(u*this.width+d)*4,f=e[m+4]>=.5,x=e[m],y=f?$((x-t)/r,0,1):0,v=Math.max(0,Math.min(255,Math.round(y*255)));n[p]=v,n[p+1]=v,n[p+2]=v,n[p+3]=f?255:0}}let s=new OffscreenCanvas(this.width,this.height),l=s.getContext("2d");if(!l)throw new Error("Failed to create a terrain export canvas.");l.putImageData(new ImageData(n,this.width,this.height),0,0);let o=await(await s.convertToBlob({type:"image/png"})).arrayBuffer();return{width:this.width,height:this.height,minTerrain:t,maxTerrain:i,data:o}}#t(e,t){return e>0?e*.88+t*.12:t}#r(e){let t=Math.max(0,Number(e)||0);this.lastErosionPassMs=t,this.lastCpuErosionPassMs=t,this.avgErosionPassMs=this.#t(this.avgErosionPassMs,t),this.avgCpuErosionPassMs=this.#t(this.avgCpuErosionPassMs,t)}#e(e){let t=Math.max(0,Number(e)||0);this.lastRenderPassMs=t,this.lastCpuRenderPassMs=t,this.avgRenderPassMs=this.#t(this.avgRenderPassMs,t),this.avgCpuRenderPassMs=this.#t(this.avgCpuRenderPassMs,t)}#y(e){let t=Math.max(0,Number(e)||0);this.lastGpuErosionPassMs=t,this.avgGpuErosionPassMs=this.#t(this.avgGpuErosionPassMs,t)}#k(e){let t=Math.max(0,Number(e)||0);this.lastGpuRenderPassMs=t,this.avgGpuRenderPassMs=this.#t(this.avgGpuRenderPassMs,t)}getStats(){let e=this.simulationTime,t=Math.max(0,this.simulationParams.rainDuration??0),i=Math.max(0,e-this.simulationParams.sourceTimeOffset),r=t<=0||i<t||this.simulationParams.pulse2Duration>0&&i<t+this.simulationParams.pulse2Duration;return{width:this.width,height:this.height,cellCount:this.cellCount,ready:this.ready,iterationCount:this.iterationCount,simTime:e,rainActive:r,rainDuration:t,sourceFieldBuildCount:this.sourceFieldBuildCount,layeredDemEnabled:this.layeredDemEnabled,layerConfigMode:this.layerConfigMode,activeLayerCount:[0,1,2,3].reduce((n,s)=>n+(this.layerMaterialValues[s*4+2]>.5?1:0),0),lastErosionPassMs:this.lastErosionPassMs,lastRenderPassMs:this.lastRenderPassMs,avgErosionPassMs:this.avgErosionPassMs,avgRenderPassMs:this.avgRenderPassMs,lastCpuErosionPassMs:this.lastCpuErosionPassMs,lastCpuRenderPassMs:this.lastCpuRenderPassMs,avgCpuErosionPassMs:this.avgCpuErosionPassMs,avgCpuRenderPassMs:this.avgCpuRenderPassMs,gpuTimingMode:this.device?.features.has("timestamp-query")?"timestamp-query":"completion-time",lastGpuErosionPassMs:this.lastGpuErosionPassMs,lastGpuRenderPassMs:this.lastGpuRenderPassMs,avgGpuErosionPassMs:this.avgGpuErosionPassMs,avgGpuRenderPassMs:this.avgGpuRenderPassMs,...this.latestReadbackStats??{}}}#x(){if(!this.canvas)return;if(typeof this.canvas.getBoundingClientRect!="function"){let l=Math.max(1,Math.round(this.canvas.width||1)),a=Math.max(1,Math.round(this.canvas.height||1));this.canvas.width!==l&&(this.canvas.width=l),this.canvas.height!==a&&(this.canvas.height=a);return}let e=this.canvas.getBoundingClientRect(),t=Math.max(1,Math.round(e.width||this.canvas.width||1)),i=Math.max(1,Math.round(e.height||this.canvas.height||1)),r=Math.max(1,Math.min(globalThis.devicePixelRatio||1,2)),n=Math.max(1,Math.round(t*r)),s=Math.max(1,Math.round(i*r));this.canvas.width!==n&&(this.canvas.width=n),this.canvas.height!==s&&(this.canvas.height=s)}#l(){if(!this.device||!this.canvas)return;let e=Math.max(1,this.canvas.width||this.canvas.clientWidth||1),t=Math.max(1,this.canvas.height||this.canvas.clientHeight||1);this.depthTexture&&this.depthTextureSize.width===e&&this.depthTextureSize.height===t||(this.depthTexture?.destroy?.(),this.depthTexture=this.device.createTexture({size:{width:e,height:t},format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT,label:"paper-depth"}),this.depthTextureView=this.depthTexture.createView(),this.depthTextureSize={width:e,height:t})}#c(){if(!this.device||!this.canvas)return;let e=Math.max(1,this.canvas.width||this.canvas.clientWidth||1),t=Math.max(1,this.canvas.height||this.canvas.clientHeight||1);this.sceneColorTexture&&this.sceneColorSize.width===e&&this.sceneColorSize.height===t||(this.sceneColorTexture?.destroy?.(),this.sceneColorTexture=this.device.createTexture({size:{width:e,height:t},format:this.presentationFormat,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,label:"paper-scene-color"}),this.sceneColorView=this.sceneColorTexture.createView(),this.sceneColorSize={width:e,height:t},this.stateBuffers.length===2&&this.sceneSampler&&this.#g())}async#v(){this.#T();let e=this.device.createCommandEncoder({label:"paper-erosion-readback"});e.copyBufferToBuffer(this.stateBuffers[0],0,this.readbackBuffer,0,this.stateByteLength),this.device.queue.submit([e.finish()]),await this.readbackBuffer.mapAsync(GPUMapMode.READ);let i=this.readbackBuffer.getMappedRange().slice(0);return this.readbackBuffer.unmap(),new Float32Array(i)}#m(){if(!this.device)return;let e=Math.max(4,this.cellCount*4);this.customSpringBuffer&&this.allocatedPaintedSpringByteLength===e||(this.customSpringBuffer?.destroy?.(),this.customSpringBuffer=this.device.createBuffer({size:e,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:"paper-cached-source-rates"}),this.allocatedPaintedSpringByteLength=e,this.#b())}#f(){if(this.sourceFieldDirty=!0,!!this.device&&(this.#m(),!!this.customSpringBuffer)){if(!(this.paintedSpringMap instanceof Float32Array)||this.paintedSpringMap.length!==this.cellCount){this.#b();return}this.device.queue.writeBuffer(this.customSpringBuffer,0,this.paintedSpringMap)}}#T(){this.device&&(this.readbackBuffer&&this.allocatedReadbackByteLength===this.stateByteLength||(this.readbackBuffer?.destroy?.(),this.readbackBuffer=this.device.createBuffer({size:this.stateByteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,label:"paper-readback"}),this.allocatedReadbackByteLength=this.stateByteLength))}#R(){(!(this.paintedSpringMap instanceof Float32Array)||this.paintedSpringMap.length!==this.cellCount)&&(this.paintedSpringMap=new Float32Array(this.cellCount))}#b(){if(this.sourceFieldDirty=!0,!this.device||(this.#m(),!this.customSpringBuffer))return;let e=this.device.createCommandEncoder({label:"paper-clear-painted-springs"});e.clearBuffer(this.customSpringBuffer),this.device.queue.submit([e.finish()])}#w(){!this.device||!this.presentationFormat||this.dummySceneTexture||(this.dummySceneTexture=this.device.createTexture({size:{width:1,height:1},format:this.presentationFormat,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST,label:"paper-dummy-scene-color"}),this.dummySceneView=this.dummySceneTexture.createView(),this.device.queue.writeTexture({texture:this.dummySceneTexture},new Uint8Array([14,17,20,255]),{bytesPerRow:256},{width:1,height:1,depthOrArrayLayers:1}))}#p(){this.terrainRenderBundle=null,this.finalRenderBundle=null,this.terrainRenderBundleList.length=0,this.finalRenderBundleList.length=0}#E(){if(!this.device||!this.renderPipeline||!this.compositePipeline||!this.waterRenderPipeline||!this.renderBindGroupTerrain||!this.renderBindGroupWater||!this.compositeBindGroup||this.terrainRenderBundle&&this.finalRenderBundle)return;let e=this.drawCount||Math.max(1,(this.width-1)*(this.height-1)),t=this.device.createRenderBundleEncoder({colorFormats:[this.presentationFormat],depthStencilFormat:"depth24plus"});t.setPipeline(this.renderPipeline),t.setBindGroup(0,this.renderBindGroupTerrain),t.draw(6,e,0,0),this.terrainRenderBundle=t.finish({label:"paper-render-terrain-bundle"}),this.terrainRenderBundleList[0]=this.terrainRenderBundle;let i=this.device.createRenderBundleEncoder({colorFormats:[this.presentationFormat],depthStencilFormat:"depth24plus"});i.setPipeline(this.compositePipeline),i.setBindGroup(0,this.compositeBindGroup),i.draw(3,1,0,0),i.setPipeline(this.waterRenderPipeline),i.setBindGroup(0,this.renderBindGroupWater),i.draw(6,e,0,0),this.finalRenderBundle=i.finish({label:"paper-render-final-bundle"}),this.finalRenderBundleList[0]=this.finalRenderBundle}#M(e,t,i=null){if(!this.ready)return;let r=this.dispatchX||Math.ceil(this.width/Et),n=this.dispatchY||Math.ceil(this.height/Ft);this.#I(t);let s=this.stepParamStride>>>2,l=this.stepParamValuesCpu,a=this.paramValues[3],o=this.simulationTime,h=this.#P(a);for(let v=0;v<t;v++){let b=v*s;l[b+25]=o+v*a,this.#C(l,b,o+v*a,h)}this.device.queue.writeBuffer(this.stepParamBuffer,0,l.buffer,0,this.stepParamStride*t);let u=this.fluxPipeline,d=this.flowPipeline,g=this.thermalOutflowPipeline,m=this.erosionPipeline,p=this.computeBindGroup01,f=this.computeBindGroup10,x=this.computeDynamicOffset,y=e.beginComputePass({label:"paper-erosion-compute",timestampWrites:i?.writes()});this.sourceFieldDirty&&this.simulationParams.sourceEnabled&&this.simulationParams.sourceLayoutMode>=.5&&(x[0]=0,y.setBindGroup(0,p,x),y.setPipeline(this.sourcePipeline),y.dispatchWorkgroups(r,n),this.sourceFieldDirty=!1,this.sourceFieldBuildCount++);for(let v=0;v<t;v++)x[0]=v*this.stepParamStride,y.setBindGroup(0,p,x),y.setPipeline(u),y.dispatchWorkgroups(r,n),y.setPipeline(d),y.dispatchWorkgroups(r,n),y.setBindGroup(0,f,x),this.simulationParams.bendSedimentRate>0&&this.simulationParams.riverBendLength>0&&this.simulationParams.hydraulicErosionEnabled&&(y.setPipeline(this.channelPipeline),y.dispatchWorkgroups(this.height),y.setPipeline(this.channelBendPipeline),y.dispatchWorkgroups(Math.ceil(this.height/64))),(this.simulationParams.thermalRate>0||(this.simulationParams.bankErosionRate>0||this.simulationParams.bendSedimentRate>0)&&this.simulationParams.hydraulicErosionEnabled)&&(y.setPipeline(g),y.dispatchWorkgroups(r,n)),y.setPipeline(m),y.dispatchWorkgroups(r,n);y.end(),this.iterationCount+=t,this.simulationTime+=t*a,this.#A(this.iterationCount,this.paramValues)}#S(e,t,i=null){this.#E();{let r=e.beginRenderPass({timestampWrites:i?.writes(0),colorAttachments:[{view:this.sceneColorView,clearValue:{r:.055,g:.065,b:.08,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:this.depthTextureView?{view:this.depthTextureView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}:void 0});if(this.terrainRenderBundle)r.executeBundles(this.terrainRenderBundleList);else{let n=this.drawCount||Math.max(1,(this.width-1)*(this.height-1));r.setPipeline(this.renderPipeline),r.setBindGroup(0,this.renderBindGroupTerrain),r.draw(6,n,0,0)}r.end()}{let r=e.beginRenderPass({timestampWrites:i?.writes(1),colorAttachments:[{view:t,clearValue:{r:.055,g:.065,b:.08,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:this.depthTextureView?{view:this.depthTextureView,depthLoadOp:"load",depthStoreOp:"discard"}:void 0});if(this.finalRenderBundle)r.executeBundles(this.finalRenderBundleList);else{let n=this.drawCount||Math.max(1,(this.width-1)*(this.height-1));r.setPipeline(this.compositePipeline),r.setBindGroup(0,this.compositeBindGroup),r.draw(3,1,0,0),r.setPipeline(this.waterRenderPipeline),r.setBindGroup(0,this.renderBindGroupWater),r.draw(6,n,0,0)}r.end()}}#h(){if(!this.renderUniformBufferTerrain&&!this.renderUniformBufferWater||!this.device)return;let e=this.canvas?Math.max((this.canvas.width||1)/Math.max(this.canvas.height||1,1),1e-6):1,t=pt(this.simulationParams.cameraAzimuthDeg),i=pt(this.simulationParams.cameraElevationDeg),r=this.width>1||this.height>1?2/Math.max(this.width-1,this.height-1,1):1,n=[this.simulationParams.cameraPosX??1.5238999619464006,this.simulationParams.cameraPosY??1.9404787584406888,this.simulationParams.cameraPosZ??1.5238999619464006],s=tt([-Math.cos(i)*Math.cos(t),-Math.sin(i),-Math.cos(i)*Math.sin(t)]),l=[n[0]+s[0],n[1]+s[1],n[2]+s[2]],a=si(n,l,[0,1,0]),o=ni(pt(50),e,.01,32),h=ri(o,a),u=Math.max(.25,(this.simulationParams.metersPerPixel??100)/100);this.renderUniformValues.set(h,0),this.renderUniformValues[16]=this.width,this.renderUniformValues[17]=this.height,this.renderUniformValues[18]=u,this.renderUniformValues[19]=this.simulationParams.renderHeightScale,this.renderUniformValues[20]=this.simulationParams.waterOpacity,this.renderUniformValues[21]=this.simulationParams.sedimentTint,this.renderUniformValues[22]=this.simulationParams.renderMode,this.renderUniformValues[23]=this.simulationParams.thermalVisualizationScale,this.renderUniformValues[24]=r,this.renderUniformValues[25]=this.simulationParams.waterHeightScale,this.renderUniformValues[27]=5e-4;let d=tt([.45,.82,.36]);this.renderUniformValues[28]=d[0],this.renderUniformValues[29]=d[1],this.renderUniformValues[30]=d[2];let g=Math.max(1,Math.floor(this.simulationParams.renderGridStep)||Math.ceil(Math.max(this.width-1,this.height-1,1)/512)),m=Math.max(1,Math.ceil((this.width-1)/g)*Math.ceil((this.height-1)/g));this.drawCount!==m&&(this.drawCount=m,this.#p()),this.renderUniformValues[31]=g,this.renderUniformValues[32]=n[0],this.renderUniformValues[33]=n[1],this.renderUniformValues[34]=n[2],this.renderUniformValues[35]=this.layeredDemEnabled?1:0,this.renderUniformValues[36]=this.simulationTime,this.renderUniformValues[37]=this.iterationCount,this.renderUniformValues[38]=this.canvas?.width||1,this.renderUniformValues[39]=this.canvas?.height||1,this.renderUniformValues[26]=0,this.device.queue.writeBuffer(this.renderUniformBufferTerrain,0,this.renderUniformValues),this.renderUniformValues[26]=1,this.device.queue.writeBuffer(this.renderUniformBufferWater,0,this.renderUniformValues)}#F(){if(this.stateBuffers.length!==2||!this.fluxBuffer||!this.velocityBuffer||!this.thermalPipeBufferA||!this.thermalPipeBufferB||!this.layerTopBuffer||this.allocatedStateByteLength!==this.stateByteLength||this.allocatedVec4ByteLength!==this.vec4ByteLength||this.velocityBuffer?.size!==this.vec4ByteLength+this.height*32){for(let t of this.stateBuffers)t?.destroy?.();this.fluxBuffer?.destroy?.(),this.velocityBuffer?.destroy?.(),this.thermalPipeBufferA?.destroy?.(),this.thermalPipeBufferB?.destroy?.(),this.readbackBuffer?.destroy?.(),this.customSpringBuffer?.destroy?.(),this.layerTopBuffer?.destroy?.(),this.customSpringBuffer=null,this.layerTopBuffer=null,this.stateBuffers=[0,1].map(t=>this.device.createBuffer({size:this.stateByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:`paper-state-${t}`})),this.fluxBuffer=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:"paper-flux"}),this.velocityBuffer=this.device.createBuffer({size:this.vec4ByteLength+this.height*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:"paper-velocity"}),this.thermalPipeBufferA=this.device.createBuffer({size:this.vec4ByteLength*2,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,label:"paper-thermal-pipes-a"}),this.thermalPipeBufferB=this.device.createBuffer({size:this.vec4ByteLength*2,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-thermal-pipes-b"}),this.layerTopBuffer=this.device.createBuffer({size:this.vec4ByteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:"paper-layer-top-heights"}),this.readbackBuffer=null,this.allocatedReadbackByteLength=0,this.allocatedStateByteLength=this.stateByteLength,this.allocatedVec4ByteLength=this.vec4ByteLength,this.#g()}}#P(e){let t=Math.max(e,0)*.5,i=Math.max(this.simulationParams.evaporationRate,0),r=Math.max(this.simulationParams.rainDuration??0,0);return{half:t,rate:i,decay:Math.exp(-i*t),fullWeight:i>0?-Math.expm1(-i*t)/i:t,end:r>0?this.simulationParams.sourceTimeOffset+r+Math.max(this.simulationParams.pulse2Duration??0,0):1/0}}#C(e,t,i,r){let{half:n,rate:s,decay:l,fullWeight:a,end:o}=r,h=$(o-i,0,n*2),u=Math.min(h,n),d=Math.max(h-n,0);e[t+40]=u===n?a:s>0?-Math.expm1(-s*u)/s*Math.exp(-s*(n-u)):u,e[t+41]=l,e[t+42]=d===n?a:s>0?-Math.expm1(-s*d)/s*Math.exp(-s*(n-d)):d,e[t+43]=u,e[t+47]=d}#A(e,t=this.paramValues){let i=Math.max(.25,(this.simulationParams.metersPerPixel??100)/100);t[0]=this.width,t[1]=this.height,t[2]=i,t[3]=this.simulationParams.timeStep,t[4]=this.simulationParams.rainRate,t[5]=this.simulationParams.evaporationRate,t[6]=this.simulationParams.pipeArea,t[7]=this.simulationParams.gravity,t[8]=this.simulationParams.capacityScale,t[9]=this.simulationParams.suspensionRate,t[10]=this.simulationParams.depositionRate,t[11]=this.simulationParams.softeningRate,t[12]=this.simulationParams.maxErosionDepth,t[13]=this.simulationParams.thermalRate,t[14]=this.simulationParams.talusSlopeCoeff,t[15]=this.simulationParams.talusSlopeBias,t[16]=this.simulationParams.sourceLayoutMode,t[17]=this.simulationParams.randomSpringCount,t[18]=this.simulationParams.pulse2Duration,t[19]=this.simulationParams.edgeDrainStrength,t[20]=this.simulationParams.sourceCenterX,t[21]=this.simulationParams.sourceCenterY,t[22]=this.simulationParams.sourceRadius,t[23]=this.simulationParams.sourceStrength,t[24]=this.simulationParams.sourceEnabled?1:0,t[25]=this.simulationTime,t[26]=Math.max(0,this.simulationParams.rainDuration??0),t[27]=this.simulationParams.sourceTimeOffset,t[28]=this.simulationParams.historyDecay,t[29]=this.simulationParams.hardnessBase,t[30]=this.simulationParams.closedBoundaries?1:0,t[31]=4;let r=t===this.paramValues?this.paramBits:new Uint32Array(t.buffer,t.byteOffset,t.length);return r[32]=Math.floor(Math.max(0,this.simulationParams.sourceSeed??1))>>>0,t[33]=this.simulationParams.metersPerPixel??100,t[34]=this.simulationParams.hydraulicErosionEnabled?1:0,t[35]=Math.max(0,this.simulationParams.edgeWaterFloor??0),t[36]=this.simulationParams.hardnessBase,t[37]=this.simulationParams.hardnessVariation,t[38]=Math.max(0,this.simulationParams.flowDrag??2),t[39]=$(this.simulationParams.maxOutflowFraction??.5,.05,1),t[44]=Math.max(0,this.simulationParams.bankErosionRate??0),t[45]=Math.max(0,this.simulationParams.bankBendBias??24),t[46]=Math.max(0,this.simulationParams.sourceSedimentConcentration??0),t[47]=0,t[48]=Math.max(0,this.simulationParams.bendSedimentRate??0),t[49]=Math.max(0,this.simulationParams.straightBankErosionFactor??.08),t[50]=Math.max(0,this.simulationParams.riverBendLength??0),t[51]=Math.max(0,Math.min(15,Math.floor(this.simulationParams.openBoundaryEdges??15))),this.#C(t,0,this.simulationTime,this.#P(t[3])),t}#n(){this.#A(this.iterationCount,this.paramValues),this.stepParamTemplateDirty=!0,this.stepParamPreparedIterations=0,this.paramBuffer&&this.device.queue.writeBuffer(this.paramBuffer,0,this.paramValues)}#I(e=1){if(this.#B(e),!this.stepParamValuesCpu||!this.stepParamTemplateDirty&&this.stepParamPreparedIterations>=e)return;let t=this.stepParamStride>>>2,i=this.stepParamTemplateDirty?0:this.stepParamPreparedIterations;for(let r=i;r<e;r++)this.stepParamValuesCpu.set(this.paramValues,r*t);this.stepParamTemplateDirty=!1,this.stepParamPreparedIterations=Math.max(this.stepParamPreparedIterations,e)}#B(e=1){let t=Math.max(1,e|0);if(this.stepParamBuffer&&this.stepParamCapacity>=t&&this.stepParamValuesCpu&&this.stepParamValuesCpuCapacity>=t)return;let i=Math.max(8,t,this.stepParamCapacity>0?this.stepParamCapacity*2:0),r=this.stepParamStride>>>2;(!this.stepParamValuesCpu||this.stepParamValuesCpuCapacity<t)&&(this.stepParamValuesCpu=new Float32Array(r*i),this.stepParamValuesCpuCapacity=i,this.stepParamTemplateDirty=!0,this.stepParamPreparedIterations=0),(!this.stepParamBuffer||this.stepParamCapacity<t)&&(this.stepParamBuffer?.destroy?.(),this.stepParamBuffer=this.device.createBuffer({size:this.stepParamStride*i,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"paper-step-params"}),this.stepParamCapacity=i,this.stepParamTemplateDirty=!0,this.stepParamPreparedIterations=0,this.stateBuffers.length===2&&this.#g())}async#O(){let e=this.device.createShaderModule({code:ai,label:"paper-erosion-compute-wgsl"}),t=this.device.createShaderModule({code:Lt,label:"paper-erosion-render-wgsl"}),i=this.device.createShaderModule({code:oi,label:"paper-erosion-composite-wgsl"}),r=typeof this.device.createComputePipelineAsync=="function"?o=>this.device.createComputePipelineAsync(o):async o=>this.device.createComputePipeline(o),n=typeof this.device.createRenderPipelineAsync=="function"?o=>this.device.createRenderPipelineAsync(o):async o=>this.device.createRenderPipeline(o);this.computeBindGroupLayout=this.device.createBindGroupLayout({label:"paper-compute-bgl",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"uniform",hasDynamicOffset:!0}},{binding:1,visibility:GPUShaderStage.COMPUTE|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]}),this.renderBindGroupLayout=this.device.createBindGroupLayout({label:"paper-render-bgl",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:4,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:5,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}},{binding:6,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]}),this.compositeBindGroupLayout=this.device.createBindGroupLayout({label:"paper-composite-bgl",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});let s=this.device.createPipelineLayout({label:"paper-compute-layout",bindGroupLayouts:[this.computeBindGroupLayout]}),l=this.device.createPipelineLayout({label:"paper-render-layout",bindGroupLayouts:[this.renderBindGroupLayout]}),a=this.device.createPipelineLayout({label:"paper-composite-layout",bindGroupLayouts:[this.compositeBindGroupLayout]});[this.fluxPipeline,this.sourcePipeline,this.flowPipeline,this.erosionPipeline,this.thermalOutflowPipeline,this.channelPipeline,this.channelBendPipeline,this.renderPipeline,this.waterRenderPipeline,this.compositePipeline]=await Promise.all([r({label:"paper-flux-pipeline",layout:s,compute:{module:e,entryPoint:"fluxMain"}}),r({label:"paper-source-pipeline",layout:s,compute:{module:e,entryPoint:"sourceMain"}}),r({label:"paper-flow-pipeline",layout:s,compute:{module:e,entryPoint:"flowMain"}}),r({label:"paper-erosion-pipeline",layout:s,compute:{module:e,entryPoint:"erosionMain"}}),r({label:"paper-thermal-outflow-pipeline",layout:s,compute:{module:e,entryPoint:"thermalOutflowMain"}}),r({label:"river-channel-guide-pipeline",layout:s,compute:{module:e,entryPoint:"channelMain"}}),r({label:"paper-channel-bend-pipeline",layout:s,compute:{module:e,entryPoint:"channelBendMain"}}),n({label:"paper-render-pipeline",layout:l,vertex:{module:t,entryPoint:"vsMesh"},fragment:{module:t,entryPoint:"fsMesh",targets:[{format:this.presentationFormat}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),n({label:"paper-water-render-pipeline",layout:l,vertex:{module:t,entryPoint:"vsMesh"},fragment:{module:t,entryPoint:"fsMesh",targets:[{format:this.presentationFormat,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:!1,depthCompare:"less-equal"}}),n({label:"paper-composite-pipeline",layout:a,vertex:{module:i,entryPoint:"vsComposite"},fragment:{module:i,entryPoint:"fsComposite",targets:[{format:this.presentationFormat}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:!1,depthCompare:"always"}})])}#g(){!this.sceneColorView||!this.sceneSampler||(this.#p(),this.#w(),this.#m(),this.#B(1),!(!this.dummySceneView||!this.stepParamBuffer||!this.customSpringBuffer||!this.layerTopBuffer||!this.layerMaterialBuffer)&&(this.computeBindGroup01=this.device.createBindGroup({label:"paper-compute-0to1",layout:this.computeBindGroupLayout,entries:[{binding:0,resource:{buffer:this.stepParamBuffer,size:this.paramValues.byteLength}},{binding:1,resource:{buffer:this.stateBuffers[0]}},{binding:2,resource:{buffer:this.stateBuffers[1]}},{binding:3,resource:{buffer:this.fluxBuffer}},{binding:4,resource:{buffer:this.velocityBuffer}},{binding:5,resource:{buffer:this.thermalPipeBufferA}},{binding:6,resource:{buffer:this.thermalPipeBufferB}},{binding:7,resource:{buffer:this.customSpringBuffer}},{binding:8,resource:{buffer:this.layerTopBuffer}},{binding:9,resource:{buffer:this.layerMaterialBuffer}}]}),this.computeBindGroup10=this.device.createBindGroup({label:"paper-compute-1to0",layout:this.computeBindGroupLayout,entries:[{binding:0,resource:{buffer:this.stepParamBuffer,size:this.paramValues.byteLength}},{binding:1,resource:{buffer:this.stateBuffers[1]}},{binding:2,resource:{buffer:this.stateBuffers[0]}},{binding:3,resource:{buffer:this.fluxBuffer}},{binding:4,resource:{buffer:this.velocityBuffer}},{binding:5,resource:{buffer:this.thermalPipeBufferA}},{binding:6,resource:{buffer:this.thermalPipeBufferB}},{binding:7,resource:{buffer:this.customSpringBuffer}},{binding:8,resource:{buffer:this.layerTopBuffer}},{binding:9,resource:{buffer:this.layerMaterialBuffer}}]}),this.renderBindGroupTerrain=this.device.createBindGroup({label:"paper-render-terrain",layout:this.renderBindGroupLayout,entries:[{binding:0,resource:{buffer:this.renderUniformBufferTerrain}},{binding:1,resource:{buffer:this.stateBuffers[0]}},{binding:2,resource:this.sceneSampler},{binding:3,resource:this.dummySceneView},{binding:4,resource:{buffer:this.layerTopBuffer}},{binding:5,resource:{buffer:this.layerMaterialBuffer}},{binding:6,resource:{buffer:this.velocityBuffer}}]}),this.renderBindGroupWater=this.device.createBindGroup({label:"paper-render-water",layout:this.renderBindGroupLayout,entries:[{binding:0,resource:{buffer:this.renderUniformBufferWater}},{binding:1,resource:{buffer:this.stateBuffers[0]}},{binding:2,resource:this.sceneSampler},{binding:3,resource:this.sceneColorView},{binding:4,resource:{buffer:this.layerTopBuffer}},{binding:5,resource:{buffer:this.layerMaterialBuffer}},{binding:6,resource:{buffer:this.velocityBuffer}}]}),this.compositeBindGroup=this.device.createBindGroup({label:"paper-composite",layout:this.compositeBindGroupLayout,entries:[{binding:0,resource:this.sceneSampler},{binding:1,resource:this.sceneColorView}]})))}},ai=`
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
`,oi=`
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
`;function Ut(){return Lt}var Lt=`
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
`;var De=class{constructor({budgetMs:e=8,maximumSteps:t=64}={}){this.budgetMs=e,this.maximumSteps=t,this.reset()}reset(e=4){this.steps=Math.max(1,Math.min(this.maximumSteps,Math.floor(Number(e))||1))}observe(e,t){if(!Number.isFinite(t)||t<=0)return;let i=Math.max(.5,Math.min(1.25,this.budgetMs/Math.max(t,.25))),r=Math.round(e*i);this.reset(i>=1.2?Math.max(e+1,r):r)}};var li=`
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
`,Ve=class{constructor(e){this.sim=e,this.buffers=[]}async initialize(e){let{sim:t}=this,i=t.device,r=e.positionsGrid,n=e.indices;if(!r?.length||!n?.length)throw Error("No triangles were produced. Try a finer spacing.");for(let m of n)if(m>=r.length/2)throw Error("Invalid mesh index.");for(let m of r)if(!Number.isFinite(m))throw Error("Invalid mesh position.");this.indexCount=n.length;let s=(m,p)=>{let f=i.createBuffer({size:m.byteLength,usage:p|GPUBufferUsage.COPY_DST});return i.queue.writeBuffer(f,0,m),this.buffers.push(f),f};this.positions=s(r,GPUBufferUsage.STORAGE),this.indices=s(n,GPUBufferUsage.STORAGE|GPUBufferUsage.INDEX),this.style=s(new Float32Array([.55,0,0,0]),GPUBufferUsage.UNIFORM),this.dummyWater=s(new Float32Array(4),GPUBufferUsage.STORAGE),this.dummyGeometry=s(new Float32Array(12),GPUBufferUsage.STORAGE);let l=GPUShaderStage.VERTEX,a=GPUShaderStage.FRAGMENT,o=i.createBindGroupLayout({entries:[{binding:0,visibility:l|a,buffer:{type:"uniform"}},{binding:1,visibility:l|a,buffer:{type:"read-only-storage"}},{binding:2,visibility:a,sampler:{type:"filtering"}},{binding:3,visibility:a,texture:{sampleType:"float"}},{binding:4,visibility:l,buffer:{type:"read-only-storage"}},{binding:5,visibility:l,buffer:{type:"uniform"}},{binding:6,visibility:l,buffer:{type:"read-only-storage"}},{binding:7,visibility:l,buffer:{type:"read-only-storage"}},{binding:8,visibility:l,buffer:{type:"read-only-storage"}},{binding:9,visibility:a,buffer:{type:"uniform"}},{binding:10,visibility:l,buffer:{type:"read-only-storage"}},{binding:11,visibility:l,buffer:{type:"read-only-storage"}}]}),h=i.createPipelineLayout({bindGroupLayouts:[o]}),u=i.createShaderModule({code:Ut()+li}),d={color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha"}},g=(m,p,f,x)=>i.createRenderPipelineAsync({layout:h,vertex:{module:u,entryPoint:m},fragment:{module:u,entryPoint:p,targets:[{format:t.presentationFormat,...f?{blend:d}:{}}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:x,depthCompare:"less-equal"}});return[this.terrain,this.water,this.wire,this.sweWater]=await Promise.all([g("vsPost","fsMesh",!1,!0),g("vsPost","fsMesh",!0,!1),g("vsWire","fsWire",!0,!1),g("vsSWE","fsMesh",!0,!1)]),this.groupLayout=o,this.refreshGroups(),this}refreshGroups(e=null){let t=this.sim,i=t.device,r=[t.stateBuffers[0],t.sceneColorView,t.layerTopBuffer,t.velocityBuffer,e?.states[0],e?.geometry];if(this.groupKey?.every((s,l)=>s===r[l]))return;this.groupKey=r;let n=(s,l=this.dummyWater)=>i.createBindGroup({layout:this.groupLayout,entries:[{binding:0,resource:{buffer:s?t.renderUniformBufferWater:t.renderUniformBufferTerrain}},{binding:1,resource:{buffer:t.stateBuffers[0]}},{binding:2,resource:t.sceneSampler},{binding:3,resource:s?t.sceneColorView:t.dummySceneView},{binding:4,resource:{buffer:t.layerTopBuffer}},{binding:5,resource:{buffer:t.layerMaterialBuffer}},{binding:6,resource:{buffer:t.velocityBuffer}},{binding:7,resource:{buffer:this.positions}},{binding:8,resource:{buffer:this.indices}},{binding:9,resource:{buffer:this.style}},{binding:10,resource:{buffer:l}},{binding:11,resource:{buffer:e?.geometry||this.dummyGeometry}}]});this.terrainGroup=n(!1),this.waterGroup=n(!0),this.sweGroups=e?e.states.map(s=>n(!0,s)):null}render({adaptive:e=!0,wireframe:t=!0,wireOpacity:i=.55,swe:r=null}={}){let n=this.sim,s=n.device.createCommandEncoder();this.refreshGroups(r),n.device.queue.writeBuffer(this.style,0,new Float32Array([Math.max(0,Math.min(1,i)),0,0,0])),n.setSimulationParams({});let l=s.beginRenderPass({colorAttachments:[{view:n.sceneColorView,clearValue:{r:.055,g:.065,b:.08,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:n.depthTextureView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});l.setPipeline(e?this.terrain:n.renderPipeline),l.setBindGroup(0,e?this.terrainGroup:n.renderBindGroupTerrain),e?(l.setIndexBuffer(this.indices,"uint32"),l.drawIndexed(this.indexCount)):l.draw(6,n.drawCount),l.end();let a=s.beginRenderPass({colorAttachments:[{view:n.context.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:.055,g:.065,b:.08,a:1}}],depthStencilAttachment:{view:n.depthTextureView,depthLoadOp:"load",depthStoreOp:"discard"}});a.setPipeline(n.compositePipeline),a.setBindGroup(0,n.compositeBindGroup),a.draw(3),a.setPipeline(r?this.sweWater:e?this.water:n.waterRenderPipeline),a.setBindGroup(0,r?this.sweGroups[r.current]:e?this.waterGroup:n.renderBindGroupWater),r?a.draw(this.indexCount):e?(a.setIndexBuffer(this.indices,"uint32"),a.drawIndexed(this.indexCount)):a.draw(6,n.drawCount),t&&(a.setPipeline(this.wire),a.setBindGroup(0,this.terrainGroup),a.draw(this.indexCount)),a.end(),n.device.queue.submit([s.finish()])}destroy(){for(let e of this.buffers)e.destroy();this.buffers=[]}};var ci=`
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
`,hi=`
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
`;function ui(c,{cellSize:e=1,closedBoundaries:t=!0,openBoundaryEdges:i=15,width:r=0,height:n=0,sources:s=[]}={}){let l=c.positionsGrid,a=c.indices,o=a.length/3,h=new ArrayBuffer(o*48),u=new Float32Array(h),d=new Uint32Array(h),g=[],m=new Map;for(let y=0;y<o;y++){let v=[a[y*3],a[y*3+1],a[y*3+2]],b=v.map(R=>[l[R*2],l[R*2+1]]),w=(b[1][0]-b[0][0])*(b[2][1]-b[0][1])-(b[2][0]-b[0][0])*(b[1][1]-b[0][1]);if(!Number.isFinite(w)||Math.abs(w)<1e-10)throw Error("Water mesh contains a degenerate triangle.");let S=(b[0][0]+b[1][0]+b[2][0])/3,A=(b[0][1]+b[1][1]+b[2][1])/3,C=y*12;u[C]=Math.abs(w)*.5*e*e,u[C+2]=S,u[C+3]=A;let B=0;for(let R of s){let H=Math.max(.5,R.radius||1),z=((S-R.x)**2+(A-R.y)**2)/(H*H);z<1&&(B+=(R.strength||0)*Math.exp(-z*4))}u[C+8]=B;for(let R=0;R<3;R++){let H=v[R],z=v[(R+1)%3],E=Math.min(H,z)*(l.length/2)+Math.max(H,z),F=m.get(E);if(F!==void 0){if(g[F].right!==4294967295)throw Error("Water mesh has a non-manifold edge.");g[F].right=y}else{let M=b[R],k=b[(R+1)%3],I=k[0]-M[0],q=k[1]-M[1],D=Math.hypot(I,q),j=Math.sign(w),O=0;Math.max(M[0],k[0])<1e-4?O=1:Math.min(M[0],k[0])>r-1-1e-4?O=2:Math.max(M[1],k[1])<1e-4?O=4:Math.min(M[1],k[1])>n-1-1e-4&&(O=8),F=g.length,g.push({left:y,right:4294967295,open:!t&&!!(O&i),nx:j*q/D,ny:-j*I/D,len:D*e}),m.set(E,F)}d[C+4+R]=F}}let p=new ArrayBuffer(g.length*32),f=new Float32Array(p),x=new Uint32Array(p);return g.forEach((y,v)=>{x[v*8]=y.left,x[v*8+1]=y.right,x[v*8+2]=y.open?1:0,f[v*8+4]=y.nx,f[v*8+5]=y.ny,f[v*8+6]=y.len}),{geometry:h,edges:p,triangles:o,edgeCount:g.length}}var He=class{constructor(e,t,i){this.sim=e,this.topology=t,this.renderer=i,this.buffers=[],this.current=0,this.steps=0}async initialize(e={}){let t=this.sim,i=t.device,r=t.simulationParams;this.options={maxTimeStep:.02,cfl:.35,roughness:.025,rainRate:0,evaporationRate:0,...e};let n=ui(this.topology,{cellSize:r.cellSize||1,width:t.width,height:t.height,closedBoundaries:r.closedBoundaries,openBoundaryEdges:r.openBoundaryEdges??15,sources:r.sourceEnabled?t.getSourcePoints():[]});this.triangles=n.triangles,this.edgeCount=n.edgeCount,this.groups=Math.ceil(this.triangles/128);let s=new Float32Array(n.geometry);this.areas=Float32Array.from({length:this.triangles},(f,x)=>s[x*12]),this.dispatchCounts=[Math.ceil(this.edgeCount/128),this.groups,1,this.groups];let l=(f,x=GPUBufferUsage.STORAGE)=>{let y=typeof f=="number"?f:f.byteLength,v=i.createBuffer({size:y,usage:x|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});return typeof f!="number"&&i.queue.writeBuffer(v,0,f),this.buffers.push(v),v};this.geometry=l(n.geometry),this.edges=l(n.edges),this.states=[l(this.triangles*16),l(this.triangles*16)],this.fluxes=l(this.edgeCount*16),this.partials=l(this.groups*4),this.clock=l(16),this.uniform=l(48,GPUBufferUsage.UNIFORM);let a=i.createShaderModule({code:ci}),o=i.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},...[1,2,3,4,5,6,7].map(f=>({binding:f,visibility:GPUShaderStage.COMPUTE,buffer:{type:f<4?"read-only-storage":"storage"}}))]}),h=i.createPipelineLayout({bindGroupLayouts:[o]});this.pipelines=await Promise.all(["edgeFlux","localCFL","globalCFL","update"].map(f=>i.createComputePipelineAsync({layout:h,compute:{module:a,entryPoint:f}}))),this.bindings=[0,1].map(f=>i.createBindGroup({layout:o,entries:[this.uniform,this.geometry,this.edges,this.states[f],this.states[1-f],this.fluxes,this.partials,this.clock].map((x,y)=>({binding:y,resource:{buffer:x}}))}));let u=await i.createComputePipelineAsync({layout:"auto",compute:{module:i.createShaderModule({code:hi}),entryPoint:"initialize"}}),d=l(new Uint32Array([this.triangles,t.width,t.height,0,0,0,0,0]),GPUBufferUsage.UNIFORM),g=i.createBindGroup({layout:u.getBindGroupLayout(0),entries:[d,this.geometry,this.renderer.positions,this.renderer.indices,t.stateBuffers[0],this.states[0]].map((f,x)=>({binding:x,resource:{buffer:f}}))}),m=i.createCommandEncoder(),p=m.beginComputePass();return p.setPipeline(u),p.setBindGroup(0,g),p.dispatchWorkgroups(this.groups),p.end(),i.queue.submit([m.finish()]),this.updateOptions({}),this}updateOptions(e={}){Object.assign(this.options,e);let t=this.options,i=new ArrayBuffer(48),r=new Uint32Array(i),n=new Float32Array(i);r.set([this.triangles,this.edgeCount,this.groups,0]),n.set([Math.max(.001,this.sim.simulationParams.gravity||9.81),Math.max(1e-5,Math.min(1,t.maxTimeStep)),Math.max(.05,Math.min(.45,t.cfl)),Math.max(0,t.roughness)],4),n.set([Math.max(0,t.rainRate),Math.max(0,t.evaporationRate),0,0],8),this.sim.device.queue.writeBuffer(this.uniform,0,i)}step(e=1,t=null){let i=this.sim.device,r=i.createCommandEncoder(),n=r.beginComputePass({timestampWrites:t?.writes()});for(let s=0;s<Math.max(1,Math.min(256,e|0));s++){n.setBindGroup(0,this.bindings[this.current]);for(let l=0;l<4;l++)n.setPipeline(this.pipelines[l]),n.dispatchWorkgroups(this.dispatchCounts[l]);this.current=1-this.current,this.steps++}n.end(),t?.resolve(r),i.queue.submit([r.finish()])}async benchmark(e=100){e=Math.max(1,Math.min(256,e|0));let t=this.sim.device,i=t.features.has("timestamp-query")?new xe(t,1):null,r=performance.now();this.step(e,i),await t.queue.onSubmittedWorkDone();let n=performance.now()-r,s=i?await i.read():null;return{steps:e,completionMs:n,gpuMs:s,gpuMsPerStep:s===null?null:s/e}}async readStats(){let e=this.steps,t=this.sim.device,i=t.createBuffer({size:this.triangles*16+16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});try{let r=t.createCommandEncoder();r.copyBufferToBuffer(this.states[this.current],0,i,0,this.triangles*16),r.copyBufferToBuffer(this.clock,0,i,this.triangles*16,16),t.queue.submit([r.finish()]),await i.mapAsync(GPUMapMode.READ);let n=new Float32Array(i.getMappedRange()).slice();i.unmap();let s=0,l=0,a=0,o=0;for(let h=0;h<this.triangles;h++){let u=n[h*4],d=n[h*4+1],g=n[h*4+2];if(!Number.isFinite(u+d+g)||u<0){o++;continue}s+=u*this.areas[h],l=Math.max(l,u),u>1e-7&&(a=Math.max(a,Math.hypot(d,g)/u))}return{values:n.slice(0,-4),time:n.at(-4),dt:n.at(-3),steps:e,totalWater:s,maxDepth:l,maxSpeed:a,invalid:o}}finally{i.destroy()}}destroy(){for(let e of this.buffers)e.destroy();this.buffers=[]}};function Wt(c,e,t,i=2,r=null){if(!Number.isInteger(e)||!Number.isInteger(t)||e<1||t<1||c.length!==e*t||r&&r.length!==c.length||!Number.isFinite(i))throw new Error("Invalid relief raster dimensions or radius.");i=Math.max(0,Math.floor(i));let n=new Float32Array(c.length),s=new Float32Array(c.length),l=new Float32Array(c.length),a=new Int32Array(Math.max(e,t)),o=new Int32Array(a.length);for(let h=0;h<t;h++){let u=0,d=0,g=0,m=0,p=0;for(let f=0;f<e;f++){let x=Math.min(e-1,f+i);for(;p<=x;){let y=h*e+p,v=c[y];if((!r||r[y])&&Number.isFinite(v)){for(;d>u&&c[h*e+a[d-1]]>=v;)d--;for(;m>g&&c[h*e+o[m-1]]<=v;)m--;a[d++]=p,o[m++]=p}p++}for(;d>u&&a[u]<f-i;)u++;for(;m>g&&o[g]<f-i;)g++;n[h*e+f]=d>u?c[h*e+a[u]]:1/0,s[h*e+f]=m>g?c[h*e+o[g]]:-1/0}}for(let h=0;h<e;h++){let u=0,d=0,g=0,m=0,p=0;for(let f=0;f<t;f++){let x=Math.min(t-1,f+i);for(;p<=x;){let v=p*e+h;for(;d>u&&n[a[d-1]*e+h]>=n[v];)d--;for(;m>g&&s[o[m-1]*e+h]<=s[v];)m--;a[d++]=p,o[m++]=p,p++}for(;d>u&&a[u]<f-i;)u++;for(;m>g&&o[g]<f-i;)g++;let y=f*e+h;(!r||r[y])&&Number.isFinite(c[y])&&(l[y]=s[o[g]*e+h]-n[a[u]*e+h])}}return l}var zt=Math.pow(2,-52),it=new Uint32Array(512);function di(c,e,t,i,r,n){let s=c-r,l=e-n,a=t-r,o=i-n;return l*a-s*o}function mi(c,e,t,i,r,n){let s=t-c,l=i-e,a=r-c,o=n-e,h=s*s+l*l,u=a*a+o*o,d=s*o-l*a;if(d===0)return 1/0;let g=.5/d,m=(o*h-l*u)*g,p=(s*u-a*h)*g;return m*m+p*p}function Fe(c,e,t){let i=Math.abs(c)+Math.abs(e);if(i===0)return 0;let r=c/i;return((e>0?3-r:1+r)*t*.25|0)%t}function fi(c,e,t,i){for(let r=t+1;r<=i;r++){let n=c[r],s=e[n],l=r-1;for(;l>=t&&e[c[l]]>s;)c[l+1]=c[l],l--;c[l+1]=n}}function Ie(c,e,t){let i=c[e];c[e]=c[t],c[t]=i}function Oe(c,e,t,i){if(i-t<=20){fi(c,e,t,i);return}let r=t+i>>1;e[c[t]]>e[c[i]]&&Ie(c,t,i),e[c[t]]>e[c[r]]&&Ie(c,t,r),e[c[r]]>e[c[i]]&&Ie(c,r,i);let n=c[r],s=e[n];Ie(c,r,i);let l=t;for(let h=t;h<i;h++)e[c[h]]<s&&(Ie(c,h,l),l++);Ie(c,l,i);let a=l-t,o=i-l;a<o?(Oe(c,e,t,l-1),Oe(c,e,l+1,i)):(Oe(c,e,l+1,i),Oe(c,e,t,l-1))}function pi(c){return c[0]}function gi(c){return c[1]}var Ue=class c{coords;triangles;halfedges;hull;pointCount;capacity;_cx;_cy;_hullStart;_hullPrev;_hullNext;_hullTri;_hullHash;_ids;_dists;_triangles;_halfedges;_hashSize;trianglesLen;static from(e,t=pi,i=gi,r=e.length){let n=e.length;if(r<n)throw new RangeError(`capacity ${r} is smaller than point count ${n}`);let s=new Float64Array(r*2);for(let l=0,a=0;l<n;l++,a+=2){let o=e[l];s[a]=t(o),s[a+1]=i(o)}return new c(s,n)}static withCapacity(e){return new c(e,0)}constructor(e,t){if(typeof e=="number"){let i=Math.max(0,Math.floor(e));if(this.coords=new Float64Array(i*2),this.capacity=i,this.pointCount=t===void 0?0:Math.max(0,Math.floor(t)),this.pointCount>this.capacity)throw new RangeError(`pointCount ${this.pointCount} exceeds capacity ${this.capacity}`)}else if(this.coords=e,this.capacity=e.length>>1,this.pointCount=t===void 0?this.capacity:Math.max(0,Math.floor(t)),this.pointCount>this.capacity)throw new RangeError(`pointCount ${this.pointCount} exceeds capacity ${this.capacity}`);this._allocate(this.capacity),this.update()}get activeCoords(){return this.coords.subarray(0,this.pointCount<<1)}clear(){this.pointCount=0,this.update()}reserve(e){let t=Math.max(0,Math.floor(e));if(t<=this.capacity)return;let i=new Float64Array(t*2);i.set(this.coords.subarray(0,this.pointCount<<1)),this.coords=i,this.capacity=t,this._allocate(t)}ensureCapacity(e){let t=Math.max(0,Math.floor(e));if(t<=this.capacity)return;let i=this.capacity>0?this.capacity:1;for(;i<t;)i*=2;this.reserve(i)}setPointCount(e){let t=Math.max(0,Math.floor(e));this.ensureCapacity(t),this.pointCount=t}setPoint(e,t,i){let r=Math.floor(e);if(r<0)throw new RangeError(`index ${e} must be >= 0`);this.ensureCapacity(r+1);let n=r<<1;this.coords[n]=t,this.coords[n+1]=i,r>=this.pointCount&&(this.pointCount=r+1)}setPointsFlat(e,t=e.length>>1){let i=Math.max(0,Math.floor(t));this.setPointCount(i);let r=i<<1;for(let n=0;n<r;n++)this.coords[n]=e[n]}update(e=this.pointCount){this.setPointCount(e);let t=this.coords,i=this._hullPrev,r=this._hullNext,n=this._hullTri,s=this._hullHash,l=this._ids,a=this._dists,o=this.pointCount;if(o===0){this.hull=new Uint32Array(0),this.trianglesLen=0,this.triangles=this._triangles.subarray(0,0),this.halfedges=this._halfedges.subarray(0,0);return}if(o<3){let T=new Uint32Array(o);for(let P=0;P<o;P++)T[P]=P;this.hull=T,this.trianglesLen=0,this.triangles=this._triangles.subarray(0,0),this.halfedges=this._halfedges.subarray(0,0);return}let h=1/0,u=1/0,d=-1/0,g=-1/0;for(let T=0,P=0;T<o;T++,P+=2){let N=t[P],G=t[P+1];N<h&&(h=N),G<u&&(u=G),N>d&&(d=N),G>g&&(g=G),l[T]=T}let m=(h+d)*.5,p=(u+g)*.5,f=0,x=1/0;for(let T=0,P=0;T<o;T++,P+=2){let N=m-t[P],G=p-t[P+1],K=N*N+G*G;K<x&&(f=T,x=K)}let y=f<<1,v=t[y],b=t[y+1],w=-1;x=1/0;for(let T=0,P=0;T<o;T++,P+=2){if(T===f)continue;let N=v-t[P],G=b-t[P+1],K=N*N+G*G;K>0&&K<x&&(w=T,x=K)}if(w===-1){this.hull=new Uint32Array([f]),this.trianglesLen=0,this.triangles=this._triangles.subarray(0,0),this.halfedges=this._halfedges.subarray(0,0);return}let S=w<<1,A=t[S],C=t[S+1],B=-1,R=1/0;for(let T=0,P=0;T<o;T++,P+=2){if(T===f||T===w)continue;let N=mi(v,b,A,C,t[P],t[P+1]);N<R&&(B=T,R=N)}if(B===-1||R===1/0){let T=t[0],P=t[1];for(let X=0,re=0;X<o;X++,re+=2)a[X]=t[re]-T||t[re+1]-P;Oe(l,a,0,o-1);let N=new Uint32Array(o),G=0,K=-1/0;for(let X=0;X<o;X++){let re=l[X],_=a[re];_>K&&(N[G++]=re,K=_)}this.hull=N.subarray(0,G),this.trianglesLen=0,this.triangles=this._triangles.subarray(0,0),this.halfedges=this._halfedges.subarray(0,0);return}let H=B<<1,z=t[H],E=t[H+1];if(di(v,b,A,C,z,E)<0){let T=w;w=B,B=T;let P=A;A=z,z=P;let N=C;C=E,E=N,S=w<<1,H=B<<1}let F=A-v,M=C-b,k=z-v,I=E-b,q=F*F+M*M,D=k*k+I*I,O=.5/(F*I-M*k),Y=v+(I*q-M*D)*O,V=b+(F*D-k*q)*O;this._cx=Y,this._cy=V;for(let T=0,P=0;T<o;T++,P+=2){let N=t[P]-Y,G=t[P+1]-V;a[T]=N*N+G*G}Oe(l,a,0,o-1),this._hullStart=f;let Q=3;r[f]=w,i[B]=w,r[w]=B,i[f]=B,r[B]=f,i[w]=f,n[f]=0,n[w]=1,n[B]=2,s.fill(-1);let ie=this._hashSize;s[Fe(v-Y,b-V,ie)]=f,s[Fe(A-Y,C-V,ie)]=w,s[Fe(z-Y,E-V,ie)]=B,this.trianglesLen=0,this._addTriangle(f,w,B,-1,-1,-1);let ge=0,Se=0;for(let T=0;T<o;T++){let P=l[T],N=P<<1,G=t[N],K=t[N+1];if(T>0&&Math.abs(G-ge)<=zt&&Math.abs(K-Se)<=zt||(ge=G,Se=K,P===f||P===w||P===B))continue;let X=-1,re=Fe(G-Y,K-V,ie);for(let he=0;he<ie&&(X=s[re],!(X!==-1&&X!==r[X]));he++)re++,re===ie&&(re=0);if(X===-1)continue;X=i[X];let _=X,te=0;for(;;){te=r[_];let he=_<<1,Pe=te<<1,Ce=t[Pe],Ae=t[Pe+1];if((K-Ae)*(t[he]-Ce)-(G-Ce)*(t[he+1]-Ae)<0)break;if(_=te,_===X){_=-1;break}}if(_===-1)continue;let Te=this._addTriangle(_,P,r[_],-1,-1,n[_]);n[P]=this._legalize(Te+2),n[_]=Te,Q++;let ye=r[_];for(;;){te=r[ye];let he=ye<<1,Pe=te<<1,Ce=t[Pe],Ae=t[Pe+1];if((K-Ae)*(t[he]-Ce)-(G-Ce)*(t[he+1]-Ae)>=0)break;Te=this._addTriangle(ye,P,te,n[P],-1,n[ye]),n[P]=this._legalize(Te+2),r[ye]=ye,Q--,ye=te}if(_===X)for(;;){te=i[_];let he=te<<1,Pe=_<<1,Ce=t[Pe],Ae=t[Pe+1];if((K-Ae)*(t[he]-Ce)-(G-Ce)*(t[he+1]-Ae)>=0)break;Te=this._addTriangle(te,P,_,-1,n[_],n[te]),this._legalize(Te+2),n[te]=Te,r[_]=_,Q--,_=te}this._hullStart=i[P]=_,r[_]=i[ye]=P,r[P]=ye,s[Fe(G-Y,K-V,ie)]=P;let Tt=_<<1;s[Fe(t[Tt]-Y,t[Tt+1]-V,ie)]=_}let ce=new Uint32Array(Q);for(let T=0,P=this._hullStart;T<Q;T++)ce[T]=P,P=r[P];this.hull=ce,this.triangles=this._triangles.subarray(0,this.trianglesLen),this.halfedges=this._halfedges.subarray(0,this.trianglesLen)}_allocate(e){let t=Math.max(1,Math.ceil(Math.sqrt(e)));this._hashSize=t,this._hullPrev=new Uint32Array(e),this._hullNext=new Uint32Array(e),this._hullTri=new Uint32Array(e),this._hullHash=new Int32Array(t),this._ids=new Uint32Array(e),this._dists=new Float64Array(e);let i=Math.max(2*e-5,0);this._triangles=new Uint32Array(i*3),this._halfedges=new Int32Array(i*3),this.trianglesLen=0}_legalize(e){let t=this._triangles,i=this._halfedges,r=this.coords,n=this._hullStart,s=this._hullTri,l=this._hullPrev,a=0,o=0;for(;;){let h=i[e];if(h===-1){if(a===0)break;e=it[--a];continue}let u=e-e%3,d=e-u;o=d===0?u+2:e-1;let g=d===2?u:e+1,m=h-h%3,p=h-m,f=p===0?m+2:h-1,x=p===2?m:h+1,y=t[o],v=t[e],b=t[g],w=t[f],S=y<<1,A=v<<1,C=b<<1,B=w<<1,R=r[S],H=r[S+1],z=r[A],E=r[A+1],F=r[C],M=r[C+1],k=r[B],I=r[B+1],q=R-k,D=H-I,j=z-k,O=E-I,Y=F-k,V=M-I,Q=q*q+D*D,ie=j*j+O*O,ge=Y*Y+V*V;if(q*(O*ge-ie*V)-D*(j*ge-ie*Y)+Q*(j*V-O*Y)<0){t[e]=w,t[h]=y;let ce=i[f];if(ce===-1){let P=n;do{if(s[P]===f){s[P]=e;break}P=l[P]}while(P!==n)}i[e]=ce,ce!==-1&&(i[ce]=e);let T=i[o];i[h]=T,T!==-1&&(i[T]=h),i[o]=f,i[f]=o,a<it.length&&(it[a++]=x)}else{if(a===0)break;e=it[--a]}}return o}_addTriangle(e,t,i,r,n,s){let l=this.trianglesLen,a=this._triangles,o=this._halfedges;return a[l]=e,a[l+1]=t,a[l+2]=i,o[l]=r,r!==-1&&(o[r]=l),o[l+1]=n,n!==-1&&(o[n]=l+1),o[l+2]=s,s!==-1&&(o[s]=l+2),this.trianglesLen=l+3,l}};var ue=1e-6,me=256,rt=1e9;function U(c,e,t){return c<e?e:c>t?t:c}function ae(c,e,t){return c+(e-c)*t}function nt(c,e,t){let i=U((t-c)/Math.max(e-c,ue),0,1);return i*i*(3-2*i)}function yi(c,e,t,i=16){if(i<=1||t<=e+ue)return c;let r=U((c-e)/Math.max(t-e,ue),0,1),n=Math.round(r*(i-1))/(i-1);return ae(e,t,n)}function qe(c){return Number.isFinite(c)}function W(c,e,t){return e*t+c}function xi(c,e){let t=new Uint8Array(c*e);return t.fill(1),t}function Nt(c,e,t,i,r){let n=W(i,r,e);if(c[n]===0)return!1;if(i===0||r===0||i===e-1||r===t-1)return!0;let s=W(i,r-1,e),l=W(i,r+1,e),a=W(i-1,r,e),o=W(i+1,r,e);return c[s]===0||c[l]===0||c[a]===0||c[o]===0}function vi(c,e){let t=1/0,i=-1/0;for(let r=0;r<c.length;r++){if(e&&e[r]===0)continue;let n=c[r];qe(n)&&(n<t&&(t=n),n>i&&(i=n))}return!qe(t)||!qe(i)?{min:0,max:1}:Math.abs(i-t)<=ue?{min:t,max:t+1}:{min:t,max:i}}function ve(c,e){let t=new Float32Array(c.length),{min:i,max:r}=vi(c,e),n=1/Math.max(r-i,ue);for(let s=0;s<c.length;s++)e&&e[s]===0||(t[s]=U((c[s]-i)*n,0,1));return t}function bi(c,e){let t=new Float32Array(c.length);for(let i=0;i<c.length;i++)t[i]=Math.abs(c[i]);return ve(t,e)}function wi(c,e,t,i,r){let n=U(Math.floor(i),0,e-1),s=U(Math.floor(r),0,t-1),l=U(n+1,0,e-1),a=U(s+1,0,t-1),o=U(i-n,0,1),h=U(r-s,0,1),u=c[W(n,s,e)],d=c[W(l,s,e)],g=c[W(n,a,e)],m=c[W(l,a,e)],p=ae(u,d,o),f=ae(g,m,o);return ae(p,f,h)}function Mi(c,e,t,i,r){let n=U(Math.round(i),0,e-1),s=U(Math.round(r),0,t-1);return c[W(n,s,e)]}function Si(c){return c.complete&&c.naturalWidth>0&&c.naturalHeight>0?Promise.resolve():typeof c.decode=="function"?c.decode().catch(()=>new Promise((e,t)=>{c.onload=()=>e(),c.onerror=()=>t(new Error("Failed to decode image source."))})):new Promise((e,t)=>{c.onload=()=>e(),c.onerror=()=>t(new Error("Failed to load image source."))})}async function Pi(c,e={}){let t=e.valueFromRGBA??((u,d,g)=>(u+d+g)/765),i=0,r=0,n;if(typeof ImageData<"u"&&c instanceof ImageData)i=c.width,r=c.height,n=c;else{if(typeof HTMLImageElement<"u"&&c instanceof HTMLImageElement)await Si(c),i=c.naturalWidth||c.width,r=c.naturalHeight||c.height;else if(typeof ImageBitmap<"u"&&c instanceof ImageBitmap)i=c.width,r=c.height;else if(typeof HTMLCanvasElement<"u"&&c instanceof HTMLCanvasElement)i=c.width,r=c.height;else throw new Error("Unsupported image source.");let u,d;if(typeof OffscreenCanvas<"u")u=new OffscreenCanvas(i,r),d=u.getContext("2d");else if(typeof document<"u")u=document.createElement("canvas"),u.width=i,u.height=r,d=u.getContext("2d");else throw new Error("No canvas implementation available for image rasterization.");if(!d)throw new Error("Failed to acquire a 2D canvas context.");d.drawImage(c,0,0,i,r),n=d.getImageData(0,0,i,r)}let s=new Float32Array(i*r),l=new Uint8Array(i*r),a=n.data,o=!!e.flipY,h=!!e.alphaMeansInvalid;for(let u=0;u<r;u++){let d=o?r-1-u:u;for(let g=0;g<i;g++){let m=(d*i+g)*4,p=W(g,u,i),f=a[m],x=a[m+1],y=a[m+2],v=a[m+3];s[p]=t(f,x,y,v,p),l[p]=h&&v===0?0:1}}return{width:i,height:r,values:s,mask:l}}var Ye=class{options;width=0;height=0;heights=null;mask=null;analysis=null;topology=null;constructor(e={}){let t=Math.max(e.cellSize??1,ue),i=Math.max(e.minSpacing??t,ue),r=Math.max(e.maxSpacing??t*16,i);this.options={mode:e.mode??"hybrid",cellSize:t,worldMinX:e.worldMinX??0,worldMinY:e.worldMinY??0,minSpacing:i,maxSpacing:r,borderSpacing:Math.max(e.borderSpacing??r,ue),wetnessBins:Math.max(4,Math.floor(e.wetnessBins??21)),channelThreshold:Math.max(1,Math.floor(e.channelThreshold??128)),floodplainChannelDistanceCells:Math.max(1,Math.floor(e.floodplainChannelDistanceCells??6)),candidateStride:Math.max(1,Math.floor(e.candidateStride??1)),maxPoints:Math.max(3,Math.floor(e.maxPoints??Number.MAX_SAFE_INTEGER)),preserveBorder:e.preserveBorder??!0,preserveChannels:e.preserveChannels??!0,preserveMaskBoundary:e.preserveMaskBoundary??!0,preserveInnerBorder:e.preserveInnerBorder??!0,slopeWeight:Math.max(0,e.slopeWeight??.35),curvatureWeight:Math.max(0,e.curvatureWeight??.15),reliefWeight:Math.max(0,e.reliefWeight??.2),wetnessWeight:Math.max(0,e.wetnessWeight??.4),channelWeight:Math.max(0,e.channelWeight??.2),floodplainWeight:Math.max(0,e.floodplainWeight??.25),relaxIterations:Math.max(0,Math.floor(e.relaxIterations??0)),forcedWorldPoints:e.forcedWorldPoints??null}}setRaster(e){if(!e||e.width<=0||e.height<=0)throw new Error("Raster must have positive width and height.");if(!(e.values instanceof Float32Array))throw new Error("Raster values must be a Float32Array.");if(e.values.length!==e.width*e.height)throw new Error("Raster value count does not match width * height.");let t=e.mask&&e.mask.length===e.width*e.height?e.mask instanceof Uint8Array?e.mask:new Uint8Array(e.mask):xi(e.width,e.height);this.width=e.width,this.height=e.height,this.heights=e.values,this.mask=t,this.invalidate()}async setImage(e,t={}){let i=await Pi(e,t);this.setRaster(i)}setHeightValue(e,t,i){this.ensureRaster();let r=U(Math.floor(e),0,this.width-1),n=U(Math.floor(t),0,this.height-1),s=W(r,n,this.width);this.heights[s]=i,this.analysis=null,this.topology=null}async buildWithForcedWorldPoints(e){this.ensureRaster();let t=this.analysis??this.buildAnalysis(),i=this.options.forcedWorldPoints;try{this.options.forcedWorldPoints=e??i??null;let r=this.buildTopology(t);return this.analysis=t,this.topology=r,r}finally{this.options.forcedWorldPoints=i}}getAnalysis(){return this.analysis}getTopology(){return this.topology}async build(){this.ensureRaster();let e=this.buildAnalysis(),t=this.buildTopology(e);return this.analysis=e,this.topology=t,t}async rebuild(){return this.build()}resampleVertexHeights(){if(this.ensureRaster(),!this.topology)throw new Error("Build topology before resampling heights.");let e=this.sampleVertexHeights(this.topology.positionsGrid);return this.topology.heights=e,e}invalidate(){this.analysis=null,this.topology=null}ensureRaster(){if(!this.heights||!this.mask||this.width<=0||this.height<=0)throw new Error("No heightfield raster is loaded.")}buildAnalysis(){let e=this.width,t=this.height,i=this.options.cellSize,r=this.heights,n=this.mask,s=this.computeSlopeTan(r,e,t,i,n),l=new Float32Array(e*t);for(let B=0;B<s.length;B++)l[B]=Math.atan(s[B])*(180/Math.PI);let a=this.computeCurvature(r,e,t,i,n),o=this.computeLocalRelief(r,e,t,n),h=this.computeFlowReceiver(r,e,t,i,n),u=this.computeFlowAccumulation(h,e,t,n),d=this.computeWetness(u,s,e,t,i,n),g=this.extractChannels(u,e,t,n),m=this.computeChannelDistance(g,e,t,n),p=this.extractFloodplain(d,s,m,e,t,n),f=this.computeTerrainPriority(s,a,o,n),x=this.computeHydroPriority(d,g,m,p,n),y=0,v=0;for(let B=0;B<n.length;B++)n[B]&&y++,g[B]&&v++;let b=y*i*i,w=v*i,S=w>ue?b/(2*w):this.options.maxSpacing,A=this.computeImportance(f,x,n),C=this.computeSpacing(f,x,g,p,n,S);return{width:e,height:t,cellSize:i,heights:r,mask:n,slopeTan:s,slopeDegrees:l,curvature:a,localRelief:o,flowReceiver:h,flowAccumulation:u,wetness:d,channels:g,channelDistance:m,floodplain:p,terrainPriority:f,hydroPriority:x,importance:A,spacing:C,validCellCount:y,totalArea:b,totalChannelLength:w,hillslopeLength:S}}computeSlopeTan(e,t,i,r,n){let s=new Float32Array(t*i),l=1/(2*r);for(let a=0;a<i;a++){let o=a>0?a-1:a,h=a<i-1?a+1:a;for(let u=0;u<t;u++){let d=W(u,a,t);if(!n[d])continue;let g=u>0?u-1:u,m=u<t-1?u+1:u,p=e[W(g,a,t)],f=e[W(m,a,t)],x=e[W(u,o,t)],y=e[W(u,h,t)],v=(f-p)*l,b=(y-x)*l;s[d]=Math.hypot(v,b)}}return s}computeCurvature(e,t,i,r,n){let s=new Float32Array(t*i),l=1/(r*r);for(let a=0;a<i;a++){let o=a>0?a-1:a,h=a<i-1?a+1:a;for(let u=0;u<t;u++){let d=W(u,a,t);if(!n[d])continue;let g=u>0?u-1:u,m=u<t-1?u+1:u,p=e[d],f=e[W(g,a,t)],x=e[W(m,a,t)],y=e[W(u,o,t)],v=e[W(u,h,t)];s[d]=(f+x+y+v-4*p)*l}}return s}computeLocalRelief(e,t,i,r){return Wt(e,t,i,1,r)}computeFlowReceiver(e,t,i,r,n){let s=new Int32Array(t*i);s.fill(-1);let l=Math.SQRT2*r;for(let a=0;a<i;a++)for(let o=0;o<t;o++){let h=W(o,a,t);if(!n[h])continue;let u=e[h],d=-1,g=0;for(let m=-1;m<=1;m++)for(let p=-1;p<=1;p++){if(p===0&&m===0)continue;let f=o+p,x=a+m;if(f<0||x<0||f>=t||x>=i)continue;let y=W(f,x,t);if(!n[y])continue;let v=u-e[y];if(v<=0)continue;let w=v/(p!==0&&m!==0?l:r);w>g&&(g=w,d=y)}s[h]=d}return s}computeFlowAccumulation(e,t,i,r){let n=t*i,s=new Int32Array(n),l=new Float32Array(n),a=new Int32Array(n);for(let u=0;u<n;u++){if(!r[u])continue;l[u]=1;let d=e[u];d>=0&&s[d]++}let o=0,h=0;for(let u=0;u<n;u++)r[u]&&s[u]===0&&(a[h++]=u);for(;o<h;){let u=a[o++],d=e[u];d>=0&&(l[d]+=l[u],s[d]--,s[d]===0&&(a[h++]=d))}return l}computeWetness(e,t,i,r,n,s){let l=new Float32Array(i*r);for(let a=0;a<l.length;a++){if(!s[a])continue;let o=e[a]*n;l[a]=Math.log(Math.max(o,ue)/Math.max(t[a],ue))}return l}extractChannels(e,t,i,r){let n=new Uint8Array(t*i),s=this.options.channelThreshold;for(let l=0;l<n.length;l++)r[l]&&e[l]>=s&&(n[l]=1);return n}computeChannelDistance(e,t,i,r){let n=new Float32Array(t*i),s=new Int32Array(t*i),l=0,a=0;for(let o=0;o<n.length;o++){if(!r[o]){n[o]=rt;continue}e[o]?(n[o]=0,s[a++]=o):n[o]=rt}for(;l<a;){let o=s[l++],h=o%t,u=o/t|0,d=n[o];for(let g=-1;g<=1;g++)for(let m=-1;m<=1;m++){if(m===0&&g===0)continue;let p=h+m,f=u+g;if(p<0||f<0||p>=t||f>=i)continue;let x=W(p,f,t);if(!r[x])continue;let v=d+(m!==0&&g!==0?Math.SQRT2:1);v<n[x]&&(n[x]=v,s[a++]=x)}}return n}extractFloodplain(e,t,i,r,n,s){let l=ve(e,s),a=ve(t,s),o=new Uint8Array(r*n),h=this.options.floodplainChannelDistanceCells;for(let u=0;u<o.length;u++){if(!s[u])continue;let d=i[u]<=h,g=l[u]>=.58,m=a[u]<=.42;d&&g&&m&&(o[u]=1)}return o}computeTerrainPriority(e,t,i,r){let n=ve(e,r),s=bi(t,r),l=ve(i,r),a=new Float32Array(e.length);for(let o=0;o<a.length;o++){if(!r[o])continue;let h=n[o]*.48+l[o]*.34+s[o]*.18,u=Math.max(n[o],l[o]),d=Math.max(s[o],(n[o]+l[o])*.5),g=Math.max(h,u*.97,d*.9);a[o]=U(Math.pow(g,.7),0,1)}return ve(a,r)}computeHydroPriority(e,t,i,r,n){let s=ve(e,n),l=new Float32Array(e.length),a=0;for(let o=0;o<i.length;o++)n[o]&&i[o]<rt&&i[o]>a&&(a=i[o]);for(let o=0;o<l.length;o++){if(!n[o])continue;let h=i[o]>=rt?a:i[o],u=1-U(h/Math.max(a,1),0,1),d=s[o]*.46+u*.18+t[o]*.22+r[o]*.14;l[o]=U(d,0,1)}return ve(l,n)}computeImportance(e,t,i){let r=new Float32Array(e.length),n=this.options.mode;for(let s=0;s<r.length;s++){if(!i[s])continue;let l=e[s],a=t[s];n==="traditional"?r[s]=l:n==="hydrological"?r[s]=Math.max(l*.8,a):n==="hydrographic"?r[s]=Math.max(l,a*.55)*.96+Math.min(l,a)*.04:r[s]=Math.max(l,a*.5)*.97+Math.min(l,a)*.03,r[s]=U(r[s],0,1)}return ve(r,i)}computeSpacing(e,t,i,r,n,s){let l=new Float32Array(e.length),a=this.options.cellSize,o=Math.max(this.options.minSpacing,a),h=Math.max(o,Math.min(this.options.maxSpacing,Math.max(s,o))),u=this.options.mode;for(let d=0;d<l.length;d++){if(!n[d])continue;let g=U(e[d],0,1),m=U(t[d],0,1),p=nt(.08,.94,g),f=ae(h,o,Math.pow(p,.62)),x=f;if(u==="hydrological"){let y=nt(.42,.97,m),v=ae(h,o*1.15,Math.pow(y,1.15));x=Math.min(f,v)}else if(i[d]){let y=Math.max(.82,nt(.45,.98,m)),v=ae(h,o*1.02,Math.pow(y,1.05));x=Math.min(x,v)}else if(r[d]){let y=Math.max(.62,nt(.45,.95,m)),v=ae(h,o*1.28,Math.pow(y,1.15));x=Math.min(x,v)}else if(m>=.9&&g>=.35){let y=ae(h,o*1.6,Math.pow(m,1.35));x=Math.min(x,y)}g>=.82?x=Math.min(x,ae(h,o,.92)):g>=.68?x=Math.min(x,ae(h,o,.8)):g>=.52&&(x=Math.min(x,ae(h,o,.64))),l[d]=U(x,o,h)}return l}buildTopology(e){let t=[],i=new Uint8Array(e.width*e.height),r=Math.max(e.cellSize,this.options.minSpacing*.5,this.options.maxSpacing*.25),n=this.options.worldMinX,s=this.options.worldMinY,l=n+Math.max(0,e.width-1)*e.cellSize,a=s+Math.max(0,e.height-1)*e.cellSize,o=Math.max(1,Math.floor((l-n)/r)+1),h=Math.max(1,Math.floor((a-s)/r)+1),u=Array.from({length:o*h},()=>[]),d=(E,F,M,k)=>{if(E<0||F<0||E>=e.width||F>=e.height)return!1;let I=W(E,F,e.width);if(!e.mask[I]||i[I])return!1;let q=this.options.worldMinX+E*e.cellSize,D=this.options.worldMinY+F*e.cellSize,j=U(Math.floor((q-n)/r),0,o-1),O=U(Math.floor((D-s)/r),0,h-1),Y=Math.max(M,r*.75),V=Math.max(1,Math.ceil(Y/r)),Q=Math.max(0,j-V),ie=Math.min(o-1,j+V),ge=Math.max(0,O-V),Se=Math.min(h-1,O+V);for(let T=ge;T<=Se;T++){let P=T*o;for(let N=Q;N<=ie;N++){let G=u[P+N];for(let K=0;K<G.length;K++){let X=t[G[K]],re=q-X.x,_=D-X.y,te=Math.max(M,X.radius);if(re*re+_*_<te*te)return!1}}}let ce=t.length;return t.push({x:q,y:D,gx:E,gy:F,radius:M,fixed:k}),i[I]=1,u[O*o+j].push(ce),!0};if(this.emitForcedPoints(e,d),this.emitAdaptivePoints(e,d),t.length<3)throw new Error("Adaptive sampling retained fewer than 3 valid points.");this.options.relaxIterations>0&&this.relaxAcceptedPoints(t,e);let g=t.length,m=new Float32Array(g*2),p=new Float32Array(g*2),f=new Float32Array(g*2),x=new Float32Array(g),y=new Float32Array(g),v=new Uint8Array(g);for(let E=0;E<g;E++){let F=t[E],M=E*2;m[M]=F.x,m[M+1]=F.y,p[M]=F.gx,p[M+1]=F.gy,f[M]=e.width>1?F.gx/(e.width-1):0,f[M+1]=e.height>1?F.gy/(e.height-1):0;let k=U(Math.round(F.gx),0,e.width-1),I=U(Math.round(F.gy),0,e.height-1),q=W(k,I,e.width);x[E]=e.importance[q],y[E]=e.wetness[q],v[E]=Nt(e.mask,e.width,e.height,k,I)?1:0}let b=new Float64Array(g*2);for(let E=0;E<m.length;E++)b[E]=m[E];let w=new Ue(b),S=this.filterTrianglesToMask(w.triangles,p,e),A=this.buildHalfedges(S),{centers:C,areas:B}=this.buildTriangleGeometry(S,m),{offsets:R,neighbors:H}=this.buildVertexAdjacency(S,g),z=this.sampleVertexHeights(p);return{positions2D:m,positionsGrid:p,uvs:f,heights:z,indices:S,halfedges:A,triangleCenters:C,triangleAreas:B,vertexNeighborOffsets:R,vertexNeighbors:H,vertexBoundaryMask:v,vertexImportance:x,vertexWetness:y,analysis:e}}emitForcedPoints(e,t){let i=Math.max(this.options.minSpacing,e.cellSize),r=Math.max(this.options.maxSpacing,i),n=Math.max(1,Math.round(this.options.borderSpacing/e.cellSize)),s=Math.max(1,Math.round(r/e.cellSize)),l=Math.max(2,Math.round(i*2.25/e.cellSize)),a=Math.max(1,Math.round(r/e.cellSize)),o=Math.max(r*.5,e.cellSize),h=Math.max(r*.5,e.cellSize),u=Math.max(i*1,e.cellSize),d=Math.max(i*.35,e.cellSize*.5);if(this.options.preserveBorder){for(let m=0;m<e.width;m+=n)t(m,0,o,!0),t(m,e.height-1,o,!0);for(let m=0;m<e.height;m+=n)t(0,m,o,!0),t(e.width-1,m,o,!0);if(t(0,0,o,!0),t(e.width-1,0,o,!0),t(0,e.height-1,o,!0),t(e.width-1,e.height-1,o,!0),this.options.preserveInnerBorder&&e.width>2&&e.height>2){let m=U(a,1,e.width-2),p=U(e.width-1-a,1,e.width-2),f=U(a,1,e.height-2),x=U(e.height-1-a,1,e.height-2);for(let y=m;y<=p;y+=s)t(y,f,h,!0),t(y,x,h,!0);for(let y=f;y<=x;y+=s)t(m,y,h,!0),t(p,y,h,!0)}}if(this.options.preserveMaskBoundary)for(let m=0;m<e.height;m++)for(let p=0;p<e.width;p++){if(!Nt(e.mask,e.width,e.height,p,m))continue;(p===0||m===0||p===e.width-1||m===e.height-1||p%s===0||m%s===0)&&t(p,m,h,!0)}if(this.options.preserveChannels)for(let m=0;m<e.height;m++)for(let p=0;p<e.width;p++){let f=W(p,m,e.width);if(!e.channels[f])continue;let x=this.countMarkedNeighbors(e.channels,e.width,e.height,p,m),y=x<=1,v=x>=3,b=p%l===0&&m%l===0||(p+m)%l===0&&x===2;(y||v||b)&&t(p,m,u,!0)}let g=this.options.forcedWorldPoints;if(g&&g.length>=2)for(let m=0;m+1<g.length;m+=2){let p=Number(g[m]),f=Number(g[m+1]);if(!qe(p)||!qe(f))continue;let x=Math.round((p-this.options.worldMinX)/e.cellSize),y=Math.round((f-this.options.worldMinY)/e.cellSize);t(x,y,d,!0)}}countMarkedNeighbors(e,t,i,r,n){let s=0;for(let l=-1;l<=1;l++)for(let a=-1;a<=1;a++){if(a===0&&l===0)continue;let o=r+a,h=n+l;o<0||h<0||o>=t||h>=i||e[W(o,h,t)]!==0&&s++}return s}emitAdaptivePoints(e,t){let i=this.options.candidateStride,r=Array.from({length:me},()=>[]),n=Array.from({length:me},()=>[]),s=Array.from({length:me},()=>[]);for(let g=0;g<e.height;g+=i)for(let m=0;m<e.width;m+=i){let p=W(m,g,e.width);if(!e.mask[p])continue;let f=U(e.terrainPriority[p],0,1),x=U(e.hydroPriority[p],0,1),y=U(Math.max(f,x*.45),0,1),v=U(Math.floor(Math.pow(f,.85)*(me-1)),0,me-1),b=U(Math.floor(Math.pow(x,1.55)*(me-1)),0,me-1),w=U(Math.floor(Math.pow(y,.95)*(me-1)),0,me-1);r[v].push(p),n[b].push(p),s[w].push(p)}let l=new Uint8Array(e.width*e.height),a=0,o=(g,m,p,f)=>{for(let x=me-1;x>=0;x--){let y=g[x];for(let v=0;v<y.length;v++){if(a>=this.options.maxPoints||a>=f)return;let b=y[v];if(l[b])continue;let w=b%e.width,S=b/e.width|0,A=e.spacing[b],C=U(e.terrainPriority[b],0,1),B=U(e.hydroPriority[b],0,1),R=Math.max(A*m,e.cellSize*.7);C>=.85?R*=.64:C>=.7?R*=.74:C>=.55&&(R*=.84),p&&(e.channels[b]||e.floodplain[b])&&(R*=.95),R=yi(R,Math.max(e.cellSize*.7,this.options.minSpacing*.32),Math.max(this.options.minSpacing,this.options.maxSpacing*.62),16),t(w,S,R,!1)&&(l[b]=1,a++)}}},h=this.options.maxPoints,u=Math.floor(h*.82),d=Math.floor(h*.92);o(r,.54,!1,u),o(s,.56,!1,d),o(n,.6,!0,h)}relaxAcceptedPoints(e,t){if(!(e.length<3))for(let i=0;i<this.options.relaxIterations;i++){let r=new Float64Array(e.length*2);for(let o=0;o<e.length;o++)r[o*2]=e[o].x,r[o*2+1]=e[o].y;let n=new Ue(r),s=new Float64Array(e.length),l=new Float64Array(e.length),a=new Uint32Array(e.length);for(let o=0;o<n.triangles.length;o+=3){let h=n.triangles[o],u=n.triangles[o+1],d=n.triangles[o+2];this.accumulateNeighbor(h,u,e,s,l,a),this.accumulateNeighbor(h,d,e,s,l,a),this.accumulateNeighbor(u,h,e,s,l,a),this.accumulateNeighbor(u,d,e,s,l,a),this.accumulateNeighbor(d,h,e,s,l,a),this.accumulateNeighbor(d,u,e,s,l,a)}for(let o=0;o<e.length;o++){if(e[o].fixed||a[o]===0)continue;let h=s[o]/a[o],u=l[o]/a[o],d=U(Math.round((h-this.options.worldMinX)/t.cellSize),0,t.width-1),g=U(Math.round((u-this.options.worldMinY)/t.cellSize),0,t.height-1);t.mask[W(d,g,t.width)]&&(e[o].gx=d,e[o].gy=g,e[o].x=this.options.worldMinX+d*t.cellSize,e[o].y=this.options.worldMinY+g*t.cellSize)}}}accumulateNeighbor(e,t,i,r,n,s){r[e]+=i[t].x,n[e]+=i[t].y,s[e]++}filterTrianglesToMask(e,t,i){let r=[];for(let n=0;n<e.length;n+=3){let s=e[n],l=e[n+1],a=e[n+2],o=s*2,h=l*2,u=a*2,d=(t[o]+t[h]+t[u])/3,g=(t[o+1]+t[h+1]+t[u+1])/3;Mi(i.mask,i.width,i.height,d,g)&&r.push(s,l,a)}return Uint32Array.from(r)}buildHalfedges(e){let t=new Int32Array(e.length);t.fill(-1);let i=new Map;for(let r=0;r<e.length;r+=3)for(let n=0;n<3;n++){let s=e[r+n],l=e[r+(n+1)%3],a=`${s}:${l}`,o=`${l}:${s}`;if(i.has(o)){let h=i.get(o);t[r+n]=h,t[h]=r+n}else i.set(a,r+n)}return t}buildTriangleGeometry(e,t){let i=e.length/3,r=new Float32Array(i*2),n=new Float32Array(i);for(let s=0;s<i;s++){let l=e[s*3]*2,a=e[s*3+1]*2,o=e[s*3+2]*2,h=t[l],u=t[l+1],d=t[a],g=t[a+1],m=t[o],p=t[o+1];r[s*2]=(h+d+m)/3,r[s*2+1]=(u+g+p)/3;let f=Math.abs((d-h)*(p-u)-(g-u)*(m-h));n[s]=.5*f}return{centers:r,areas:n}}buildVertexAdjacency(e,t){let i=Array.from({length:t},()=>new Set);for(let a=0;a<e.length;a+=3){let o=e[a],h=e[a+1],u=e[a+2];i[o].add(h),i[o].add(u),i[h].add(o),i[h].add(u),i[u].add(o),i[u].add(h)}let r=new Uint32Array(t+1),n=0;for(let a=0;a<t;a++)r[a]=n,n+=i[a].size;r[t]=n;let s=new Uint32Array(n),l=0;for(let a=0;a<t;a++){let o=Array.from(i[a]);o.sort((h,u)=>h-u);for(let h=0;h<o.length;h++)s[l++]=o[h]}return{offsets:r,neighbors:s}}sampleVertexHeights(e){let t=new Float32Array(e.length>>1);for(let i=0,r=0;i<t.length;i++,r+=2)t[i]=wi(this.heights,this.width,this.height,e[r],e[r+1]);return t}};function Gt(c,e,t){return c<e?e:c>t?t:c}function gt(c,e,t,i){let r=t-c,n=i-e;return Math.hypot(r,n)}function Le(c,e,t,i,r){let n=Gt(Math.round(i),0,e-1),s=Gt(Math.round(r),0,t-1);return c[s*e+n]}var yt=class{ids;keys;size;constructor(e){this.ids=new Int32Array(e),this.keys=new Float64Array(e),this.size=0}push(e,t){let i=this.size++;for(this.ids[i]=e,this.keys[i]=t;i>0;){let r=i-1>>1;if(this.keys[r]<=t)break;this.ids[i]=this.ids[r],this.keys[i]=this.keys[r],i=r}this.ids[i]=e,this.keys[i]=t}pop(){if(this.size===0)return null;let e=this.ids[0],t=this.keys[0];if(this.size--,this.size>0){let i=this.ids[this.size],r=this.keys[this.size],n=0;for(;;){let s=n*2+1;if(s>=this.size||(s+1<this.size&&this.keys[s+1]<this.keys[s]&&s++,this.keys[s]>=r))break;this.ids[n]=this.ids[s],this.keys[n]=this.keys[s],n=s}this.ids[n]=i,this.keys[n]=r}return{id:e,key:t}}},je=class{options;constructor(e={}){this.options={maxNeighborDistance:Math.max(e.maxNeighborDistance??Number.POSITIVE_INFINITY,0),boundaryEpsilon:Math.max(e.boundaryEpsilon??.5,0),minDrop:Math.max(e.minDrop??1e-5,0),flatTolerance:Math.max(e.flatTolerance??1e-4,0),sinkMaxRise:Math.max(e.sinkMaxRise??.02,0),sinkClimbPenalty:Math.max(e.sinkClimbPenalty??2.5,0)}}buildFromTopology(e){if(e.heights.length===0)return{edges:[],downstream:new Int32Array(0),edgeIndexBySource:new Int32Array(0),vertexFlowWeight:new Float32Array(0),upstreamCount:new Uint32Array(0),boundaryMask:new Uint8Array(0),maxWeight:0,minWeight:0};let i=this.buildNeighbors(e),r=this.computeBoundaryMask(e.positionsGrid,e.vertexBoundaryMask),n=this.chooseDownstream(e,i,r),{edges:s,edgeIndexBySource:l,vertexFlowWeight:a,upstreamCount:o,minWeight:h,maxWeight:u}=this.accumulateFlow(e,n);return{edges:s,downstream:n,edgeIndexBySource:l,vertexFlowWeight:a,upstreamCount:o,boundaryMask:r,minWeight:h,maxWeight:u}}buildSegments(e,t,i){let r=this.collectActiveEdgeMask(t,i);return this.buildSegmentsFromMask(e,t,r)}buildSegmentsFromMask(e,t,i){let r=0;for(let a=0;a<i.length;a++)i[a]&&r++;let n=e.positionsGrid,s=new Float32Array(r*4),l=0;for(let a=0;a<t.edges.length;a++){if(!i[a])continue;let o=t.edges[a],h=o.source*2,u=o.target*2;s[l++]=n[h],s[l++]=n[h+1],s[l++]=n[u],s[l++]=n[u+1]}return s}buildActiveArtifacts(e,t,i){let r=this.collectActiveEdgeMask(t,i),n=new Uint8Array(t.vertexFlowWeight.length),s=0,l=0;for(let p=0;p<t.edges.length;p++){if(!r[p])continue;l++;let f=t.edges[p];n[f.source]||(n[f.source]=1,s++),n[f.target]||(n[f.target]=1,s++)}let a=new Array(l),o=new Float32Array(l*4),h=e.positionsGrid,u=0,d=0;for(let p=0;p<t.edges.length;p++){if(!r[p])continue;let f=t.edges[p];a[u++]=f;let x=f.source*2,y=f.target*2;o[d++]=h[x],o[d++]=h[x+1],o[d++]=h[y],o[d++]=h[y+1]}let g=new Uint32Array(s),m=0;for(let p=0;p<n.length;p++)n[p]&&(g[m++]=p);return{mask:r,activeEdges:a,activeVertices:g,segments:o}}collectActiveEdges(e,t){let i=this.collectActiveEdgeMask(e,t),r=[];for(let n=0;n<e.edges.length;n++)i[n]&&r.push(e.edges[n]);return r}collectActiveVertices(e,t){let i=this.collectActiveEdgeMask(e,t);return this.collectActiveVerticesFromMask(e,i)}collectActiveVerticesFromMask(e,t){let i=new Uint8Array(e.vertexFlowWeight.length),r=0;for(let l=0;l<e.edges.length;l++){if(!t[l])continue;let a=e.edges[l];i[a.source]||(i[a.source]=1,r++),i[a.target]||(i[a.target]=1,r++)}let n=new Uint32Array(r),s=0;for(let l=0;l<i.length;l++)i[l]&&(n[s++]=l);return n}buildGuidePoints(e,t,i){let r=Math.max(i.minWeight,0),n=Math.max(i.step,.5),s=Math.max(0,Math.floor(i.maxPoints??6e4)),l=this.collectActiveEdges(t,r),a=e.positionsGrid,o=[],h=1/0,u=1/0,d=-1/0,g=-1/0;for(let y=0;y<a.length;y+=2){let v=a[y],b=a[y+1];v<h&&(h=v),b<u&&(u=b),v>d&&(d=v),b>g&&(g=b)}let m=4,p=Math.max(1,Math.floor((d-h)*m)+3),f=new Set,x=(y,v)=>{let b=Math.round((y-h)*m),S=Math.round((v-u)*m)*p+b;f.has(S)||(f.add(S),o.push(y,v))};for(let y=0;y<l.length&&!(o.length>>1>=s);y++){let v=l[y],b=v.source*2,w=v.target*2,S=a[b],A=a[b+1],C=a[w],B=a[w+1],R=Math.max(gt(S,A,C,B),1e-6),H=Math.max(1,Math.ceil(R/n));for(let z=0;z<=H&&!(o.length>>1>=s);z++){let E=z/H;x(S+(C-S)*E,A+(B-A)*E)}}return Float32Array.from(o)}buildNeighbors(e){let t=e.heights.length,i=Array.from({length:t},()=>[]),r=e.positions2D,n=e.vertexNeighborOffsets,s=e.vertexNeighbors,l=this.options.maxNeighborDistance;for(let a=0;a<t;a++)for(let o=n[a];o<n[a+1];o++){let h=s[o];if(h===a)continue;let u=a*2,d=h*2;gt(r[u],r[u+1],r[d],r[d+1])<=l+1e-6&&i[a].push(h)}return i.map(a=>Uint32Array.from(a))}computeBoundaryMask(e,t){if(t&&t.length===e.length>>1)return new Uint8Array(t);let i=e.length>>1,r=new Uint8Array(i),n=1/0,s=1/0,l=-1/0,a=-1/0;for(let h=0;h<e.length;h+=2){let u=e[h],d=e[h+1];u<n&&(n=u),d<s&&(s=d),u>l&&(l=u),d>a&&(a=d)}let o=this.options.boundaryEpsilon;for(let h=0;h<i;h++){let u=e[h*2],d=e[h*2+1];(u<=n+o||u>=l-o||d<=s+o||d>=a-o)&&(r[h]=1)}return r}chooseDownstream(e,t,i){let r=e.heights.length,n=new Int32Array(r);n.fill(-1);let s=e.heights,l=e.positions2D,a=e.positionsGrid,o=e.vertexWetness,h=e.analysis,u=h.width,d=h.height,g=h.flowAccumulation,m=h.flowReceiver;for(let f=0;f<r;f++){let x=f*2,y=a[x],v=a[x+1],b=s[f],w=Le(g,u,d,y,v),S=Le(m,u,d,y,v),A=S>=0?S%u:-1,C=S>=0?Math.floor(S/u):-1,B=A>=0?A-y:0,R=C>=0?C-v:0,H=Math.hypot(B,R),z=-1,E=-1e30;for(let F=0;F<t[f].length;F++){let M=t[f][F],k=M*2,I=s[M],q=b-I,D=q>this.options.minDrop,j=I<=b+this.options.flatTolerance&&M>f;if(!D&&!j)continue;let O=l[k]-l[x],Y=l[k+1]-l[x+1],V=Math.max(Math.hypot(O,Y),1e-6),Q=a[k],ie=a[k+1],ge=Le(g,u,d,Q,ie),Se=0;H>1e-6&&(Se=(O*B+Y*R)/(V*H));let ce=D?q/V*6:.05/V,T=Math.log1p(Math.max(0,ge-w))*.12,P=o[M]*.035,N=Math.max(0,Se)*.7,G=i[M]?.08:0,K=D?.08:0,X=ce+T+P+N+G+K;X>E&&(E=X,z=M)}n[f]=z}let p=this.computeEscapeDownstream(e,t,i);for(let f=0;f<r;f++)n[f]<0&&!i[f]&&(n[f]=p[f]);return n}computeEscapeDownstream(e,t,i){let r=e.heights.length,n=new Int32Array(r);n.fill(-1);let s=new Float64Array(r);s.fill(1e30);let l=e.heights,a=e.positions2D,o=e.positionsGrid,h=e.analysis,u=h.width,d=h.height,g=h.flowAccumulation,m=h.flowReceiver,p=Math.max(this.options.sinkMaxRise,this.options.flatTolerance*64),f=this.options.sinkClimbPenalty,x=new yt(r*4+16);for(let y=0;y<r;y++)i[y]&&(s[y]=0,x.push(y,0));for(;x.size>0;){let y=x.pop();if(!y)break;let v=y.id,b=y.key;if(!(b>s[v]+1e-6))for(let w=0;w<t[v].length;w++){let S=t[v][w],A=this.transitionCost(S,v,e,g,m,u,d,p,f);if(!isFinite(A))continue;let C=b+A;C+1e-6<s[S]&&(s[S]=C,n[S]=v,x.push(S,C))}}return n}transitionCost(e,t,i,r,n,s,l,a,o){let h=i.heights,u=i.positions2D,d=i.positionsGrid,g=e*2,m=t*2,p=u[g],f=u[g+1],x=u[m],y=u[m+1],v=Math.max(gt(p,f,x,y),1e-6),b=h[e],w=h[t],S=Math.max(0,w-b);if(S>a+1e-6)return 1e30;let A=d[g],C=d[g+1],B=Le(r,s,l,A,C),R=Le(n,s,l,A,C),H=R>=0?R%s:-1,z=R>=0?Math.floor(R/s):-1,E=H>=0?H-A:0,F=z>=0?z-C:0,M=Math.hypot(E,F),k=0;if(M>1e-6){let V=d[m]-A,Q=d[m+1]-C;k=(V*E+Q*F)/(Math.max(Math.hypot(V,Q),1e-6)*M)}let I=d[m],q=d[m+1],D=Le(r,s,l,I,q),j=D+1e-6>=B?0:.35*(1-D/Math.max(B,1)),O=M>1e-6?.5*(1-Math.max(-.25,k)):.2,Y=o*(S/Math.max(v,1e-6));return v*(1+O+j)+Y}accumulateFlow(e,t){let i=e.heights.length,r=new Float32Array(i);r.fill(1);let n=new Uint32Array(i),s=new Int32Array(i);s.fill(-1);let l=new Uint32Array(i);for(let m=0;m<i;m++)l[m]=m;let a=e.heights,o=e.vertexWetness;l.sort((m,p)=>{let f=a[p]-a[m];if(Math.abs(f)>1e-6)return f;let x=o[p]-o[m];return Math.abs(x)>1e-6?x:p-m});for(let m=0;m<l.length;m++){let p=l[m],f=t[p];f>=0&&f!==p&&(r[f]+=r[p],n[f]++)}let h=[],u=e.positions2D,d=1/0,g=0;for(let m=0;m<i;m++){let p=t[m];if(p<0||p===m)continue;let f=m*2,x=p*2,y=u[x]-u[f],v=u[x+1]-u[f+1],b=Math.max(Math.hypot(y,v),1e-6),w=r[m],S={source:m,target:p,weight:w,length:b};s[m]=h.length,h.push(S),w<d&&(d=w),w>g&&(g=w)}return isFinite(d)||(d=0),{edges:h,edgeIndexBySource:s,vertexFlowWeight:r,upstreamCount:n,minWeight:d,maxWeight:g}}collectActiveEdgeMask(e,t){let i=Math.max(t,0),r=new Uint8Array(e.edges.length),n=e.vertexFlowWeight.length;for(let s=0;s<n;s++){if(e.vertexFlowWeight[s]+1e-6<i)continue;let l=s,a=0;for(;l>=0&&a<n;){let o=e.edgeIndexBySource[l];if(o<0||r[o])break;r[o]=1,l=e.downstream[l],a++}}return r}};var xt={preview:{minSpacing:6,maxSpacing:40,maxTriangles:4e3},balanced:{minSpacing:3,maxSpacing:20,maxTriangles:16e3},high:{minSpacing:1,maxSpacing:10,maxTriangles:6e4}};var $e=class{constructor(e,t=()=>{},{execution:i="worker"}={}){this.sim=e,this.onstatus=t,this.snapshots=0,this.execution=i,this.view={adaptive:!0,wireframe:!0,wireOpacity:.55}}async capture(){let e=this.sim;if(this.snapshot?.iteration===e.iterationCount&&this.snapshot.buffer===e.stateBuffers[0])return this.snapshot;let t=e.device.createBuffer({size:e.cellCount*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{let i=e.device.createCommandEncoder();i.copyBufferToBuffer(e.stateBuffers[0],0,t,0,t.size),e.device.queue.submit([i.finish()]),await t.mapAsync(GPUMapMode.READ);let r=new Float32Array(t.getMappedRange()),n=new Float32Array(e.cellCount),s=new Uint8Array(e.cellCount);for(let l=0;l<e.cellCount;l++)n[l]=r[l*8],s[l]=r[l*8+4]>.5?1:0;return t.unmap(),this.snapshots++,this.snapshot={width:e.width,height:e.height,values:n,mask:s,iteration:e.iterationCount,buffer:e.stateBuffers[0]}}finally{t.destroy()}}async build(e={}){this.swe?.destroy(),this.swe=null,this.onstatus("Capturing paused terrain\u2026");let t=await this.capture(),i=performance.now();if(this.execution==="worker"){let{default:g}=await Promise.resolve().then(()=>(Vt(),Dt));this.coordinator??=new g({debugLogging:!1,onstatus:this.onstatus}),await this.coordinator.setRaster({width:t.width,height:t.height,values:t.values.slice(),mask:t.mask.slice()})}let r=Math.max(1,Math.min(256,Number(e.minSpacing)||3)),n=Math.max(r,Math.min(512,Number(e.maxSpacing)||20)),s=Math.max(128,Math.min(25e4,Math.floor(Number(e.maxTriangles)||16e3))),l=1/0,a=-1/0;for(let g=0;g<t.values.length;g++)t.mask[g]&&(l=Math.min(l,t.values[g]),a=Math.max(a,t.values[g]));let o=Math.max(a-l,.001),h;for(let g=0;g<8;g++){let m={topologyBuildOptions:{mode:e.mode||"hybrid",cellSize:1,minSpacing:r,maxSpacing:n,borderSpacing:n,wetnessBins:21,channelThreshold:256,candidateStride:1,relaxIterations:0,preserveBorder:!0,preserveChannels:!0,preserveMaskBoundary:!0,preserveInnerBorder:!0},riverOptions:{maxNeighborDistance:1/0,boundaryEpsilon:.5,minDrop:1e-5,flatTolerance:1e-4,sinkMaxRise:o*.015,sinkClimbPenalty:2.5},guideOptions:{minWeight:3,step:Math.max(r*1.75,Math.min(n*.6,r*5)),maxPoints:Math.min(16e3,Math.floor(t.values.length*.45))},floodplainOptions:{maxRise:o*.05,maxClimbStep:o*.015,maxEdgeSlope:.75,maxWorldDistance:Math.max(r*6,n*1.5),minTriangleFloodplainVerts:2},minRiverWeight:6,guideEnabled:e.guideEnabled!==!1};if(this.execution==="worker")h=await this.coordinator.build(m);else{this.onstatus("Building hydraulic topology in the background\u2026");let x=new Ye(m.topologyBuildOptions);x.setRaster(t);let y=await x.build();if(m.guideEnabled){let v=new je(m.riverOptions),b=v.buildFromTopology(y),w=v.buildGuidePoints(y,b,m.guideOptions);w.length&&(y=await x.buildWithForcedWorldPoints(w))}h={topology:y}}let p=h.topology.indices.length/3;if(p<=s)break;if(g===7)throw Error("This boundary needs more triangles. Increase the triangle limit.");let f=Math.max(1.2,Math.sqrt(p/s)*1.1);r*=f,n*=f,this.onstatus(`Fitting the ${s.toLocaleString()} triangle limit\u2026`)}let u=new Ve(this.sim);try{await u.initialize(h.topology)}catch(g){throw u.destroy(),g}this.renderer?.destroy(),this.renderer=u,this.topology=h.topology;let d=Ii(t,h.topology);return this.info={vertices:h.topology.positionsGrid.length/2,triangles:h.topology.indices.length/3,gridTriangles:(t.width-1)*(t.height-1)*2,buildMs:performance.now()-i,minSpacing:r,maxSpacing:n,maxTriangles:s,iteration:t.iteration,time:this.sim.getStats().simTime,snapshots:this.snapshots,...d},this.render(),this.info}render(e={}){Object.assign(this.view,e),this.renderer?this.renderer.render({...this.view,swe:this.swe}):this.sim.render()}async startWater(e={}){if(!this.renderer)throw Error("Build a hydraulic mesh first.");if(this.swe)this.swe.updateOptions(e);else{let t=new He(this.sim,this.topology,this.renderer);try{await t.initialize(e)}catch(i){throw t.destroy(),i}this.swe=t}this.view.adaptive=!0,this.render()}resetWater(){this.swe?.destroy(),this.swe=null,this.render()}stepWater(e){this.swe?.step(e),this.render()}invalidate(){this.swe?.destroy(),this.swe=null,this.renderer?.destroy(),this.renderer=null,this.topology=null,this.snapshot=null,this.info=null}destroy(){this.invalidate(),this.coordinator?.dispose(),this.coordinator=null}};function Ii(c,e){let{width:t,height:i,values:r}=c,n=e.positionsGrid,s=e.indices,l=(d,g)=>{d=Math.max(0,Math.min(t-1,d)),g=Math.max(0,Math.min(i-1,g));let m=Math.floor(d),p=Math.floor(g),f=d-m,x=g-p,y=r[p*t+m],v=r[p*t+Math.min(m+1,t-1)],b=r[Math.min(p+1,i-1)*t+m],w=r[Math.min(p+1,i-1)*t+Math.min(m+1,t-1)];return(y*(1-f)+v*f)*(1-x)+(b*(1-f)+w*f)*x},a=0,o=0,h=0,u=[[1/3,1/3,1/3],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6],[.5,.5,0],[.5,0,.5],[0,.5,.5]];for(let d=0;d<s.length;d+=3){let g=s[d]*2,m=s[d+1]*2,p=s[d+2]*2,f=Math.abs((n[m]-n[g])*(n[p+1]-n[g+1])-(n[p]-n[g])*(n[m+1]-n[g+1]))*.5,x=[l(n[g],n[g+1]),l(n[m],n[m+1]),l(n[p],n[p+1])];for(let y of u){let v=y[0]*n[g]+y[1]*n[m]+y[2]*n[p],b=y[0]*n[g+1]+y[1]*n[m+1]+y[2]*n[p+1],w=Math.abs(y[0]*x[0]+y[1]*x[1]+y[2]*x[2]-l(v,b));a+=w*w*f,o+=f,h=Math.max(h,w)}}return{sampledRmsError:Math.sqrt(a/Math.max(o,1e-12)),sampledMaxError:h}}function wt({onbuild:c,onview:e,onwater:t}={}){let i=document.createElement("fieldset");i.className="mesh-controls",i.style.cssText="border:1px solid #46636f;border-radius:10px;padding:12px;margin:12px 0;color:#d9e8ed";let r=document.createElement("legend");r.textContent="Final hydraulic mesh \xB7 build while paused",i.append(r);let n=document.createElement("div");n.style.cssText="display:flex;flex-wrap:wrap;align-items:center;gap:10px",i.append(n);let s=n;function l(M){s=document.createElement("div"),s.style.cssText=n.style.cssText+";border-top:1px solid #46636f;padding-top:10px;margin-top:12px";let k=document.createElement("strong");k.textContent=M,k.style.cssText="font-size:12px;flex-basis:100%",s.append(k),i.append(s)}let a={};function o(M,k,I,q,D){let j=document.createElement("label");j.style.cssText="display:flex;align-items:center;gap:6px",j.textContent=k+" ";let O=document.createElement(I==="select"?"select":"input");if(O.setAttribute("aria-label",k),O.dataset.mesh=M,D)for(let[Y,V]of D){let Q=document.createElement("option");Q.value=Y,Q.textContent=V,O.append(Q)}else O.type=I;return O.value=q,I==="checkbox"&&(O.checked=q),O.style.cssText="background:#20333f;color:#e6f0f2;border:1px solid #46636f;border-radius:6px;padding:6px;max-width:170px",I==="number"&&(O.min="1",O.max=M==="budget"?"250000":"512",O.style.width="90px",O.step="1"),j.append(O),s.append(j),a[M]=O,O}let h=o("fidelity","Mesh fidelity","select","balanced",[["preview","Preview"],["balanced","Balanced"],["high","High"],["custom","Custom"]]);o("mode","Mesh priority","select","hybrid",[["hybrid","Terrain + rivers"],["traditional","Relief"],["hydrographic","Channels"],["hydrological","Catchments"]]),o("min","Min spacing","number",3),o("max","Max spacing","number",20),o("budget","Triangle limit","number",16e3),o("guides","River guides","checkbox",!0),a.budget.min=128;let u=M=>{let k=document.createElement("button");return k.type="button",k.textContent=M,k.style.cssText="background:#24585c;color:#e6f0f2;border:1px solid #46636f;border-radius:6px;padding:8px;cursor:pointer",s.append(k),k},d=u("Build hydraulic mesh");l("Surface comparison"),o("surface","Mesh surface","select","adaptive",[["adaptive","Adaptive triangles"],["grid","Full grid"]]),o("wire","Mesh wireframe","checkbox",!0);let g=o("opacity","Wire opacity","range",.55);g.min=0,g.max=1,g.step=.05,g.value=.55,l("Water over the frozen mesh");let m=u("Run mesh water"),p=u("Reset mesh water"),f=u("Inspect mesh water"),x=o("roughness","Water roughness","number",.025);x.min=0,x.max=.2,x.step=.005;let y=o("dt","Water max step","number",.02);y.min=1e-4,y.max=1,y.step=.005;let v=o("rain","Mesh rain","number",.001);v.min=0,v.max=1,v.step=.001;let b=document.createElement("div");b.style.cssText="font-size:12px;color:#afc5cf;margin-top:10px;white-space:pre-line",i.append(b);let w=null,S=!1,A=!1,C=!1,B=0,R=!1,H=()=>({mode:a.mode.value,minSpacing:Number(a.min.value),maxSpacing:Number(a.max.value),maxTriangles:Number(a.budget.value),guideEnabled:a.guides.checked}),z=()=>({adaptive:a.surface.value==="adaptive",wireframe:a.wire.checked,wireOpacity:Number(a.opacity.value)}),E=()=>({roughness:Number(x.value),maxTimeStep:Number(y.value),rainRate:Number(v.value)});h.onchange=()=>{let M=xt[h.value];M&&(a.min.value=M.minSpacing,a.max.value=M.maxSpacing,a.budget.value=M.maxTriangles)};for(let M of["min","max","budget"])a[M].oninput=()=>{h.value="custom"};for(let M of["surface","wire","opacity"])a[M].oninput=()=>e?.(z());d.onclick=()=>c?.(H()),m.onclick=()=>t?.(C?"pause":"run",E()),p.onclick=()=>t?.("reset",E()),f.onclick=()=>t?.("inspect",E());function F(M={}){"info"in M&&(w=M.info),"busy"in M&&(S=M.busy),"running"in M&&(A=M.running),"waterRunning"in M&&(C=M.waterRunning),"ready"in M&&(R=M.ready),"waterSteps"in M&&(B=M.waterSteps);for(let[k,I]of Object.entries(a))I.disabled=S||A||C&&!["wire","opacity"].includes(k)||!w&&["surface","wire","opacity"].includes(k);d.disabled=S||A||C||!R,m.disabled=p.disabled=S||A||!w,f.disabled=S||A||C||!w,m.textContent=C?"Pause mesh water":"Run mesh water",C&&(a.surface.value="adaptive"),b.textContent=M.message||(w?`${w.triangles.toLocaleString()} triangles / ${w.gridTriangles.toLocaleString()} grid \xB7 ${w.vertices.toLocaleString()} vertices \xB7 ${(w.buildMs/1e3).toFixed(2)} s build
Built spacing ${w.minSpacing.toFixed(1)}\u2013${w.maxSpacing.toFixed(1)} cells \xB7 limit ${w.maxTriangles.toLocaleString()} triangles
Sampled height error: RMS ${w.sampledRmsError.toFixed(5)} \xB7 max ${w.sampledMaxError.toFixed(5)} \xB7 frozen at ${w.time.toFixed(2)} s${B?`
Mesh water: ${B.toLocaleString()} steps \xB7 ${C?"running":"paused"}`:""}
Mesh water keeps the terrain fixed; resuming erosion returns to the grid.`:"Pause erosion to build. Fidelity and the triangle limit control detail and size. Water over the fixed mesh runs separately from erosion.")}return F({ready:!1}),{root:i,update:F,options:H,view:z,waterOptions:E}}var se={iterationsPerFrame:4,stepIterations:160,timeStep:.02,rainRate:.001,evaporationRate:.025,pipeArea:16,gravity:9.81,capacityScale:1.2,flowDrag:2,maxOutflowFraction:.5,suspensionRate:.35,depositionRate:1.4,softeningRate:1.2,maxErosionDepth:.04,thermalRate:.25,talusCoeff:.012,talusBias:.002,erodabilityMin:.24,erodabilityMax:.36,sourceStrength:.06,sourceRadius:4,sourceEnabled:!1,sourceSedimentConcentration:0,bankErosionRate:0,bankBendBias:24,bendSedimentRate:0,straightBankErosionFactor:.08,riverBendLength:0,openBoundaryEdges:15,sourceLayoutMode:1,randomSpringCount:3,sourceSeed:7,rainDuration:0,pulse2Duration:0,metersPerPixel:100,renderHeightScale:.45,waterHeightScale:.45,waterOpacity:.28,edgeWaterFloor:0,renderMode:8,hydraulicErosionEnabled:!0,closedBoundaries:!1},Ht={paper_balanced:{...se,label:"Balanced drainage",description:"Steady light rain cuts connected drainage networks. Start with the alpine catchment and compare after 20\u201360 simulated seconds."},river_cut:{...se,label:"Spring-fed channels",rainRate:25e-5,flowDrag:1.4,sourceEnabled:!0,sourceStrength:.12,randomSpringCount:3,capacityScale:1.8,suspensionRate:.55,depositionRate:1.2,thermalRate:.18,description:"Localized springs carve channels with sediment bars downstream. Use the alpine catchment; change Spring seed for a different source layout."},delta_depositor:{...se,label:"Alluvial fan",rainRate:.0025,evaporationRate:.012,flowDrag:3.5,capacityScale:.7,suspensionRate:.4,depositionRate:3,erodabilityMin:.4,erodabilityMax:.6,thermalRate:.08,rainDuration:25,edgeWaterFloor:.015,description:"Runoff leaves a narrow mountain outlet and spreads across a gentle plain. Use the fan example; the 25-second storm is followed by settling."},flash_flood:{...se,label:"Flash flood and dry-down",rainRate:.018,flowDrag:2.5,evaporationRate:.18,capacityScale:3.2,suspensionRate:1.2,depositionRate:2.4,rainDuration:6,thermalRate:.08,erodabilityMin:.35,erodabilityMax:.55,description:"A six-second storm strips loose soil, then deposits its load as water recedes. Run the fan or badlands for 40\u201380 simulated seconds."},badlands:{...se,label:"Soft badlands",rainRate:.002,evaporationRate:.065,capacityScale:1.8,suspensionRate:.55,depositionRate:1.8,erodabilityMin:.55,erodabilityMax:.75,thermalRate:.45,talusCoeff:.006,talusBias:.001,description:"Soft terrain develops rills while unstable banks slump. Use the badlands example and compare after 30\u201390 simulated seconds."},canyon_carver:{...se,label:"Layered canyon",rainRate:6e-4,sourceEnabled:!0,sourceStrength:.16,randomSpringCount:2,capacityScale:2.8,suspensionRate:.7,depositionRate:1.1,thermalRate:.12,erodabilityMin:.12,erodabilityMax:.22,description:"Slow incision preserves resistant shelves. The canyon example includes a softer upper layer over resistant bedrock; run for 60\u2013180 simulated seconds."},river_delta:{...se,label:"River delta",rainRate:0,evaporationRate:0,sourceEnabled:!0,sourceLayoutMode:0,sourceStrength:.45,sourceSedimentConcentration:.12,flowDrag:3,capacityScale:1.2,suspensionRate:.3,depositionRate:1.4,thermalRate:.04,erodabilityMin:.45,erodabilityMax:.6,edgeWaterFloor:.035,waterOpacity:.6,description:"A sediment-fed river enters a shallow coastal basin. Upstream sediment replenishes the bed and builds mouth bars. Adjust River sediment load in the live preview; Run continues until paused."},mountain_canyon:{...se,label:"Mountain river canyon",rainRate:1e-4,evaporationRate:.012,sourceEnabled:!0,sourceLayoutMode:0,sourceStrength:.4,flowDrag:1.4,capacityScale:2.8,suspensionRate:.75,depositionRate:1.2,thermalRate:.08,erodabilityMin:.25,erodabilityMax:.4,waterOpacity:.6,description:"A headwater spring follows a descending mountain canyon. A thin erodible surface overlies resistant rock; compare incision and downstream bars after 60\u2013120 simulated seconds."},volcanic_cone:{...se,label:"Volcanic radial canyons",rainRate:.006,rainDuration:35,closedBoundaries:!0,evaporationRate:.075,flowDrag:1.8,capacityScale:3.4,suspensionRate:1.25,depositionRate:2.5,thermalRate:.06,erodabilityMin:.74,erodabilityMax:.9,talusCoeff:.016,waterOpacity:.45,description:"Longonot-inspired broad crater and radial drainage. A 35-second storm incises loose ash and thick tephra above resistant tuff and lava. Run 90\u2013150 seconds to see deepening gullies, exposed layers and sediment aprons."},volcanic_lake:{...se,label:"Crater lake and flank erosion",rainRate:.006,rainDuration:20,closedBoundaries:!0,evaporationRate:.035,flowDrag:2,capacityScale:2.8,suspensionRate:.95,depositionRate:2,thermalRate:.04,erodabilityMin:.65,erodabilityMax:.8,talusCoeff:.02,waterOpacity:.6,description:"A twenty-second storm cuts the layered volcanic flanks while the crater collects runoff. Compare evolving radial gullies and the sheltered lake after 60\u201390 seconds."},lowland_meanders:{...se,label:"Lowland meandering river",rainRate:0,evaporationRate:.002,sourceEnabled:!0,sourceLayoutMode:0,sourceStrength:.015,sourceSedimentConcentration:.1,flowDrag:2,pipeArea:20,capacityScale:1.2,suspensionRate:.12,depositionRate:3,bankErosionRate:3,bankBendBias:80,thermalRate:.01,talusCoeff:.008,bendSedimentRate:4,straightBankErosionFactor:.005,riverBendLength:.045,openBoundaryEdges:8,erodabilityMin:.75,erodabilityMax:.9,waterOpacity:.65,renderHeightScale:.9,waterHeightScale:.9,description:"A nearly straight river develops bends through layered alluvium. Inner-bank sediment bars deflect flow, outer banks retreat, and new channels carve while abandoned reaches fill. Run continuously or compare after 360 seconds. Headwater and side boundaries are walls; the downstream edge is open."},thermal_heavy:{...se,label:"Dry talus relaxation",rainRate:0,sourceEnabled:!1,hydraulicErosionEnabled:!1,thermalRate:3,talusCoeff:.005,talusBias:.001,erodabilityMin:.55,erodabilityMax:.7,description:"Dry cliffs shed material into talus aprons without water erosion. Use the talus example; inspect the thermal view after 10\u201330 simulated seconds."},gentle_weathering:{...se,label:"Gentle weathering",rainRate:4e-4,capacityScale:1.2,suspensionRate:.3,thermalRate:.08,erodabilityMin:.12,erodabilityMax:.2,description:"Slow changes retain the starting relief. Use the alpine catchment for long runs, or as a restrained starting point for imported DEMs."}};var Ke={alpine:{label:"Alpine catchment",mode:0,preset:"paper_balanced",layered:!0},fan:{label:"Mountain outlet and fan",mode:1,preset:"delta_depositor",layered:!0},badlands:{label:"Warped badlands",mode:2,preset:"badlands",layered:!0},canyon:{label:"Layered canyon",mode:3,preset:"canyon_carver",layered:!0},talus:{label:"Cliffs and talus",mode:4,preset:"thermal_heavy",layered:!0},basin:{label:"Closed basin",mode:5,preset:"flash_flood",closed:!0,layered:!0},river_delta:{label:"River delta",mode:6,preset:"river_delta",layered:!0},mountain_canyon:{label:"Mountain river canyon",mode:7,preset:"mountain_canyon",layered:!0},volcano:{label:"Longonot-inspired volcano",mode:8,preset:"volcanic_cone",closed:!0,layered:!0},crater_lake:{label:"Volcanic crater lake",mode:8,preset:"volcanic_lake",closed:!0,layered:!0},lowland_river:{label:"Lowland river and floodplain",mode:9,preset:"lowland_meanders",layered:!0}},fe={label:"Resistant bedrock",hardnessMin:0,hardnessMax:.02,thermalEnabled:!1},We={alpine:[{label:"Mountain soil"},{label:"Weathered rock",hardnessMin:.1,hardnessMax:.18},fe],fan:[{label:"Loose fan sediment"},{label:"Older alluvium",hardnessMin:.28,hardnessMax:.45},fe],badlands:[{label:"Soft surface beds"},{label:"Cemented ledges",hardnessMin:.08,hardnessMax:.16,thermalEnabled:!1},{label:"Weak sediment beds",hardnessMin:.45,hardnessMax:.65},fe],canyon:[{label:"Upper sediment"},{label:"Resistant shelves",hardnessMin:.04,hardnessMax:.09,thermalEnabled:!1},{label:"Weathered lower beds",hardnessMin:.18,hardnessMax:.3},fe],talus:[{label:"Loose talus"},{label:"Fractured cliff rock",hardnessMin:.25,hardnessMax:.4},fe],basin:[{label:"Basin sediment"},{label:"Compacted substrate",hardnessMin:.12,hardnessMax:.22},fe],river_delta:[{label:"Young delta sediment"},{label:"Older alluvium",hardnessMin:.18,hardnessMax:.3},fe],lowland_river:[{label:"Loose floodplain alluvium"},{label:"Older river deposits",hardnessMin:.4,hardnessMax:.6},fe],mountain_canyon:[{label:"Thin canyon soil"},{label:"Weathered rock",hardnessMin:.05,hardnessMax:.09,thermalEnabled:!1},fe],volcano:[{label:"Loose volcanic ash"},{label:"Unconsolidated tephra",hardnessMin:.5,hardnessMax:.72},{label:"Resistant tuff",hardnessMin:.08,hardnessMax:.16,thermalEnabled:!1},{...fe,label:"Lava core"}]};We.crater_lake=We.volcano;var Oi=`
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
`;async function qt(c,e="alpine",t=512,i=7){let r=Ke[e];if(!r)throw new Error(`Unknown terrain example: ${e}`);t=Math.max(64,Math.min(1024,Math.floor(t)||512));let n=t*t*16,s=c.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),l=c.createBuffer({size:n,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),a=c.createBuffer({size:n,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{let o=await c.createComputePipelineAsync({layout:"auto",compute:{module:c.createShaderModule({code:Oi}),entryPoint:"generate"}}),h=c.createBindGroup({layout:o.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:l}}]});c.queue.writeBuffer(s,0,new Uint32Array([t,r.mode,i>>>0,0]));let u=c.createCommandEncoder(),d=u.beginComputePass();d.setPipeline(o),d.setBindGroup(0,h),d.dispatchWorkgroups(Math.ceil(t/8),Math.ceil(t/8)),d.end(),u.copyBufferToBuffer(l,0,a,0,n),c.queue.submit([u.finish()]),await a.mapAsync(GPUMapMode.READ);let g=new Float32Array(a.getMappedRange()),m=Array.from({length:We[e].length},(x,y)=>Float32Array.from({length:t*t},(v,b)=>g[b*4+y])),p=m[0];a.unmap();let f={width:t,height:t,values:p};return r.layered&&(f.bands=m),(e==="river_delta"||e==="mountain_canyon"||e==="crater_lake"||e==="lowland_river")&&(f.water=Float32Array.from(p,(x,y)=>{let v=y%t/(t-1),b=Math.floor(y/t)/(t-1);if(e==="crater_lake")return Math.hypot((v-.51)*1.06,(b-.49)*.94)<.2?Math.max(.34-x,0):0;let w=e==="lowland_river"?.065+.04*(1-b)+.008:e==="river_delta"?Math.max(.12,.045+.16*(1-b)+.012):.075+.23*(1-b)+.012;return Math.max(w-x,0)})),f}finally{s.destroy(),l.destroy(),a.destroy()}}function Yt(c,e){return Ke[c]?.layered?{layerConfig:{mode:"packed_rgba",layers:We[c].map(t=>({enabled:!0,heightMin:0,heightMax:1,hardnessMin:e.erodabilityMin,hardnessMax:e.erodabilityMax,thermalEnabled:!0,...t}))}}:{}}function jt(c,e,t){if(t!=="river_delta"&&t!=="mountain_canyon"&&t!=="lowland_river")return;let i=Math.round((e.height-1)*(t==="lowland_river"?.035:.08)),r=Math.round(e.width*.5);for(let n=Math.floor(e.width*.2);n<e.width*.8;n++)e.values[i*e.width+n]<e.values[i*e.width+r]&&(r=n);c.paintSpringBrush({x:r,y:i,radius:Math.max(2,4*e.width/256),strength:c.simulationParams.sourceStrength,hardness:1})}var de=document.createElement("canvas");de.width=800;de.height=600;de.setAttribute("aria-label","Live erosion preview");de.style.touchAction="none";var be=document.createElement("div");be.className="live-panel";var St=document.createElement("div");St.className="live-toolbar";var dt=c=>{let e=document.createElement("button");return e.type="button",e.textContent=c,e},ot=dt("Pause"),Pt=dt("Reset"),lt=dt("View result"),Xt=dt("Close"),we=document.createElement("select"),Ct=document.createElement("label");Ct.textContent="River sediment load ";var le=document.createElement("input");le.type="number";le.min="0";le.max="1";le.step="0.01";le.style.width="80px";le.title="Sediment supplied per unit of spring water. Zero supplies clear water.";Ct.append(le);we.setAttribute("aria-label","Live view");for(let[c,e]of[[7,"Natural"],[8,"Slope relief"],[9,"River flow"],[4,"Erodability"],[6,"Erosion and deposition"]]){let t=document.createElement("option");t.value=c,t.textContent=e,we.append(t)}var mt=document.createElement("span");mt.setAttribute("role","status");mt.className="live-status";St.append(ot,Pt,lt,we,Ct,Xt,mt);var $t=document.createElement("p");$t.textContent="Drag to orbit \xB7 scroll to zoom. Run continues until paused. View result advances to the pictured time. River sediment load controls material arriving with spring water.";var ct=document.createElement("div");ct.className="layer-legend";var Ze=wt({onbuild:async c=>{if(oe||ee||Me)return;ee=!0,J("Building hydraulic mesh\u2026");let e;try{ne??=new $e(L,t=>Ze.update({message:t,busy:!0})),await ne.build(c)}catch(t){e=t.message}finally{ee=!1,J(e)}},onview:c=>{ne?.render(c)},onwater:async(c,e)=>{if(ee||Me)return;if(c==="inspect"){try{let i=await ne?.swe?.readStats();if(!i)throw Error("Run mesh water first.");Ze.update({message:`Mesh water: ${i.time.toFixed(2)} s \xB7 ${i.steps} steps \xB7 stable step ${i.dt.toFixed(5)} s
Water volume ${i.totalWater.toPrecision(6)} \xB7 depth max ${i.maxDepth.toFixed(4)} \xB7 ${i.invalid} invalid cells`})}catch(i){Ze.update({message:i.message})}return}if(c==="pause"){ke=!1,pe();return}if(c==="reset"){ke=!1,pe(),ne?.resetWater(),J();return}if(!ne?.renderer)return;ee=!0,J("Preparing water over the fixed mesh\u2026");let t;try{await ne.startWater(e),ke=!0,oe=!0;let i=++Qe;ht.reset(8),Ne=requestAnimationFrame(()=>Bt(i))}catch(i){t=i.message}finally{ee=!1,J(t)}}});be.append(St,de,$t,ct,Ze.root);var L,Z,Mt,ee=!1,oe=!1,Me=!1,Re=0,Ne=0,Qe=0,ne=null,ke=!1;function ft(){ke=!1,ne?.destroy(),ne=null}function Kt(){ne?.renderer?ne.render():L.render()}var ht=new De,ut=45,Je=42,ze=2.9,Be=null;function J(c){mt.textContent=c||`${L?.getStats().simTime.toFixed(2)??"0.00"} simulated seconds${oe?" \xB7 running":" \xB7 paused"}`,ot.textContent=oe?"Pause":"Run",ot.disabled=Pt.disabled=lt.disabled=we.disabled=ee||Me||!L?.ready||!Z,le.disabled=we.disabled||!Z?.preset.sourceEnabled,lt.disabled||=!L?.ready||L.getStats().iterationCount>=(Z?.picturedSteps??0),be.dataset.running=String(oe),be.dataset.time=String(L?.getStats().simTime??0),Ze.update({info:ne?.info||null,busy:ee||Me,running:oe&&!ke,waterRunning:ke,waterSteps:ne?.swe?.steps||0,ready:!!L?.ready&&!!Z,message:c})}function pe(){oe=!1,ke=!1,Qe++,cancelAnimationFrame(Ne),Ne=0,Z&&J()}function At(){if(!L?.ready)return;let c=ut*Math.PI/180,e=Je*Math.PI/180;L.setSimulationParams({cameraAzimuthDeg:ut,cameraElevationDeg:Je,cameraPosX:Math.cos(e)*Math.cos(c)*ze,cameraPosY:.12+Math.sin(e)*ze,cameraPosZ:Math.cos(e)*Math.sin(c)*ze}),Kt()}async function Bt(c){if(Ne=0,!(!oe||c!==Qe||ee||Me))try{let e=ht.steps,t=performance.now();if(ke?ne.stepWater(e):L.stepAndRender(e),await L.device.queue.onSubmittedWorkDone(),!oe||c!==Qe)return;ht.observe(e,performance.now()-t),J(),Ne=requestAnimationFrame(()=>Bt(c))}catch(e){pe(),J(e.message)}}function kt(){if(!L?.ready||ee||Me||oe)return;ft(),oe=!0;let c=++Qe;ht.reset(Z.preset.iterationsPerFrame),J(),Ne=requestAnimationFrame(()=>Bt(c))}async function Zt(){ft(),await L.device.queue.onSubmittedWorkDone();let c=Z.preset;L.setSimulationParams({...c,closedBoundaries:!!Ke[Z.terrain].closed,talusSlopeCoeff:c.talusCoeff,talusSlopeBias:c.talusBias,hardnessBase:c.erodabilityMin,hardnessVariation:c.erodabilityMax-c.erodabilityMin,renderGridStep:0,renderMode:Number(we.value),sourceTimeOffset:0,sourceSedimentConcentration:Math.max(0,Math.min(1,Number(le.value)||0))}),await L.setDEM(Mt,Yt(Z.terrain,c)),jt(L,Mt,Z.terrain),Z.seconds=Z.picturedSeconds,Z.picturedSteps=Math.round(Z.picturedSeconds/c.timeStep),At(),J()}async function Ui(c){pe();let e=++Re;ee=!0;let t=c.querySelector(".run-sim");t.disabled=!0,c.append(be),be.hidden=!1,J("Preparing terrain\u2026");try{if(L||(L=new _e({canvas:de}),await L.initialize()),await L.device.queue.onSubmittedWorkDone(),e!==Re)return;let i=c.dataset.terrain,r=Ht[c.dataset.preset];if(!r||!Ke[i])throw Error("Unknown example.");if(Z={card:c,terrain:i,preset:r,picturedSeconds:Number(c.dataset.seconds),seconds:Number(c.dataset.seconds)},Mt=await qt(L.device,i,Number(c.dataset.size)||256,Number(c.dataset.seed)||7),e!==Re)return;ut=45,Je=i==="lowland_river"?65:42,ze=2.9,we.value="8",le.value=String(r.sourceSedimentConcentration||0),ct.replaceChildren(),We[i].forEach((n,s)=>{let l=document.createElement("span");l.textContent=n.label,l.style.borderColor=["#9c8a66","#856647","#6e7066","#4a4f4d"][s],ct.append(l)}),await Zt(),ee=!1,kt()}catch(i){ee=!1,J(i.message)}finally{t.disabled=!1,e!==Re&&(ee=!1,J())}}document.querySelectorAll(".run-sim").forEach(c=>c.addEventListener("click",()=>{if(ee||Me)return;let e=c.closest("article");Z?.card===e&&L?.ready?(e.append(be),be.hidden=!1,kt()):Ui(e)}));ot.addEventListener("click",()=>oe?pe():kt());Pt.addEventListener("click",async()=>{pe(),ee=!0,J("Restoring terrain\u2026");let c;try{await Zt()}catch(e){c=e.message}finally{ee=!1,J(c)}});lt.addEventListener("click",async()=>{pe(),ft(),Me=!0,J("Advancing to the pictured result\u2026");let c=Re,e;try{for(;c===Re&&L.getStats().iterationCount<Z.picturedSteps;){let t=Math.min(128,Z.picturedSteps-L.getStats().iterationCount);L.step(t),L.render(),await L.device.queue.onSubmittedWorkDone(),J(`Advancing\u2026 ${L.getStats().simTime.toFixed(1)} / ${Z.picturedSeconds} seconds`),await new Promise(i=>requestAnimationFrame(i))}}catch(t){e=t.message}finally{Me=!1,J(e)}});we.addEventListener("change",()=>{L.setSimulationParams({renderMode:Number(we.value)}),Kt()});le.addEventListener("input",()=>{L.setSimulationParams({sourceSedimentConcentration:Math.max(0,Math.min(1,Number(le.value)||0))})});Xt.addEventListener("click",()=>{pe(),Re++,be.hidden=!0});document.addEventListener("visibilitychange",()=>{document.hidden&&pe()});window.addEventListener("pagehide",()=>{pe(),ft(),L?.destroy()});de.addEventListener("pointerdown",c=>{Be={id:c.pointerId,x:c.clientX,y:c.clientY},de.setPointerCapture(c.pointerId)});de.addEventListener("pointermove",c=>{!Be||c.pointerId!==Be.id||(ut+=(c.clientX-Be.x)*.35,Je=Math.max(12,Math.min(85,Je+(c.clientY-Be.y)*.25)),Be.x=c.clientX,Be.y=c.clientY,At())});for(let c of["pointerup","pointercancel","lostpointercapture"])de.addEventListener(c,()=>{Be=null});de.addEventListener("wheel",c=>{c.preventDefault(),ze=Math.max(1.1,Math.min(5,ze*Math.exp(c.deltaY*.001))),At()},{passive:!1});})();
