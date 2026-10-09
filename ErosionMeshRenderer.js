import { getTerrainRenderShader } from './WebGPUTerrainErosion.js';

// Display-only topology. Heights, material layers and water are sampled on the
// GPU from the paused solver state; this renderer never changes that state.
const postShader = /* wgsl */ `
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
`;

export default class ErosionMeshRenderer {
  constructor(sim) { this.sim = sim; this.buffers = []; }

  async initialize(topology) {
    const { sim } = this, d = sim.device;
    const positions = topology.positionsGrid, indices = topology.indices;
    if (!positions?.length || !indices?.length) throw Error('No triangles were produced. Try a finer spacing.');
    for (const i of indices) if (i >= positions.length / 2) throw Error('Invalid mesh index.');
    for (const p of positions) if (!Number.isFinite(p)) throw Error('Invalid mesh position.');
    this.indexCount = indices.length;
    const upload = (values, usage) => {
      const b = d.createBuffer({ size: values.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(b, 0, values); this.buffers.push(b); return b;
    };
    this.positions = upload(positions, GPUBufferUsage.STORAGE);
    this.indices = upload(indices, GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX);
    this.style = upload(new Float32Array([.55,0,0,0]), GPUBufferUsage.UNIFORM);
    this.dummyWater = upload(new Float32Array(4), GPUBufferUsage.STORAGE);
    this.dummyGeometry = upload(new Float32Array(12), GPUBufferUsage.STORAGE);
    const V = GPUShaderStage.VERTEX, F = GPUShaderStage.FRAGMENT;
    const groupLayout = d.createBindGroupLayout({ entries: [
      { binding: 0, visibility: V | F, buffer: { type: 'uniform' } },
      { binding: 1, visibility: V | F, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: F, sampler: { type: 'filtering' } },
      { binding: 3, visibility: F, texture: { sampleType: 'float' } },
      { binding: 4, visibility: V, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: V, buffer: { type: 'uniform' } },
      { binding: 6, visibility: V, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: V, buffer: { type: 'read-only-storage' } },
      { binding: 8, visibility: V, buffer: { type: 'read-only-storage' } },
      { binding: 9, visibility: F, buffer: { type: 'uniform' } },
      { binding: 10, visibility: V, buffer: { type: 'read-only-storage' } },
      { binding: 11, visibility: V, buffer: { type: 'read-only-storage' } },
    ] });
    const layout = d.createPipelineLayout({ bindGroupLayouts: [groupLayout] });
    const module = d.createShaderModule({ code: getTerrainRenderShader() + postShader });
    const blend = { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
    const pipeline = (vertex, fragment, blended, write) => d.createRenderPipelineAsync({
      layout, vertex: { module, entryPoint: vertex },
      fragment: { module, entryPoint: fragment, targets: [{ format: sim.presentationFormat, ...(blended ? { blend } : {}) }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: write, depthCompare: 'less-equal' },
    });
    [this.terrain, this.water, this.wire, this.sweWater] = await Promise.all([
      pipeline('vsPost','fsMesh',false,true), pipeline('vsPost','fsMesh',true,false), pipeline('vsWire','fsWire',true,false),
      pipeline('vsSWE','fsMesh',true,false),
    ]);
    this.groupLayout=groupLayout;
    this.refreshGroups();
    return this;
  }

  refreshGroups(swe=null) {
    const sim=this.sim,d=sim.device;
    const key=[sim.stateBuffers[0],sim.sceneColorView,sim.layerTopBuffer,sim.velocityBuffer,swe?.states[0],swe?.geometry];
    if(this.groupKey?.every((v,i)=>v===key[i]))return;
    this.groupKey=key;
    const group = (water, waterState=this.dummyWater) => d.createBindGroup({ layout: this.groupLayout, entries: [
      { binding: 0, resource: { buffer: water ? sim.renderUniformBufferWater : sim.renderUniformBufferTerrain } },
      { binding: 1, resource: { buffer: sim.stateBuffers[0] } },
      { binding: 2, resource: sim.sceneSampler },
      { binding: 3, resource: water ? sim.sceneColorView : sim.dummySceneView },
      { binding: 4, resource: { buffer: sim.layerTopBuffer } },
      { binding: 5, resource: { buffer: sim.layerMaterialBuffer } },
      { binding: 6, resource: { buffer: sim.velocityBuffer } },
      { binding: 7, resource: { buffer: this.positions } },
      { binding: 8, resource: { buffer: this.indices } },
      { binding: 9, resource: { buffer: this.style } },
      { binding: 10, resource: { buffer: waterState } },
      { binding: 11, resource: { buffer: swe?.geometry || this.dummyGeometry } },
    ] });
    this.terrainGroup = group(false); this.waterGroup = group(true);
    this.sweGroups=swe?swe.states.map(b=>group(true,b)):null;
  }

  render({ adaptive = true, wireframe = true, wireOpacity = .55, swe = null } = {}) {
    const s = this.sim, e = s.device.createCommandEncoder();
    this.refreshGroups(swe);
    s.device.queue.writeBuffer(this.style,0,new Float32Array([Math.max(0,Math.min(1,wireOpacity)),0,0,0]));
    // Synchronize the shared camera uniforms without drawing or advancing time.
    s.setSimulationParams({});
    const first = e.beginRenderPass({
      colorAttachments: [{ view: s.sceneColorView, clearValue: { r: .055, g: .065, b: .08, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: s.depthTextureView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    first.setPipeline(adaptive ? this.terrain : s.renderPipeline);
    first.setBindGroup(0, adaptive ? this.terrainGroup : s.renderBindGroupTerrain);
    if (adaptive) { first.setIndexBuffer(this.indices, 'uint32'); first.drawIndexed(this.indexCount); }
    else first.draw(6, s.drawCount);
    first.end();
    const final = e.beginRenderPass({
      colorAttachments: [{ view: s.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: .055, g: .065, b: .08, a: 1 } }],
      depthStencilAttachment: { view: s.depthTextureView, depthLoadOp: 'load', depthStoreOp: 'discard' },
    });
    final.setPipeline(s.compositePipeline); final.setBindGroup(0, s.compositeBindGroup); final.draw(3);
    final.setPipeline(swe ? this.sweWater : adaptive ? this.water : s.waterRenderPipeline);
    final.setBindGroup(0, swe ? this.sweGroups[swe.current] : adaptive ? this.waterGroup : s.renderBindGroupWater);
    if (swe) final.draw(this.indexCount);
    else if (adaptive) { final.setIndexBuffer(this.indices, 'uint32'); final.drawIndexed(this.indexCount); }
    else final.draw(6, s.drawCount);
    if (wireframe) { final.setPipeline(this.wire); final.setBindGroup(0, this.terrainGroup); final.draw(this.indexCount); }
    final.end(); s.device.queue.submit([e.finish()]);
  }

  destroy() { for (const b of this.buffers) b.destroy(); this.buffers = []; }
}
