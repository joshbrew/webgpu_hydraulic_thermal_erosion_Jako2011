// Hierarchical reductions leave only 64 bytes to map, even for a large DEM.
const STATISTICS_WGSL = /* wgsl */ `
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
`;

export default class GpuStatistics {
  constructor(device) {
    this.device = device;
    this.pipelines = null;
    this.resources = [];
    this.count = 0;
    this.pending = Promise.resolve();
  }

  async initialize() {
    if (!this.pipelines) {
      const module = this.device.createShaderModule({ code: STATISTICS_WGSL, label: 'erosion-mass-reduction' });
      this.pipelines = Promise.all(['reduceCells', 'reducePartials'].map(entryPoint =>
        this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint } })));
    }
    return this.pipelines;
  }

  read(state, cellCount, snapshot = () => ({})) {
    // Mapping the same staging buffer twice is invalid. Serialize inspections.
    const result = this.pending.then(() => this.reduce(state, cellCount, snapshot));
    this.pending = result.catch(() => {});
    return result;
  }

  async reduce(state, cellCount, snapshot) {
    const [first, next] = await this.initialize();
    if (this.count !== cellCount) {
      this.destroyResources();
      this.count = cellCount;
      const groups = Math.max(1, Math.ceil(cellCount / 256));
      this.buffers = [0, 1].map(() => this.device.createBuffer({
        size: groups * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        label: 'erosion-mass-partials',
      }));
      this.staging = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      this.resources.push(...this.buffers, this.staging);
      this.uniforms = [];
      for (let count = cellCount; ; count = Math.ceil(count / 256)) {
        const uniform = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(uniform, 0, new Uint32Array([count, 0, 0, 0]));
        this.resources.push(uniform); this.uniforms.push(uniform);
        if (Math.ceil(count / 256) <= 1) break;
      }
    }
    const encoder = this.device.createCommandEncoder({ label: 'erosion-inspect-mass' });
    const pass = encoder.beginComputePass();
    let count = cellCount, outputIndex = 0, stage = 0;
    while (true) {
      const pipeline = stage === 0 ? first : next;
      const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.uniforms[stage] } },
        { binding: stage === 0 ? 1 : 2, resource: { buffer: stage === 0 ? state : this.buffers[1 - outputIndex] } },
        { binding: 3, resource: { buffer: this.buffers[outputIndex] } },
      ] });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      count = Math.ceil(count / 256);
      pass.dispatchWorkgroups(count);
      if (count <= 1) break;
      outputIndex = 1 - outputIndex; stage++;
    }
    pass.end();
    encoder.copyBufferToBuffer(this.buffers[outputIndex], 0, this.staging, 0, 64);
    const metadata = snapshot();
    this.device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ);
    let data;
    try { data = new Float32Array(this.staging.getMappedRange()).slice(); }
    finally { this.staging.unmap(); }
    const validCells = data[3];
    const range = (index) => validCells ? { min: data[4 + index], max: data[8 + index] } : null;
    return {
      ...metadata,
      validCells, invalidCells: data[15],
      totalTerrain: data[0], totalWater: data[1], totalSediment: data[2], totalSolid: data[0] + data[2],
      terrainRange: range(0), waterRange: range(1), sedimentRange: range(2),
      historyRange: validCells ? { min: data[12], max: data[13] } : null,
      averageHardness: validCells ? data[14] / validCells : 0,
      statisticsReadbackBytes: 64,
    };
  }

  destroyResources() {
    for (const resource of this.resources) resource.destroy();
    this.resources = []; this.count = 0;
  }

  destroy() { this.destroyResources(); }
}
