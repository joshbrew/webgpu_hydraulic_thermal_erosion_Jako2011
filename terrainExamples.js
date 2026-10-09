export const TERRAIN_EXAMPLES = {
  alpine: { label: 'Alpine catchment', mode: 0, preset: 'paper_balanced', layered: true },
  fan: { label: 'Mountain outlet and fan', mode: 1, preset: 'delta_depositor', layered: true },
  badlands: { label: 'Warped badlands', mode: 2, preset: 'badlands', layered: true },
  canyon: { label: 'Layered canyon', mode: 3, preset: 'canyon_carver', layered: true },
  talus: { label: 'Cliffs and talus', mode: 4, preset: 'thermal_heavy', layered: true },
  basin: { label: 'Closed basin', mode: 5, preset: 'flash_flood', closed: true, layered: true },
  river_delta: { label: 'River delta', mode: 6, preset: 'river_delta', layered: true },
  mountain_canyon: { label: 'Mountain river canyon', mode: 7, preset: 'mountain_canyon', layered: true },
  volcano: { label: 'Longonot-inspired volcano', mode: 8, preset: 'volcanic_cone', closed: true, layered: true },
  crater_lake: { label: 'Volcanic crater lake', mode: 8, preset: 'volcanic_lake', closed: true, layered: true },
  lowland_river: { label: 'Lowland river and floodplain', mode: 9, preset: 'lowland_meanders', layered: true },
};

// Top to bottom. Only the surface follows the selected erosion profile;
// deeper units retain their individual strengths when changing that profile.
const bedrock = { label: 'Resistant bedrock', hardnessMin: 0, hardnessMax: 0.02, thermalEnabled: false };
export const EXAMPLE_MATERIALS = {
  alpine: [{ label: 'Mountain soil' }, { label: 'Weathered rock', hardnessMin: 0.1, hardnessMax: 0.18 }, bedrock],
  fan: [{ label: 'Loose fan sediment' }, { label: 'Older alluvium', hardnessMin: 0.28, hardnessMax: 0.45 }, bedrock],
  badlands: [{ label: 'Soft surface beds' }, { label: 'Cemented ledges', hardnessMin: 0.08, hardnessMax: 0.16, thermalEnabled: false }, { label: 'Weak sediment beds', hardnessMin: 0.45, hardnessMax: 0.65 }, bedrock],
  canyon: [{ label: 'Upper sediment' }, { label: 'Resistant shelves', hardnessMin: 0.04, hardnessMax: 0.09, thermalEnabled: false }, { label: 'Weathered lower beds', hardnessMin: 0.18, hardnessMax: 0.3 }, bedrock],
  talus: [{ label: 'Loose talus' }, { label: 'Fractured cliff rock', hardnessMin: 0.25, hardnessMax: 0.4 }, bedrock],
  basin: [{ label: 'Basin sediment' }, { label: 'Compacted substrate', hardnessMin: 0.12, hardnessMax: 0.22 }, bedrock],
  river_delta: [{ label: 'Young delta sediment' }, { label: 'Older alluvium', hardnessMin: 0.18, hardnessMax: 0.3 }, bedrock],
  lowland_river: [{ label: 'Loose floodplain alluvium' }, { label: 'Older river deposits', hardnessMin: 0.4, hardnessMax: 0.6 }, bedrock],
  mountain_canyon: [{ label: 'Thin canyon soil' }, { label: 'Weathered rock', hardnessMin: 0.05, hardnessMax: 0.09, thermalEnabled: false }, bedrock],
  volcano: [{ label: 'Loose volcanic ash' }, { label: 'Unconsolidated tephra', hardnessMin: 0.5, hardnessMax: 0.72 }, { label: 'Resistant tuff', hardnessMin: 0.08, hardnessMax: 0.16, thermalEnabled: false }, { ...bedrock, label: 'Lava core' }],
};
EXAMPLE_MATERIALS.crater_lake = EXAMPLE_MATERIALS.volcano;

// Compact 2-D simplex + domain-warped FBM, adapted from the noiseCompute
// shader approach. Integer hashing replaces its permutation-table binding.
export const TERRAIN_EXAMPLE_WGSL = /* wgsl */ `
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
`;

export async function generateTerrainExample(device, name = 'alpine', size = 512, seed = 7) {
  const example = TERRAIN_EXAMPLES[name];
  if (!example) throw new Error(`Unknown terrain example: ${name}`);
  size = Math.max(64, Math.min(1024, Math.floor(size) || 512));
  const bytes = size * size * 16;
  const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto', compute: { module: device.createShaderModule({ code: TERRAIN_EXAMPLE_WGSL }), entryPoint: 'generate' },
    });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: output } },
    ] });
    device.queue.writeBuffer(uniform, 0, new Uint32Array([size, example.mode, seed >>> 0, 0]));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const packed = new Float32Array(readback.getMappedRange());
    const bands = Array.from({ length: EXAMPLE_MATERIALS[name].length }, (_, band) => Float32Array.from({ length: size * size }, (_, i) => packed[i * 4 + band]));
    const values = bands[0];
    readback.unmap();
    const raster = { width: size, height: size, values };
    if (example.layered) raster.bands = bands;
    if (name === 'river_delta' || name === 'mountain_canyon' || name === 'crater_lake' || name === 'lowland_river') {
      raster.water = Float32Array.from(values, (h, i) => {
        const x = (i % size) / (size - 1), y = Math.floor(i / size) / (size - 1);
        if (name === 'crater_lake') return Math.hypot((x - 0.51) * 1.06, (y - 0.49) * 0.94) < 0.2 ? Math.max(0.34 - h, 0) : 0;
        const level = name === 'lowland_river' ? 0.065 + 0.04 * (1 - y) + 0.008 : name === 'river_delta' ? Math.max(0.12, 0.045 + 0.16 * (1 - y) + 0.012) : 0.075 + 0.23 * (1 - y) + 0.012;
        return Math.max(level - h, 0);
      });
    }
    return raster;
  } finally {
    uniform.destroy(); output.destroy(); readback.destroy();
  }
}

// Shared by the demo and regression/gallery fixtures, including DEM resets.
export function terrainExampleOptions(name, preset) {
  if (!TERRAIN_EXAMPLES[name]?.layered) return {};
  return { layerConfig: { mode: 'packed_rgba', layers: EXAMPLE_MATERIALS[name].map(material => ({
    enabled: true, heightMin: 0, heightMax: 1, hardnessMin: preset.erodabilityMin, hardnessMax: preset.erodabilityMax, thermalEnabled: true, ...material,
  })) } };
}

export function seedTerrainExampleSources(sim, raster, name) {
  if (name !== 'river_delta' && name !== 'mountain_canyon' && name !== 'lowland_river') return;
  const y = Math.round((raster.height - 1) * (name === 'lowland_river' ? 0.035 : 0.08));
  let x = Math.round(raster.width * 0.5);
  for (let candidate = Math.floor(raster.width * 0.2); candidate < raster.width * 0.8; candidate++) {
    if (raster.values[y * raster.width + candidate] < raster.values[y * raster.width + x]) x = candidate;
  }
  sim.paintSpringBrush({ x, y, radius: Math.max(2, 4 * raster.width / 256), strength: sim.simulationParams.sourceStrength, hardness: 1 });
}
