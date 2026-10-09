

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
