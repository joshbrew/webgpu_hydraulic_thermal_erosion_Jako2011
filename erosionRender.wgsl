

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
