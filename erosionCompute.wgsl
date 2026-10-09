

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
