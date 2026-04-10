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

