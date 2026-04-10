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
