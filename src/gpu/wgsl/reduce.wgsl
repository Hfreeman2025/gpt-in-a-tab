// Sums S partial results: out[i] (+)= Σ_s parts[off + s·stride + i].
// The second half of every split-K matmul and column reduction.

struct Params { len: u32, S: u32, stride: u32, off: u32 };

override ACCUM: bool = false;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> parts: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.len) { return; }
  var s = 0.0;
  for (var z = 0u; z < P.S; z++) { s += parts[P.off + z * P.stride + i]; }
  if (ACCUM) { s += out[i]; }
  out[i] = s;
}
