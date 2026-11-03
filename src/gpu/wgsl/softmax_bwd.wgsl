// Causal softmax backward, in place: d ← y ⊙ (d − Σ y·d) on the visible prefix, 0 beyond it.

struct Params { rows: u32, T: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> d: array<f32>;

var<workgroup> red: array<f32, 64>;

fn sum64(v: f32, li: u32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let r = wid.y * nw.x + wid.x;
  if (r >= P.rows) { return; }
  let base = r * P.T;
  let len = (r % P.T) + 1u;

  var s = 0.0;
  for (var j = li; j < len; j += 64u) { s += y[base + j] * d[base + j]; }
  let dotYD = sum64(s, li);

  for (var j = li; j < P.T; j += 64u) {
    var v = 0.0;
    if (j < len) { v = y[base + j] * (d[base + j] - dotYD); }
    d[base + j] = v;
  }
}
