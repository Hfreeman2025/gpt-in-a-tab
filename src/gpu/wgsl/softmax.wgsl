// Causal softmax over attention-score rows of length T, one workgroup per row.
// Row r is query time t = r mod T; it may see keys 0..t and gives the future probability zero.

struct Params { rows: u32, T: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

var<workgroup> red: array<f32, 64>;

fn max64(v: f32, li: u32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] = max(red[li], red[li + s]); }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

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

  var mx = -3.0e38;
  for (var j = li; j < len; j += 64u) { mx = max(mx, x[base + j]); }
  mx = max64(mx, li);

  var s = 0.0;
  for (var j = li; j < len; j += 64u) { s += exp(x[base + j] - mx); }
  let inv = 1.0 / sum64(s, li);

  for (var j = li; j < P.T; j += 64u) {
    var p = 0.0;
    if (j < len) { p = exp(x[base + j] - mx) * inv; }
    y[base + j] = p;
  }
}
