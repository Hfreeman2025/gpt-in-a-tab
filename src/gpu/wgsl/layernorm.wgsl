// Layer norm, one 64-thread workgroup per row. Two-pass (mean, then variance) for stability.
// Saves the mean and reciprocal standard deviation for the backward pass.

struct Params { N: u32, C: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read_write> mean: array<f32>;
@group(0) @binding(6) var<storage, read_write> rstd: array<f32>;

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
  let n = wid.y * nw.x + wid.x;
  if (n >= P.N) { return; }
  let base = n * P.C;
  let fc = f32(P.C);

  var s = 0.0;
  for (var c = li; c < P.C; c += 64u) { s += x[base + c]; }
  let m = sum64(s, li) / fc;

  var v = 0.0;
  for (var c = li; c < P.C; c += 64u) {
    let d = x[base + c] - m;
    v += d * d;
  }
  let r = 1.0 / sqrt(sum64(v, li) / fc + P.eps);

  for (var c = li; c < P.C; c += 64u) {
    y[base + c] = (x[base + c] - m) * r * g[c] + b[c];
  }
  if (li == 0u) {
    mean[n] = m;
    rstd[n] = r;
  }
}
