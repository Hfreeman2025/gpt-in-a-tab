// Layer-norm backward for the input, one workgroup per row:
//   dx (+)= rstd · (dy·g − mean(dy·g) − x̂ · mean(dy·g·x̂))
// Accumulating into dx adds the branch gradient onto the residual stream in place.

struct Params { N: u32, C: u32, _pad0: u32, _pad1: u32 };

override ACCUM: bool = true;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> rstd: array<f32>;
@group(0) @binding(5) var<storage, read> g: array<f32>;
@group(0) @binding(6) var<storage, read_write> dx: array<f32>;

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
  let m = mean[n];
  let r = rstd[n];

  var s1 = 0.0;
  var s2 = 0.0;
  for (var c = li; c < P.C; c += 64u) {
    let xhat = (x[base + c] - m) * r;
    let d = dy[base + c] * g[c];
    s1 += d;
    s2 += d * xhat;
  }
  let fc = f32(P.C);
  let meanD = sum64(s1, li) / fc;
  let meanDX = sum64(s2, li) / fc;

  for (var c = li; c < P.C; c += 64u) {
    let xhat = (x[base + c] - m) * r;
    let d = dy[base + c] * g[c];
    var v = r * (d - meanD - xhat * meanDX);
    if (ACCUM) { v += dx[base + c]; }
    dx[base + c] = v;
  }
}
