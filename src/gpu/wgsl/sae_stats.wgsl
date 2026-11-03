// Per feature, over a batch of activations f[B×F]: how many rows it fired on, the sum and the
// sum of squares of its activations, and its K largest activations with the rows they came from
// (kept sorted, largest first, across calls). Rows are numbered from `rowBase`.

struct Params { B: u32, F: u32, K: u32, rowBase: u32, track: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> f: array<f32>;
@group(0) @binding(2) var<storage, read_write> sums: array<f32>;
@group(0) @binding(3) var<storage, read_write> topv: array<f32>;
@group(0) @binding(4) var<storage, read_write> topi: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= P.F) { return; }
  var n = 0.0;
  var s = 0.0;
  var s2 = 0.0;
  let base = j * P.K;
  for (var b = 0u; b < P.B; b++) {
    let v = f[b * P.F + j];
    if (v <= 0.0) { continue; }
    n += 1.0;
    s += v;
    s2 += v * v;
    if (P.track == 0u || v <= topv[base + P.K - 1u]) { continue; }
    var k = P.K - 1u;
    loop {
      if (k == 0u || topv[base + k - 1u] >= v) { break; }
      topv[base + k] = topv[base + k - 1u];
      topi[base + k] = topi[base + k - 1u];
      k -= 1u;
    }
    topv[base + k] = v;
    topi[base + k] = P.rowBase + b;
  }
  sums[j * 3u] += n;
  sums[j * 3u + 1u] += s;
  sums[j * 3u + 2u] += s2;
}
