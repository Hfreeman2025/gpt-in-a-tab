// Rescales every row of the decoder (one row per feature, C long) to unit length, after each
// optimiser step, so the L1 penalty cannot be dodged by shrinking features and growing their
// directions (Bricken et al. 2023).

struct Params { F: u32, C: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x;
  if (j >= P.F) { return; }
  var s = 0.0;
  for (var c = 0u; c < P.C; c++) {
    let v = w[j * P.C + c];
    s += v * v;
  }
  let k = 1.0 / max(sqrt(s), 1e-8);
  for (var c = 0u; c < P.C; c++) { w[j * P.C + c] *= k; }
}
