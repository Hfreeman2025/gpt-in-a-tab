// Global gradient norm, stage two: sum the partial squares, then store the norm (in the ring
// and as "current") and the clipping multiplier the optimiser will apply.

struct Params { groups: u32, ring: u32, current: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> parts: array<f32>;
@group(0) @binding(2) var<storage, read> step: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var s = 0.0;
  for (var i = li; i < P.groups; i += 256u) { s += parts[i]; }
  red[li] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] += red[li + k]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    let norm = sqrt(red[0]);
    let clipNorm = bitcast<f32>(step[8]);
    stats[P.ring + step[3]] = norm;
    stats[P.current] = norm;
    stats[P.current + 1u] = min(1.0, clipNorm / (norm + 1e-6));
  }
}
