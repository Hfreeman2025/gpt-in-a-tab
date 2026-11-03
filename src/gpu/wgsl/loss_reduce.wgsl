// Mean of the per-row losses, written into a ring of results at this step's slot so the CPU
// can read many steps back at once instead of stalling on every step.

struct Params { N: u32, ring: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> losses: array<f32>;
@group(0) @binding(2) var<storage, read> step: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var s = 0.0;
  for (var i = li; i < P.N; i += 256u) { s += losses[i]; }
  red[li] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] += red[li + k]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    stats[P.ring + step[3]] = red[0] / f32(P.N);
  }
}
