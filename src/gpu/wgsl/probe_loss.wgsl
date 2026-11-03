// Mean loss over two ranges of positions in every row (the first and second copy of the
// induction test, or any two spans), written into a probe slot at the given offset.

struct Params { B: u32, T: u32, lo0: u32, hi0: u32, lo1: u32, hi1: u32, base: u32, stride: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> losses: array<f32>;
@group(0) @binding(2) var<storage, read> slot: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> r0: array<f32, 256>;
var<workgroup> r1: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  let n0 = P.hi0 - P.lo0;
  let n1 = P.hi1 - P.lo1;
  var a = 0.0;
  for (var k = li; k < P.B * n0; k += 256u) { a += losses[(k / n0) * P.T + P.lo0 + k % n0]; }
  var c = 0.0;
  for (var k = li; k < P.B * n1; k += 256u) { c += losses[(k / n1) * P.T + P.lo1 + k % n1]; }
  r0[li] = a;
  r1[li] = c;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) {
      r0[li] += r0[li + s];
      r1[li] += r1[li + s];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let o = slot[0] * P.stride + P.base;
    out[o] = r0[0] / f32(P.B * n0);
    out[o + 1u] = r1[0] / f32(P.B * n1);
  }
}
