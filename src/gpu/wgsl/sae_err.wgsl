// Squared reconstruction error and squared length, summed over a batch: out[slot·2] += Σ (x̂−x)²
// and out[slot·2+1] += Σ x² (for the fraction of variance explained). One workgroup.

struct Params { n: u32, slot: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> xhat: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> e: array<f32, 256>;
var<workgroup> q: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var a = 0.0;
  var c = 0.0;
  for (var i = li; i < P.n; i += 256u) {
    let d = xhat[i] - x[i];
    a += d * d;
    c += x[i] * x[i];
  }
  e[li] = a;
  q[li] = c;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) {
      e[li] += e[li + s];
      q[li] += q[li + s];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    out[P.slot * 2u] += e[0];
    out[P.slot * 2u + 1u] += q[0];
  }
}
