// The induction test (Olsson et al. 2022), per head of one layer, over a batch of random runs
// repeated with the given period: the prefix-matching score (mean attention from each position
// of the second copy to the character after the previous occurrence of its own character) and
// the previous-character score. One workgroup per head; the results land in this probe's slot.
// CPU reference: scoreProbe in src/model/probe.ts.

struct Params { B: u32, T: u32, H: u32, period: u32, base: u32, stride: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read> slot: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> pm: array<f32, 256>;
var<workgroup> pt: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let h = wid.x;
  let T = P.T;
  let span = T - P.period;
  let nPm = P.B * span;
  let nPt = P.B * (T - 1u);
  var a = 0.0;
  for (var k = li; k < nPm; k += 256u) {
    let b = k / span;
    let i = P.period + k % span;
    a += probs[((b * P.H + h) * T + i) * T + i - P.period + 1u];
  }
  var c = 0.0;
  for (var k = li; k < nPt; k += 256u) {
    let b = k / (T - 1u);
    let i = 1u + k % (T - 1u);
    c += probs[((b * P.H + h) * T + i) * T + i - 1u];
  }
  pm[li] = a;
  pt[li] = c;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) {
      pm[li] += pm[li + s];
      pt[li] += pt[li + s];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let o = slot[0] * P.stride + P.base + h * 2u;
    out[o] = pm[0] / f32(nPm);
    out[o + 1u] = pt[0] / f32(nPt);
  }
}
