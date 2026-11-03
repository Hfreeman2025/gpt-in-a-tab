// Global gradient norm, stage one: each workgroup sums the squares of one chunk of the flat
// gradient buffer.

struct Params { n: u32, perGroup: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> parts: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let start = wid.x * P.perGroup;
  let end = min(P.n, start + P.perGroup);
  var s = 0.0;
  for (var i = start + li; i < end; i += 256u) {
    let g = grads[i];
    s += g * g;
  }
  red[li] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] += red[li + k]; }
    workgroupBarrier();
  }
  if (li == 0u) { parts[wid.x] = red[0]; }
}
