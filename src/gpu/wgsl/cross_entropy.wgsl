// Cross-entropy, forward and backward fused, one workgroup per row:
//   loss[n] = logsumexp(logits[n]) − logits[n][target]
//   dlogits[n] = (softmax(logits[n]) − onehot(target)) / N

struct Params { N: u32, V: u32, invN: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> dlogits: array<f32>;
@group(0) @binding(4) var<storage, read_write> losses: array<f32>;

var<workgroup> red: array<f32, 64>;

fn max64(v: f32, li: u32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] = max(red[li], red[li + s]); }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

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
  let base = n * P.V;

  var mx = -3.0e38;
  for (var v = li; v < P.V; v += 64u) { mx = max(mx, logits[base + v]); }
  mx = max64(mx, li);

  var s = 0.0;
  for (var v = li; v < P.V; v += 64u) { s += exp(logits[base + v] - mx); }
  let total = sum64(s, li);
  let inv = 1.0 / total;
  let t = targets[n];

  for (var v = li; v < P.V; v += 64u) {
    let p = exp(logits[base + v] - mx) * inv;
    var hot = 0.0;
    if (v == t) { hot = 1.0; }
    dlogits[base + v] = (p - hot) * P.invN;
  }
  if (li == 0u) {
    losses[n] = log(total) + mx - logits[base + t];
  }
}
