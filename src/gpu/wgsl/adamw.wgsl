// AdamW over the whole flat parameter buffer in one dispatch. Matrices sit before decayEnd and
// get decoupled weight decay; biases and layer-norm parameters do not. The learning rate and
// bias corrections come from this step's header, the clip multiplier from grad_norm_final.

struct Params { n: u32, decayEnd: u32, clipAt: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> step: array<u32>;
@group(0) @binding(2) var<storage, read> grads: array<f32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> params: array<f32>;
@group(0) @binding(5) var<storage, read_write> m: array<f32>;
@group(0) @binding(6) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.n) { return; }
  let lr = bitcast<f32>(step[0]);
  let bc1 = bitcast<f32>(step[1]);
  let bc2 = bitcast<f32>(step[2]);
  let wd = bitcast<f32>(step[4]);
  let b1 = bitcast<f32>(step[5]);
  let b2 = bitcast<f32>(step[6]);
  let eps = bitcast<f32>(step[7]);

  let g = grads[i] * stats[P.clipAt];
  let mi = b1 * m[i] + (1.0 - b1) * g;
  let vi = b2 * v[i] + (1.0 - b2) * g * g;
  m[i] = mi;
  v[i] = vi;
  let p = params[i];
  var decay = 0.0;
  if (i < P.decayEnd) { decay = wd * p; }
  params[i] = p - lr * (mi / bc1 / (sqrt(vi / bc2) + eps) + decay);
}
