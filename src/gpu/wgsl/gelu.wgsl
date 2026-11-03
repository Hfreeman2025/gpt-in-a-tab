// GELU, tanh approximation (GPT-2). tanh is written out with a clamp because the built-in
// can return NaN for large arguments on some drivers.

struct Params { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

const K0: f32 = 0.7978845608028654; // sqrt(2/pi)
const K1: f32 = 0.044715;

fn tanh_safe(u: f32) -> f32 {
  let e = exp(2.0 * clamp(u, -15.0, 15.0));
  return (e - 1.0) / (e + 1.0);
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.n) { return; }
  let v = x[i];
  y[i] = 0.5 * v * (1.0 + tanh_safe(K0 * (v + K1 * v * v * v)));
}
