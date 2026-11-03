// The sparse autoencoder's loss gradients, elementwise.
//   mode 0, reconstruction: d[i] = 2 (x̂[i] − x[i]) / B          (a = x̂, b = x)
//   mode 1, through the ReLU and the L1 penalty: d[i] = (d[i] + λ/B) · [f[i] > 0]   (a = f)

struct Params { n: u32, mode: u32, k: f32, _p: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> d: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.n) { return; }
  if (P.mode == 0u) {
    d[i] = P.k * (a[i] - b[i]);
  } else {
    d[i] = select(0.0, d[i] + P.k, a[i] > 0.0);
  }
}
