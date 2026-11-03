// f = max(pre, 0), in place: the sparse autoencoder's feature activations.

struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> f: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.n) { return; }
  f[i] = max(f[i], 0.0);
}
