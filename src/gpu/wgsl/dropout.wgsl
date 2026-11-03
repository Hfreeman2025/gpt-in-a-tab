// Dropout mask applied to a gradient (the backward of dropout is the same multiplication):
// out = d ⊙ mask(site).

struct Params { n: u32, site: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> d: array<f32>;
@group(0) @binding(2) var<storage, read> drop: Dropout;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.n) { return; }
  let dd = drop;
  out[i] = d[i] * drop_mask(dd, site_seed(dd.seed, P.site), i);
}
