// Token + position embedding, with dropout: x[n] = dropout(wte[token[n]] + wpe[n mod T]).

struct Params { N: u32, T: u32, C: u32, site: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read> wte: array<f32>;
@group(0) @binding(3) var<storage, read> wpe: array<f32>;
@group(0) @binding(4) var<storage, read> drop: Dropout;
@group(0) @binding(5) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.N * P.C) { return; }
  let n = i / P.C;
  let c = i % P.C;
  let d = drop;
  let e = wte[tokens[n] * P.C + c] + wpe[(n % P.T) * P.C + c];
  x[i] = e * drop_mask(d, site_seed(d.seed, P.site), i);
}
