// Position-embedding gradient: dwpe[t] = Σ_b dx[b·T + t].

struct Params { B: u32, T: u32, C: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> dx: array<f32>;
@group(0) @binding(2) var<storage, read_write> dwpe: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.T * P.C) { return; }
  var acc = 0.0;
  for (var b = 0u; b < P.B; b++) { acc += dx[b * P.T * P.C + i]; }
  dwpe[i] = acc;
}
