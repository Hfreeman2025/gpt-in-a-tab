// Edits one layer's attention output, per head, before the output projection: each head's
// slice is scaled by keep[h] and has patch[h] times the same slice of another run added. Keep 1,
// patch 0 leaves it alone; keep 0, patch 0 knocks the head out; keep 0, patch 1 patches in the
// head's output from the other run (activation patching).

struct Params { N: u32, C: u32, H: u32, _p: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> edit: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.N * P.C) { return; }
  let h = (i % P.C) / (P.C / P.H);
  y[i] = edit[h] * y[i] + edit[P.H + h] * src[i];
}
