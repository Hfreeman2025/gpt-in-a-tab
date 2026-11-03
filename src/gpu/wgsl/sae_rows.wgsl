// A batch of residual-stream vectors for the sparse autoencoder, scaled so the average vector
// has squared length C. Rows are drawn at random from the collected activations (hash of seed
// and row), or taken in order from `first` when seed is zero.

struct Params { B: u32, C: u32, nRows: u32, seed: u32, scale: f32, first: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> acts: array<f32>;
@group(0) @binding(2) var<storage, read> step: array<u32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.B * P.C) { return; }
  let b = i / P.C;
  let c = i % P.C;
  var row = P.first + b;
  if (P.seed != 0u) {
    // Word 9 of the step header is the step number: a fresh draw every step.
    row = hash32(hash32(P.seed ^ step[9]) ^ (b * 0x9e3779b9u)) % P.nRows;
  }
  x[i] = acts[row * P.C + c] * P.scale;
}
