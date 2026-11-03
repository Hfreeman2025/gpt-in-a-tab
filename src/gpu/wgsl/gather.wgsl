// Builds a batch on the GPU: B windows of the corpus at the offsets written for this step,
// with targets shifted by one. The step buffer is [16-word header | B offsets]; an offset with
// its top bit set describes a window of repeated random characters (see REPEAT_FLAG).

struct Params { B: u32, T: u32, V: u32, _pad1: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> corpus: array<u32>;
@group(0) @binding(2) var<storage, read> step: array<u32>;
@group(0) @binding(3) var<storage, read_write> inputs: array<u32>;
@group(0) @binding(4) var<storage, read_write> targets: array<u32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.B * P.T) { return; }
  let b = i / P.T;
  let t = i % P.T;
  let raw = step[16u + b];
  if ((raw & 0x80000000u) != 0u) {
    // Random characters, a run of `period` repeated; the same hash as repeatToken in data.ts.
    let seed = raw & 0xffffffu;
    let period = (raw >> 24u) & 0x7fu;
    inputs[i] = hash32(seed * 0x9e3779b9u + t % period) % P.V;
    targets[i] = hash32(seed * 0x9e3779b9u + (t + 1u) % period) % P.V;
  } else {
    inputs[i] = corpus[raw + t];
    targets[i] = corpus[raw + t + 1u];
  }
}
