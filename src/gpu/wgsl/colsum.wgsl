// Partial column sums of a rows×cols matrix over S chunks of rows: parts[s][j] = Σ x[r][j].
// Bias gradients are column sums of the upstream gradient; reduce.wgsl finishes the job.

struct Params { rows: u32, cols: u32, S: u32, chunk: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> parts: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.cols * P.S) { return; }
  let j = i % P.cols;
  let s = i / P.cols;
  let r0 = s * P.chunk;
  let r1 = min(P.rows, r0 + P.chunk);
  var acc = 0.0;
  for (var r = r0; r < r1; r++) { acc += x[r * P.cols + j]; }
  parts[s * P.cols + j] = acc;
}
