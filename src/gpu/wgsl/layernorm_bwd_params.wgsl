// Partial sums for the layer-norm gain and bias gradients over S chunks of rows:
//   parts[s][c] = Σ dy·x̂,   parts[s][C + c] = Σ dy
// reduce.wgsl then sums the chunks into dg and db.

struct Params { N: u32, C: u32, S: u32, chunk: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> rstd: array<f32>;
@group(0) @binding(5) var<storage, read_write> parts: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  if (i >= P.C * P.S) { return; }
  let c = i % P.C;
  let s = i / P.C;
  let r0 = s * P.chunk;
  let r1 = min(P.N, r0 + P.chunk);
  var dg = 0.0;
  var db = 0.0;
  for (var n = r0; n < r1; n++) {
    let d = dy[n * P.C + c];
    dg += d * (x[n * P.C + c] - mean[n]) * rstd[n];
    db += d;
  }
  parts[s * 2u * P.C + c] = dg;
  parts[s * 2u * P.C + P.C + c] = db;
}
