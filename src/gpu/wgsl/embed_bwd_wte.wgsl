// Token-embedding gradient without float atomics: each thread owns one (token, channel, chunk)
// and scans its chunk of rows for that token. parts[s][v][c] = Σ_{n in chunk s, token[n] = v} dx[n][c].
// reduce.wgsl then adds the chunks onto the output-head gradient (the weights are tied).

struct Params { N: u32, C: u32, V: u32, S: u32, chunk: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read> dx: array<f32>;
@group(0) @binding(3) var<storage, read_write> parts: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nw: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let i = (wid.y * nw.x + wid.x) * 256u + li;
  let VC = P.V * P.C;
  if (i >= VC * P.S) { return; }
  let s = i / VC;
  let vc = i % VC;
  let v = vc / P.C;
  let c = vc % P.C;
  let r0 = s * P.chunk;
  let r1 = min(P.N, r0 + P.chunk);
  var acc = 0.0;
  for (var n = r0; n < r1; n++) {
    if (tokens[n] == v) { acc += dx[n * P.C + c]; }
  }
  parts[i] = acc;
}
