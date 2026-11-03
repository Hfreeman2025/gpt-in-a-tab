// Tiled, batched matrix multiply: C = alpha · op(A) · op(B) (+ bias) (+ C).
//
// Each 16×16 workgroup computes a 64×64 tile of C; each thread owns a 4×4 block of it,
// strided by 16 so that neighbouring threads touch neighbouring memory. A and B are staged
// through workgroup memory in 16-deep slices of K.
//
// Transposes, bias and accumulation are pipeline-override constants, so every variant compiles
// to straight-line code. Strides make the same kernel serve the linear layers (one batch),
// attention (batch × head) and split-K weight gradients (batch = K split).

struct Params {
  M: u32, N: u32, K: u32, nb2: u32,
  lda: u32, ldb: u32, ldc: u32, alpha: f32,
  aOff: u32, bOff: u32, cOff: u32, _pad0: u32,
  aS1: u32, aS2: u32, bS1: u32, bS2: u32,
  cS1: u32, cS2: u32, _pad1: u32, _pad2: u32,
};

override TRANS_A: bool = false; // A is stored K×M
override TRANS_B: bool = false; // B is stored N×K
override HAS_BIAS: bool = false;
override ACCUM: bool = false;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;

var<workgroup> As: array<f32, 1024>; // [BK][BM]
var<workgroup> Bs: array<f32, 1024>; // [BK][BN]

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let z1 = wid.z / P.nb2;
  let z2 = wid.z % P.nb2;
  let aBase = P.aOff + z1 * P.aS1 + z2 * P.aS2;
  let bBase = P.bOff + z1 * P.bS1 + z2 * P.bS2;
  let cBase = P.cOff + z1 * P.cS1 + z2 * P.cS2;
  let row0 = wid.y * BM;
  let col0 = wid.x * BN;

  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

  for (var k0 = 0u; k0 < P.K; k0 += BK) {
    // Stage a BM×BK slice of A, reading along whichever axis is contiguous in memory.
    for (var s = 0u; s < 4u; s++) {
      let e = li + s * 256u;
      var m: u32;
      var k: u32;
      if (TRANS_A) { k = e / BM; m = e % BM; } else { m = e / BK; k = e % BK; }
      let gm = row0 + m;
      let gk = k0 + k;
      var v = 0.0;
      if (gm < P.M && gk < P.K) {
        if (TRANS_A) { v = A[aBase + gk * P.lda + gm]; } else { v = A[aBase + gm * P.lda + gk]; }
      }
      As[k * BM + m] = v;
    }
    // And a BK×BN slice of B.
    for (var s = 0u; s < 4u; s++) {
      let e = li + s * 256u;
      var n: u32;
      var k: u32;
      if (TRANS_B) { n = e / BK; k = e % BK; } else { k = e / BN; n = e % BN; }
      let gn = col0 + n;
      let gk = k0 + k;
      var v = 0.0;
      if (gn < P.N && gk < P.K) {
        if (TRANS_B) { v = B[bBase + gn * P.ldb + gk]; } else { v = B[bBase + gk * P.ldb + gn]; }
      }
      Bs[k * BN + n] = v;
    }
    workgroupBarrier();

    for (var k = 0u; k < BK; k++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { a[i] = As[k * BM + lid.y + 16u * i]; }
      for (var j = 0u; j < 4u; j++) { b[j] = Bs[k * BN + lid.x + 16u * j]; }
      for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
          acc[i * 4u + j] = fma(a[i], b[j], acc[i * 4u + j]);
        }
      }
    }
    workgroupBarrier();
  }

  for (var i = 0u; i < 4u; i++) {
    let gm = row0 + lid.y + 16u * i;
    if (gm >= P.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let gn = col0 + lid.x + 16u * j;
      if (gn >= P.N) { continue; }
      var v = acc[i * 4u + j] * P.alpha;
      if (HAS_BIAS) { v += bias[gn]; }
      let idx = cBase + gm * P.ldc + gn;
      if (ACCUM) { v += C[idx]; }
      C[idx] = v;
    }
  }
}
