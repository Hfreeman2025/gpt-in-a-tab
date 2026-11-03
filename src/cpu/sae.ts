/**
 * The sparse autoencoder of Bricken et al. (2023) and Cunningham et al. (2023), as a plain
 * TypeScript reference for the WGSL version (src/gpu/sae.ts) and for clamping features while
 * sampling. On a batch of residual-stream vectors x (already scaled by the run's constant):
 *
 *   f = ReLU(x · W_enc + b_enc)        F features, mostly zero
 *   x̂ = f · W_dec + b_dec              each decoder row a unit-length direction in the residual
 *   loss = mean ‖x̂ − x‖² + λ · mean Σ f
 *
 * Parameters live in one flat buffer: W_enc [C×F], b_enc [F], W_dec [F×C], b_dec [C].
 */
export interface SaeLayout {
  readonly C: number;
  readonly F: number;
  readonly wEnc: number;
  readonly bEnc: number;
  readonly wDec: number;
  readonly bDec: number;
  readonly total: number;
}

export function saeLayout(C: number, F: number): SaeLayout {
  const wEnc = 0;
  const bEnc = C * F;
  const wDec = bEnc + F;
  const bDec = wDec + F * C;
  return { C, F, wEnc, bEnc, wDec, bDec, total: bDec + C };
}

export interface SaeForward {
  readonly f: Float32Array;
  readonly xhat: Float32Array;
}

export function saeForward(p: Float32Array, lay: SaeLayout, x: Float32Array, B: number): SaeForward {
  const { C, F } = lay;
  const f = new Float32Array(B * F);
  const xhat = new Float32Array(B * C);
  for (let b = 0; b < B; b++) {
    for (let j = 0; j < F; j++) {
      let s = p[lay.bEnc + j];
      for (let c = 0; c < C; c++) s += x[b * C + c] * p[lay.wEnc + c * F + j];
      f[b * F + j] = Math.max(s, 0);
    }
    for (let c = 0; c < C; c++) {
      let s = p[lay.bDec + c];
      for (let j = 0; j < F; j++) s += f[b * F + j] * p[lay.wDec + j * C + c];
      xhat[b * C + c] = s;
    }
  }
  return { f, xhat };
}

export interface SaeLoss extends SaeForward {
  readonly loss: number;
  readonly mse: number;
  readonly l1: number;
  readonly grads: Float32Array;
}

/** Loss and gradients, in float64 sums. */
export function saeLoss(p: Float32Array, lay: SaeLayout, x: Float32Array, B: number, lambda: number): SaeLoss {
  const { C, F } = lay;
  const { f, xhat } = saeForward(p, lay, x, B);
  const g = new Float32Array(lay.total);
  let mse = 0;
  let l1 = 0;
  const dxh = new Float64Array(B * C);
  for (let i = 0; i < B * C; i++) {
    const d = xhat[i] - x[i];
    mse += d * d;
    dxh[i] = (2 * d) / B;
  }
  for (let i = 0; i < B * F; i++) l1 += f[i];
  mse /= B;
  l1 /= B;
  for (let b = 0; b < B; b++) {
    for (let c = 0; c < C; c++) g[lay.bDec + c] += dxh[b * C + c];
    for (let j = 0; j < F; j++) {
      const fj = f[b * F + j];
      let df = 0;
      for (let c = 0; c < C; c++) {
        g[lay.wDec + j * C + c] += fj * dxh[b * C + c];
        df += dxh[b * C + c] * p[lay.wDec + j * C + c];
      }
      if (fj <= 0) continue;
      const dpre = df + lambda / B;
      g[lay.bEnc + j] += dpre;
      for (let c = 0; c < C; c++) g[lay.wEnc + c * F + j] += x[b * C + c] * dpre;
    }
  }
  return { f, xhat, loss: mse + lambda * l1, mse, l1, grads: g };
}

/** Unit-length decoder rows. */
export function normaliseDecoder(p: Float32Array, lay: SaeLayout): void {
  const { C, F } = lay;
  for (let j = 0; j < F; j++) {
    let s = 0;
    for (let c = 0; c < C; c++) s += p[lay.wDec + j * C + c] ** 2;
    const k = 1 / Math.max(Math.sqrt(s), 1e-8);
    for (let c = 0; c < C; c++) p[lay.wDec + j * C + c] *= k;
  }
}

/**
 * One feature, pinned: the residual vector x (unscaled) with feature j's activation replaced by
 * `value`, everything the autoencoder does not explain left as it was:
 *   x' = x + (value − f_j(x)) · d_j / scale
 * Returns f_j(x) before the change.
 */
export interface FeatureClamp {
  /** W_enc column j and b_enc[j], W_dec row j: the only parts of the autoencoder needed. */
  readonly enc: Float32Array;
  readonly bias: number;
  readonly dec: Float32Array;
  readonly scale: number;
  readonly value: number;
}

export function clampFeature(x: Float32Array, k: FeatureClamp): number {
  let s = k.bias;
  for (let c = 0; c < x.length; c++) s += x[c] * k.scale * k.enc[c];
  const f = Math.max(s, 0);
  const delta = (k.value - f) / k.scale;
  for (let c = 0; c < x.length; c++) x[c] += delta * k.dec[c];
  return f;
}
