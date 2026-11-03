/**
 * Plain TypeScript reference implementation of every operation the GPU runs.
 *
 * Each function works on Float32Array (for training and parity tests) or Float64Array (for
 * finite-difference gradient checks). Outputs are preallocated by the caller. Unless a parameter
 * is documented as accumulating, outputs are overwritten.
 *
 * Shapes are row-major. N is the number of token rows (batch × time).
 */
import { type DropoutStep, hash32, siteSeed } from '../model/dropout';

export type F = Float32Array | Float64Array;

// ---------------------------------------------------------------------------------------------
// Matrix multiplication. Three layouts cover every product in the model.

/** out[M×N] = a[M×K] · b[K×N] (+ bias[N]). */
export function matmul(out: F, a: F, b: F, M: number, K: number, N: number, bias: F | null = null): void {
  for (let i = 0; i < M; i++) {
    const o = i * N;
    if (bias) for (let j = 0; j < N; j++) out[o + j] = bias[j];
    else out.fill(0, o, o + N);
    const ar = i * K;
    for (let k = 0; k < K; k++) {
      const aik = a[ar + k];
      const br = k * N;
      for (let j = 0; j < N; j++) out[o + j] += aik * b[br + j];
    }
  }
}

/** out[M×N] = a[M×K] · bᵀ where b is stored N×K. */
export function matmulNT(out: F, a: F, b: F, M: number, K: number, N: number): void {
  for (let i = 0; i < M; i++) {
    const ar = i * K;
    for (let j = 0; j < N; j++) {
      const br = j * K;
      let s = 0;
      for (let k = 0; k < K; k++) s += a[ar + k] * b[br + k];
      out[i * N + j] = s;
    }
  }
}

/** out[K×N] = aᵀ · b where a is stored M×K and b is M×N. */
export function matmulTN(out: F, a: F, b: F, M: number, K: number, N: number): void {
  out.fill(0, 0, K * N);
  for (let i = 0; i < M; i++) {
    const ar = i * K;
    const br = i * N;
    for (let k = 0; k < K; k++) {
      const aik = a[ar + k];
      if (aik === 0) continue;
      const o = k * N;
      for (let j = 0; j < N; j++) out[o + j] += aik * b[br + j];
    }
  }
}

/** Column sums: out[N] = Σᵢ x[i, :]. */
export function colSum(out: F, x: F, M: number, N: number): void {
  out.fill(0, 0, N);
  for (let i = 0; i < M; i++) {
    const r = i * N;
    for (let j = 0; j < N; j++) out[j] += x[r + j];
  }
}

/**
 * Backward of y = x·W + b. Any of the outputs may be null.
 * dx[M×K] = dy·Wᵀ, dW[K×N] = xᵀ·dy, db[N] = Σᵢ dy.
 */
export function linearBackward(dx: F | null, dW: F | null, db: F | null, dy: F, x: F, W: F, M: number, K: number, N: number): void {
  if (dx) matmulNT(dx, dy, W, M, N, K);
  if (dW) matmulTN(dW, x, dy, M, K, N);
  if (db) colSum(db, dy, M, N);
}

// ---------------------------------------------------------------------------------------------
// Layer norm

export const LN_EPS = 1e-5;

export function layerNorm(out: F, mean: F, rstd: F, x: F, g: F, b: F, N: number, C: number): void {
  for (let n = 0; n < N; n++) {
    const r = n * C;
    let m = 0;
    for (let c = 0; c < C; c++) m += x[r + c];
    m /= C;
    let v = 0;
    for (let c = 0; c < C; c++) {
      const d = x[r + c] - m;
      v += d * d;
    }
    v /= C;
    const s = 1 / Math.sqrt(v + LN_EPS);
    for (let c = 0; c < C; c++) out[r + c] = (x[r + c] - m) * s * g[c] + b[c];
    mean[n] = m;
    rstd[n] = s;
  }
}

/** dx is ACCUMULATED (it carries the residual-stream gradient); dg and db are overwritten. */
export function layerNormBackward(dx: F, dg: F, db: F, dy: F, x: F, mean: F, rstd: F, g: F, N: number, C: number): void {
  dg.fill(0, 0, C);
  db.fill(0, 0, C);
  for (let n = 0; n < N; n++) {
    const r = n * C;
    const m = mean[n];
    const s = rstd[n];
    let sumD = 0;
    let sumDX = 0;
    for (let c = 0; c < C; c++) {
      const xhat = (x[r + c] - m) * s;
      const d = dy[r + c] * g[c];
      sumD += d;
      sumDX += d * xhat;
      dg[c] += dy[r + c] * xhat;
      db[c] += dy[r + c];
    }
    sumD /= C;
    sumDX /= C;
    for (let c = 0; c < C; c++) {
      const xhat = (x[r + c] - m) * s;
      const d = dy[r + c] * g[c];
      dx[r + c] += s * (d - sumD - xhat * sumDX);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// GELU (tanh approximation, as in GPT-2)

const GELU_K = Math.sqrt(2 / Math.PI);
const GELU_C = 0.044715;

export function gelu(out: F, x: F, n: number): void {
  for (let i = 0; i < n; i++) {
    const v = x[i];
    out[i] = 0.5 * v * (1 + Math.tanh(GELU_K * (v + GELU_C * v * v * v)));
  }
}

/** dx = gelu'(x) ⊙ dy. Safe to call with dx === dy. */
export function geluBackward(dx: F, dy: F, x: F, n: number): void {
  for (let i = 0; i < n; i++) {
    const v = x[i];
    const u = GELU_K * (v + GELU_C * v * v * v);
    const t = Math.tanh(u);
    const du = GELU_K * (1 + 3 * GELU_C * v * v);
    dx[i] = dy[i] * (0.5 * (1 + t) + 0.5 * v * (1 - t * t) * du);
  }
}

// ---------------------------------------------------------------------------------------------
// Causal softmax over rows of length T. Row r sits at time t = r mod T and may only see
// positions 0..t; later positions get probability zero.

export function causalSoftmax(out: F, x: F, rows: number, T: number): void {
  for (let r = 0; r < rows; r++) {
    const base = r * T;
    const len = (r % T) + 1;
    let mx = -Infinity;
    for (let j = 0; j < len; j++) mx = Math.max(mx, x[base + j]);
    let sum = 0;
    for (let j = 0; j < len; j++) {
      const e = Math.exp(x[base + j] - mx);
      out[base + j] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let j = 0; j < len; j++) out[base + j] *= inv;
    for (let j = len; j < T; j++) out[base + j] = 0;
  }
}

/** dx = y ⊙ (dy − Σ y·dy), restricted to the causal part of each row. */
export function causalSoftmaxBackward(dx: F, dy: F, y: F, rows: number, T: number): void {
  for (let r = 0; r < rows; r++) {
    const base = r * T;
    const len = (r % T) + 1;
    let dot = 0;
    for (let j = 0; j < len; j++) dot += y[base + j] * dy[base + j];
    for (let j = 0; j < len; j++) dx[base + j] = y[base + j] * (dy[base + j] - dot);
    for (let j = len; j < T; j++) dx[base + j] = 0;
  }
}

// ---------------------------------------------------------------------------------------------
// Causal multi-head self-attention on packed qkv rows [q | k | v], each C wide.

export interface AttentionShape {
  readonly B: number;
  readonly T: number;
  readonly C: number;
  readonly H: number;
}

/**
 * out[N×C] = softmax(q·kᵀ/√hs) · v per head. probs[B×H×T×T] is saved for backward;
 * scores is scratch of the same size.
 */
export function attention(out: F, probs: F, scores: F, qkv: F, s: AttentionShape): void {
  const { B, T, C, H } = s;
  const hs = C / H;
  const scale = 1 / Math.sqrt(hs);
  const C3 = 3 * C;
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const pBase = (b * H + h) * T * T;
      for (let t = 0; t < T; t++) {
        const q = (b * T + t) * C3 + h * hs;
        for (let j = 0; j < T; j++) {
          let dot = 0;
          if (j <= t) {
            const k = (b * T + j) * C3 + C + h * hs;
            for (let d = 0; d < hs; d++) dot += qkv[q + d] * qkv[k + d];
          }
          scores[pBase + t * T + j] = dot * scale;
        }
      }
    }
  }
  causalSoftmax(probs, scores, B * H * T, T);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const pBase = (b * H + h) * T * T;
      for (let t = 0; t < T; t++) {
        const o = (b * T + t) * C + h * hs;
        for (let d = 0; d < hs; d++) out[o + d] = 0;
        for (let j = 0; j <= t; j++) {
          const p = probs[pBase + t * T + j];
          const v = (b * T + j) * C3 + 2 * C + h * hs;
          for (let d = 0; d < hs; d++) out[o + d] += p * qkv[v + d];
        }
      }
    }
  }
}

/** dqkv is overwritten. dScores is scratch of size B×H×T×T. */
export function attentionBackward(dqkv: F, dScores: F, dout: F, qkv: F, probs: F, s: AttentionShape): void {
  const { B, T, C, H } = s;
  const hs = C / H;
  const scale = 1 / Math.sqrt(hs);
  const C3 = 3 * C;
  dqkv.fill(0, 0, B * T * C3);
  // dP[t, j] = dout_t · v_j ; dV_j += P[t, j] · dout_t
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const pBase = (b * H + h) * T * T;
      for (let t = 0; t < T; t++) {
        const o = (b * T + t) * C + h * hs;
        for (let j = 0; j < T; j++) {
          let dp = 0;
          if (j <= t) {
            const v = (b * T + j) * C3 + 2 * C + h * hs;
            const p = probs[pBase + t * T + j];
            for (let d = 0; d < hs; d++) {
              dp += dout[o + d] * qkv[v + d];
              dqkv[v + d] += p * dout[o + d];
            }
          }
          dScores[pBase + t * T + j] = dp;
        }
      }
    }
  }
  causalSoftmaxBackward(dScores, dScores, probs, B * H * T, T);
  // dQ_t = scale Σ_j dS[t, j] k_j ; dK_j = scale Σ_t dS[t, j] q_t
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const pBase = (b * H + h) * T * T;
      for (let t = 0; t < T; t++) {
        const q = (b * T + t) * C3 + h * hs;
        for (let j = 0; j <= t; j++) {
          const ds = dScores[pBase + t * T + j] * scale;
          if (ds === 0) continue;
          const k = (b * T + j) * C3 + C + h * hs;
          for (let d = 0; d < hs; d++) {
            dqkv[q + d] += ds * qkv[k + d];
            dqkv[k + d] += ds * qkv[q + d];
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Cross-entropy, fused forward and backward (as on the GPU).

/**
 * losses[n] = −log softmax(logits[n])[target[n]], returns their mean.
 * dlogits = (softmax − onehot) / N, i.e. the gradient of the mean loss.
 */
export function crossEntropy(losses: F, dlogits: F | null, logits: F, targets: ArrayLike<number>, N: number, V: number): number {
  let total = 0;
  for (let n = 0; n < N; n++) {
    const r = n * V;
    let mx = -Infinity;
    for (let v = 0; v < V; v++) mx = Math.max(mx, logits[r + v]);
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(logits[r + v] - mx);
    const t = targets[n];
    const loss = Math.log(sum) + mx - logits[r + t];
    losses[n] = loss;
    total += loss;
    if (dlogits) {
      const inv = 1 / sum;
      for (let v = 0; v < V; v++) {
        const p = Math.exp(logits[r + v] - mx) * inv;
        dlogits[r + v] = (p - (v === t ? 1 : 0)) / N;
      }
    }
  }
  return total / N;
}

// ---------------------------------------------------------------------------------------------
// Embeddings

export function embed(out: F, tokens: ArrayLike<number>, wte: F, wpe: F, B: number, T: number, C: number): void {
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const n = b * T + t;
      const e = tokens[n] * C;
      const p = t * C;
      for (let c = 0; c < C; c++) out[n * C + c] = wte[e + c] + wpe[p + c];
    }
  }
}

/** dwte is ACCUMULATED (the tied output head writes it first); dwpe is overwritten. */
export function embedBackward(dwte: F, dwpe: F, dout: F, tokens: ArrayLike<number>, B: number, T: number, C: number): void {
  dwpe.fill(0, 0, T * C);
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      const n = b * T + t;
      const e = tokens[n] * C;
      for (let c = 0; c < C; c++) {
        const g = dout[n * C + c];
        dwte[e + c] += g;
        dwpe[t * C + c] += g;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Optimiser

export function add(out: F, a: F, b: F, n: number): void {
  for (let i = 0; i < n; i++) out[i] = a[i] + b[i];
}

export function l2Norm(x: F): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s);
}

export function clipScale(norm: number, maxNorm: number): number {
  return Math.min(1, maxNorm / (norm + 1e-6));
}

export interface AdamWStep {
  readonly lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly weightDecay: number;
  /** 1 − β₁ᵗ and 1 − β₂ᵗ. */
  readonly bc1: number;
  readonly bc2: number;
  /** Gradient multiplier from global-norm clipping. */
  readonly clip: number;
  /** Elements before this index are weight-decayed. */
  readonly decayEnd: number;
}

/** Decoupled weight decay (Loshchilov & Hutter), as in PyTorch's AdamW. */
export function adamw(p: F, g: F, m: F, v: F, s: AdamWStep): void {
  const { lr, beta1, beta2, eps, weightDecay, bc1, bc2, clip, decayEnd } = s;
  for (let i = 0; i < p.length; i++) {
    const gi = g[i] * clip;
    const mi = beta1 * m[i] + (1 - beta1) * gi;
    const vi = beta2 * v[i] + (1 - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    const update = mi / bc1 / (Math.sqrt(vi / bc2) + eps);
    const decay = i < decayEnd ? weightDecay * p[i] : 0;
    p[i] -= lr * (update + decay);
  }
}

// ---------------------------------------------------------------------------------------------
// Dropout (counter-based masks, see src/model/dropout.ts). Forward and backward are the same
// operation: multiply by the mask. Safe in place.

export function dropout(out: F, x: F, n: number, d: DropoutStep, site: number): void {
  if (d.threshold === 0) {
    if (out !== x) out.set(x.subarray(0, n));
    return;
  }
  const s = siteSeed(d.seed, site);
  for (let i = 0; i < n; i++) out[i] = hash32((i ^ s) >>> 0) >= d.threshold ? x[i] * d.scale : 0;
}
