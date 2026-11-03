import { describe, expect, it } from 'vitest';
import * as op from '../src/cpu/ops';
import { Rng } from '../src/model/rng';
import { dot, gradCheck, randn } from './helpers';

// Float64 finite differences with eps = 1e-6 are accurate to ~1e-9; anything above 1e-6
// relative to the gradient's scale is a real bug, not noise.
const TOL = 1e-6;

describe('finite-difference gradient checks (float64)', () => {
  const rng = new Rng(1234);

  it('linear: y = x·W + b', () => {
    const [M, K, N] = [5, 7, 6];
    const x = randn(M * K, rng);
    const W = randn(K * N, rng);
    const b = randn(N, rng);
    const w = randn(M * N, rng);
    const y = new Float64Array(M * N);
    const loss = () => {
      op.matmul(y, x, W, M, K, N, b);
      return dot(y, w);
    };
    const dx = new Float64Array(M * K);
    const dW = new Float64Array(K * N);
    const db = new Float64Array(N);
    op.linearBackward(dx, dW, db, w, x, W, M, K, N);
    expect(gradCheck(loss, x, dx)).toBeLessThan(TOL);
    expect(gradCheck(loss, W, dW)).toBeLessThan(TOL);
    expect(gradCheck(loss, b, db)).toBeLessThan(TOL);
  });

  it('matmulNT and matmulTN agree with their definitions', () => {
    const [M, K, N] = [4, 3, 5];
    const a = randn(M * K, rng);
    const bt = randn(N * K, rng);
    const out = new Float64Array(M * N);
    op.matmulNT(out, a, bt, M, K, N);
    for (let i = 0; i < M; i++)
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += a[i * K + k] * bt[j * K + k];
        expect(out[i * N + j]).toBeCloseTo(s, 12);
      }
    const c = randn(M * N, rng);
    const tn = new Float64Array(K * N);
    op.matmulTN(tn, a, c, M, K, N);
    for (let k = 0; k < K; k++)
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let i = 0; i < M; i++) s += a[i * K + k] * c[i * N + j];
        expect(tn[k * N + j]).toBeCloseTo(s, 12);
      }
  });

  it('layer norm', () => {
    const [N, C] = [4, 9];
    const x = randn(N * C, rng, 2);
    const g = randn(C, rng);
    const b = randn(C, rng);
    const w = randn(N * C, rng);
    const y = new Float64Array(N * C);
    const mean = new Float64Array(N);
    const rstd = new Float64Array(N);
    const loss = () => {
      op.layerNorm(y, mean, rstd, x, g, b, N, C);
      return dot(y, w);
    };
    loss();
    const dx = new Float64Array(N * C);
    const dg = new Float64Array(C);
    const db = new Float64Array(C);
    op.layerNormBackward(dx, dg, db, w, x, mean, rstd, g, N, C);
    expect(gradCheck(loss, x, dx)).toBeLessThan(TOL);
    expect(gradCheck(loss, g, dg)).toBeLessThan(TOL);
    expect(gradCheck(loss, b, db)).toBeLessThan(TOL);
  });

  it('layer norm backward accumulates into dx', () => {
    const [N, C] = [2, 4];
    const x = randn(N * C, rng);
    const g = randn(C, rng);
    const b = randn(C, rng);
    const dy = randn(N * C, rng);
    const y = new Float64Array(N * C);
    const mean = new Float64Array(N);
    const rstd = new Float64Array(N);
    op.layerNorm(y, mean, rstd, x, g, b, N, C);
    const fresh = new Float64Array(N * C);
    const acc = Float64Array.from({ length: N * C }, (_, i) => i);
    op.layerNormBackward(fresh, new Float64Array(C), new Float64Array(C), dy, x, mean, rstd, g, N, C);
    op.layerNormBackward(acc, new Float64Array(C), new Float64Array(C), dy, x, mean, rstd, g, N, C);
    for (let i = 0; i < N * C; i++) expect(acc[i]).toBeCloseTo(fresh[i] + i, 12);
  });

  it('GELU', () => {
    const n = 40;
    const x = randn(n, rng, 3);
    const w = randn(n, rng);
    const y = new Float64Array(n);
    const loss = () => {
      op.gelu(y, x, n);
      return dot(y, w);
    };
    const dx = new Float64Array(n);
    op.geluBackward(dx, w, x, n);
    expect(gradCheck(loss, x, dx)).toBeLessThan(TOL);
  });

  it('causal softmax', () => {
    const [rows, T] = [12, 6];
    const x = randn(rows * T, rng, 2);
    const w = randn(rows * T, rng);
    const y = new Float64Array(rows * T);
    const loss = () => {
      op.causalSoftmax(y, x, rows, T);
      return dot(y, w);
    };
    loss();
    // Rows sum to one over the visible prefix, zero beyond it.
    for (let r = 0; r < rows; r++) {
      let s = 0;
      for (let j = 0; j < T; j++) {
        if (j > r % T) expect(y[r * T + j]).toBe(0);
        s += y[r * T + j];
      }
      expect(s).toBeCloseTo(1, 12);
    }
    const dx = new Float64Array(rows * T);
    op.causalSoftmaxBackward(dx, w, y, rows, T);
    expect(gradCheck(loss, x, dx)).toBeLessThan(TOL);
  });

  it('causal multi-head attention', () => {
    const s = { B: 2, T: 5, C: 8, H: 2 };
    const N = s.B * s.T;
    const qkv = randn(N * 3 * s.C, rng);
    const w = randn(N * s.C, rng);
    const out = new Float64Array(N * s.C);
    const probs = new Float64Array(s.B * s.H * s.T * s.T);
    const scores = new Float64Array(probs.length);
    const loss = () => {
      op.attention(out, probs, scores, qkv, s);
      return dot(out, w);
    };
    loss();
    const dqkv = new Float64Array(qkv.length);
    op.attentionBackward(dqkv, new Float64Array(probs.length), w, qkv, probs, s);
    expect(gradCheck(loss, qkv, dqkv)).toBeLessThan(TOL);
  });

  it('attention is causal: the future cannot change the past', () => {
    const s = { B: 1, T: 6, C: 4, H: 2 };
    const qkv = randn(s.T * 3 * s.C, rng);
    const out = new Float64Array(s.T * s.C);
    const probs = new Float64Array(s.H * s.T * s.T);
    const scores = new Float64Array(probs.length);
    op.attention(out, probs, scores, qkv, s);
    const before = out.slice(0, 3 * s.C);
    for (let i = 4 * 3 * s.C; i < qkv.length; i++) qkv[i] += 10; // perturb times 4 and 5
    op.attention(out, probs, scores, qkv, s);
    expect(Array.from(out.slice(0, 3 * s.C))).toEqual(Array.from(before));
  });

  it('cross-entropy', () => {
    const [N, V] = [6, 7];
    const logits = randn(N * V, rng, 2);
    const targets = Array.from({ length: N }, () => rng.int(V));
    const losses = new Float64Array(N);
    const loss = () => op.crossEntropy(losses, null, logits, targets, N, V);
    const d = new Float64Array(N * V);
    op.crossEntropy(losses, d, logits, targets, N, V);
    expect(gradCheck(loss, logits, d)).toBeLessThan(TOL);
    // Uniform logits give log V.
    const flat = new Float64Array(N * V);
    expect(op.crossEntropy(losses, null, flat, targets, N, V)).toBeCloseTo(Math.log(V), 12);
  });

  it('cross-entropy is stable for huge logits', () => {
    const logits = Float64Array.from([1000, 0, -1000]);
    const losses = new Float64Array(1);
    const d = new Float64Array(3);
    expect(op.crossEntropy(losses, d, logits, [0], 1, 3)).toBeCloseTo(0, 12);
    expect(op.crossEntropy(losses, d, logits, [1], 1, 3)).toBeCloseTo(1000, 9);
    expect(d.every(Number.isFinite)).toBe(true);
  });

  it('token and position embeddings', () => {
    const [B, T, C, V] = [2, 4, 3, 5];
    const tokens = [0, 1, 1, 4, 2, 2, 2, 3];
    const wte = randn(V * C, rng);
    const wpe = randn(T * C, rng);
    const w = randn(B * T * C, rng);
    const out = new Float64Array(B * T * C);
    const loss = () => {
      op.embed(out, tokens, wte, wpe, B, T, C);
      return dot(out, w);
    };
    const dwte = new Float64Array(V * C);
    const dwpe = new Float64Array(T * C);
    op.embedBackward(dwte, dwpe, w, tokens, B, T, C);
    expect(gradCheck(loss, wte, dwte)).toBeLessThan(TOL);
    expect(gradCheck(loss, wpe, dwpe)).toBeLessThan(TOL);
  });
});

describe('AdamW', () => {
  it('matches a hand-computed first step, with decay only before decayEnd', () => {
    const p = Float64Array.from([1, -2, 3]);
    const g = Float64Array.from([0.5, 0.5, -1]);
    const m = new Float64Array(3);
    const v = new Float64Array(3);
    const s = { lr: 0.1, beta1: 0.9, beta2: 0.99, eps: 1e-8, weightDecay: 0.5, bc1: 0.1, bc2: 0.01, clip: 1, decayEnd: 2 };
    op.adamw(p, g, m, v, s);
    // First step: m̂ = g, v̂ = g², so the update is sign(g) (up to eps).
    expect(p[0]).toBeCloseTo(1 - 0.1 * (1 + 0.5 * 1), 6);
    expect(p[1]).toBeCloseTo(-2 - 0.1 * (1 + 0.5 * -2), 6);
    expect(p[2]).toBeCloseTo(3 - 0.1 * -1, 6); // no decay past decayEnd
  });

  it('clipping scales the gradient before the moments', () => {
    expect(op.clipScale(4, 1)).toBeCloseTo(0.25, 6);
    expect(op.clipScale(0.5, 1)).toBe(1);
    const p = Float64Array.from([0]);
    const m = new Float64Array(1);
    const v = new Float64Array(1);
    op.adamw(p, Float64Array.from([8]), m, v, { lr: 1, beta1: 0.9, beta2: 0.99, eps: 0, weightDecay: 0, bc1: 1, bc2: 1, clip: 0.25, decayEnd: 0 });
    expect(m[0]).toBeCloseTo(0.1 * 2, 12);
    expect(v[0]).toBeCloseTo(0.01 * 4, 12);
  });
});
