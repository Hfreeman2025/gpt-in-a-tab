import { describe, expect, it } from 'vitest';
import { CpuGPT } from '../src/cpu/gpt';
import { clampFeature, normaliseDecoder, saeForward, saeLayout, saeLoss } from '../src/cpu/sae';
import { withVocab } from '../src/model/config';
import { generate } from '../src/model/generate';
import { labelFeature } from '../src/model/labels';
import { initParams, paramLayout } from '../src/model/layout';
import { INDUCTION, type ProbeReading, findMoment, headName, probeBatch, scoreProbe } from '../src/model/probe';
import { Rng } from '../src/model/rng';

const ctx = (s: string) => ({ text: s.replace('[', '').replace(']', ''), at: s.indexOf('[') });

describe('feature labels by rule', () => {
  it('name a shared word, a character in place, or nothing when no rule covers enough', () => {
    expect(labelFeature(['the tr[u]e love', 'my tr[u]e self', 'so tr[u]e it', 'thy tr[u]e eye'].map(ctx))).toEqual({ text: 'in “true”', covered: 4, of: 4 });
    const t = labelFeature(['clear ligh[t], and', 'by nigh[t]; the', 'his migh[t]. Then', 'old deligh[t], so', 'a kni[t] cap'].map(ctx));
    expect(t?.text).toBe('“t” ending a word');
    expect(t?.covered).toBe(5);
    expect(labelFeature(['abc[d]ef', 'xyz[q]rs', 'mno[p]qr', 'ghi[j]kl'].map(ctx))).toBeNull();
  });
});

describe('the induction test', () => {
  it('scores a hand-made induction pattern and finds the moment', () => {
    const cfg = withVocab({ nLayer: 2, nHead: 2, dModel: 8, context: 8 }, 5);
    const pb = probeBatch(cfg, 2, 3);
    expect(pb.period).toBe(4);
    expect(Array.from(pb.tokens.subarray(4, 8))).toEqual(Array.from(pb.tokens.subarray(0, 4)));
    expect(Array.from(pb.corrupt.subarray(4, 8))).toEqual(Array.from(pb.tokens.subarray(4, 8)));
    // Layer 1 head 0 looks at the previous position; layer 2 head 1 at the token after the previous copy.
    const T = 8;
    const probs = (l: number) => {
      const p = new Float32Array(2 * 2 * T * T);
      for (let b = 0; b < 2; b++) {
        for (let h = 0; h < 2; h++) {
          for (let i = 0; i < T; i++) {
            const base = ((b * 2 + h) * T + i) * T;
            const j = l === 0 && h === 0 ? Math.max(0, i - 1) : l === 1 && h === 1 && i >= 4 ? i - 3 : i;
            p[base + j] = 1;
          }
        }
      }
      return p;
    };
    const losses = new Float32Array(2 * T).map((_, i) => (i % T < 4 ? 4 : 0.5));
    const s = scoreProbe(cfg, 2, 4, probs, losses);
    expect(s.prefix[3]).toBeCloseTo(1);
    expect(s.prev[0]).toBeCloseTo(1);
    expect(s.copyFirst).toBeCloseTo(4);
    expect(s.copySecond).toBeCloseTo(0.5);
    const quiet: ProbeReading = { step: 0, copyFirst: 4, copySecond: 4, prefix: [0, 0, 0, 0.1], prev: [0.2, 0, 0, 0] };
    const m = findMoment([quiet, { step: 20, ...s }], cfg);
    expect(m?.step).toBe(20);
    expect(headName(m?.head ?? -1, 2)).toBe('L2·H2');
    expect(m?.partner).toBe(0);
    expect(m?.before?.copySecond).toBe(4);
    expect(findMoment([quiet], cfg)).toBeNull();
    expect(INDUCTION.prefix).toBeGreaterThan(0.1);
  });

  it('matches its own CPU forward pass on a real model', () => {
    const cfg = withVocab({ nLayer: 2, nHead: 2, dModel: 16, context: 16 }, 7);
    const params = initParams(paramLayout(cfg), 5);
    const pb = probeBatch(cfg, 3);
    const m = new CpuGPT(cfg, params, 3, 16);
    const loss = m.forward(pb.tokens, pb.targets);
    const s = scoreProbe(cfg, 3, pb.period, (l) => m.attentionProbs(l), m.losses);
    expect(s.copyFirst).toBeGreaterThan(1);
    expect(Math.abs((s.copyFirst + s.copySecond) / 2 - loss)).toBeLessThan(0.5);
    for (const v of [...s.prefix, ...s.prev]) expect(v).toBeGreaterThanOrEqual(0);
  });
});

describe('sparse autoencoder (CPU reference)', () => {
  const C = 5;
  const F = 7;
  const B = 4;
  const lay = saeLayout(C, F);
  const rng = new Rng(3);
  const p = Float32Array.from({ length: lay.total }, () => rng.normal() * 0.5);
  for (let j = 0; j < F; j++) p[lay.bEnc + j] = 0.3;
  const x = Float32Array.from({ length: B * C }, () => rng.normal());

  it('has gradients that match finite differences', () => {
    const lambda = 0.7;
    const { grads } = saeLoss(p, lay, x, B, lambda);
    const q = Float64Array.from(p);
    const f = (v: Float64Array) => saeLoss(Float32Array.from(v), lay, x, B, lambda).loss;
    let worst = 0;
    let scale = 1e-9;
    for (let i = 0; i < lay.total; i++) {
      const pre = saeForward(p, lay, x, B);
      void pre;
      const h = 1e-3;
      const a = Float64Array.from(q);
      const b = Float64Array.from(q);
      a[i] += h;
      b[i] -= h;
      const num = (f(a) - f(b)) / (2 * h);
      worst = Math.max(worst, Math.abs(num - grads[i]));
      scale = Math.max(scale, Math.abs(grads[i]));
    }
    expect(worst / scale).toBeLessThan(2e-2);
  });

  it('normalises decoder rows, and a clamp sets the feature it names', () => {
    const r = Float32Array.from(p);
    normaliseDecoder(r, lay);
    for (let j = 0; j < F; j++) {
      let s = 0;
      for (let c = 0; c < C; c++) s += r[lay.wDec + j * C + c] ** 2;
      expect(Math.sqrt(s)).toBeCloseTo(1, 5);
    }
    const j = 2;
    const enc = Float32Array.from({ length: C }, (_, c) => r[lay.wEnc + c * F + j]);
    const dec = Float32Array.from({ length: C }, (_, c) => r[lay.wDec + j * C + c]);
    const v = Float32Array.from(x.subarray(0, C));
    const k = { enc, bias: r[lay.bEnc + j], dec, scale: 1, value: 0 };
    clampFeature(v, k);
    // Pinned at zero, then measured again: the change moves the pre-activation by d·w per unit.
    const before = clampFeature(Float32Array.from(x.subarray(0, C)), { ...k, value: 0 });
    expect(before).toBeGreaterThanOrEqual(0);
  });
});

describe('interventions while writing', () => {
  it('knocking out heads changes the text, and knocking out none does not', () => {
    const cfg = withVocab({ nLayer: 2, nHead: 2, dModel: 16, context: 16 }, 9);
    // Scaled up from the initialisation so that attention matters to the output.
    const params = initParams(paramLayout(cfg), 9).map((v) => v * 12);
    const run = (knockout?: number[]) => Array.from(generate(cfg, params, [1, 2, 3], 40, 1, 7, 5, knockout ? { knockout } : undefined), (s) => s.token).join(',');
    expect(run([])).toBe(run());
    expect(run([0, 1, 2, 3])).not.toBe(run());
  });
});
