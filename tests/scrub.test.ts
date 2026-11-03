import { describe, expect, it } from 'vitest';
import { CpuGPT } from '../src/cpu/gpt';
import { withVocab } from '../src/model/config';
import { attentionWhenWriting, generate } from '../src/model/generate';
import { fromHalf, fromHalfBits, toHalf, toHalfBits } from '../src/model/half';
import { initParams, paramLayout } from '../src/model/layout';
import { Rng } from '../src/model/rng';
import { isCheckpoint } from '../src/model/schedule';

describe('half precision', () => {
  it('round-trips exactly representable values and rounds the rest to nearest', () => {
    for (const x of [0, -0, 1, -2, 0.5, 65504, 2 ** -14, 2 ** -24, 1024.5]) expect(fromHalfBits(toHalfBits(x))).toBe(x === 1024.5 ? 1024 : x);
    expect(fromHalfBits(toHalfBits(1e6))).toBe(Infinity);
    expect(fromHalfBits(toHalfBits(1e-9))).toBe(0);
    expect(Number.isNaN(fromHalfBits(toHalfBits(NaN)))).toBe(true);
    expect(toHalfBits(1 + 2 ** -11)).toBe(toHalfBits(1)); // a tie rounds to even
    expect(toHalfBits(1 + 3 * 2 ** -11)).toBe(toHalfBits(1 + 2 ** -9));
  });

  it('keeps weights to within half-precision rounding', () => {
    const rng = new Rng(1);
    const x = Float32Array.from({ length: 20000 }, () => rng.normal() * 0.05);
    const y = fromHalf(toHalf(x));
    for (let i = 0; i < x.length; i++) expect(Math.abs(y[i] - x[i])).toBeLessThanOrEqual(Math.max(Math.abs(x[i]) * 2 ** -11, 2 ** -25));
  });
});

describe('checkpoint schedule', () => {
  it('is dense early and stays under about seventy for long runs', () => {
    const at = (total: number) => Array.from({ length: total + 1 }, (_, s) => s).filter((s) => isCheckpoint(s, total));
    expect(at(1700).slice(0, 12)).toEqual([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125]);
    expect(at(1700).length).toBeLessThan(50);
    expect(at(12000).length).toBeLessThan(75);
  });
});

describe('attention when writing', () => {
  const cfg = withVocab({ nLayer: 2, nHead: 2, dModel: 16, context: 8 }, 9);
  const params = initParams(paramLayout(cfg), 4);
  const rng = new Rng(8);
  for (let i = 0; i < params.length; i++) params[i] += rng.normal() * 0.3;

  it('replays the exact window the sampler used, across slides of the context', () => {
    const prompt = [1, 2, 3];
    const sample = Array.from(generate(cfg, params, prompt, 30, 0, 1), (s) => s.token); // greedy
    for (let m = 0; m < sample.length; m++) {
      const a = attentionWhenWriting(cfg, params, prompt, sample, m);
      let best = 0;
      for (let v = 1; v < a.logits.length; v++) if (a.logits[v] > a.logits[best]) best = v;
      expect(best, `character ${m}`).toBe(sample[m]);
      expect(a.start + a.length).toBe(prompt.length + m); // the query is the character before m
      for (let h = 0; h < cfg.nLayer * cfg.nHead; h++) {
        let sum = 0;
        for (let j = 0; j < a.length; j++) sum += a.weights[h * a.length + j];
        expect(sum).toBeCloseTo(1, 5);
      }
    }
  });

  it('captures the same attention as a full forward pass over that window', () => {
    const seq = [4, 1, 7, 2, 2, 5];
    const a = attentionWhenWriting(cfg, params, seq.slice(0, 1), seq.slice(1), seq.length - 2);
    const full = new CpuGPT(cfg, params, 1, a.length);
    full.forward(seq.slice(a.start, a.start + a.length));
    const t = a.length - 1;
    for (let l = 0; l < cfg.nLayer; l++)
      for (let h = 0; h < cfg.nHead; h++)
        for (let j = 0; j < a.length; j++)
          expect(a.weights[(l * cfg.nHead + h) * a.length + j]).toBeCloseTo(full.attentionProbs(l)[(h * a.length + t) * a.length + j], 5);
  });
});
