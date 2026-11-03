import { describe, expect, it } from 'vitest';
import { withVocab } from '../src/model/config';
import { REPEAT_FLAG, REPEAT_PERIOD, gatherBatch, repeatToken, repeatWord, sampleOffsets, splitCorpus } from '../src/model/data';
import { ALIGN, initParams, paramLayout } from '../src/model/layout';
import { Rng } from '../src/model/rng';
import { sampleToken } from '../src/model/sample';
import { dataSteps, dropoutFor, learningRate, planSteps } from '../src/model/schedule';
import { buildTokenizer } from '../src/model/tokenizer';

describe('parameter layout', () => {
  const cfg = withVocab({ nLayer: 4, nHead: 4, dModel: 128, context: 128 }, 65);
  const layout = paramLayout(cfg);

  it('counts parameters like GPT-2 with a tied head', () => {
    const C = 128;
    const perLayer = 12 * C * C + 13 * C;
    expect(layout.count).toBe(65 * C + 128 * C + 4 * perLayer + 2 * C);
  });

  it('aligns every tensor to 256 bytes and decays matrices only', () => {
    for (const t of layout.tensors) {
      expect(t.offset % ALIGN).toBe(0);
      expect(t.decay).toBe(t.shape.length === 2);
      expect(t.decay).toBe(t.offset < layout.decayEnd);
    }
  });

  it('initialises deterministically', () => {
    const a = initParams(layout, 42);
    const b = initParams(layout, 42);
    expect(a).toEqual(b);
    const g = layout.get('h0.ln1.g');
    expect(Array.from(a.subarray(g.offset, g.offset + g.size)).every((x) => x === 1)).toBe(true);
  });
});

describe('tokenizer', () => {
  it('round-trips text, including astral code points', () => {
    const text = 'Hello, 🌍! naïve café\n';
    const tok = buildTokenizer(text);
    expect(tok.decode(tok.encode(text))).toBe(text);
    expect(tok.size).toBe(new Set(Array.from(text)).size);
  });

  it('caps the vocabulary and maps rare characters to an unknown token', () => {
    const chars = Array.from({ length: 300 }, (_, i) => String.fromCodePoint(0x4e00 + i));
    const text = chars.join('') + 'aaaa';
    const tok = buildTokenizer(text, 64);
    expect(tok.size).toBe(64);
    expect(tok.unknown).not.toBeNull();
    expect(tok.encode(text).length).toBe(304);
  });
});

describe('schedule', () => {
  const cfg = { peakLr: 1, minLr: 0.1, warmupSteps: 10 };
  it('warms up linearly, peaks, then decays to the minimum', () => {
    expect(learningRate(0, 100, cfg)).toBeCloseTo(0.1, 12);
    expect(learningRate(9, 100, cfg)).toBeCloseTo(1, 12);
    expect(learningRate(10, 100, cfg)).toBeCloseTo(1, 12);
    expect(learningRate(55, 100, cfg)).toBeCloseTo(0.55, 12);
    expect(learningRate(100, 100, cfg)).toBeCloseTo(0.1, 12);
    expect(learningRate(500, 100, cfg)).toBeCloseTo(0.1, 12);
  });

  it('plans a step budget from measured speed, within bounds', () => {
    const c = { targetSeconds: 90, minSteps: 600, maxSteps: 12000 };
    expect(planSteps(40, c)).toBe(3600);
    expect(planSteps(1, c)).toBe(600);
    expect(planSteps(1e6, c)).toBe(12000);
    expect(planSteps(NaN, c)).toBe(600);
  });
});

describe('batches', () => {
  it('hold out the tail only when there is room, and windows stay inside their split', () => {
    const T = 8;
    expect(splitCorpus(new Uint32Array(30), T).val).toBeNull();
    const corpus = splitCorpus(Uint32Array.from({ length: 1000 }, (_, i) => i), T);
    expect(corpus.val).toEqual({ start: 900, length: 100 });
    const rng = new Rng(1);
    for (let i = 0; i < 200; i++) {
      for (const o of sampleOffsets(rng, corpus.train, 4, T)) expect(o + T + 1).toBeLessThanOrEqual(900);
      for (const o of sampleOffsets(rng, corpus.val!, 4, T)) {
        expect(o).toBeGreaterThanOrEqual(900);
        expect(o + T + 1).toBeLessThanOrEqual(1000);
      }
    }
    const { inputs, targets } = gatherBatch(corpus.tokens, Uint32Array.from([3, 50]), T, 1000);
    expect(Array.from(inputs.subarray(0, 3))).toEqual([3, 4, 5]);
    expect(Array.from(targets.subarray(T, T + 3))).toEqual([51, 52, 53]);
  });

  it('flag every k-th window as a repeating run of random characters, its period varying', () => {
    const T = 128;
    const corpus = splitCorpus(Uint32Array.from({ length: 5000 }, (_, i) => i % 300), T);
    const offsets = sampleOffsets(new Rng(2), corpus.train, 8, T, 4);
    expect(Array.from(offsets, (o) => (o & REPEAT_FLAG ? 1 : 0))).toEqual([0, 0, 0, 1, 0, 0, 0, 1]);
    const periods = new Set<number>();
    for (let i = 0; i < 50; i++) {
      for (const o of sampleOffsets(new Rng(i), corpus.train, 4, T, 2)) {
        if (!(o & REPEAT_FLAG)) continue;
        const p = (o >>> 24) & 0x7f;
        expect(p).toBeGreaterThanOrEqual(REPEAT_PERIOD.min);
        expect(p).toBeLessThanOrEqual(REPEAT_PERIOD.max);
        periods.add(p);
      }
    }
    expect(periods.size).toBeGreaterThan(10);
    const V = 50;
    const P = 5;
    const { inputs, targets } = gatherBatch(corpus.tokens, Uint32Array.from([10, repeatWord(12345, P)]), 8, V);
    expect(Array.from(inputs.subarray(0, 8))).toEqual([10, 11, 12, 13, 14, 15, 16, 17]);
    const run = [0, 1, 2, 3, 4].map((k) => repeatToken(12345, k, V));
    expect(new Set(run).size).toBeGreaterThan(1);
    expect(Array.from(inputs.subarray(8, 16))).toEqual([...run, run[0], run[1], run[2]]);
    expect(Array.from(targets.subarray(8, 16))).toEqual([run[1], run[2], run[3], run[4], run[0], run[1], run[2], run[3]]);
  });
});

describe('sampling', () => {
  it('is greedy at temperature 0 and reports calibrated probabilities', () => {
    const logits = Float32Array.from([0, 2, 1]);
    const s = sampleToken(logits, 0, new Rng(1), new Float32Array(3));
    expect(s.token).toBe(1);
    const e = [1, Math.exp(2), Math.exp(1)];
    const z = e[0] + e[1] + e[2];
    expect(s.p).toBeCloseTo(e[1] / z, 6);
    expect(s.top.map((t) => t[0])).toEqual([1, 2, 0]);
  });

  it('draws in proportion to probability', () => {
    const logits = Float32Array.from([Math.log(1), Math.log(3)]);
    const rng = new Rng(3);
    const scratch = new Float32Array(2);
    let ones = 0;
    for (let i = 0; i < 20000; i++) ones += sampleToken(logits, 1, rng, scratch).token;
    expect(ones / 20000).toBeGreaterThan(0.73);
    expect(ones / 20000).toBeLessThan(0.77);
  });
});

describe('run planning', () => {
  const c = { minSteps: 600, maxSteps: 12000 };
  it('caps small texts by passes over the data and big ones by the step ceiling', () => {
    expect(dataSteps(85_000, 4096, c)).toBe(1650); // the sonnets: 80 passes
    expect(dataSteps(2_000, 4096, c)).toBe(600); // a paragraph: the floor
    expect(dataSteps(50_000_000, 4096, c)).toBe(12000);
  });
  it('regularises small texts harder', () => {
    expect(dropoutFor(95_000)).toBe(0.2);
    expect(dropoutFor(2_000_000)).toBe(0.1);
  });
});

describe('tokenizer cap edge case', () => {
  it('never reuses a kept character as the unknown token, even if both usual glyphs are kept', () => {
    const common = '·�'.repeat(50) + 'abc'.repeat(50);
    const rare = Array.from({ length: 80 }, (_, i) => String.fromCodePoint(0x4e00 + i)).join('');
    const tok = buildTokenizer(common + rare, 16);
    expect(tok.size).toBe(16);
    expect(new Set(tok.chars).size).toBe(16);
    expect(tok.decode(tok.encode('·�abc'))).toBe('·�abc');
  });
});
