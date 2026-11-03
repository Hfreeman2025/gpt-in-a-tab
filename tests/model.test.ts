import { describe, expect, it } from 'vitest';
import { CpuGPT, Decoder } from '../src/cpu/gpt';
import { CpuTrainer } from '../src/cpu/trainer';
import { type TrainConfig, withVocab } from '../src/model/config';
import { splitCorpus } from '../src/model/data';
import { NO_DROPOUT, dropoutStep, maskValue } from '../src/model/dropout';
import { generate } from '../src/model/generate';
import { initParams, paramLayout } from '../src/model/layout';
import { Rng } from '../src/model/rng';
import { buildTokenizer } from '../src/model/tokenizer';
import { gradCheck } from './helpers';

const tiny = withVocab({ nLayer: 2, nHead: 2, dModel: 8, context: 6 }, 7);

describe('whole-model gradient check (float64)', () => {
  for (const [label, drop] of [
    ['without dropout', NO_DROPOUT],
    ['with dropout (fixed masks)', dropoutStep(0.3, 1234)],
  ] as const)
  it(`backward matches finite differences for every tensor, ${label}`, () => {
    const layout = paramLayout(tiny);
    const params = Float64Array.from(initParams(layout, 3));
    // Larger weights than the 0.02 init so that every path carries a measurable gradient.
    const rng = new Rng(9);
    for (let i = 0; i < params.length; i++) if (params[i] !== 0 && params[i] !== 1) params[i] = rng.normal() * 0.4;
    for (let i = 0; i < params.length; i++) if (params[i] === 0) params[i] = rng.normal() * 0.1;
    const B = 2;
    const T = tiny.context;
    const model = new CpuGPT(tiny, params, B, T, Float64Array);
    const tokens = Array.from({ length: B * T }, () => rng.int(tiny.vocabSize));
    const targets = Array.from({ length: B * T }, () => rng.int(tiny.vocabSize));
    model.forward(tokens, targets, drop);
    model.backward();
    const grads = Float64Array.from(model.grads);
    for (const t of layout.tensors) {
      const x = params.subarray(t.offset, t.offset + t.size);
      const g = grads.subarray(t.offset, t.offset + t.size);
      const err = gradCheck(() => model.forward(tokens, targets, drop), x, g, { maxChecks: 24, rng });
      expect(err, t.name).toBeLessThan(1e-6);
    }
  });
});

describe('dropout masks', () => {
  it('drop about p of the elements and rescale the rest', () => {
    const d = dropoutStep(0.2, 99);
    let kept = 0;
    let sum = 0;
    const n = 200000;
    for (let i = 0; i < n; i++) {
      const m = maskValue(d, 3, i);
      if (m !== 0) {
        kept++;
        expect(m).toBeCloseTo(1.25, 12);
      }
      sum += m;
    }
    expect(kept / n).toBeGreaterThan(0.795);
    expect(kept / n).toBeLessThan(0.805);
    expect(sum / n).toBeGreaterThan(0.99); // unbiased in expectation
    expect(sum / n).toBeLessThan(1.01);
  });

  it('differ between sites and steps, and vanish at p = 0', () => {
    const d = dropoutStep(0.5, 7);
    const a = Array.from({ length: 64 }, (_, i) => maskValue(d, 1, i));
    const b = Array.from({ length: 64 }, (_, i) => maskValue(d, 2, i));
    const c = Array.from({ length: 64 }, (_, i) => maskValue(dropoutStep(0.5, 8), 1, i));
    expect(a).not.toEqual(b);
    expect(a).not.toEqual(c);
    expect(maskValue(NO_DROPOUT, 1, 5)).toBe(1);
  });
});

describe('decoder', () => {
  it('KV-cache decoding gives the same logits as a full forward pass', () => {
    const cfg = withVocab({ nLayer: 2, nHead: 2, dModel: 16, context: 10 }, 11);
    const params = initParams(paramLayout(cfg), 5);
    const rng = new Rng(4);
    for (let i = 0; i < params.length; i++) params[i] += rng.normal() * 0.2;
    const seq = Array.from({ length: cfg.context }, () => rng.int(cfg.vocabSize));
    const full = new CpuGPT(cfg, params, 1, cfg.context);
    full.forward(seq);
    const dec = new Decoder(cfg, params);
    for (let t = 0; t < seq.length; t++) {
      const logits = dec.push(seq[t]);
      for (let v = 0; v < cfg.vocabSize; v++) expect(logits[v]).toBeCloseTo(full.logits[t * cfg.vocabSize + v], 4);
    }
    expect(dec.full).toBe(true);
    expect(() => dec.push(0)).toThrow();
  });

  it('generates exactly the requested length, sliding past the context window, deterministically', () => {
    const cfg = withVocab({ nLayer: 1, nHead: 2, dModel: 8, context: 6 }, 5);
    const params = initParams(paramLayout(cfg), 3);
    const run = () => Array.from(generate(cfg, params, [1, 2], 40, 0.9, 11), (s) => s.token);
    const a = run();
    expect(a).toHaveLength(40);
    expect(a).toEqual(run());
    expect(a.every((t) => t >= 0 && t < cfg.vocabSize)).toBe(true);
    expect(() => generate(cfg, params, [], 3, 1, 1).next()).toThrow();
  });

  it('after the window slides, a token depends only on the recent half-context', () => {
    // Greedy decoding of a model that has memorised a cycle keeps producing the cycle across slides.
    const text = 'abcdefgh'.repeat(20);
    const tok = buildTokenizer(text);
    const cfg = withVocab({ nLayer: 1, nHead: 1, dModel: 16, context: 8 }, tok.size);
    const corpus = splitCorpus(tok.encode(text), cfg.context);
    const t = new CpuTrainer(cfg, { ...train, peakLr: 2e-2 }, corpus, initParams(paramLayout(cfg), 1), 1, 300);
    for (let i = 0; i < 300; i++) t.runStep();
    const out = tok.decode(Array.from(generate(cfg, t.params, Array.from(tok.encode('abc')), 30, 0, 1), (s) => s.token));
    expect(out).toBe('defghabcdefghabcdefghabcdefgha');
  });
});

const train: TrainConfig = {
    batch: 4,
    peakLr: 1e-2,
    minLr: 1e-3,
    warmupSteps: 10,
    weightDecay: 0,
    beta1: 0.9,
    beta2: 0.99,
    eps: 1e-8,
    clipNorm: 1,
    dropout: 0,
    repeatEvery: 0,
    targetSeconds: 1,
    minSteps: 1,
    maxSteps: 1000,
  };

describe('training', () => {
  it('overfits a tiny text: loss is driven near zero', () => {
    const text = 'the quick brown fox jumps over the lazy dog. ';
    const tok = buildTokenizer(text);
    const cfg = withVocab({ nLayer: 2, nHead: 2, dModel: 32, context: 16 }, tok.size);
    const corpus = splitCorpus(tok.encode(text.repeat(3)), cfg.context);
    const trainer = new CpuTrainer(cfg, train, corpus, initParams(paramLayout(cfg), 1), 1, 400);
    const first = trainer.runStep().loss;
    expect(first).toBeGreaterThan(Math.log(tok.size) - 0.3);
    let last = first;
    for (let i = 1; i < 400; i++) last = trainer.runStep().loss;
    expect(last).toBeLessThan(0.05);

    // And the model has learned the text: greedy decoding reproduces it.
    const dec = new Decoder(cfg, trainer.params);
    const prompt = tok.encode('the quick');
    let logits: Float32Array = new Float32Array(0);
    for (const t of prompt) logits = dec.push(t);
    let out = '';
    for (let i = 0; i < 8; i++) {
      let best = 0;
      for (let v = 1; v < logits.length; v++) if (logits[v] > logits[best]) best = v;
      out += tok.chars[best];
      if (!dec.full) logits = dec.push(best);
    }
    expect('the quick' + out).toBe('the quick brown f');
  });

  it('keeps training at the final learning rate: extending the run does not restart the schedule', () => {
    const text = 'abcabcabdabcabcabd'.repeat(4);
    const tok = buildTokenizer(text);
    const cfg = withVocab({ nLayer: 1, nHead: 1, dModel: 8, context: 8 }, tok.size);
    const t = new CpuTrainer(cfg, train, splitCorpus(tok.encode(text), cfg.context), initParams(paramLayout(cfg), 2), 2, 30);
    let last = 0;
    for (let i = 0; i < 30; i++) last = t.runStep().lr;
    expect(last).toBeLessThan(train.minLr * 1.1); // the cosine reaches the floor at the planned length
    t.totalSteps += 20; // what "keep training" does
    for (let i = 0; i < 20; i++) expect(t.runStep().lr).toBeCloseTo(train.minLr, 12);
  });

  it('is deterministic for a seed', () => {
    const text = 'abcabcabdabcabcabd'.repeat(4);
    const tok = buildTokenizer(text);
    const cfg = withVocab({ nLayer: 1, nHead: 1, dModel: 8, context: 8 }, tok.size);
    const run = () => {
      const corpus = splitCorpus(tok.encode(text), cfg.context);
      const t = new CpuTrainer(cfg, train, corpus, initParams(paramLayout(cfg), 2), 2, 20);
      for (let i = 0; i < 5; i++) t.runStep();
      return t.runStep().loss;
    };
    expect(run()).toBe(run());
  });
});
