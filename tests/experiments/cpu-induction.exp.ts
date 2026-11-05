// Dev experiment (not part of npm test): does the CPU fallback model form an induction head?
//   npx vitest run --config tests/experiments/vitest.config.ts
import { it } from 'vitest';
import { CpuGPT } from '../../src/cpu/gpt';
import { CpuTrainer } from '../../src/cpu/trainer';
import { CPU_SHAPE, CPU_TRAIN, withVocab } from '../../src/model/config';
import { splitCorpus } from '../../src/model/data';
import { initParams, paramLayout } from '../../src/model/layout';
import { probeBatch, scoreProbe } from '../../src/model/probe';
import { buildTokenizer } from '../../src/model/tokenizer';
import fs from 'node:fs';

it('cpu induction', () => {
  const text = fs.readFileSync('src/samples/sonnets.txt', 'utf8');
  const tok = buildTokenizer(text);
  const cfg = withVocab(CPU_SHAPE, tok.size);
  const corpus = splitCorpus(tok.encode(text), cfg.context);
  const repeat = Number(process.env.REPEAT ?? 4);
  const steps = Number(process.env.STEPS ?? 3000);
  const train = { ...CPU_TRAIN, repeatEvery: repeat, minLr: Number(process.env.MINLR ?? CPU_TRAIN.minLr), batch: Number(process.env.BATCH ?? CPU_TRAIN.batch) };
  const t = new CpuTrainer(cfg, train, corpus, initParams(paramLayout(cfg), 1), 2026, steps);
  const pb = probeBatch(cfg, 8);
  const m = new CpuGPT(cfg, t.params, 8, cfg.context);
  const t0 = Date.now();
  for (let s = 0; s <= steps; s++) {
    if (s % 100 === 0) {
      m.forward(pb.tokens, pb.targets);
      const r = scoreProbe(cfg, 8, pb.period, (l) => m.attentionProbs(l), m.losses);
      const best = r.prefix.reduce((bi, v, i, a) => (v > a[bi] ? i : bi), 0);
      const bp = r.prev.reduce((bi, v, i, a) => (v > a[bi] ? i : bi), 0);
      process.stderr.write(
        `${s} ${((Date.now() - t0) / 1000).toFixed(0)}s copy ${r.copyFirst.toFixed(2)} → ${r.copySecond.toFixed(2)}  prefix ${best} ${r.prefix[best].toFixed(2)}  prev ${bp} ${r.prev[bp].toFixed(2)}\n`,
      );
    }
    if (s < steps) t.runStep();
  }
}, 3_600_000);
