/**
 * Dev experiment: train the shipped run, then the sparse autoencoder on its residual stream, and
 * log reconstruction, sparsity, dead features and a few features' top contexts.
 *   ?lambda=3&steps=3000&features=1024&site=3
 */
import { GPU_SHAPE, GPU_TRAIN, withVocab } from '../../src/model/config';
import { splitCorpus } from '../../src/model/data';
import { initParams, paramLayout } from '../../src/model/layout';
import { dataSteps, dropoutFor, textTokensPerStep } from '../../src/model/schedule';
import { buildTokenizer } from '../../src/model/tokenizer';
import { Gpu, acquireDevice } from '../../src/gpu/context';
import { PIPELINE_VARIANTS } from '../../src/gpu/model';
import { GpuSae, saeDefaults } from '../../src/gpu/sae';
import { GpuTrainer } from '../../src/gpu/trainer';
import sonnets from '../../src/samples/sonnets.txt?raw';

declare global {
  interface Window {
    __exp?: unknown;
  }
}
const out = document.getElementById('out') as HTMLElement;
const log = (s: string) => {
  out.textContent += s + '\n';
  console.log(s);
};

async function main() {
  const q = new URLSearchParams(location.search);
  const text = sonnets;
  const tok = buildTokenizer(text);
  const cfg = withVocab(GPU_SHAPE, tok.size);
  const corpus = splitCorpus(tok.encode(text), cfg.context);
  const train = { ...GPU_TRAIN, dropout: dropoutFor(text.length) };
  const steps = Number(q.get('train') ?? dataSteps(corpus.train.length, textTokensPerStep(train, cfg.context), train));
  const acq = await acquireDevice();
  if ('error' in acq) throw new Error(acq.error);
  const g = new Gpu(acq.device);
  await g.prepare(PIPELINE_VARIANTS);
  const trainer = new GpuTrainer(g, cfg, train, corpus, initParams(paramLayout(cfg), 1), 2026, steps);
  const t0 = performance.now();
  const inflight: Promise<undefined>[] = [];
  for (let s = 0; s < steps; s++) {
    trainer.submitStep();
    inflight.push(acq.device.queue.onSubmittedWorkDone());
    if (inflight.length >= 3) await inflight.shift();
    if (s % 500 === 499) await trainer.readStats();
  }
  await Promise.all(inflight);
  log(`trained ${steps} steps in ${((performance.now() - t0) / 1000).toFixed(1)} s`);

  const d = saeDefaults(cfg.nLayer);
  const opts = {
    ...d,
    lambda: Number(q.get('lambda') ?? d.lambda),
    steps: Number(q.get('steps') ?? d.steps),
    features: Number(q.get('features') ?? d.features),
    site: Number(q.get('site') ?? d.site),
    lr: Number(q.get('lr') ?? d.lr),
  };
  const sae = new GpuSae(trainer.model, opts);
  const t1 = performance.now();
  const rows = await sae.collect(corpus);
  log(`collected ${rows} positions at site ${opts.site} in ${((performance.now() - t1) / 1000).toFixed(2)} s, scale ${sae.scale.toFixed(4)}`);
  const t2 = performance.now();
  for (let s = 0; s < opts.steps; s++) {
    const measure = s % 250 === 249;
    sae.submitStep(measure);
    if (measure) {
      const r = await sae.readStep();
      log(`  sae step ${r.step}  fvu ${r.mse.toFixed(4)}  L0 ${r.l0.toFixed(1)}  t=${((performance.now() - t2) / 1000).toFixed(1)}s`);
    }
  }
  const t3 = performance.now();
  const res = await sae.analyse();
  log(`analysed ${res.positions} positions in ${((performance.now() - t3) / 1000).toFixed(2)} s: fvu ${res.fvu.toFixed(4)}  L0 ${res.l0.toFixed(2)}  dead ${(res.dead * 100).toFixed(1)}%`);
  const dens = res.features.map((f) => f.density).filter((x) => x > 0).sort((a, b) => a - b);
  const qq = (p: number) => dens[Math.floor(p * (dens.length - 1))];
  log(`density quantiles (live): 10% ${qq(0.1).toExponential(1)}  50% ${qq(0.5).toExponential(1)}  90% ${qq(0.9).toExponential(1)}  max ${qq(1).toExponential(1)}`);
  const show = (pos: number) => {
    const a = Math.max(0, pos - 18);
    const s = tok.decode(Array.from(corpus.tokens.subarray(a, pos + 4)));
    const at = pos - a;
    return `${s.slice(0, at)}[${s[at]}]${s.slice(at + 1)}`.replace(/\n/g, '⏎');
  };
  const live = res.features.filter((f) => f.top.length >= 6).sort((a, b) => b.top[0].value - a.top[0].value);
  for (const f of live.filter((_, i) => i % Math.max(1, Math.floor(live.length / 14)) === 0).slice(0, 14)) {
    log(`\n#${f.id} density ${(f.density * 100).toFixed(2)}%  max ${f.top[0].value.toFixed(2)}`);
    for (const t of f.top.slice(0, 5)) log(`    ${t.value.toFixed(2)}  ${show(t.pos)}`);
  }
  window.__exp = { done: true };
}

main().catch((e) => {
  log(String(e));
  window.__exp = { done: true, error: String(e) };
});
