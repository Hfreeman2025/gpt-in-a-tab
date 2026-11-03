/**
 * Dev experiment: train on a bundled sample with the shipped GPU configuration and log the
 * train/held-out loss curve and samples at milestones. Used to tune the run length.
 *   ?steps=6000&batch=32&sample=sonnets&lr=0.003
 */
import { Decoder } from '../../src/cpu/gpt';
import { GPU_SHAPE, GPU_TRAIN, withVocab } from '../../src/model/config';
import { splitCorpus } from '../../src/model/data';
import { initParams, paramLayout } from '../../src/model/layout';
import { Rng } from '../../src/model/rng';
import { sampleToken } from '../../src/model/sample';
import { buildTokenizer } from '../../src/model/tokenizer';
import { Gpu, acquireDevice } from '../../src/gpu/context';
import { PIPELINE_VARIANTS } from '../../src/gpu/model';
import { GpuTrainer } from '../../src/gpu/trainer';
import sonnets from '../../src/samples/sonnets.txt?raw';

declare global {
  interface Window {
    __exp?: unknown;
  }
}

const out = document.getElementById('out') as HTMLElement;
const log = (s: string) => (out.textContent += s + '\n');

async function main() {
  const q = new URLSearchParams(location.search);
  const steps = Number(q.get('steps') ?? 6000);
  const batch = Number(q.get('batch') ?? GPU_TRAIN.batch);
  const peakLr = Number(q.get('lr') ?? GPU_TRAIN.peakLr);
  const text = sonnets;
  const tok = buildTokenizer(text);
  const cfg = withVocab(GPU_SHAPE, tok.size);
  const corpus = splitCorpus(tok.encode(text), cfg.context);
  const dropout = Number(q.get('dropout') ?? GPU_TRAIN.dropout);
  const train = { ...GPU_TRAIN, batch, peakLr, minLr: peakLr / 10, dropout };
  const acq = await acquireDevice();
  if ('error' in acq) throw new Error(acq.error);
  const g = new Gpu(acq.device);
  await g.prepare(PIPELINE_VARIANTS);
  const layout = paramLayout(cfg);
  const trainer = new GpuTrainer(g, cfg, train, corpus, initParams(layout, 1), 1, steps);
  log(`chars ${text.length}, vocab ${tok.size}, params ${layout.count}, B=${batch}, steps ${steps}, epochs ${((steps * batch * cfg.context) / corpus.train.length).toFixed(0)}`);

  const sample = (params: Float32Array, n = 240) => {
    const dec = new Decoder(cfg, params);
    const rng = new Rng(77);
    const scratch = new Float32Array(cfg.vocabSize);
    let logits = dec.push(tok.encode('\n')[0]);
    let s = '';
    for (let i = 0; i < n; i++) {
      const t = sampleToken(logits, 0.8, rng, scratch).token;
      s += tok.chars[t];
      if (dec.full) {
        dec.reset();
        const ctx = tok.encode(s.slice(-cfg.context / 2));
        for (const c of ctx) logits = dec.push(c);
      } else logits = dec.push(t);
    }
    return s;
  };

  const milestones = new Set([0, 100, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000]);
  const t0 = performance.now();
  const inflight: Promise<undefined>[] = [];
  const curve: string[] = [];
  let lastLoss = NaN;
  for (let s = 0; s <= steps; s++) {
    if (milestones.has(s)) {
      const snap = await trainer.snapshot();
      log(`\n— step ${snap.step} (${((performance.now() - t0) / 1000).toFixed(1)} s) —\n${sample(snap.params)}`);
    }
    if (s % 200 === 0) trainer.submitEval();
    if (s === steps) break;
    trainer.submitStep();
    inflight.push(acq.device.queue.onSubmittedWorkDone());
    if (inflight.length >= 3) await inflight.shift();
    if (s % 200 === 199) {
      const st = await trainer.readStats();
      lastLoss = st.steps[st.steps.length - 1]?.loss ?? NaN;
      for (const e of st.evals) curve.push(`${String(e.step).padStart(5)}  train ${lastLoss.toFixed(3)}  val ${e.loss.toFixed(3)}  t=${((performance.now() - t0) / 1000).toFixed(1)}s`);
    }
  }
  const st = await trainer.readStats();
  for (const e of st.evals) curve.push(`${String(e.step).padStart(5)}  val ${e.loss.toFixed(3)}`);
  log('\n' + curve.join('\n'));
  window.__exp = { done: true };
}

main().catch((e) => {
  log(String(e));
  window.__exp = { done: true };
});
