/**
 * Dev experiment: train with the shipped GPU configuration and, every `every` steps, run the
 * induction test (Olsson et al. 2022) on a CPU copy of the weights: repeated random characters,
 * per-head prefix-matching and previous-token scores, and the loss on the first and second copy.
 *   ?steps=1700&sample=sonnets&every=25&dropout=0.2
 */
import { CpuGPT } from '../../src/cpu/gpt';
import { GPU_SHAPE, GPU_TRAIN, withVocab } from '../../src/model/config';
import { splitCorpus } from '../../src/model/data';
import { dataSteps } from '../../src/model/schedule';
import { initParams, paramLayout } from '../../src/model/layout';
import { Rng } from '../../src/model/rng';
import { buildTokenizer } from '../../src/model/tokenizer';
import { Gpu, acquireDevice } from '../../src/gpu/context';
import { PIPELINE_VARIANTS } from '../../src/gpu/model';
import { GpuTrainer } from '../../src/gpu/trainer';
import alice from '../../src/samples/alice.txt?raw';
import sonnets from '../../src/samples/sonnets.txt?raw';
import tragedies from '../../src/samples/tragedies.txt?raw';

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
  const name = q.get('sample') ?? 'sonnets';
  const text = name === 'alice' ? alice : name === 'tragedies' ? tragedies : sonnets;
  const tok = buildTokenizer(text);
  const cfg = withVocab({ ...GPU_SHAPE, nLayer: Number(q.get('layers') ?? GPU_SHAPE.nLayer) }, tok.size);
  const corpus = splitCorpus(tok.encode(text), cfg.context);
  const dropout = Number(q.get('dropout') ?? GPU_TRAIN.dropout);
  const peakLr = Number(q.get('lr') ?? GPU_TRAIN.peakLr);
  const repeatEvery = Number(q.get('repeat') ?? 0);
  const minLr = Number(q.get('minlr') ?? peakLr / 10);
  const train = { ...GPU_TRAIN, dropout, peakLr, minLr, repeatEvery };
  const planned = dataSteps(corpus.train.length, train.batch * cfg.context, train);
  const steps = Number(q.get('steps') ?? planned);
  const every = Number(q.get('every') ?? 25);
  const acq = await acquireDevice();
  if ('error' in acq) throw new Error(acq.error);
  const g = new Gpu(acq.device);
  await g.prepare(PIPELINE_VARIANTS);
  const layout = paramLayout(cfg);
  const trainer = new GpuTrainer(g, cfg, train, corpus, initParams(layout, Number(q.get('init') ?? 1)), Number(q.get('seed') ?? 2026), steps);
  log(
    `${name} probe=${q.get('dist') ?? 'uniform'}: vocab ${tok.size}, layers ${cfg.nLayer}, steps ${steps} (planned ${planned}), dropout ${dropout}, lr ${peakLr}, repeat every ${repeatEvery}`,
  );

  // The probe: B sequences of T/2 random characters, each repeated once.
  const B = 4;
  const T = cfg.context;
  const half = T / 2;
  const rng = new Rng(7);
  const tokens = new Uint32Array(B * T);
  const targets = new Uint32Array(B * T);
  // uniform: any character; unigram: characters drawn as often as they occur in the text.
  const dist = q.get('dist') ?? 'uniform';
  const enc = tok.encode(text);
  const draw = () => (dist === 'unigram' ? enc[rng.int(enc.length)] : rng.int(tok.size));
  for (let b = 0; b < B; b++) {
    const s = Array.from({ length: half }, draw);
    for (let t = 0; t < T; t++) tokens[b * T + t] = s[t % half];
    for (let t = 0; t < T; t++) targets[b * T + t] = s[(t + 1) % half];
  }
  const H = cfg.nHead;
  const probe = (params: Float32Array) => {
    const m = new CpuGPT(cfg, params, B, T);
    m.forward(tokens, targets);
    let first = 0;
    let second = 0;
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < half - 1; t++) first += m.losses[b * T + t];
      for (let t = half; t < T - 1; t++) second += m.losses[b * T + t];
    }
    first /= B * (half - 1);
    second /= B * (half - 1);
    const prefix: number[] = [];
    const prev: number[] = [];
    for (let l = 0; l < cfg.nLayer; l++) {
      const p = m.attentionProbs(l);
      for (let h = 0; h < H; h++) {
        let pm = 0;
        let pt = 0;
        for (let b = 0; b < B; b++) {
          const base = (b * H + h) * T * T;
          for (let i = half; i < T; i++) pm += p[base + i * T + (i - half + 1)];
          for (let i = 1; i < T; i++) pt += p[base + i * T + (i - 1)];
        }
        prefix.push(pm / (B * half));
        prev.push(pt / (B * (T - 1)));
      }
    }
    return { first, second, prefix, prev };
  };

  if (q.has('check')) {
    trainer.submitStep();
    const ins = new Uint32Array(await g.read(trainer.model.inputs));
    const tg = new Uint32Array(await g.read(trainer.model.targets));
    for (const b of [0, 1, 2, 3])
      log(
        `row ${b}: ${Array.from(ins.subarray(b * T, b * T + 12)).join(',')} … ${Array.from(ins.subarray(b * T + half, b * T + half + 12)).join(',')} | targets ${Array.from(tg.subarray(b * T, b * T + 6)).join(',')}`,
      );
    window.__exp = { done: true };
    return;
  }
  const t0 = performance.now();
  const inflight: Promise<undefined>[] = [];
  let lastLoss = NaN;
  const rows: unknown[] = [];
  for (let s = 0; s <= steps; s++) {
    if (s % every === 0) {
      const st = await trainer.readStats();
      if (st.steps.length) lastLoss = st.steps[st.steps.length - 1].loss;
      const snap = await trainer.snapshot();
      const r = probe(snap.params);
      const best = r.prefix.reduce((bi, v, i, a) => (v > a[bi] ? i : bi), 0);
      const bestPrev = r.prev.reduce((bi, v, i, a) => (v > a[bi] ? i : bi), 0);
      const name = (i: number) => `L${Math.floor(i / H) + 1}H${(i % H) + 1}`;
      log(
        `${String(snap.step).padStart(5)}  t=${((performance.now() - t0) / 1000).toFixed(1).padStart(5)}s  train ${lastLoss.toFixed(3)}  copy1 ${r.first.toFixed(3)}  copy2 ${r.second.toFixed(3)}  best prefix ${name(best)} ${r.prefix[best].toFixed(2)}  best prev ${name(bestPrev)} ${r.prev[bestPrev].toFixed(2)}  | ${r.prefix.map((v) => v.toFixed(2)).join(' ')}`,
      );
      rows.push({ step: snap.step, train: lastLoss, ...r });
    }
    if (s === steps) break;
    trainer.submitStep();
    inflight.push(acq.device.queue.onSubmittedWorkDone());
    if (inflight.length >= 3) await inflight.shift();
  }
  window.__exp = { done: true, rows };
}

main().catch((e) => {
  log(String(e));
  window.__exp = { done: true, error: String(e) };
});
