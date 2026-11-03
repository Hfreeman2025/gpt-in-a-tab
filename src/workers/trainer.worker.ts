/// <reference lib="webworker" />
/**
 * The no-WebGPU fallback: the CPU reference implementation trains a smaller model here, off the
 * main thread, reporting results in batches and handing over weight snapshots on request.
 */
import { CpuGPT } from '../cpu/gpt';
import { CpuTrainer } from '../cpu/trainer';
import { PROBE_EVERY } from '../app/engine';
import { type ProbeBatch, type ProbeReading, probeBatch, scoreProbe } from '../model/probe';
import { splitCorpus } from '../model/data';
import { initParams, paramLayout } from '../model/layout';
import { Rng } from '../model/rng';
import { isCheckpoint } from '../model/schedule';
import type { EvalStats, StepStats } from '../gpu/trainer';
import type { TrainerRequest, TrainerResponse } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

let trainer: CpuTrainer | null = null;
let running = false;
let looping = false;
const evalRng = new Rng(0x5eed);
let pendingSteps: StepStats[] = [];
let pendingEvals: EvalStats[] = [];
let pendingProbes: ProbeReading[] = [];
let probe: { model: CpuGPT; batch: ProbeBatch } | null = null;

/** The induction test (src/model/probe.ts) on the live weights, eight rows. */
function runProbe(): void {
  if (!trainer) return;
  if (!probe) {
    const batch = probeBatch(trainer.cfg, 8);
    probe = { model: new CpuGPT(trainer.cfg, trainer.params, batch.rows, batch.T), batch };
  }
  const { model, batch } = probe;
  model.forward(batch.tokens, batch.targets);
  pendingProbes.push({ step: trainer.step, ...scoreProbe(trainer.cfg, batch.rows, batch.period, (l) => model.attentionProbs(l), model.losses) });
}
let busyMs = 0;

const post = (m: TrainerResponse, transfer: Transferable[] = []) => self.postMessage(m, transfer);
const tick = () => new Promise<void>((r) => setTimeout(r, 0));

function snapshot(req: number): void {
  if (!trainer) return;
  const params = Float32Array.from(trainer.params);
  post({ t: 'snapshot', req, step: trainer.step, params }, [params.buffer]);
}

function flush(): void {
  if (!trainer) return;
  post({ t: 'stats', steps: pendingSteps, evals: pendingEvals, probes: pendingProbes, completed: trainer.step, busyMs });
  pendingSteps = [];
  pendingEvals = [];
  pendingProbes = [];
}

async function loop(): Promise<void> {
  if (looping) return;
  looping = true;
  let lastFlush = performance.now();
  try {
    while (running && trainer && trainer.step < trainer.totalSteps) {
      const t = trainer;
      const evalEvery = Math.max(25, Math.round(t.scheduleSteps / 50 / 25) * 25);
      const t0 = performance.now();
      if (t.step % evalEvery === 0) {
        const loss = t.evaluate(evalRng, 1);
        if (loss !== null) pendingEvals.push({ step: t.step, loss });
      }
      if (t.step % PROBE_EVERY === 0) runProbe();
      pendingSteps.push(t.runStep());
      busyMs += performance.now() - t0;
      if (isCheckpoint(t.step, t.scheduleSteps) && t.step < t.totalSteps) snapshot(-1);
      if (performance.now() - lastFlush > 150) {
        flush();
        lastFlush = performance.now();
      }
      await tick();
    }
    if (trainer && trainer.step >= trainer.totalSteps) {
      const loss = trainer.evaluate(evalRng, 1);
      if (loss !== null) pendingEvals.push({ step: trainer.step, loss });
      runProbe();
      flush();
      snapshot(-1);
      running = false;
      post({ t: 'finished', step: trainer.step });
    } else flush();
  } catch (e) {
    post({ t: 'error', message: String(e) });
  } finally {
    looping = false;
  }
}

self.onmessage = (e: MessageEvent<TrainerRequest>) => {
  const m = e.data;
  switch (m.t) {
    case 'start': {
      const corpus = splitCorpus(m.tokens, m.cfg.context);
      trainer = new CpuTrainer(m.cfg, m.train, corpus, initParams(paramLayout(m.cfg), m.seed), m.seed, m.totalSteps);
      probe = null;
      running = true;
      snapshot(-1);
      void loop();
      break;
    }
    case 'pause':
      running = false;
      break;
    case 'resume':
      running = true;
      void loop();
      break;
    case 'plan':
      if (trainer) {
        trainer.totalSteps = Math.max(trainer.step + 1, m.totalSteps);
        trainer.scheduleSteps = trainer.totalSteps;
      }
      break;
    case 'extend':
      if (trainer) trainer.totalSteps = m.totalSteps;
      running = true;
      void loop();
      break;
    case 'snapshot':
      snapshot(m.req);
      break;
  }
};
