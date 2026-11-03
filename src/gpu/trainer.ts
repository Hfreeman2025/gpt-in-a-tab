import type { ModelConfig, TrainConfig } from '../model/config';
import { type Corpus, sampleOffsets } from '../model/data';
import { dropoutStep, stepSeed } from '../model/dropout';
import { Rng } from '../model/rng';
import { biasCorrections, learningRate } from '../model/schedule';
import type { Gpu } from './context';
import type { ProbeReading } from '../model/probe';
import { GpuGPT, PROBE_RING, RING, STATS, probeStride } from './model';

export interface StepStats {
  readonly step: number;
  readonly loss: number;
  readonly gradNorm: number;
  readonly lr: number;
}

export interface EvalStats {
  readonly step: number;
  readonly loss: number;
}

export interface StatsBatch {
  readonly steps: readonly StepStats[];
  readonly evals: readonly EvalStats[];
  /** Steps known to have finished on the GPU when this batch was read. */
  readonly completed: number;
  /** Induction tests run since the last read. */
  readonly probes: readonly ProbeReading[];
}

/**
 * Drives the GPU model: draws batches (from the same random stream the CPU reference uses),
 * writes each step's header, submits, and reads results back from the GPU-side ring in bulk.
 */
export class GpuTrainer {
  readonly model: GpuGPT;
  private readonly rng: Rng;
  private readonly evalRng: Rng;
  private readonly lrs: number[] = [];
  /** Steps submitted so far. */
  step = 0;
  /** Where training stops. */
  totalSteps: number;
  /** The length the learning-rate schedule decays over; steps past it run at the minimum. */
  scheduleSteps: number;
  private evalCount = 0;
  private readonly pendingEvals: { slot: number; step: number; batches: number }[] = [];
  private lastRead = 0;
  private probeCount = 0;
  private readonly pendingProbes: { slot: number; step: number }[] = [];
  private readonly evalBatches: number;
  private readonly seed: number;

  constructor(
    readonly gpu: Gpu,
    readonly cfg: ModelConfig,
    readonly train: TrainConfig,
    readonly corpus: Corpus,
    initial: Float32Array,
    seed: number,
    totalSteps: number,
    evalBatches = 2,
  ) {
    this.model = new GpuGPT(gpu, cfg, train, corpus.tokens, initial, train.batch);
    this.rng = new Rng(seed);
    this.evalRng = new Rng(seed ^ 0x5eed);
    this.totalSteps = totalSteps;
    this.scheduleSteps = totalSteps;
    this.evalBatches = evalBatches;
    this.seed = seed;
  }

  get tokensPerStep(): number {
    return this.train.batch * this.cfg.context;
  }

  submitStep(): void {
    const t = this.step;
    const offsets = sampleOffsets(this.rng, this.corpus.train, this.train.batch, this.cfg.context, this.train.repeatEvery);
    const lr = learningRate(t, this.scheduleSteps, this.train);
    const [bc1, bc2] = biasCorrections(t + 1, this.train.beta1, this.train.beta2);
    this.lrs[t % RING] = lr;
    this.model.writeDropout(dropoutStep(this.train.dropout, stepSeed(this.seed, t)));
    this.model.writeTrainStep(
      {
        lr,
        bc1,
        bc2,
        slot: t % RING,
        weightDecay: this.train.weightDecay,
        beta1: this.train.beta1,
        beta2: this.train.beta2,
        eps: this.train.eps,
        clipNorm: this.train.clipNorm,
      },
      offsets,
    );
    this.model.submit(this.model.trainProgram);
    this.step++;
  }

  /** Queue a held-out evaluation of the current weights. No-op without a held-out split. */
  submitEval(): boolean {
    const val = this.corpus.val;
    if (!val) return false;
    const first = this.evalCount;
    for (let b = 0; b < this.evalBatches; b++) {
      const slot = this.evalCount++ % RING;
      this.model.writeEvalStep(slot, sampleOffsets(this.evalRng, val, this.train.batch, this.cfg.context));
      this.model.submit(this.model.evalProgram);
    }
    this.pendingEvals.push({ slot: first % RING, step: this.step, batches: this.evalBatches });
    return true;
  }

  /** Queue the induction test (src/model/probe.ts) on the current weights. */
  submitProbe(): void {
    const slot = this.probeCount++ % PROBE_RING;
    this.model.submitProbe(slot);
    this.pendingProbes.push({ slot, step: this.step });
  }

  /** Reads every result produced since the last call. Must run at least every RING steps. */
  async readStats(): Promise<StatsBatch> {
    const upTo = this.step;
    const evals = this.pendingEvals.splice(0);
    const probes = this.pendingProbes.splice(0);
    const [stats, results] = await Promise.all([this.model.readStats(), probes.length ? this.model.readProbes() : null]);
    const steps: StepStats[] = [];
    for (let t = this.lastRead; t < upTo; t++) {
      steps.push({ step: t, loss: stats[STATS.trainLoss + (t % RING)], gradNorm: stats[STATS.gradNorm + (t % RING)], lr: this.lrs[t % RING] });
    }
    this.lastRead = upTo;
    const evalStats = evals.map((e) => {
      let s = 0;
      for (let b = 0; b < e.batches; b++) s += stats[STATS.valLoss + ((e.slot + b) % RING)];
      return { step: e.step, loss: s / e.batches };
    });
    const stride = probeStride(this.cfg);
    const heads = this.cfg.nLayer * this.cfg.nHead;
    const readings: ProbeReading[] = results
      ? probes.map(({ slot, step }) => {
          const o = slot * stride;
          return {
            step,
            copyFirst: results[o],
            copySecond: results[o + 1],
            prefix: Array.from({ length: heads }, (_, i) => results[o + 2 + 2 * i]),
            prev: Array.from({ length: heads }, (_, i) => results[o + 3 + 2 * i]),
          };
        })
      : [];
    return { steps, evals: evalStats, completed: upTo, probes: readings };
  }

  /** The weights after every step submitted so far. */
  async snapshot(): Promise<{ step: number; params: Float32Array }> {
    const step = this.step;
    return { step, params: await this.model.readParams() };
  }

  destroy(): void {
    this.gpu.destroy();
  }
}
