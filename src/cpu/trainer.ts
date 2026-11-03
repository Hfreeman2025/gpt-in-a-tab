import type { ModelConfig, TrainConfig } from '../model/config';
import { type Corpus, gatherBatch, sampleOffsets } from '../model/data';
import { dropoutStep, stepSeed } from '../model/dropout';
import { Rng } from '../model/rng';
import { biasCorrections, learningRate } from '../model/schedule';
import { CpuGPT } from './gpt';
import { adamw, clipScale, l2Norm } from './ops';

export interface StepResult {
  readonly step: number;
  readonly loss: number;
  readonly gradNorm: number;
  readonly lr: number;
}

/** Plain-JavaScript training loop: the reference for the GPU trainer and the no-WebGPU fallback. */
export class CpuTrainer {
  readonly model: CpuGPT;
  readonly m: Float32Array;
  readonly v: Float32Array;
  private readonly rng: Rng;
  private readonly seed: number;
  step = 0;
  /** Where training stops. */
  totalSteps: number;
  /** The length the learning-rate schedule decays over; steps past it run at the minimum. */
  scheduleSteps: number;

  constructor(
    readonly cfg: ModelConfig,
    readonly train: TrainConfig,
    readonly corpus: Corpus,
    params: Float32Array,
    seed: number,
    totalSteps: number,
  ) {
    this.model = new CpuGPT(cfg, params, train.batch, cfg.context);
    this.m = new Float32Array(params.length);
    this.v = new Float32Array(params.length);
    this.rng = new Rng(seed);
    this.seed = seed;
    this.totalSteps = totalSteps;
    this.scheduleSteps = totalSteps;
  }

  get params(): Float32Array {
    return this.model.params as Float32Array;
  }

  /** Offsets for the next batch; drawn from the same stream the GPU trainer uses. */
  nextOffsets(): Uint32Array {
    return sampleOffsets(this.rng, this.corpus.train, this.train.batch, this.cfg.context, this.train.repeatEvery);
  }

  runStep(offsets = this.nextOffsets()): StepResult {
    const T = this.cfg.context;
    const { inputs, targets } = gatherBatch(this.corpus.tokens, offsets, T, this.cfg.vocabSize);
    const loss = this.model.forward(inputs, targets, dropoutStep(this.train.dropout, stepSeed(this.seed, this.step)));
    this.model.backward();
    const gradNorm = l2Norm(this.model.grads);
    const lr = learningRate(this.step, this.scheduleSteps, this.train);
    const [bc1, bc2] = biasCorrections(this.step + 1, this.train.beta1, this.train.beta2);
    adamw(this.model.params, this.model.grads, this.m, this.v, {
      lr,
      beta1: this.train.beta1,
      beta2: this.train.beta2,
      eps: this.train.eps,
      weightDecay: this.train.weightDecay,
      bc1,
      bc2,
      clip: clipScale(gradNorm, this.train.clipNorm),
      decayEnd: this.model.layout.decayEnd,
    });
    const result = { step: this.step, loss, gradNorm, lr };
    this.step++;
    return result;
  }

  /** Mean loss over a batch drawn from the held-out split, without touching the weights. */
  evaluate(rng: Rng, batches = 1): number | null {
    const val = this.corpus.val;
    if (!val) return null;
    let total = 0;
    for (let i = 0; i < batches; i++) {
      const offsets = sampleOffsets(rng, val, this.train.batch, this.cfg.context);
      const { inputs, targets } = gatherBatch(this.corpus.tokens, offsets, this.cfg.context, this.cfg.vocabSize);
      total += this.model.forward(inputs, targets);
    }
    return total / batches;
  }
}
