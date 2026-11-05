/** Shape of the transformer. Everything else (layout, buffers, kernels) is derived from this. */
export interface ModelConfig {
  readonly vocabSize: number;
  readonly nLayer: number;
  readonly nHead: number;
  readonly dModel: number;
  /** Context length T: the longest sequence the model can attend over. */
  readonly context: number;
}

export interface TrainConfig {
  readonly batch: number;
  readonly peakLr: number;
  readonly minLr: number;
  readonly warmupSteps: number;
  readonly weightDecay: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly clipNorm: number;
  /** Dropout on the embeddings and both residual branches of every block. */
  readonly dropout: number;
  /**
   * Every this-many-th window of a batch is a passage shown twice (its first half, repeated), so
   * that copying from earlier in the window pays off; 0 for none. See REPEAT_FLAG.
   */
  readonly repeatEvery: number;
  /** The run is planned to take about this long; the step budget is fixed after a short calibration. */
  readonly targetSeconds: number;
  readonly minSteps: number;
  readonly maxSteps: number;
}

export type ModelShape = Omit<ModelConfig, 'vocabSize'>;

/** Sized to learn visibly in 60–90 s on an M-series GPU. */
export const GPU_SHAPE: ModelShape = { nLayer: 4, nHead: 4, dModel: 128, context: 128 };
export const GPU_TRAIN: TrainConfig = {
  batch: 32,
  peakLr: 3e-3,
  minLr: 3e-4,
  warmupSteps: 100,
  weightDecay: 0.1,
  beta1: 0.9,
  beta2: 0.99,
  eps: 1e-8,
  clipNorm: 1.0,
  dropout: 0.1,
  // One window in four is repeated random characters: copying has to pay for an induction head
  // to form within a minute (see README, "Induction heads").
  repeatEvery: 4,
  targetSeconds: 90,
  minSteps: 2000,
  maxSteps: 12000,
};

/** The no-WebGPU fallback: small enough that plain JavaScript can train it in a worker. */
export const CPU_SHAPE: ModelShape = { nLayer: 2, nHead: 2, dModel: 48, context: 48 };
export const CPU_TRAIN: TrainConfig = {
  batch: 8,
  peakLr: 4e-3,
  minLr: 4e-4,
  warmupSteps: 40,
  weightDecay: 0.05,
  beta1: 0.9,
  beta2: 0.99,
  eps: 1e-8,
  clipNorm: 1.0,
  dropout: 0,
  repeatEvery: 4,
  targetSeconds: 120,
  minSteps: 300,
  maxSteps: 6000,
};

export function withVocab(shape: ModelShape, vocabSize: number): ModelConfig {
  return { ...shape, vocabSize };
}

export function validateConfig(cfg: ModelConfig): void {
  const ints: (keyof ModelConfig)[] = ['vocabSize', 'nLayer', 'nHead', 'dModel', 'context'];
  for (const k of ints) {
    const v = cfg[k];
    if (!Number.isInteger(v) || v <= 0) throw new Error(`config.${k} must be a positive integer, got ${v}`);
  }
  if (cfg.dModel % cfg.nHead !== 0) throw new Error('dModel must be divisible by nHead');
}
