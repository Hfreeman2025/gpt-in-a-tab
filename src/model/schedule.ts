import type { TrainConfig } from './config';

/**
 * Linear warm-up to the peak learning rate, then cosine decay to the minimum over the planned
 * run. Steps past the plan stay at the minimum (that is what "keep training" uses).
 */
export function learningRate(step: number, totalSteps: number, cfg: Pick<TrainConfig, 'peakLr' | 'minLr' | 'warmupSteps'>): number {
  const { peakLr, minLr, warmupSteps } = cfg;
  if (step < warmupSteps) return (peakLr * (step + 1)) / warmupSteps;
  const span = Math.max(1, totalSteps - warmupSteps);
  const progress = Math.min(1, (step - warmupSteps) / span);
  return minLr + 0.5 * (peakLr - minLr) * (1 + Math.cos(Math.PI * progress));
}

/** Adam bias corrections for 1-based step t. */
export function biasCorrections(t: number, beta1: number, beta2: number): [number, number] {
  return [1 - beta1 ** t, 1 - beta2 ** t];
}

/** Pick a step budget that should take about targetSeconds at the measured speed. */
export function planSteps(stepsPerSecond: number, cfg: Pick<TrainConfig, 'targetSeconds' | 'minSteps' | 'maxSteps'>): number {
  if (!Number.isFinite(stepsPerSecond) || stepsPerSecond <= 0) return cfg.minSteps;
  const raw = Math.round((stepsPerSecond * cfg.targetSeconds) / 50) * 50;
  return Math.min(cfg.maxSteps, Math.max(cfg.minSteps, raw));
}

/**
 * Passes over the text a run aims for. Past roughly this point a model this size, with
 * dropout, stops improving on held-out text and starts memorising (measured on the sonnets).
 */
export const TARGET_EPOCHS = 80;

/** Tokens of the text itself in one step: the windows that are not repeated random characters. */
export function textTokensPerStep(train: Pick<TrainConfig, 'batch' | 'repeatEvery'>, context: number): number {
  const repeated = train.repeatEvery > 0 ? Math.floor(train.batch / train.repeatEvery) : 0;
  return (train.batch - repeated) * context;
}

/** The step budget the data supports, before the clock is taken into account. */
export function dataSteps(trainTokens: number, tokensPerStep: number, cfg: Pick<TrainConfig, 'minSteps' | 'maxSteps'>): number {
  const raw = Math.round((TARGET_EPOCHS * trainTokens) / tokensPerStep / 50) * 50;
  return Math.min(cfg.maxSteps, Math.max(cfg.minSteps, raw));
}

/** More regularisation for less text. */
export function dropoutFor(chars: number): number {
  return chars < 400_000 ? 0.2 : 0.1;
}

/**
 * Where checkpoints are kept for the scrubber: densely early, where the model changes fastest,
 * then more sparsely, so a whole run stays under about seventy snapshots.
 */
export function isCheckpoint(step: number, totalSteps: number): boolean {
  if (step === 0) return true;
  if (step < 100) return step % 10 === 0;
  if (step < 400) return step % 25 === 0;
  if (step < 1000) return step % 50 === 0;
  const late = Math.max(100, Math.ceil(totalSteps / 40 / 50) * 50);
  return step % late === 0;
}
