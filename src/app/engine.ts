import type { ModelConfig, TrainConfig } from '../model/config';
import type { EvalStats, StepStats } from '../gpu/trainer';
import type { Profile } from '../gpu/profile';
import type { ProbeReading } from '../model/probe';
import type { Knockouts, Patching } from '../gpu/microscope';
import type { SaeOptions, SaeResult, SaeStep } from '../gpu/sae';

export interface Snapshot {
  /** Number of optimiser steps the weights have seen. */
  readonly step: number;
  readonly params: Float32Array;
}

export interface Progress {
  readonly completed: number;
  readonly totalSteps: number;
  /** Seconds spent training (pauses excluded). */
  readonly elapsed: number;
  readonly tokensPerSecond: number;
  readonly tokensSeen: number;
  /** How many times the training text has been read, on average. */
  readonly epochs: number;
  /** Why the run is as long as it is. */
  readonly plannedBy: 'data' | 'time';
}

export interface EngineInfo {
  readonly kind: 'gpu' | 'cpu';
  readonly device: string;
  readonly cfg: ModelConfig;
  readonly train: TrainConfig;
  readonly paramCount: number;
  readonly memory: readonly { readonly category: string; readonly bytes: number }[];
  readonly dispatchesPerStep: number;
  readonly flopsPerStep: number;
}

export type EngineEvent =
  | {
      readonly type: 'stats';
      readonly steps: readonly StepStats[];
      readonly evals: readonly EvalStats[];
      readonly probes: readonly ProbeReading[];
      readonly progress: Progress;
    }
  | { readonly type: 'milestone'; readonly snapshot: Snapshot }
  | { readonly type: 'state'; readonly state: RunState }
  | { readonly type: 'error'; readonly message: string };

export type RunState = 'training' | 'paused' | 'finished' | 'failed';

/** The trained model taken apart: causal experiments on its heads, and a sparse autoencoder. */
export interface Dissection {
  /** The weights' step. */
  readonly step: number;
  readonly knockouts: Knockouts;
  readonly patching: Patching;
  /** Every head's attention on the first row of the induction test: T×T each, row = query. */
  readonly patterns: readonly Float32Array[];
  readonly sae: DissectedSae;
}

export interface DissectedSae {
  readonly opts: SaeOptions;
  readonly curve: readonly SaeStep[];
  readonly result: SaeResult;
  /** W_enc, b_enc, W_dec, b_dec (src/cpu/sae.ts layout), for clamping while sampling. */
  readonly params: Float32Array;
  readonly seconds: number;
}

export type DissectProgress =
  | { readonly stage: 'heads' }
  | { readonly stage: 'heads-done'; readonly knockouts: Knockouts; readonly patching: Patching; readonly patterns: readonly Float32Array[] }
  | { readonly stage: 'collect' }
  | { readonly stage: 'sae'; readonly at: SaeStep | null; readonly step: number; readonly total: number }
  | { readonly stage: 'analyse' };

export interface Engine {
  readonly info: EngineInfo;
  readonly state: RunState;
  start(): void;
  pause(): void;
  resume(): void;
  /** Train for `steps` more, at the final (minimum) learning rate. */
  extend(steps: number): void;
  /** The weights after every step submitted so far. */
  snapshot(): Promise<Snapshot>;
  profile(): Promise<Profile | null>;
  /** Every head's attention on the latest induction test, [L·H][T·T], or null if there is none. */
  probePatterns(): Promise<Float32Array | null>;
  /** Knockouts, patching and the sparse autoencoder; null where the engine cannot (the CPU fallback). */
  dissect(onProgress: (p: DissectProgress) => void): Promise<Dissection | null>;
  on(listener: (e: EngineEvent) => void): () => void;
  dispose(): void;
}

/** How often (in steps) the induction test runs during training. */
export const PROBE_EVERY = 20;

/** Wall-clock bookkeeping shared by both engines: elapsed time and a windowed throughput. */
export class Clock {
  private accumulated = 0;
  private since: number | null = null;
  private readonly samples: { t: number; done: number }[] = [];

  start(): void {
    if (this.since === null) this.since = performance.now();
  }

  stop(): void {
    if (this.since !== null) this.accumulated += performance.now() - this.since;
    this.since = null;
    this.samples.length = 0;
  }

  get elapsed(): number {
    return (this.accumulated + (this.since === null ? 0 : performance.now() - this.since)) / 1000;
  }

  /** Records that `done` steps have completed; returns steps per second over the last ~2 s. */
  mark(done: number): number {
    const t = this.elapsed;
    this.samples.push({ t, done });
    while (this.samples.length > 2 && t - this.samples[0].t > 2) this.samples.shift();
    const first = this.samples[0];
    const dt = t - first.t;
    return dt > 0.05 ? (done - first.done) / dt : 0;
  }
}

export class Emitter {
  private readonly listeners = new Set<(e: EngineEvent) => void>();

  on(l: (e: EngineEvent) => void): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  emit(e: EngineEvent): void {
    for (const l of this.listeners) l(e);
  }
}
