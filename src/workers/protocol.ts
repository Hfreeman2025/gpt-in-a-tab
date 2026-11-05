import type { GenerationEdit } from '../model/generate';
import type { ProbeReading } from '../model/probe';
import type { ModelConfig, TrainConfig } from '../model/config';
import type { EvalStats, StepStats } from '../gpu/trainer';

// ---- sampling workers. Every job carries its own weights (half precision), so any worker can
// take any job and nothing has to be kept in step between them.

export interface WeightsRef {
  /** Identifies the checkpoint, so a worker can reuse weights it has already unpacked. */
  readonly key: string;
  readonly half: Uint16Array;
}

export type SamplerRequest =
  | {
      readonly t: 'sample';
      readonly job: number;
      readonly cfg: ModelConfig;
      readonly weights: WeightsRef;
      readonly prompt: readonly number[];
      readonly length: number;
      readonly temperature: number;
      readonly seed: number;
      readonly edit?: GenerationEdit;
    }
  | {
      readonly t: 'attention';
      readonly job: number;
      readonly cfg: ModelConfig;
      readonly weights: WeightsRef;
      readonly prompt: readonly number[];
      readonly sample: readonly number[];
      readonly index: number;
    }
  | { readonly t: 'cancel'; readonly job: number };

export type SamplerResponse =
  | {
      readonly t: 'sample';
      readonly job: number;
      readonly tokens: Uint32Array;
      readonly probs: Float32Array;
      /** Top alternatives per token, flattened as [token, p, token, p, …] (TOP_K pairs each). */
      readonly tops: Float32Array;
      readonly ms: number;
    }
  | {
      readonly t: 'attention';
      readonly job: number;
      readonly start: number;
      readonly length: number;
      readonly promptOffset: number;
      readonly weights: Float32Array;
    }
  | { readonly t: 'error'; readonly job: number; readonly message: string };

export const TOP_K = 5;

// ---- CPU trainer worker (the no-WebGPU fallback)

export type TrainerRequest =
  | {
      readonly t: 'start';
      readonly cfg: ModelConfig;
      readonly train: TrainConfig;
      readonly tokens: Uint32Array;
      readonly totalSteps: number;
      readonly seed: number;
    }
  | { readonly t: 'pause' }
  | { readonly t: 'resume' }
  | { readonly t: 'extend'; readonly totalSteps: number }
  | { readonly t: 'plan'; readonly totalSteps: number }
  | { readonly t: 'snapshot'; readonly req: number };

export type TrainerResponse =
  | {
      readonly t: 'stats';
      readonly steps: readonly StepStats[];
      readonly evals: readonly EvalStats[];
      readonly probes: readonly ProbeReading[];
      readonly completed: number;
      readonly busyMs: number;
    }
  | { readonly t: 'snapshot'; readonly req: number; readonly step: number; readonly params: Float32Array }
  | { readonly t: 'finished'; readonly step: number }
  | { readonly t: 'error'; readonly message: string };
