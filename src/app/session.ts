import type { ModelConfig } from '../model/config';
import type { Corpus } from '../model/data';
import type { Moment, ProbeReading } from '../model/probe';
import type { Tokenizer } from '../model/tokenizer';
import type { Profile } from '../gpu/profile';
import type { StepStats } from '../gpu/trainer';
import type { Engine, Progress } from './engine';
import type { HeadAttention, Sample, Ticket } from './sampler';

/** Characters drawn at every checkpoint. */
export const SAMPLE_LENGTH = 480;
/** Every checkpoint is sampled with the same random numbers, so differences are the model's alone. */
export const SAMPLE_SEED = 1609;
/** Checkpoints whose weights are kept (half precision, about 1.6 MB each for the GPU model). */
export const MAX_CHECKPOINTS = 80;

/** A snapshot of the weights, and the text drawn from it at the current settings. */
export interface Checkpoint {
  readonly step: number;
  readonly key: string;
  readonly half: Uint16Array;
  sample: Sample | null;
  /** The sampling settings `sample` was drawn with. */
  epoch: number;
  ticket: Ticket<Sample> | null;
  urgent: boolean;
}

/** One training run and everything the page keeps about it. */
export interface Session {
  readonly id: number;
  readonly engine: Engine;
  readonly tok: Tokenizer;
  readonly cfg: ModelConfig;
  readonly corpus: Corpus;
  /** The token every sample starts from when there is no prompt: a line break if the text has one. */
  readonly start: number;
  readonly checkpoints: Checkpoint[];
  /** Bumped whenever the prompt or temperature changes; samples from an older epoch are stale. */
  epoch: number;
  prompt: number[];
  promptChars: string[];
  /** Prompt characters the text never uses, so the model cannot read them. */
  dropped: number;
  temperature: number;
  /** Following the run (true) or replaying the checkpoint under the cursor. */
  live: boolean;
  cursor: number;
  shown: number;
  progress: Progress | null;
  last: StepStats | null;
  heldOut: number | null;
  sixty: { step: number; loss: number; heldOut: number | null; tps: number } | null;
  profile: Profile | null;
  profiling: boolean;
  profileError: string | null;
  inspect: { index: number; ticket: Ticket<HeadAttention> | null; timer: number } | null;
  attention: { key: string; index: number; value: HeadAttention } | null;
  /** The induction test's readings, and the moment a head learned to copy. */
  probes: ProbeReading[];
  moment: Moment | null;
  dissecting: boolean;
  /** Every step's loss and learning rate, and every held-out reading, as [step, value]. */
  trace: { steps: [number, number][]; lr: [number, number][]; evals: [number, number][] };
}

/** What sampling starts from: the typed prompt, or the start token. */
export function promptFor(s: Session): number[] {
  return s.prompt.length ? s.prompt : [s.start];
}
