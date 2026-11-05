import type { RecordedRun, RecordedWeights } from '../../app/recorded';
import type { Knockouts, Patching } from '../../gpu/microscope';
import type { SaeFeature, SaeOptions, SaeStep } from '../../gpu/sae';
import type { ModelConfig } from '../../model/config';
import type { GenerationEdit } from '../../model/generate';
import type { Moment, ProbeReading } from '../../model/probe';

export type Mode = 'train' | 'heads' | 'features';
export type Pane = 'heads' | 'features';
/** Which model a pane shows: the run in this tab, or the recorded one. */
export type Which = 'live' | 'recorded';

/** A model's sparse autoencoder, as far as it has got (training, or trained and analysed). */
export interface SubjectSae {
  readonly opts: SaeOptions;
  curve: SaeStep[];
  fvu: number | null;
  l0: number | null;
  dead: number | null;
  positions: number;
  seconds: number | null;
  scale: number;
  features: SaeFeature[] | null;
  params: Float32Array | null;
  progress: { step: number; total: number } | null;
}

/** What the panes can show about one model, live or recorded. */
export interface Subject {
  readonly which: Which;
  readonly cfg: ModelConfig;
  /** The text, as tokens, and how to spell them, for feature contexts. */
  readonly tokens: Uint32Array;
  readonly chars: readonly string[];
  latest: ProbeReading | null;
  moment: Moment | null;
  knockouts: Knockouts | null;
  patching: Patching | null;
  patterns: Map<number, Float32Array>;
  sae: SubjectSae | null;
  /** What it is doing, when that is worth saying (taking the model apart, or why it cannot). */
  stage: string | null;
}

/** What the page's controller does for the microscope. */
export interface MicroscopeHost {
  /**
   * Writes text from the live model (its latest weights) or the recorded one, with an
   * intervention. 'free': from the prompt, sampled. 'copy': greedily continuing the copying
   * demonstration (random characters shown twice).
   */
  write(which: Which, edit: GenerationEdit | null, weights: Float32Array | null, kind: 'free' | 'copy'): Promise<string | null>;
  /** The copying demonstration's prompt and the continuation a perfect copier would write. */
  copyPrompt(which: Which): { prompt: string; expected: string } | null;
  /** The head whose attention lights the text when a character is inspected (null: all of them). */
  selectHead(head: number | null): void;
  /** Attention of every head on the latest induction test, for the live model. */
  livePatterns(): Promise<Float32Array | null>;
}

/** What a pane asks of the microscope around it. */
export interface PaneContext {
  readonly host: MicroscopeHost;
  readonly mode: () => Mode;
  /** The pane's chosen model, or whichever exists. */
  shown(pane: Pane): Subject | null;
  /** Exactly this model, if it exists. */
  subject(which: Which): Subject | null;
  view(pane: Pane): Which;
  live(): Subject | null;
  run(): RecordedRun | null;
  recordedWeights(): RecordedWeights | null;
  /** The Live/Recorded switch for a pane's strip. */
  viewSwitch(pane: Pane): HTMLElement;
  /** The recorded model's download, offered where writing with it needs its weights. */
  weightsGate(box: HTMLElement, which: Which): void;
  scheduleWrite(): void;
}
