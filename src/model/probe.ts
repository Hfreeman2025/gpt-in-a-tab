import type { ModelConfig } from './config';
import { Rng } from './rng';

/**
 * The induction test of Olsson et al. (2022), "In-context Learning and Induction Heads": feed the
 * model random characters, the same run twice over, and measure two things.
 *
 * - The loss on each copy. The first copy is unpredictable (about ln V nats a character); a model
 *   that can copy from earlier in its window predicts the second copy almost perfectly.
 * - For every head, the prefix-matching score: how much attention each position of the second
 *   copy pays to the character just after the previous occurrence of its own character, which is
 *   where an induction head looks. Also each head's attention to the previous character, the job
 *   of the head an induction head works with.
 *
 * The test's run is half the window long, longer than any repeat the model trains on, so a head
 * cannot pass it by looking a fixed distance back.
 */
export const PROBE_SEED = 1012;

export interface ProbeBatch {
  readonly rows: number;
  readonly T: number;
  readonly period: number;
  /** Clean: the run, repeated. */
  readonly tokens: Uint32Array;
  readonly targets: Uint32Array;
  /** Corrupted: the first copy replaced by different random characters, the second unchanged. */
  readonly corrupt: Uint32Array;
  readonly corruptTargets: Uint32Array;
}

export function probeBatch(cfg: ModelConfig, rows: number, seed = PROBE_SEED): ProbeBatch {
  const T = cfg.context;
  const period = T >> 1;
  const rng = new Rng(seed);
  const tokens = new Uint32Array(rows * T);
  const targets = new Uint32Array(rows * T);
  const corrupt = new Uint32Array(rows * T);
  const corruptTargets = new Uint32Array(rows * T);
  for (let b = 0; b < rows; b++) {
    const s = Array.from({ length: period }, () => rng.int(cfg.vocabSize));
    const r = Array.from({ length: period }, () => rng.int(cfg.vocabSize));
    for (let t = 0; t < T; t++) {
      tokens[b * T + t] = s[t % period];
      targets[b * T + t] = s[(t + 1) % period];
      corrupt[b * T + t] = t < period ? r[t] : s[t % period];
      corruptTargets[b * T + t] = t < period - 1 ? r[t + 1] : s[(t + 1) % period];
    }
  }
  return { rows, T, period, tokens, targets, corrupt, corruptTargets };
}

export interface ProbeScores {
  /** Mean loss on the first copy, and on the second (nats per character). */
  readonly copyFirst: number;
  readonly copySecond: number;
  /** Per head, layer-major: prefix-matching score and previous-character score, each 0–1. */
  readonly prefix: readonly number[];
  readonly prev: readonly number[];
}

export interface ProbeReading extends ProbeScores {
  readonly step: number;
}

/** Positions scored for each copy: the first copy minus its last character (whose target starts the repeat), and the second minus the window's last. */
export function copyRanges(T: number, period: number): { first: readonly [number, number]; second: readonly [number, number] } {
  return { first: [0, period - 1], second: [period, T - 1] };
}

/**
 * Scores one forward pass over a probe batch. `probs(l)` is layer l's attention, laid out
 * [rows][H][T][T]; `losses` the per-position loss, [rows][T]. The CPU reference for the WGSL
 * kernels probe_attn and probe_loss.
 */
export function scoreProbe(cfg: ModelConfig, rows: number, period: number, probs: (l: number) => ArrayLike<number>, losses: ArrayLike<number>): ProbeScores {
  const { context: T, nHead: H, nLayer: L } = cfg;
  const { first, second } = copyRanges(T, period);
  const mean = (lo: number, hi: number) => {
    let s = 0;
    for (let b = 0; b < rows; b++) for (let t = lo; t < hi; t++) s += losses[b * T + t];
    return s / (rows * (hi - lo));
  };
  const prefix: number[] = [];
  const prev: number[] = [];
  for (let l = 0; l < L; l++) {
    const p = probs(l);
    for (let h = 0; h < H; h++) {
      let pm = 0;
      let pt = 0;
      for (let b = 0; b < rows; b++) {
        const base = (b * H + h) * T * T;
        for (let i = period; i < T; i++) pm += p[base + i * T + (i - period + 1)];
        for (let i = 1; i < T; i++) pt += p[base + i * T + (i - 1)];
      }
      prefix.push(pm / (rows * (T - period)));
      prev.push(pt / (rows * (T - 1)));
    }
  }
  return { copyFirst: mean(first[0], first[1]), copySecond: mean(second[0], second[1]), prefix, prev };
}

/** Below this drop (nats) the second copy counts as no easier than the first: the plateau. */
const PLATEAU = 0.25;

/** When a head counts as an induction head: it matches prefixes, and the model copies. */
export const INDUCTION = {
  /** Prefix-matching score at or above this (an even spread over 64–128 positions scores about 0.01). */
  prefix: 0.3,
  /** The second copy at least this many nats easier than the first. */
  drop: 1,
} as const;

export interface Moment {
  readonly step: number;
  /** Head index, layer-major. */
  readonly head: number;
  readonly prefix: number;
  /** Its partner: the strongest previous-character head in an earlier layer, if any. */
  readonly partner: number | null;
  readonly partnerPrev: number;
  /** Copy losses on the plateau before copying began, and at the moment. */
  readonly before: { readonly step: number; readonly copyFirst: number; readonly copySecond: number } | null;
  readonly copyFirst: number;
  readonly copySecond: number;
}

/** The first reading at which some head passes INDUCTION, or null. */
export function findMoment(readings: readonly ProbeReading[], cfg: ModelConfig): Moment | null {
  const H = cfg.nHead;
  for (let k = 0; k < readings.length; k++) {
    const r = readings[k];
    if (r.copyFirst - r.copySecond < INDUCTION.drop) continue;
    let head = -1;
    for (let i = 0; i < r.prefix.length; i++) if (r.prefix[i] >= INDUCTION.prefix && (head < 0 || r.prefix[i] > r.prefix[head])) head = i;
    if (head < 0) continue;
    let partner: number | null = null;
    const layer = Math.floor(head / H);
    for (let i = 0; i < layer * H; i++) if (partner === null || r.prev[i] > r.prev[partner]) partner = i;
    // "Before": the last reading on the plateau, when the second copy was no easier than the first.
    let b: ProbeReading | null = null;
    for (let j = k - 1; j >= 0 && !b; j--) if (readings[j].copyFirst - readings[j].copySecond < PLATEAU) b = readings[j];
    return {
      step: r.step,
      head,
      prefix: r.prefix[head],
      partner,
      partnerPrev: partner === null ? 0 : r.prev[partner],
      before: b ? { step: b.step, copyFirst: b.copyFirst, copySecond: b.copySecond } : null,
      copyFirst: r.copyFirst,
      copySecond: r.copySecond,
    };
  }
  return null;
}

/** "L2·H3": layers and heads counted from one, as people say them. */
export function headName(index: number, H: number): string {
  return `L${Math.floor(index / H) + 1}·H${(index % H) + 1}`;
}
