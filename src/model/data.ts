import { hash32 } from './dropout';
import { Rng } from './rng';

/** A contiguous run of the token stream that batches are drawn from. */
export interface Split {
  readonly start: number;
  readonly length: number;
}

export interface Corpus {
  readonly tokens: Uint32Array;
  readonly train: Split;
  /** Held-out tail of the text, or null when the text is too short to spare any. */
  readonly val: Split | null;
}

/** Hold out the last tenth of the text when that still leaves room for several windows. */
export function splitCorpus(tokens: Uint32Array, context: number): Corpus {
  const n = tokens.length;
  if (n < context + 2) throw new Error(`text is too short: need at least ${context + 2} characters`);
  const valLen = Math.floor(n * 0.1);
  if (valLen >= 4 * (context + 1) && n - valLen >= 4 * (context + 1)) {
    return { tokens, train: { start: 0, length: n - valLen }, val: { start: n - valLen, length: valLen } };
  }
  return { tokens, train: { start: 0, length: n }, val: null };
}

/**
 * An offset with this bit set is not an offset: it marks a window of random characters, a run of
 * P characters (bits 24–30) repeated to fill the window, drawn by repeatToken from the seed in
 * bits 0–23. Each position's target is the next character of the repeating run. Text can be
 * memorised; random characters cannot, so the only way to predict a repeat is to find the
 * previous occurrence of the current character and copy what followed it, which is what an
 * induction head does. The period varies from window to window, so attending a fixed distance
 * back does not work.
 */
export const REPEAT_FLAG = 0x80000000;
export const REPEAT_PERIOD = { min: 12, max: 48 } as const;

export function repeatWord(seed: number, period: number): number {
  return (REPEAT_FLAG | (period << 24) | (seed & 0xffffff)) >>> 0;
}

/** Character k of a repeated run with this seed; the same hash runs in WGSL (gather.wgsl). */
export function repeatToken(seed: number, k: number, vocab: number): number {
  return hash32((Math.imul(seed >>> 0, 0x9e3779b9) + k) >>> 0) % vocab;
}

/**
 * Random window starts such that every input window and its shifted target fit inside the split.
 * With `repeatEvery` k > 0, every k-th window is instead repeated random characters (REPEAT_FLAG).
 */
export function sampleOffsets(rng: Rng, split: Split, batch: number, T: number, repeatEvery = 0): Uint32Array {
  const span = split.length - T - 1;
  if (span < 0) throw new Error('split shorter than one window');
  const out = new Uint32Array(batch);
  for (let b = 0; b < batch; b++) {
    if (repeatEvery > 0 && b % repeatEvery === repeatEvery - 1) {
      const period = Math.min(T >> 1, REPEAT_PERIOD.min + rng.int(REPEAT_PERIOD.max - REPEAT_PERIOD.min + 1));
      out[b] = repeatWord(rng.int(0x1000000), period);
    } else out[b] = split.start + rng.int(span + 1);
  }
  return out;
}

export function gatherBatch(tokens: Uint32Array, offsets: Uint32Array, T: number, vocab: number): { inputs: Uint32Array; targets: Uint32Array } {
  const B = offsets.length;
  const inputs = new Uint32Array(B * T);
  const targets = new Uint32Array(B * T);
  for (let b = 0; b < B; b++) {
    const o = offsets[b];
    if (o & REPEAT_FLAG) {
      const seed = o & 0xffffff;
      const period = (o >>> 24) & 0x7f;
      for (let t = 0; t < T; t++) {
        inputs[b * T + t] = repeatToken(seed, t % period, vocab);
        targets[b * T + t] = repeatToken(seed, (t + 1) % period, vocab);
      }
    } else {
      inputs.set(tokens.subarray(o, o + T), b * T);
      targets.set(tokens.subarray(o + 1, o + T + 1), b * T);
    }
  }
  return { inputs, targets };
}

export { Rng };
