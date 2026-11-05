import type { ModelConfig } from '../model/config';
import type { GenerationEdit } from '../model/generate';
import { Rng } from '../model/rng';
import type { WeightsRef } from '../workers/protocol';
import { PRIORITY, type SamplerPool } from './sampler';
import { SAMPLE_SEED } from './session';

/** Characters written for the microscope's comparisons. */
export const WRITE_LENGTH = 320;

/** The copying demonstration: a prompt, and the continuation a perfect copier would write. */
export interface CopyDemo {
  readonly prompt: string;
  readonly expected: string;
}

/**
 * The induction test made readable: a run of random characters (any character of the text but
 * the line break, as in the test), shown twice, then its first few characters again. A model that
 * copies finishes the run; the continuation is known, so every character can be checked.
 */
export function copyDemo(chars: readonly string[], context: number): CopyDemo | null {
  const usable = chars.filter((c) => c !== '\n' && c !== '\r');
  if (usable.length < 8) return null;
  const rng = new Rng(31);
  const n = Math.floor(context / 4);
  let run = '';
  for (let i = 0; i < n; i++) run += usable[rng.int(usable.length)];
  const lead = 5;
  const prompt = run + run + run.slice(0, lead);
  const expected = run.repeat(4).slice(lead, lead + Math.min(n + 8, context - prompt.length - 1));
  return { prompt, expected };
}

/** Tokens for the characters of `text` the vocabulary has; the rest are skipped. */
export function encodeKnown(chars: readonly string[], text: string): number[] {
  const index = new Map(chars.map((c, i) => [c, i]));
  return Array.from(text).flatMap((c) => (index.has(c) ? [index.get(c) ?? 0] : []));
}

/**
 * Writes `length` characters with an intervention (or none), always with the same seed so two
 * writings differ only by what was changed. Resolves to null if the job was cancelled or failed.
 */
export async function write(
  pool: SamplerPool,
  cfg: ModelConfig,
  weights: WeightsRef,
  chars: readonly string[],
  prompt: readonly number[],
  length: number,
  temperature: number,
  edit: GenerationEdit | null,
): Promise<string | null> {
  const ticket = pool.sample(cfg, weights, prompt, length, temperature, SAMPLE_SEED, PRIORITY.interactive, edit ?? undefined);
  const r = await ticket.result.catch(() => null);
  return r ? Array.from(r.tokens, (k) => chars[k] ?? '').join('') : null;
}
