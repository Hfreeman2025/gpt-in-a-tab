import { Decoder } from '../cpu/gpt';
import { type FeatureClamp, clampFeature } from '../cpu/sae';
import type { ModelConfig } from './config';
import { Rng } from './rng';
import { type SampledToken, sampleToken } from './sample';

/** Interventions while writing: heads knocked out, or one autoencoder feature pinned. */
export interface GenerationEdit {
  /** Heads to silence, layer-major. */
  readonly knockout?: readonly number[];
  /** A feature pinned at a value in the residual stream entering block `site`. */
  readonly clamp?: FeatureClamp & { readonly site: number };
}

export function applyEdit(dec: Decoder, cfg: ModelConfig, edit: GenerationEdit | undefined): void {
  if (edit?.knockout?.length) {
    const scale = new Float32Array(cfg.nLayer * cfg.nHead).fill(1);
    for (const h of edit.knockout) scale[h] = 0;
    dec.headScale = scale;
  }
  const clamp = edit?.clamp;
  if (clamp) dec.residualHook = (l, x) => void (l === clamp.site && clampFeature(x, clamp));
}

/**
 * Autoregressive sampling with a key/value cache. When the context fills, the window slides:
 * the cache is rebuilt from the most recent half-context and generation carries on, so every
 * token sees between T/2 and T characters of history.
 */
export function* generate(
  cfg: ModelConfig,
  params: Float32Array,
  prompt: readonly number[],
  length: number,
  temperature: number,
  seed: number,
  topK = 5,
  edit?: GenerationEdit,
): Generator<SampledToken> {
  if (prompt.length === 0) throw new Error('generation needs at least one prompt token');
  const dec = new Decoder(cfg, params);
  applyEdit(dec, cfg, edit);
  const rng = new Rng(seed);
  const scratch = new Float32Array(cfg.vocabSize);
  const history = prompt.slice(-(cfg.context - 1));
  let logits: Float32Array = new Float32Array(0);
  for (const t of history) logits = dec.push(t);
  for (let made = 0; made < length; made++) {
    const s = sampleToken(logits, temperature, rng, scratch, topK);
    yield s;
    history.push(s.token);
    if (made + 1 === length) return;
    if (dec.full) {
      dec.reset();
      for (const t of history.slice(-Math.floor(cfg.context / 2))) logits = dec.push(t);
    } else {
      logits = dec.push(s.token);
    }
  }
}

export interface WriteAttention {
  /** Index into [trimmed prompt, sample] of the oldest character the model could see. */
  readonly start: number;
  /** How many characters it could see (the query is the last of them). */
  readonly length: number;
  /** weights[(l · nHead + h) · length + j]: how much head h of layer l attended to character start + j. */
  readonly weights: Float32Array;
  /** The logits it sampled from, for checking. */
  readonly logits: Float32Array;
  /** How many leading prompt characters were dropped to fit the context, as in `generate`. */
  readonly promptOffset: number;
}

/**
 * Where every head looked when the model wrote character `m` of a sample. Replays the exact
 * window `generate` used (the same prompt trimming, the same slide when the context fills) and
 * captures the attention of the query that produced character m: the character before it.
 */
export function attentionWhenWriting(cfg: ModelConfig, params: Float32Array, prompt: readonly number[], sample: readonly number[], m: number): WriteAttention {
  const T = cfg.context;
  const history = prompt.slice(-(T - 1));
  const P = history.length;
  if (P === 0) throw new Error('needs the prompt the sample was written from');
  if (m < 0 || m >= sample.length) throw new Error(`no character ${m} in a sample of ${sample.length}`);
  const seq = [...history, ...sample];
  let start = 0;
  let len = P;
  for (let k = 0; k < m; k++) {
    // After writing character k, generate() either slides the window or appends to it.
    if (len >= T) {
      start = P + k + 1 - Math.floor(T / 2);
      len = Math.floor(T / 2);
    } else len++;
  }
  const dec = new Decoder(cfg, params);
  const weights = new Float32Array(cfg.nLayer * cfg.nHead * len);
  let logits: Float32Array = new Float32Array(0);
  for (let j = 0; j < len; j++) logits = dec.push(seq[start + j], j === len - 1 ? weights : null);
  return { start, length: len, weights, logits: Float32Array.from(logits), promptOffset: prompt.length - P };
}
