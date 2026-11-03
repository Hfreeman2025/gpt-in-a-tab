import type { Rng } from './rng';

export interface SampledToken {
  readonly token: number;
  /** Probability the (temperature-adjusted) model gave the chosen token. */
  readonly p: number;
  /** The most likely alternatives, best first, as [token, p] pairs. */
  readonly top: readonly (readonly [number, number])[];
}

/**
 * Draw the next token from logits at the given temperature. Temperature 0 is greedy.
 * `scratch` must have the vocabulary's length.
 */
export function sampleToken(logits: Float32Array, temperature: number, rng: Rng, scratch: Float32Array, topK = 5): SampledToken {
  const V = logits.length;
  const greedy = temperature <= 1e-3;
  const invT = greedy ? 1 : 1 / temperature;
  let mx = -Infinity;
  for (let v = 0; v < V; v++) mx = Math.max(mx, logits[v] * invT);
  let sum = 0;
  for (let v = 0; v < V; v++) sum += scratch[v] = Math.exp(logits[v] * invT - mx);
  for (let v = 0; v < V; v++) scratch[v] /= sum;

  let token = 0;
  if (greedy) {
    for (let v = 1; v < V; v++) if (scratch[v] > scratch[token]) token = v;
  } else {
    const r = rng.next();
    let acc = 0;
    token = V - 1;
    for (let v = 0; v < V; v++) {
      acc += scratch[v];
      if (r < acc) {
        token = v;
        break;
      }
    }
  }

  const top: [number, number][] = [];
  for (let v = 0; v < V; v++) {
    const p = scratch[v];
    if (top.length < topK) {
      top.push([v, p]);
      top.sort((a, b) => b[1] - a[1]);
    } else if (p > top[topK - 1][1]) {
      top[topK - 1] = [v, p];
      top.sort((a, b) => b[1] - a[1]);
    }
  }
  return { token, p: scratch[token], top };
}
