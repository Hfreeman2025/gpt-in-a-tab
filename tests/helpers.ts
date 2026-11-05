import { Rng } from '../src/model/rng';

export function randn(n: number, rng: Rng, scale = 1): Float64Array {
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = rng.normal() * scale;
  return a;
}

/**
 * Central finite-difference check. `loss` reads `x` (which is perturbed in place);
 * `analytic` is the gradient to compare. Checks every element, or `maxChecks` random ones.
 * Returns the worst relative error, normalised by the gradient's overall scale so tiny
 * entries do not dominate.
 */
export function gradCheck(
  loss: () => number,
  x: Float64Array,
  analytic: ArrayLike<number>,
  opts: { eps?: number; maxChecks?: number; rng?: Rng } = {},
): number {
  const eps = opts.eps ?? 1e-6;
  const idx: number[] = [];
  if (opts.maxChecks !== undefined && opts.maxChecks < x.length) {
    const rng = opts.rng ?? new Rng(7);
    for (let i = 0; i < opts.maxChecks; i++) idx.push(rng.int(x.length));
  } else {
    for (let i = 0; i < x.length; i++) idx.push(i);
  }
  let scale = 1e-12;
  for (let i = 0; i < analytic.length; i++) scale = Math.max(scale, Math.abs(analytic[i]));
  let worst = 0;
  for (const i of idx) {
    const orig = x[i];
    x[i] = orig + eps;
    const lp = loss();
    x[i] = orig - eps;
    const lm = loss();
    x[i] = orig;
    const numeric = (lp - lm) / (2 * eps);
    worst = Math.max(worst, Math.abs(numeric - analytic[i]) / scale);
  }
  return worst;
}

/** Σ out ⊙ w: a scalar that makes `w` the upstream gradient of `out`. */
export function dot(out: ArrayLike<number>, w: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < out.length; i++) s += out[i] * w[i];
  return s;
}
