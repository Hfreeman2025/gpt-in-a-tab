/**
 * Counter-based dropout masks: whether element i survives depends only on (seed, site, i), so
 * the GPU kernels and the CPU reference draw identical masks without storing them, and the
 * backward pass recomputes the forward mask instead of saving it. The same integer hash is
 * written in WGSL (src/gpu/wgsl/common.wgsl).
 */

/** "lowbias32" integer hash (Chris Wellons). */
export function hash32(x: number): number {
  let v = x >>> 0;
  v ^= v >>> 16;
  v = Math.imul(v, 0x7feb352d) >>> 0;
  v ^= v >>> 15;
  v = Math.imul(v, 0x846ca68b) >>> 0;
  v ^= v >>> 16;
  return v >>> 0;
}

/** Per-site seed: one independent mask per dropout site per step. */
export function siteSeed(stepSeed: number, site: number): number {
  return hash32((stepSeed ^ hash32(site + 1)) >>> 0);
}

export function stepSeed(runSeed: number, step: number): number {
  return hash32((Math.imul(runSeed >>> 0, 0x9e3779b9) + step) >>> 0);
}

export interface DropoutStep {
  readonly seed: number;
  /** An element is kept when hash ≥ threshold; threshold = p · 2³². */
  readonly threshold: number;
  /** 1 / (1 − p), applied to kept elements. */
  readonly scale: number;
}

export function dropoutStep(p: number, seed: number): DropoutStep {
  if (!(p >= 0 && p < 1)) throw new Error(`dropout must be in [0, 1), got ${p}`);
  return { seed: seed >>> 0, threshold: Math.min(0xffffffff, Math.floor(p * 4294967296)), scale: 1 / (1 - p) };
}

export const NO_DROPOUT: DropoutStep = { seed: 0, threshold: 0, scale: 1 };

/** Mask value for element i at a site: 0 or scale. */
export function maskValue(d: DropoutStep, site: number, i: number): number {
  if (d.threshold === 0) return 1;
  return hash32((i ^ siteSeed(d.seed, site)) >>> 0) >= d.threshold ? d.scale : 0;
}

/** Sites: 0 is the embedding output; layer l uses 1 + 2l (attention) and 2 + 2l (MLP). */
export const SITE_EMBED = 0;
export const siteAttn = (l: number) => 1 + 2 * l;
export const siteMlp = (l: number) => 2 + 2 * l;
