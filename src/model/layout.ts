import { type ModelConfig, validateConfig } from './config';
import { Rng } from './rng';

export type Init = { readonly kind: 'normal'; readonly std: number } | { readonly kind: 'zeros' } | { readonly kind: 'ones' };

export interface TensorSpec {
  readonly name: string;
  readonly shape: readonly number[];
  readonly size: number;
  /** Offset in floats into the flat parameter buffer. Always a multiple of ALIGN. */
  readonly offset: number;
  readonly decay: boolean;
  readonly init: Init;
}

export interface ParamLayout {
  readonly tensors: readonly TensorSpec[];
  /** Length of the flat buffer in floats, including alignment padding. */
  readonly total: number;
  /** Number of real parameters (no padding). */
  readonly count: number;
  /** Tensors before this float offset get weight decay; tensors after it do not. */
  readonly decayEnd: number;
  get(name: string): TensorSpec;
}

/**
 * Every tensor starts on a 256-byte boundary so the GPU can bind it as its own sub-range
 * (minStorageBufferOffsetAlignment is 256 on every adapter we care about).
 */
export const ALIGN = 64;

const roundUp = (n: number, m: number) => Math.ceil(n / m) * m;

/**
 * GPT-2 style layout. Weights are stored (in × out) so that y = x · W.
 * Decayed tensors (all matrices, including the embeddings) come first so the optimiser can
 * decide decay with a single comparison.
 */
export function paramLayout(cfg: ModelConfig): ParamLayout {
  validateConfig(cfg);
  const { vocabSize: V, nLayer: L, dModel: C, context: T } = cfg;
  const std = 0.02;
  const projStd = 0.02 / Math.sqrt(2 * L);
  const decayed: [string, number[], Init][] = [
    ['wte', [V, C], { kind: 'normal', std }],
    ['wpe', [T, C], { kind: 'normal', std }],
  ];
  const plain: [string, number[], Init][] = [];
  for (let l = 0; l < L; l++) {
    decayed.push(
      [`h${l}.attn.w`, [C, 3 * C], { kind: 'normal', std }],
      [`h${l}.attn.proj.w`, [C, C], { kind: 'normal', std: projStd }],
      [`h${l}.mlp.fc.w`, [C, 4 * C], { kind: 'normal', std }],
      [`h${l}.mlp.proj.w`, [4 * C, C], { kind: 'normal', std: projStd }],
    );
    plain.push(
      [`h${l}.ln1.g`, [C], { kind: 'ones' }],
      [`h${l}.ln1.b`, [C], { kind: 'zeros' }],
      [`h${l}.attn.b`, [3 * C], { kind: 'zeros' }],
      [`h${l}.attn.proj.b`, [C], { kind: 'zeros' }],
      [`h${l}.ln2.g`, [C], { kind: 'ones' }],
      [`h${l}.ln2.b`, [C], { kind: 'zeros' }],
      [`h${l}.mlp.fc.b`, [4 * C], { kind: 'zeros' }],
      [`h${l}.mlp.proj.b`, [C], { kind: 'zeros' }],
    );
  }
  plain.push(['lnf.g', [C], { kind: 'ones' }], ['lnf.b', [C], { kind: 'zeros' }]);

  const tensors: TensorSpec[] = [];
  let offset = 0;
  let count = 0;
  const add =
    (decay: boolean) =>
    ([name, shape, init]: [string, number[], Init]) => {
      const size = shape.reduce((a, b) => a * b, 1);
      tensors.push({ name, shape, size, offset, decay, init });
      offset = roundUp(offset + size, ALIGN);
      count += size;
    };
  decayed.forEach(add(true));
  const decayEnd = offset;
  plain.forEach(add(false));

  const byName = new Map(tensors.map((t) => [t.name, t]));
  return {
    tensors,
    total: offset,
    count,
    decayEnd,
    get(name: string) {
      const t = byName.get(name);
      if (!t) throw new Error(`no tensor named ${name}`);
      return t;
    },
  };
}

/** Deterministic initialisation; identical for the CPU reference and the GPU model. */
export function initParams(layout: ParamLayout, seed: number): Float32Array {
  const out = new Float32Array(layout.total);
  const rng = new Rng(seed);
  for (const t of layout.tensors) {
    const view = out.subarray(t.offset, t.offset + t.size);
    if (t.init.kind === 'ones') view.fill(1);
    else if (t.init.kind === 'normal') {
      const s = t.init.std;
      for (let i = 0; i < view.length; i++) view[i] = rng.normal() * s;
    }
  }
  return out;
}

export function view<A extends Float32Array | Float64Array>(flat: A, t: TensorSpec): A {
  return flat.subarray(t.offset, t.offset + t.size) as A;
}
