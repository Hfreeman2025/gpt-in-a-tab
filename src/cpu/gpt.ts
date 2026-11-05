import type { ModelConfig } from '../model/config';
import { type DropoutStep, NO_DROPOUT, SITE_EMBED, siteAttn, siteMlp } from '../model/dropout';
import { type ParamLayout, type TensorSpec, paramLayout, view } from '../model/layout';
import * as op from './ops';
import type { F } from './ops';

type Ctor = Float32ArrayConstructor | Float64ArrayConstructor;

interface LayerActs {
  readonly x: F; // block input (residual stream)
  readonly ln1: F;
  readonly ln1Mean: F;
  readonly ln1Rstd: F;
  readonly qkv: F;
  readonly probs: F;
  readonly attnY: F;
  readonly mid: F; // x + attention branch
  readonly ln2: F;
  readonly ln2Mean: F;
  readonly ln2Rstd: F;
  readonly fcPre: F;
  readonly fcAct: F;
}

interface LayerParams {
  readonly ln1g: TensorSpec;
  readonly ln1b: TensorSpec;
  readonly attnW: TensorSpec;
  readonly attnB: TensorSpec;
  readonly projW: TensorSpec;
  readonly projB: TensorSpec;
  readonly ln2g: TensorSpec;
  readonly ln2b: TensorSpec;
  readonly fcW: TensorSpec;
  readonly fcB: TensorSpec;
  readonly fc2W: TensorSpec;
  readonly fc2B: TensorSpec;
}

export function layerTensors(layout: ParamLayout, l: number): LayerParams {
  const g = (s: string) => layout.get(`h${l}.${s}`);
  return {
    ln1g: g('ln1.g'),
    ln1b: g('ln1.b'),
    attnW: g('attn.w'),
    attnB: g('attn.b'),
    projW: g('attn.proj.w'),
    projB: g('attn.proj.b'),
    ln2g: g('ln2.g'),
    ln2b: g('ln2.b'),
    fcW: g('mlp.fc.w'),
    fcB: g('mlp.fc.b'),
    fc2W: g('mlp.proj.w'),
    fc2B: g('mlp.proj.b'),
  };
}

/**
 * The reference GPT. Forward saves every activation; backward fills `grads` with the gradient
 * of the mean cross-entropy loss. One instance is sized for a fixed (B, T).
 */
export class CpuGPT {
  readonly layout: ParamLayout;
  readonly grads: F;
  readonly N: number;
  private readonly acts: LayerActs[];
  private readonly lp: LayerParams[];
  private readonly scratch: { tmp: F; tmp2: F; big: F; scores: F; dqkv: F; gres: F; dmask: F };
  readonly xf: F; // residual stream after the last block
  readonly lnf: F;
  readonly lnfMean: F;
  readonly lnfRstd: F;
  readonly logits: F;
  readonly dlogits: F;
  readonly losses: F;
  private tokens: ArrayLike<number> = [];
  private targets: ArrayLike<number> | null = null;
  private drop: DropoutStep = NO_DROPOUT;

  constructor(
    readonly cfg: ModelConfig,
    readonly params: F,
    readonly B: number,
    readonly T: number,
    ctor: Ctor = Float32Array,
  ) {
    if (T > cfg.context) throw new Error(`T=${T} exceeds context ${cfg.context}`);
    this.layout = paramLayout(cfg);
    if (params.length !== this.layout.total) throw new Error('parameter buffer does not match layout');
    const { dModel: C, nHead: H, nLayer: L, vocabSize: V } = cfg;
    const N = B * T;
    this.N = N;
    const a = (n: number) => new ctor(n);
    this.grads = a(this.layout.total);
    this.acts = [];
    this.lp = [];
    for (let l = 0; l < L; l++) {
      this.lp.push(layerTensors(this.layout, l));
      this.acts.push({
        x: a(N * C),
        ln1: a(N * C),
        ln1Mean: a(N),
        ln1Rstd: a(N),
        qkv: a(N * 3 * C),
        probs: a(B * H * T * T),
        attnY: a(N * C),
        mid: a(N * C),
        ln2: a(N * C),
        ln2Mean: a(N),
        ln2Rstd: a(N),
        fcPre: a(N * 4 * C),
        fcAct: a(N * 4 * C),
      });
    }
    this.scratch = { tmp: a(N * C), tmp2: a(N * C), big: a(N * 4 * C), scores: a(B * H * T * T), dqkv: a(N * 3 * C), gres: a(N * C), dmask: a(N * C) };
    this.xf = a(N * C);
    this.lnf = a(N * C);
    this.lnfMean = a(N);
    this.lnfRstd = a(N);
    this.logits = a(N * V);
    this.dlogits = a(N * V);
    this.losses = a(N);
  }

  private p(t: TensorSpec): F {
    return view(this.params, t);
  }

  private g(t: TensorSpec): F {
    return view(this.grads, t);
  }

  /** The residual stream entering block l (after the forward pass), laid out [B·T][C]. */
  residual(l: number): F {
    return this.acts[l].x;
  }

  /** Layer l's per-head attention output before the projection, laid out [B·T][C]. */
  attentionOutput(l: number): F {
    return this.acts[l].attnY;
  }

  /** Attention probabilities of layer l, laid out [B][H][T][T]. */
  attentionProbs(l: number): F {
    return this.acts[l].probs;
  }

  /**
   * Runs the model; returns the mean loss if targets are given, otherwise NaN. Dropout (on the
   * embeddings and both residual branches) is applied only when `drop` is given.
   */
  /**
   * Called after each layer's attention, with that layer's per-head output (before the output
   * projection, laid out [B·T][C]), for knockouts and activation patching. The CPU reference for
   * the WGSL kernel head_edit.
   */
  editAttention: ((layer: number, y: F) => void) | null = null;

  forward(tokens: ArrayLike<number>, targets: ArrayLike<number> | null = null, drop: DropoutStep = NO_DROPOUT): number {
    const { dModel: C, nHead: H, vocabSize: V } = this.cfg;
    const { B, T, N } = this;
    if (tokens.length < N) throw new Error('not enough tokens for the batch');
    this.tokens = tokens;
    this.targets = targets;
    this.drop = drop;
    const shape = { B, T, C, H };
    const L = this.cfg.nLayer;
    const x0 = L > 0 ? this.acts[0].x : this.xf;
    op.embed(x0, tokens, this.p(this.layout.get('wte')), this.p(this.layout.get('wpe')), B, T, C);
    op.dropout(x0, x0, N * C, drop, SITE_EMBED);
    for (let l = 0; l < L; l++) {
      const a = this.acts[l];
      const w = this.lp[l];
      const out = l + 1 < L ? this.acts[l + 1].x : this.xf;
      op.layerNorm(a.ln1, a.ln1Mean, a.ln1Rstd, a.x, this.p(w.ln1g), this.p(w.ln1b), N, C);
      op.matmul(a.qkv, a.ln1, this.p(w.attnW), N, C, 3 * C, this.p(w.attnB));
      op.attention(a.attnY, a.probs, this.scratch.scores, a.qkv, shape);
      this.editAttention?.(l, a.attnY);
      op.matmul(this.scratch.tmp, a.attnY, this.p(w.projW), N, C, C, this.p(w.projB));
      op.dropout(this.scratch.tmp, this.scratch.tmp, N * C, drop, siteAttn(l));
      op.add(a.mid, a.x, this.scratch.tmp, N * C);
      op.layerNorm(a.ln2, a.ln2Mean, a.ln2Rstd, a.mid, this.p(w.ln2g), this.p(w.ln2b), N, C);
      op.matmul(a.fcPre, a.ln2, this.p(w.fcW), N, C, 4 * C, this.p(w.fcB));
      op.gelu(a.fcAct, a.fcPre, N * 4 * C);
      op.matmul(this.scratch.tmp, a.fcAct, this.p(w.fc2W), N, 4 * C, C, this.p(w.fc2B));
      op.dropout(this.scratch.tmp, this.scratch.tmp, N * C, drop, siteMlp(l));
      op.add(out, a.mid, this.scratch.tmp, N * C);
    }
    const lnfG = this.layout.get('lnf.g');
    const lnfB = this.layout.get('lnf.b');
    op.layerNorm(this.lnf, this.lnfMean, this.lnfRstd, this.xf, this.p(lnfG), this.p(lnfB), N, C);
    // Tied output head: logits = lnf · wteᵀ.
    op.matmulNT(this.logits, this.lnf, this.p(this.layout.get('wte')), N, C, V);
    if (!targets) return NaN;
    return op.crossEntropy(this.losses, this.dlogits, this.logits, targets, N, V);
  }

  backward(): void {
    if (!this.targets) throw new Error('backward needs a forward pass with targets');
    const { dModel: C, nHead: H, vocabSize: V, nLayer: L } = this.cfg;
    const { B, T, N } = this;
    const shape = { B, T, C, H };
    const { tmp, tmp2, big, scores, dqkv, gres, dmask } = this.scratch;
    const drop = this.drop;
    this.grads.fill(0);
    const wte = this.layout.get('wte');

    // Head: logits = lnf · wteᵀ. d(lnf) = dlogits · wte; d(wte) = dlogitsᵀ · lnf.
    op.matmul(tmp, this.dlogits, this.p(wte), N, V, C);
    op.matmulTN(this.g(wte), this.dlogits, this.lnf, N, V, C);
    gres.fill(0);
    op.layerNormBackward(
      gres,
      this.g(this.layout.get('lnf.g')),
      this.g(this.layout.get('lnf.b')),
      tmp,
      this.xf,
      this.lnfMean,
      this.lnfRstd,
      this.p(this.layout.get('lnf.g')),
      N,
      C,
    );

    for (let l = L - 1; l >= 0; l--) {
      const a = this.acts[l];
      const w = this.lp[l];
      // MLP branch. gres is the gradient w.r.t. this block's output.
      op.dropout(dmask, gres, N * C, drop, siteMlp(l));
      op.linearBackward(big, this.g(w.fc2W), this.g(w.fc2B), dmask, a.fcAct, this.p(w.fc2W), N, 4 * C, C);
      op.geluBackward(big, big, a.fcPre, N * 4 * C);
      op.linearBackward(tmp, this.g(w.fcW), this.g(w.fcB), big, a.ln2, this.p(w.fcW), N, C, 4 * C);
      op.layerNormBackward(gres, this.g(w.ln2g), this.g(w.ln2b), tmp, a.mid, a.ln2Mean, a.ln2Rstd, this.p(w.ln2g), N, C);
      // Attention branch.
      op.dropout(dmask, gres, N * C, drop, siteAttn(l));
      op.linearBackward(tmp2, this.g(w.projW), this.g(w.projB), dmask, a.attnY, this.p(w.projW), N, C, C);
      op.attentionBackward(dqkv, scores, tmp2, a.qkv, a.probs, shape);
      op.linearBackward(tmp, this.g(w.attnW), this.g(w.attnB), dqkv, a.ln1, this.p(w.attnW), N, C, 3 * C);
      op.layerNormBackward(gres, this.g(w.ln1g), this.g(w.ln1b), tmp, a.x, a.ln1Mean, a.ln1Rstd, this.p(w.ln1g), N, C);
    }
    op.dropout(dmask, gres, N * C, drop, SITE_EMBED);
    op.embedBackward(this.g(wte), this.g(this.layout.get('wpe')), dmask, this.tokens, B, T, C);
  }
}

/**
 * Incremental decoder with a key/value cache, for sampling. Feeding tokens one at a time gives
 * the same logits as a full forward pass over the sequence (tested).
 */
export class Decoder {
  readonly layout: ParamLayout;
  private readonly lp: LayerParams[];
  private readonly kCache: Float32Array[];
  private readonly vCache: Float32Array[];
  private readonly x: Float32Array;
  private readonly h: Float32Array;
  private readonly qkv: Float32Array;
  private readonly y: Float32Array;
  private readonly t1: Float32Array;
  private readonly fc: Float32Array;
  private readonly att: Float32Array;
  private readonly one = new Float32Array(1);
  private readonly one2 = new Float32Array(1);
  readonly logits: Float32Array;
  pos = 0;
  /** Per head (layer-major), a factor on its output: 0 knocks it out. Null leaves every head alone. */
  headScale: Float32Array | null = null;
  /** Called with the residual stream entering block `layer`, which it may change (clamping features). */
  residualHook: ((layer: number, x: Float32Array) => void) | null = null;

  constructor(
    readonly cfg: ModelConfig,
    readonly params: Float32Array,
  ) {
    this.layout = paramLayout(cfg);
    if (params.length !== this.layout.total) throw new Error('parameter buffer does not match layout');
    const { dModel: C, nLayer: L, context: T, vocabSize: V } = cfg;
    this.lp = Array.from({ length: L }, (_, l) => layerTensors(this.layout, l));
    this.kCache = Array.from({ length: L }, () => new Float32Array(T * C));
    this.vCache = Array.from({ length: L }, () => new Float32Array(T * C));
    this.x = new Float32Array(C);
    this.h = new Float32Array(C);
    this.qkv = new Float32Array(3 * C);
    this.y = new Float32Array(C);
    this.t1 = new Float32Array(C);
    this.fc = new Float32Array(4 * C);
    this.att = new Float32Array(T);
    this.logits = new Float32Array(V);
  }

  reset(): void {
    this.pos = 0;
  }

  get full(): boolean {
    return this.pos >= this.cfg.context;
  }

  private p(t: TensorSpec): Float32Array {
    return view(this.params, t);
  }

  /**
   * Appends one token at the next position and returns the logits for the token after it. With
   * `capture` (length nLayer · nHead · (pos + 1)), also records where every head attended from
   * this position: capture[(l · H + h) · (pos + 1) + j] is the weight on cache position j.
   */
  push(token: number, capture: Float32Array | null = null): Float32Array {
    const { dModel: C, nHead: H, nLayer: L, vocabSize: V } = this.cfg;
    if (this.full) throw new Error('decoder context is full; reset and re-feed a shorter window');
    const t = this.pos;
    const hs = C / H;
    const scale = 1 / Math.sqrt(hs);
    const wte = this.p(this.layout.get('wte'));
    const wpe = this.p(this.layout.get('wpe'));
    const { x, h, qkv, y, t1, fc, att, one, one2 } = this;
    for (let c = 0; c < C; c++) x[c] = wte[token * C + c] + wpe[t * C + c];
    for (let l = 0; l < L; l++) {
      const w = this.lp[l];
      const kc = this.kCache[l];
      const vc = this.vCache[l];
      this.residualHook?.(l, x);
      op.layerNorm(h, one, one2, x, this.p(w.ln1g), this.p(w.ln1b), 1, C);
      op.matmul(qkv, h, this.p(w.attnW), 1, C, 3 * C, this.p(w.attnB));
      kc.set(qkv.subarray(C, 2 * C), t * C);
      vc.set(qkv.subarray(2 * C, 3 * C), t * C);
      for (let hd = 0; hd < H; hd++) {
        const o = hd * hs;
        let mx = -Infinity;
        for (let j = 0; j <= t; j++) {
          let dot = 0;
          for (let d = 0; d < hs; d++) dot += qkv[o + d] * kc[j * C + o + d];
          att[j] = dot * scale;
          if (att[j] > mx) mx = att[j];
        }
        let sum = 0;
        for (let j = 0; j <= t; j++) {
          att[j] = Math.exp(att[j] - mx);
          sum += att[j];
        }
        for (let d = 0; d < hs; d++) y[o + d] = 0;
        const cap = capture ? (l * H + hd) * (t + 1) : -1;
        for (let j = 0; j <= t; j++) {
          const pj = att[j] / sum;
          if (cap >= 0 && capture) capture[cap + j] = pj;
          for (let d = 0; d < hs; d++) y[o + d] += pj * vc[j * C + o + d];
        }
        const k = this.headScale ? this.headScale[l * H + hd] : 1;
        if (k !== 1) for (let d = 0; d < hs; d++) y[o + d] *= k;
      }
      op.matmul(t1, y, this.p(w.projW), 1, C, C, this.p(w.projB));
      for (let c = 0; c < C; c++) x[c] += t1[c];
      op.layerNorm(h, one, one2, x, this.p(w.ln2g), this.p(w.ln2b), 1, C);
      op.matmul(fc, h, this.p(w.fcW), 1, C, 4 * C, this.p(w.fcB));
      op.gelu(fc, fc, 4 * C);
      op.matmul(t1, fc, this.p(w.fc2W), 1, 4 * C, C, this.p(w.fc2B));
      for (let c = 0; c < C; c++) x[c] += t1[c];
    }
    op.layerNorm(h, one, one2, x, this.p(this.layout.get('lnf.g')), this.p(this.layout.get('lnf.b')), 1, C);
    op.matmulNT(this.logits, h, wte, 1, C, V);
    this.pos++;
    return this.logits;
  }
}
