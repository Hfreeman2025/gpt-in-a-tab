import { type SaeLayout, saeLayout } from '../cpu/sae';
import type { Corpus } from '../model/data';
import { NO_DROPOUT } from '../model/dropout';
import { Rng } from '../model/rng';
import { type Dispatch, type Ref, gridThreads, slice } from './context';
import type { GpuGPT } from './model';
import * as op from './ops';

export interface SaeOptions {
  /** Dictionary size. */
  readonly features: number;
  /** Residual-stream vectors per optimiser step. */
  readonly batch: number;
  readonly steps: number;
  /** L1 coefficient on feature activations (inputs are scaled to average squared length C). */
  readonly lambda: number;
  readonly lr: number;
  /** Which residual stream: the input to this block (1 … nLayer − 1). */
  readonly site: number;
  /** Largest activations kept per feature, for the browser. */
  readonly topK: number;
}

export function saeDefaults(nLayer: number): SaeOptions {
  return { features: 1024, batch: 4096, steps: 3000, lambda: 2, lr: 1e-3, site: Math.max(1, nLayer - 1), topK: 12 };
}

export interface SaeStep {
  readonly step: number;
  /** Fraction of variance unexplained on this step's batch (against the variance of all positions). */
  readonly mse: number;
  /** Features active per vector, on average. */
  readonly l0: number;
}

export interface SaeFeature {
  readonly id: number;
  /** Share of all positions it fires on. */
  readonly density: number;
  readonly mean: number;
  /** Largest activations, largest first, with the corpus position they were measured at. */
  readonly top: readonly { readonly pos: number; readonly value: number }[];
}

export interface SaeResult {
  readonly features: readonly SaeFeature[];
  readonly positions: number;
  /** Fraction of variance unexplained, over every position: Σ‖x̂ − x‖² / Σ‖x − mean‖². */
  readonly fvu: number;
  readonly l0: number;
  readonly dead: number;
  readonly scale: number;
}

/**
 * A sparse autoencoder (Bricken et al. 2023; Cunningham et al. 2023) trained on the GPU, on the
 * trained model's residual stream at one site. The CPU reference is src/cpu/sae.ts.
 */
export class GpuSae {
  readonly lay: SaeLayout;
  readonly params: GPUBuffer;
  private readonly grads: GPUBuffer;
  private readonly m: GPUBuffer;
  private readonly v: GPUBuffer;
  private readonly header: GPUBuffer;
  private readonly clip: GPUBuffer;
  private acts: GPUBuffer | null = null;
  private nRows = 0;
  scale = 1;
  /** Σ‖x − mean‖² over every collected position, scaled. */
  private variance = 1;
  private readonly x: GPUBuffer;
  private readonly f: GPUBuffer;
  private readonly xhat: GPUBuffer;
  private readonly dxhat: GPUBuffer;
  private readonly df: GPUBuffer;
  private readonly parts: GPUBuffer;
  private readonly sums: GPUBuffer;
  private readonly topv: GPUBuffer;
  private readonly topi: GPUBuffer;
  private readonly err: GPUBuffer;
  private train: Dispatch[] = [];
  private stats: Dispatch[] = [];
  private analyseRows: (first: number) => Dispatch[] = () => [];
  step = 0;

  constructor(
    private readonly model: GpuGPT,
    readonly opts: SaeOptions,
  ) {
    const g = model.gpu;
    const C = model.cfg.dModel;
    const { features: F, batch: B } = opts;
    this.lay = saeLayout(C, F);
    const s = (label: string, n: number) => g.floats(label, n, 'microscope');
    this.params = s('sae params', this.lay.total);
    this.grads = s('sae grads', this.lay.total);
    this.m = s('sae adam m', this.lay.total);
    this.v = s('sae adam v', this.lay.total);
    this.header = g.buffer('sae step', 64, 'microscope');
    this.clip = g.upload('sae clip', Float32Array.of(1, 0, 0, 0), 'microscope');
    this.x = s('sae x', B * C);
    this.f = s('sae f', B * F);
    this.xhat = s('sae x̂', B * C);
    this.dxhat = s('sae dx̂', B * C);
    this.df = s('sae df', B * F);
    this.parts = s('sae partials', Math.max(op.scratchForTN(B, C, F), op.scratchForTN(B, F, C), op.scratchForColSum(B, F)));
    this.sums = s('sae feature sums', 3 * F);
    this.topv = s('sae top values', F * opts.topK);
    this.topi = g.buffer('sae top rows', F * opts.topK * 4, 'microscope');
    this.err = s('sae error', 4);
  }

  private ref(offset: number, size: number): Ref {
    return slice(this.params, offset, size);
  }

  /**
   * Runs the trained model over the whole training text in consecutive windows, without
   * dropout, and keeps the residual stream at the chosen site for every position.
   */
  async collect(corpus: Corpus): Promise<number> {
    const m = this.model;
    const g = m.gpu;
    const { context: T, dModel: C } = m.cfg;
    const B = m.B;
    const windows = Math.floor((corpus.train.length - 1) / T);
    const chunks = Math.floor(windows / B);
    if (chunks < 1) throw new Error('the text is too short for the sparse autoencoder');
    if (!this.acts || this.nRows !== chunks * B * T) this.acts = g.floats('residual stream, every position', chunks * B * T * C, 'microscope');
    this.nRows = chunks * B * T;
    this.step = 0;
    const site = m.layers[this.opts.site].x;
    for (let k = 0; k < chunks; k++) {
      const offsets = Uint32Array.from({ length: B }, (_, b) => corpus.train.start + (k * B + b) * T);
      m.writeEvalStep(0, offsets);
      m.writeDropout(NO_DROPOUT);
      const enc = g.device.createCommandEncoder();
      m.encode(enc, [m.evalProgram[0], ...m.forward]);
      enc.copyBufferToBuffer(site, 0, this.acts, k * B * T * C * 4, B * T * C * 4);
      g.device.queue.submit([enc.finish()]);
    }
    // Scale so the average vector has squared length C; keep the mean and the total variance
    // about it, for the fraction of variance the autoencoder leaves unexplained.
    const all = new Float32Array(await g.read(this.acts));
    const mean = new Float64Array(C);
    let sq = 0;
    for (let r = 0; r < this.nRows; r++) {
      for (let c = 0; c < C; c++) {
        const v = all[r * C + c];
        mean[c] += v;
        sq += v * v;
      }
    }
    let msq = 0;
    for (let c = 0; c < C; c++) {
      mean[c] /= this.nRows;
      msq += mean[c] * mean[c];
    }
    this.scale = Math.sqrt((C * this.nRows) / Math.max(sq, 1e-12));
    this.variance = (sq - this.nRows * msq) * this.scale * this.scale;
    this.init(Float32Array.from(mean));
    this.build();
    return this.nRows;
  }

  /** Decoder rows: random unit directions; encoder: their transpose; b_dec: the data mean. */
  private init(mean: Float32Array): void {
    const { C, F } = this.lay;
    const p = new Float32Array(this.lay.total);
    const rng = new Rng(4242);
    for (let j = 0; j < F; j++) {
      let n = 0;
      for (let c = 0; c < C; c++) {
        const v = rng.normal();
        p[this.lay.wDec + j * C + c] = v;
        n += v * v;
      }
      n = Math.sqrt(n);
      for (let c = 0; c < C; c++) {
        p[this.lay.wDec + j * C + c] /= n;
        p[this.lay.wEnc + c * F + j] = p[this.lay.wDec + j * C + c];
      }
    }
    for (let c = 0; c < C; c++) p[this.lay.bDec + c] = mean[c] * this.scale;
    const g = this.model.gpu;
    g.write(this.params, p);
    g.write(this.m, new Float32Array(this.lay.total));
    g.write(this.v, new Float32Array(this.lay.total));
  }

  private build(): void {
    const g = this.model.gpu;
    const { C, F } = this.lay;
    const B = this.opts.batch;
    const L = this.lay;
    const acts = this.acts as GPUBuffer;
    const dummy = this.err;
    const wEnc = this.ref(L.wEnc, C * F);
    const bEnc = this.ref(L.bEnc, F);
    const wDec = this.ref(L.wDec, F * C);
    const bDec = this.ref(L.bDec, C);
    const gr = (o: number, n: number) => slice(this.grads, o, n);
    const rows = (seed: number, first: number) =>
      g.dispatch('sae batch', 'sae_rows', [B, C, this.nRows, seed, { f: this.scale }, first, 0, 0], [acts, this.header, this.x], gridThreads(B * C));
    const forward = [
      op.linear(g, 'sae encode', { x: this.x, W: wEnc, b: bEnc, y: this.f }, B, C, F, dummy),
      g.dispatch('sae relu', 'sae_relu', [B * F, 0, 0, 0], [this.f], gridThreads(B * F)),
      op.linear(g, 'sae decode', { x: this.f, W: wDec, b: bDec, y: this.xhat }, B, F, C, dummy),
    ];
    this.train = [
      rows(0x5ae, 0),
      ...forward,
      g.dispatch('sae dx̂', 'sae_grad', [B * C, 0, { f: 2 / B }, 0], [this.xhat, this.x, this.dxhat], gridThreads(B * C)),
      ...op.linearBackward(
        g,
        'sae decode bwd',
        { dy: this.dxhat, x: this.f, W: wDec, dx: this.df, dW: gr(L.wDec, F * C), db: gr(L.bDec, C), scratch: this.parts },
        B,
        F,
        C,
        dummy,
      ),
      g.dispatch('sae df', 'sae_grad', [B * F, 1, { f: this.opts.lambda / B }, 0], [this.f, this.f, this.df], gridThreads(B * F)),
      ...op.linearBackward(
        g,
        'sae encode bwd',
        { dy: this.df, x: this.x, W: wEnc, dx: null, dW: gr(L.wEnc, C * F), db: gr(L.bEnc, F), scratch: this.parts },
        B,
        C,
        F,
        dummy,
      ),
      g.dispatch('sae adam', 'adamw', [L.total, 0, 0, 0], [this.header, this.grads, this.clip, this.params, this.m, this.v], gridThreads(L.total)),
      g.dispatch('sae decoder norm', 'sae_norm', [F, C, 0, 0], [wDec], [Math.ceil(F / 64), 1, 1]),
    ];
    this.stats = [
      g.dispatch('sae firing', 'sae_stats', [B, F, this.opts.topK, 0, 0, 0, 0, 0], [this.f, this.sums, this.topv, this.topi], [Math.ceil(F / 64), 1, 1]),
      g.dispatch('sae error', 'sae_err', [B * C, 0, 0, 0], [this.xhat, this.x, this.err], [1, 1, 1]),
    ];
    this.analyseRows = (first) => [
      rows(0, first),
      ...forward,
      g.dispatch(
        'sae top contexts',
        'sae_stats',
        [B, F, this.opts.topK, first, 1, 0, 0, 0],
        [this.f, this.sums, this.topv, this.topi],
        [Math.ceil(F / 64), 1, 1],
      ),
      g.dispatch('sae error', 'sae_err', [B * C, 0, 0, 0], [this.xhat, this.x, this.err], [1, 1, 1]),
    ];
    g.flushUniforms();
  }

  private learningRate(t: number): number {
    const warm = 100;
    const decayFrom = Math.floor(this.opts.steps * 0.8);
    if (t < warm) return (this.opts.lr * (t + 1)) / warm;
    if (t < decayFrom) return this.opts.lr;
    return this.opts.lr * Math.max(0.05, (this.opts.steps - t) / (this.opts.steps - decayFrom));
  }

  private writeHeader(t: number): void {
    const w = new Uint32Array(16);
    const f = new Float32Array(w.buffer);
    const b1 = 0.9;
    const b2 = 0.999;
    f[0] = this.learningRate(t);
    f[1] = 1 - b1 ** (t + 1);
    f[2] = 1 - b2 ** (t + 1);
    w[3] = 0;
    f[4] = 0;
    f[5] = b1;
    f[6] = b2;
    f[7] = 1e-8;
    w[9] = t + 1;
    this.model.gpu.write(this.header, w);
  }

  /** One optimiser step; every `measure` steps also measures error and firing on that batch. */
  submitStep(measure: boolean): void {
    const g = this.model.gpu;
    this.writeHeader(this.step);
    if (measure) {
      g.write(this.sums, new Float32Array(3 * this.lay.F));
      g.write(this.err, new Float32Array(4));
    }
    const enc = g.device.createCommandEncoder();
    this.model.encode(enc, measure ? [...this.train, ...this.stats] : this.train);
    g.device.queue.submit([enc.finish()]);
    this.step++;
  }

  /** The measurement taken on the last measured step: the step's batch, after its update. */
  async readStep(): Promise<SaeStep> {
    const g = this.model.gpu;
    const [sums, err] = await Promise.all([g.read(this.sums), g.read(this.err)]);
    const s = new Float32Array(sums);
    const e = new Float32Array(err);
    let fired = 0;
    for (let j = 0; j < this.lay.F; j++) fired += s[j * 3];
    const batchVariance = (this.variance / this.nRows) * this.opts.batch;
    return { step: this.step, mse: e[0] / Math.max(batchVariance, 1e-12), l0: fired / this.opts.batch };
  }

  /**
   * Every position, in order: each feature's firing count, mean activation and top contexts,
   * and the reconstruction error over the whole text.
   */
  async analyse(): Promise<SaeResult> {
    const g = this.model.gpu;
    const F = this.lay.F;
    const K = this.opts.topK;
    const B = this.opts.batch;
    g.write(this.sums, new Float32Array(3 * F));
    g.write(this.topv, new Float32Array(F * K));
    g.write(this.topi, new Uint32Array(F * K));
    g.write(this.err, new Float32Array(4));
    const chunks = Math.floor(this.nRows / B);
    const programs = Array.from({ length: chunks }, (_, k) => this.analyseRows(k * B));
    g.flushUniforms();
    for (const program of programs) {
      const enc = g.device.createCommandEncoder();
      this.model.encode(enc, program);
      g.device.queue.submit([enc.finish()]);
    }
    const [sums, tv, ti, err] = await Promise.all([g.read(this.sums), g.read(this.topv), g.read(this.topi), g.read(this.err)]);
    const s = new Float32Array(sums);
    const v = new Float32Array(tv);
    const ix = new Uint32Array(ti);
    const e = new Float32Array(err);
    const positions = chunks * B;
    const features: SaeFeature[] = [];
    let fired = 0;
    let dead = 0;
    for (let j = 0; j < F; j++) {
      const n = s[j * 3];
      fired += n;
      if (n === 0) dead++;
      const top: { pos: number; value: number }[] = [];
      for (let k = 0; k < K; k++) if (v[j * K + k] > 0) top.push({ pos: ix[j * K + k], value: v[j * K + k] });
      features.push({ id: j, density: n / positions, mean: n > 0 ? s[j * 3 + 1] / n : 0, top });
    }
    return { features, positions, fvu: e[0] / Math.max(this.variance, 1e-12), l0: fired / positions, dead: dead / F, scale: this.scale };
  }

  async readParams(): Promise<Float32Array> {
    return new Float32Array(await this.model.gpu.read(this.params));
  }
}
