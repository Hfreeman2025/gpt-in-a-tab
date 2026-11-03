import type { ModelConfig, TrainConfig } from '../model/config';
import { type DropoutStep, NO_DROPOUT, SITE_EMBED, siteAttn, siteMlp } from '../model/dropout';
import { type ParamLayout, type TensorSpec, paramLayout } from '../model/layout';
import { copyRanges, probeBatch } from '../model/probe';
import { type Constants, type Dispatch, type Gpu, type KernelName, type Ref, gridThreads, slice } from './context';
import * as op from './ops';

/** Results ring: per-step losses and norms land here so the CPU can read many steps at once. */
export const RING = 1024;
export const STATS = {
  trainLoss: 0,
  gradNorm: RING,
  valLoss: 2 * RING,
  currentNorm: 3 * RING,
  clip: 3 * RING + 1,
  length: 3 * RING + 4,
} as const;

/** Induction-test results: a ring of slots, each [copyFirst, copySecond, then (prefix, prev) per head]. */
export const PROBE_RING = 256;
export function probeStride(cfg: ModelConfig): number {
  return 2 + 2 * cfg.nLayer * cfg.nHead;
}

/** Header words of the per-step buffer; batch offsets follow from word 16. */
export const HEADER = 16;

export interface StepHeader {
  readonly lr: number;
  readonly bc1: number;
  readonly bc2: number;
  readonly slot: number;
  readonly weightDecay: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly clipNorm: number;
}

export function packStep(h: StepHeader, offsets: Uint32Array): Uint32Array {
  const words = new Uint32Array(HEADER + offsets.length);
  const f = new Float32Array(words.buffer);
  f[0] = h.lr;
  f[1] = h.bc1;
  f[2] = h.bc2;
  words[3] = h.slot;
  f[4] = h.weightDecay;
  f[5] = h.beta1;
  f[6] = h.beta2;
  f[7] = h.eps;
  f[8] = h.clipNorm;
  words.set(offsets, HEADER);
  return words;
}

/** Every pipeline variant the model uses, compiled up front in parallel. */
export const PIPELINE_VARIANTS: readonly (readonly [KernelName, Constants])[] = [
  ...[false, true].flatMap((ta) =>
    [false, true].flatMap((tb) => [false, true].map((bias) => ['matmul', { TRANS_A: ta, TRANS_B: tb, HAS_BIAS: bias, ACCUM: false }] as const)),
  ),
  ['reduce', { ACCUM: false }],
  ['reduce', { ACCUM: true }],
  ['colsum', {}],
  ['layernorm', {}],
  ['layernorm_bwd', { ACCUM: true }],
  ['layernorm_bwd', { ACCUM: false }],
  ['layernorm_bwd_params', {}],
  ['softmax', {}],
  ['softmax_bwd', {}],
  ['gelu', {}],
  ['gelu_bwd', {}],
  ['residual', {}],
  ['dropout', {}],
  ['embed', {}],
  ['embed_bwd_wpe', {}],
  ['embed_bwd_wte', {}],
  ['gather', {}],
  ['cross_entropy', {}],
  ['loss_reduce', {}],
  ['grad_norm', {}],
  ['grad_norm_final', {}],
  ['adamw', {}],
  ['probe_attn', {}],
  ['probe_loss', {}],
  ['head_edit', {}],
];

interface LayerBuffers {
  readonly x: GPUBuffer;
  readonly ln1: GPUBuffer;
  readonly ln1Mean: GPUBuffer;
  readonly ln1Rstd: GPUBuffer;
  readonly qkv: GPUBuffer;
  readonly probs: GPUBuffer;
  readonly attnY: GPUBuffer;
  readonly mid: GPUBuffer;
  readonly ln2: GPUBuffer;
  readonly ln2Mean: GPUBuffer;
  readonly ln2Rstd: GPUBuffer;
  readonly fcPre: GPUBuffer;
  readonly fcAct: GPUBuffer;
}

/**
 * The GPT on the GPU: one flat buffer each for parameters, gradients and the two Adam moments;
 * activations saved per layer for the backward pass; and three precompiled programs
 * (train step, held-out evaluation, and their shared forward pass).
 */
export class GpuGPT {
  readonly layout: ParamLayout;
  readonly N: number;
  readonly params: GPUBuffer;
  readonly grads: GPUBuffer;
  readonly m: GPUBuffer;
  readonly v: GPUBuffer;
  readonly stats: GPUBuffer;
  readonly corpus: GPUBuffer;
  readonly trainStep: GPUBuffer;
  readonly evalStep: GPUBuffer;
  /** Dropout seed and rate for whatever is submitted next (zero rate for evaluation). */
  readonly drop: GPUBuffer;
  readonly inputs: GPUBuffer;
  readonly targets: GPUBuffer;
  readonly logits: GPUBuffer;
  readonly dlogits: GPUBuffer;
  readonly losses: GPUBuffer;
  readonly layers: readonly LayerBuffers[];
  readonly forward: readonly Dispatch[];
  readonly backward: readonly Dispatch[];
  readonly optimiser: readonly Dispatch[];
  readonly trainProgram: readonly Dispatch[];
  readonly evalProgram: readonly Dispatch[];
  /** The induction test: B random runs, each repeated once (src/model/probe.ts). */
  readonly probeTokens: GPUBuffer;
  readonly probeTargets: GPUBuffer;
  readonly probeCorrupt: GPUBuffer;
  readonly probeCorruptTargets: GPUBuffer;
  readonly probePeriod: number;
  readonly probeSlot: GPUBuffer;
  readonly probeOut: GPUBuffer;
  /** Every head's attention on the first row of the latest induction test, [L][H][T][T]. */
  readonly probePatterns: GPUBuffer;
  /** Scores the forward pass just run as an induction test, into the slot in probeSlot. */
  readonly probeScores: readonly Dispatch[];
  /** Per layer: the head edit (keep[H], patch[H]) and a saved attention output to patch from. */
  readonly edits: readonly GPUBuffer[];
  readonly savedY: readonly GPUBuffer[];
  /** The forward pass with every layer's head edit applied. */
  readonly forwardEdited: readonly Dispatch[];

  constructor(
    readonly gpu: Gpu,
    readonly cfg: ModelConfig,
    readonly train: TrainConfig,
    corpusTokens: Uint32Array,
    initial: Float32Array,
    readonly B: number,
  ) {
    const { vocabSize: V, nLayer: L, nHead: H, dModel: C, context: T } = cfg;
    const layout = paramLayout(cfg);
    this.layout = layout;
    if (initial.length !== layout.total) throw new Error('initial parameters do not match the layout');
    const N = B * T;
    this.N = N;
    const g = gpu;
    const f = (label: string, n: number) => g.floats(label, n, 'activations');

    this.params = g.upload('params', initial, 'parameters');
    this.grads = g.floats('grads', layout.total, 'gradients');
    this.m = g.floats('adam m', layout.total, 'optimiser state');
    this.v = g.floats('adam v', layout.total, 'optimiser state');
    this.stats = g.floats('stats', STATS.length, 'other');
    this.corpus = g.upload('corpus', corpusTokens, 'corpus');
    this.trainStep = g.buffer('train step', (HEADER + B) * 4, 'other');
    this.evalStep = g.buffer('eval step', (HEADER + B) * 4, 'other');
    this.drop = g.buffer('dropout', 16, 'other');
    const drop = this.drop;
    this.inputs = g.buffer('inputs', N * 4, 'activations');
    this.targets = g.buffer('targets', N * 4, 'activations');

    const layers: LayerBuffers[] = [];
    for (let l = 0; l < L; l++) {
      layers.push({
        x: f(`h${l}.x`, N * C),
        ln1: f(`h${l}.ln1`, N * C),
        ln1Mean: f(`h${l}.ln1.mean`, N),
        ln1Rstd: f(`h${l}.ln1.rstd`, N),
        qkv: f(`h${l}.qkv`, N * 3 * C),
        probs: f(`h${l}.probs`, B * H * T * T),
        attnY: f(`h${l}.attn.y`, N * C),
        mid: f(`h${l}.mid`, N * C),
        ln2: f(`h${l}.ln2`, N * C),
        ln2Mean: f(`h${l}.ln2.mean`, N),
        ln2Rstd: f(`h${l}.ln2.rstd`, N),
        fcPre: f(`h${l}.fc.pre`, N * 4 * C),
        fcAct: f(`h${l}.fc.act`, N * 4 * C),
      });
    }
    this.layers = layers;
    const xf = f('xf', N * C);
    const lnf = f('lnf', N * C);
    const lnfMean = f('lnf.mean', N);
    const lnfRstd = f('lnf.rstd', N);
    this.logits = f('logits', N * V);
    this.dlogits = f('dlogits', N * V);
    this.losses = f('losses', N);

    // Scratch, reused in turn by every reduction in the backward pass.
    const s = (label: string, n: number) => g.floats(label, n, 'scratch');
    const tmp = s('tmp', N * C);
    const tmp2 = s('tmp2', N * C);
    const big = s('big', N * 4 * C);
    const dqkv = s('dqkv', N * 3 * C);
    const gres = s('gres', N * C);
    const dmask = s('dmask', N * C);
    const attScratch = s('scores', B * H * T * T);
    const partsSize = Math.max(
      op.scratchForTN(N, C, 3 * C),
      op.scratchForTN(N, C, 4 * C),
      op.scratchForTN(N, 4 * C, C),
      op.scratchForTN(N, C, C),
      op.scratchForTN(N, V, C),
      op.scratchForColSum(N, 4 * C),
      op.scratchForLayerNorm(N, C),
      op.scratchForEmbed(N, V, C),
    );
    const parts = s('partials', partsSize);
    const normGroups = Math.min(1024, Math.ceil(layout.total / 4096));
    const normParts = s('norm partials', normGroups);
    const dummy = s('dummy', 4);

    const P = (t: TensorSpec): Ref => slice(this.params, t.offset, t.size);
    const G = (t: TensorSpec): Ref => slice(this.grads, t.offset, t.size);
    const t = (name: string) => layout.get(name);
    const shape = { B, T, C, H };

    // ---- analysis buffers (the microscope): the induction test, head edits
    const probe = probeBatch(cfg, B);
    const m = (label: string, n: number) => g.floats(label, n, 'microscope');
    this.probePeriod = probe.period;
    this.probeTokens = g.upload('probe tokens', probe.tokens, 'microscope');
    this.probeTargets = g.upload('probe targets', probe.targets, 'microscope');
    this.probeCorrupt = g.upload('probe corrupted', probe.corrupt, 'microscope');
    this.probeCorruptTargets = g.upload('probe corrupted targets', probe.corruptTargets, 'microscope');
    this.probeSlot = g.buffer('probe slot', 16, 'microscope');
    this.probeOut = m('probe results', PROBE_RING * probeStride(cfg));
    this.probePatterns = m('probe attention patterns', L * H * T * T);
    this.edits = layers.map((_, l) => {
      const e = new Float32Array(2 * H);
      e.fill(1, 0, H);
      return g.upload(`h${l}.edit`, e, 'microscope');
    });
    this.savedY = layers.map((_, l) => m(`h${l}.attn.y saved`, N * C));

    // ---- forward
    const buildForward = (edited: boolean): Dispatch[] => {
      const fwd: Dispatch[] = [];
      fwd.push(op.embed(g, 'embed', this.inputs, P(t('wte')), P(t('wpe')), drop, L > 0 ? layers[0].x : xf, N, T, C, SITE_EMBED));
      for (let l = 0; l < L; l++) {
        const a = layers[l];
        const w = (n: string) => P(t(`h${l}.${n}`));
        const out = l + 1 < L ? layers[l + 1].x : xf;
        fwd.push(
          op.layerNorm(g, `h${l} ln1`, { x: a.x, g: w('ln1.g'), b: w('ln1.b'), y: a.ln1, mean: a.ln1Mean, rstd: a.ln1Rstd }, N, C),
          op.linear(g, `h${l} qkv`, { x: a.ln1, W: w('attn.w'), b: w('attn.b'), y: a.qkv }, N, C, 3 * C, dummy),
          ...op.attention(g, `h${l} attn`, { qkv: a.qkv, scores: attScratch, probs: a.probs, y: a.attnY }, shape, dummy),
        );
        if (edited) fwd.push(g.dispatch(`h${l} head edit`, 'head_edit', [N, C, H, 0], [this.edits[l], this.savedY[l], a.attnY], gridThreads(N * C)));
        fwd.push(
          op.linear(g, `h${l} proj`, { x: a.attnY, W: w('attn.proj.w'), b: w('attn.proj.b'), y: tmp }, N, C, C, dummy),
          op.residual(g, `h${l} residual 1`, a.x, tmp, drop, a.mid, N * C, siteAttn(l)),
          op.layerNorm(g, `h${l} ln2`, { x: a.mid, g: w('ln2.g'), b: w('ln2.b'), y: a.ln2, mean: a.ln2Mean, rstd: a.ln2Rstd }, N, C),
          op.linear(g, `h${l} fc`, { x: a.ln2, W: w('mlp.fc.w'), b: w('mlp.fc.b'), y: a.fcPre }, N, C, 4 * C, dummy),
          op.gelu(g, `h${l} gelu`, a.fcPre, a.fcAct, N * 4 * C),
          op.linear(g, `h${l} fc2`, { x: a.fcAct, W: w('mlp.proj.w'), b: w('mlp.proj.b'), y: tmp }, N, 4 * C, C, dummy),
          op.residual(g, `h${l} residual 2`, a.mid, tmp, drop, out, N * C, siteMlp(l)),
        );
      }
      fwd.push(
        op.layerNorm(g, 'lnf', { x: xf, g: P(t('lnf.g')), b: P(t('lnf.b')), y: lnf, mean: lnfMean, rstd: lnfRstd }, N, C),
        op.matmul(g, { label: 'head', a: lnf, b: P(t('wte')), c: this.logits, M: N, N: V, K: C, lda: C, ldb: C, ldc: V, transB: true }, dummy),
        op.crossEntropy(g, 'cross-entropy', this.logits, this.targets, this.dlogits, this.losses, N, V),
      );
      return fwd;
    };
    const fwd = buildForward(false);
    this.forwardEdited = buildForward(true);
    const stride = probeStride(cfg);
    const ranges = copyRanges(T, probe.period);
    this.probeScores = [
      ...layers.map((a, l) =>
        g.dispatch(`h${l} induction test`, 'probe_attn', [B, T, H, probe.period, 2 + 2 * H * l, stride, 0, 0], [a.probs, this.probeSlot, this.probeOut], [H, 1, 1]),
      ),
      g.dispatch('copy losses', 'probe_loss', [B, T, ranges.first[0], ranges.first[1], ranges.second[0], ranges.second[1], 0, stride], [this.losses, this.probeSlot, this.probeOut], [1, 1, 1]),
    ];
    this.forward = fwd;

    // ---- backward
    const bwd: Dispatch[] = [];
    bwd.push(
      op.matmul(g, { label: 'head dx', a: this.dlogits, b: P(t('wte')), c: tmp, M: N, N: C, K: V, lda: V, ldb: C, ldc: C }, dummy),
      ...op.matmulTNSplit(g, 'head dW', this.dlogits, lnf, G(t('wte')), parts, N, V, C, dummy),
      ...op.layerNormBackward(
        g,
        'lnf bwd',
        { dy: tmp, x: xf, mean: lnfMean, rstd: lnfRstd, g: P(t('lnf.g')), dx: gres, dg: G(t('lnf.g')), db: G(t('lnf.b')), scratch: parts },
        N,
        C,
        false,
      ),
    );
    for (let l = L - 1; l >= 0; l--) {
      const a = layers[l];
      const w = (n: string) => P(t(`h${l}.${n}`));
      const dw = (n: string) => G(t(`h${l}.${n}`));
      bwd.push(
        op.dropoutMask(g, `h${l} mlp dropout bwd`, gres, drop, dmask, N * C, siteMlp(l)),
        ...op.linearBackward(g, `h${l} fc2 bwd`, { dy: dmask, x: a.fcAct, W: w('mlp.proj.w'), dx: big, dW: dw('mlp.proj.w'), db: dw('mlp.proj.b'), scratch: parts }, N, 4 * C, C, dummy),
        op.geluBackward(g, `h${l} gelu bwd`, a.fcPre, big, N * 4 * C),
        ...op.linearBackward(g, `h${l} fc bwd`, { dy: big, x: a.ln2, W: w('mlp.fc.w'), dx: tmp, dW: dw('mlp.fc.w'), db: dw('mlp.fc.b'), scratch: parts }, N, C, 4 * C, dummy),
        ...op.layerNormBackward(
          g,
          `h${l} ln2 bwd`,
          { dy: tmp, x: a.mid, mean: a.ln2Mean, rstd: a.ln2Rstd, g: w('ln2.g'), dx: gres, dg: dw('ln2.g'), db: dw('ln2.b'), scratch: parts },
          N,
          C,
        ),
        op.dropoutMask(g, `h${l} attn dropout bwd`, gres, drop, dmask, N * C, siteAttn(l)),
        ...op.linearBackward(g, `h${l} proj bwd`, { dy: dmask, x: a.attnY, W: w('attn.proj.w'), dx: tmp2, dW: dw('attn.proj.w'), db: dw('attn.proj.b'), scratch: parts }, N, C, C, dummy),
        ...op.attentionBackward(g, `h${l} attn bwd`, { dy: tmp2, qkv: a.qkv, probs: a.probs, dScores: attScratch, dqkv }, shape, dummy),
        ...op.linearBackward(g, `h${l} qkv bwd`, { dy: dqkv, x: a.ln1, W: w('attn.w'), dx: tmp, dW: dw('attn.w'), db: dw('attn.b'), scratch: parts }, N, C, 3 * C, dummy),
        ...op.layerNormBackward(
          g,
          `h${l} ln1 bwd`,
          { dy: tmp, x: a.x, mean: a.ln1Mean, rstd: a.ln1Rstd, g: w('ln1.g'), dx: gres, dg: dw('ln1.g'), db: dw('ln1.b'), scratch: parts },
          N,
          C,
        ),
      );
    }
    bwd.push(
      op.dropoutMask(g, 'embed dropout bwd', gres, drop, dmask, N * C, SITE_EMBED),
      ...op.embedBackward(g, 'embed bwd', this.inputs, dmask, G(t('wte')), G(t('wpe')), parts, B, T, C, V),
    );
    this.backward = bwd;

    // ---- optimiser
    const perGroup = Math.ceil(layout.total / normGroups);
    this.optimiser = [
      g.dispatch('grad norm partial', 'grad_norm', [layout.total, perGroup, 0, 0], [this.grads, normParts], [normGroups, 1, 1]),
      g.dispatch('grad norm', 'grad_norm_final', [normGroups, STATS.gradNorm, STATS.currentNorm, 0], [normParts, this.trainStep, this.stats], [1, 1, 1]),
      g.dispatch('adamw', 'adamw', [layout.total, layout.decayEnd, STATS.clip, 0], [this.trainStep, this.grads, this.stats, this.params, this.m, this.v], gridThreads(layout.total)),
    ];

    const gatherTrain = g.dispatch('batch', 'gather', [B, T, V, 0], [this.corpus, this.trainStep, this.inputs, this.targets], gridThreads(N));
    const gatherEval = g.dispatch('held-out batch', 'gather', [B, T, V, 0], [this.corpus, this.evalStep, this.inputs, this.targets], gridThreads(N));
    const lossTrain = g.dispatch('loss', 'loss_reduce', [N, STATS.trainLoss, 0, 0], [this.losses, this.trainStep, this.stats], [1, 1, 1]);
    const lossEval = g.dispatch('held-out loss', 'loss_reduce', [N, STATS.valLoss, 0, 0], [this.losses, this.evalStep, this.stats], [1, 1, 1]);
    this.trainProgram = [gatherTrain, ...this.forward, lossTrain, ...this.backward, ...this.optimiser];
    this.evalProgram = [gatherEval, ...this.forward, lossEval];
    g.flushUniforms();
  }

  /** Floating-point work per training step in the matmuls (the other kernels are memory-bound). */
  get flopsPerStep(): number {
    return this.trainProgram.reduce((a, d) => a + d.flops, 0);
  }

  writeTrainStep(header: StepHeader, offsets: Uint32Array): void {
    this.gpu.write(this.trainStep, packStep(header, offsets));
  }

  writeDropout(d: DropoutStep): void {
    const w = new Uint32Array(4);
    w[0] = d.seed;
    w[1] = d.threshold;
    new Float32Array(w.buffer)[2] = d.scale;
    this.gpu.write(this.drop, w);
  }

  writeEvalStep(slot: number, offsets: Uint32Array): void {
    this.writeDropout(NO_DROPOUT);
    this.gpu.write(this.evalStep, packStep({ lr: 0, bc1: 1, bc2: 1, slot, weightDecay: 0, beta1: 0, beta2: 0, eps: 0, clipNorm: 0 }, offsets));
  }

  encode(encoder: GPUCommandEncoder, program: readonly Dispatch[]): void {
    const pass = encoder.beginComputePass();
    this.gpu.encode(pass, program);
    pass.end();
  }

  submit(program: readonly Dispatch[]): void {
    const enc = this.gpu.device.createCommandEncoder();
    this.encode(enc, program);
    this.gpu.device.queue.submit([enc.finish()]);
  }

  /**
   * Runs the induction test on the current weights into result slot `slot`: copies the probe
   * batch into the input buffers, runs the forward pass without dropout, scores it.
   */
  submitProbe(slot: number): void {
    this.writeDropout(NO_DROPOUT);
    this.gpu.write(this.probeSlot, Uint32Array.of(slot));
    const enc = this.gpu.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.probeTokens, 0, this.inputs, 0, this.inputs.size);
    enc.copyBufferToBuffer(this.probeTargets, 0, this.targets, 0, this.targets.size);
    this.encode(enc, [...this.forward, ...this.probeScores]);
    const row = this.cfg.nHead * this.cfg.context * this.cfg.context * 4;
    for (let l = 0; l < this.layers.length; l++) enc.copyBufferToBuffer(this.layers[l].probs, 0, this.probePatterns, l * row, row);
    this.gpu.device.queue.submit([enc.finish()]);
  }

  /** The attention patterns kept from the latest induction test. */
  async readProbePatterns(): Promise<Float32Array> {
    return new Float32Array(await this.gpu.read(this.probePatterns));
  }

  async readProbes(): Promise<Float32Array> {
    return new Float32Array(await this.gpu.read(this.probeOut));
  }

  async readParams(): Promise<Float32Array> {
    return new Float32Array(await this.gpu.read(this.params));
  }

  async readGrads(): Promise<Float32Array> {
    return new Float32Array(await this.gpu.read(this.grads));
  }

  async readStats(): Promise<Float32Array> {
    return new Float32Array(await this.gpu.read(this.stats, 0, STATS.length * 4));
  }
}
