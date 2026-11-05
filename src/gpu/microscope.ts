import { NO_DROPOUT } from '../model/dropout';
import { copyRanges } from '../model/probe';
import type { Dispatch } from './context';
import type { GpuGPT } from './model';

/** Per head (layer-major): what knocking it out does to the loss. */
export interface Knockouts {
  /** Loss on held-out text and on the second copy of the induction test, with every head working. */
  readonly baseText: number;
  readonly baseCopy: number;
  /** With head i zeroed: the change in each loss (nats a character). */
  readonly text: readonly number[];
  readonly copy: readonly number[];
}

/** Activation patching on the induction test (clean: the run repeated; corrupted: its first copy replaced). */
export interface Patching {
  readonly clean: number;
  readonly corrupt: number;
  /** Per head: the second copy's loss with that head's output patched in from the clean run. */
  readonly patched: readonly number[];
  /** (corrupt − patched) / (corrupt − clean): the share of the copying that head restores. */
  readonly recovered: readonly number[];
}

/**
 * Causal experiments on the GPU, run between training steps (or after the run): the forward pass
 * with per-head edits (head_edit.wgsl), scored by probe_loss into a small results buffer.
 */
export class Microscope {
  private readonly slot: GPUBuffer;
  private readonly out: GPUBuffer;
  private readonly copyLoss: Dispatch;
  private readonly textLoss: Dispatch;
  private readonly heads: number;

  constructor(private readonly model: GpuGPT) {
    const g = model.gpu;
    const { context: T, nLayer: L, nHead: H } = model.cfg;
    this.heads = L * H;
    this.slot = g.buffer('analysis slot', 16, 'microscope');
    this.out = g.floats('analysis results', 2 * (2 * this.heads + 4), 'microscope');
    const r = copyRanges(T, model.probePeriod);
    // Each run writes two numbers into its slot: [first range, second range].
    this.copyLoss = g.dispatch(
      'copy loss',
      'probe_loss',
      [model.B, T, r.first[0], r.first[1], r.second[0], r.second[1], 0, 2],
      [model.losses, this.slot, this.out],
      [1, 1, 1],
    );
    this.textLoss = g.dispatch('text loss', 'probe_loss', [model.B, T, 0, T, 0, T, 0, 2], [model.losses, this.slot, this.out], [1, 1, 1]);
    g.flushUniforms();
  }

  private identity(): Float32Array[] {
    const H = this.model.cfg.nHead;
    return this.model.edits.map(() => {
      const e = new Float32Array(2 * H);
      e.fill(1, 0, H);
      return e;
    });
  }

  /** One forward pass with `edits`, scored into `slot`. */
  private submit(slot: number, input: 'probe' | 'corrupt' | 'text', edits: Float32Array[], save = false): void {
    const m = this.model;
    const g = m.gpu;
    m.writeDropout(NO_DROPOUT);
    for (let l = 0; l < edits.length; l++) g.write(m.edits[l], edits[l]);
    g.write(this.slot, Uint32Array.of(slot));
    const enc = g.device.createCommandEncoder();
    const program: Dispatch[] = [];
    if (input === 'text') program.push(m.evalProgram[0]);
    else {
      enc.copyBufferToBuffer(input === 'probe' ? m.probeTokens : m.probeCorrupt, 0, m.inputs, 0, m.inputs.size);
      enc.copyBufferToBuffer(input === 'probe' ? m.probeTargets : m.probeCorruptTargets, 0, m.targets, 0, m.targets.size);
    }
    program.push(...m.forwardEdited, input === 'text' ? this.textLoss : this.copyLoss);
    m.encode(enc, program);
    if (save) for (let l = 0; l < m.layers.length; l++) enc.copyBufferToBuffer(m.layers[l].attnY, 0, m.savedY[l], 0, m.savedY[l].size);
    g.device.queue.submit([enc.finish()]);
  }

  private async read(): Promise<Float32Array> {
    return new Float32Array(await this.model.gpu.read(this.out));
  }

  private knockedOut(head: number): Float32Array[] {
    const H = this.model.cfg.nHead;
    const e = this.identity();
    e[Math.floor(head / H)][head % H] = 0;
    return e;
  }

  /**
   * Zero-ablation of every head in turn, on a batch of held-out text (offsets for the
   * held-out split, written to the eval step buffer) and on the induction test.
   */
  async knockouts(textOffsets: Uint32Array): Promise<Knockouts> {
    const m = this.model;
    m.writeEvalStep(0, textOffsets);
    const n = this.heads;
    // Slots: 0 baseline text, 1 baseline copy, then per head (text, copy).
    this.submit(0, 'text', this.identity());
    this.submit(1, 'probe', this.identity());
    const text: number[] = [];
    const copy: number[] = [];
    let base = [0, 0];
    for (let chunk = 0; chunk < n; chunk += 4) {
      for (let k = chunk; k < Math.min(n, chunk + 4); k++) {
        const s = 2 + 2 * (k - chunk);
        this.submit(s, 'text', this.knockedOut(k));
        this.submit(s + 1, 'probe', this.knockedOut(k));
      }
      const r = await this.read();
      if (chunk === 0) base = [r[0], r[3]];
      for (let k = chunk; k < Math.min(n, chunk + 4); k++) {
        const s = 2 + 2 * (k - chunk);
        text.push(r[2 * s] - base[0]);
        copy.push(r[2 * (s + 1) + 1] - base[1]);
      }
    }
    this.restore();
    return { baseText: base[0], baseCopy: base[1], text, copy };
  }

  /**
   * Activation patching (as in Olsson et al. 2022 and Meng et al. 2022): run the clean test and
   * keep every head's output; run the corrupted test, whose second copy has nothing earlier to
   * copy from; then rerun it with one head's output taken from the clean run, and measure how
   * much of the copying comes back.
   */
  async patching(): Promise<Patching> {
    const H = this.model.cfg.nHead;
    this.submit(0, 'probe', this.identity(), true);
    this.submit(1, 'corrupt', this.identity());
    const r0 = await this.read();
    const clean = r0[1];
    const corrupt = r0[3];
    const patched: number[] = [];
    for (let chunk = 0; chunk < this.heads; chunk += 8) {
      for (let k = chunk; k < Math.min(this.heads, chunk + 8); k++) {
        const e = this.identity();
        e[Math.floor(k / H)][k % H] = 0;
        e[Math.floor(k / H)][H + (k % H)] = 1;
        this.submit(k - chunk, 'corrupt', e);
      }
      const r = await this.read();
      for (let k = chunk; k < Math.min(this.heads, chunk + 8); k++) patched.push(r[2 * (k - chunk) + 1]);
    }
    this.restore();
    const gap = corrupt - clean;
    return { clean, corrupt, patched, recovered: patched.map((p) => (gap > 1e-6 ? (corrupt - p) / gap : 0)) };
  }

  /** Every head's attention over the first row of the induction test: T×T each, row = query. */
  async patterns(): Promise<Float32Array[]> {
    const m = this.model;
    const { context: T, nHead: H } = m.cfg;
    this.submit(0, 'probe', this.identity());
    const out: Float32Array[] = [];
    for (const layer of m.layers) {
      const all = new Float32Array(await m.gpu.read(layer.probs, 0, H * T * T * 4));
      for (let h = 0; h < H; h++) out.push(all.slice(h * T * T, (h + 1) * T * T));
    }
    return out;
  }

  private restore(): void {
    const id = this.identity();
    for (let l = 0; l < id.length; l++) this.model.gpu.write(this.model.edits[l], id[l]);
  }
}
