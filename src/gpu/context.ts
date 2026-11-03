import adamw from './wgsl/adamw.wgsl?raw';
import colsum from './wgsl/colsum.wgsl?raw';
import common from './wgsl/common.wgsl?raw';
import crossEntropy from './wgsl/cross_entropy.wgsl?raw';
import dropout from './wgsl/dropout.wgsl?raw';
import embed from './wgsl/embed.wgsl?raw';
import embedBwdWpe from './wgsl/embed_bwd_wpe.wgsl?raw';
import embedBwdWte from './wgsl/embed_bwd_wte.wgsl?raw';
import gather from './wgsl/gather.wgsl?raw';
import gelu from './wgsl/gelu.wgsl?raw';
import geluBwd from './wgsl/gelu_bwd.wgsl?raw';
import gradNorm from './wgsl/grad_norm.wgsl?raw';
import gradNormFinal from './wgsl/grad_norm_final.wgsl?raw';
import headEdit from './wgsl/head_edit.wgsl?raw';
import layernorm from './wgsl/layernorm.wgsl?raw';
import layernormBwd from './wgsl/layernorm_bwd.wgsl?raw';
import layernormBwdParams from './wgsl/layernorm_bwd_params.wgsl?raw';
import lossReduce from './wgsl/loss_reduce.wgsl?raw';
import matmul from './wgsl/matmul.wgsl?raw';
import probeAttn from './wgsl/probe_attn.wgsl?raw';
import probeLoss from './wgsl/probe_loss.wgsl?raw';
import reduce from './wgsl/reduce.wgsl?raw';
import residual from './wgsl/residual.wgsl?raw';
import saeErr from './wgsl/sae_err.wgsl?raw';
import saeGrad from './wgsl/sae_grad.wgsl?raw';
import saeNorm from './wgsl/sae_norm.wgsl?raw';
import saeRelu from './wgsl/sae_relu.wgsl?raw';
import saeRows from './wgsl/sae_rows.wgsl?raw';
import saeStats from './wgsl/sae_stats.wgsl?raw';
import softmax from './wgsl/softmax.wgsl?raw';
import softmaxBwd from './wgsl/softmax_bwd.wgsl?raw';

/** u = uniform, r = read-only storage, w = read-write storage. Binding 0 is always the uniform. */
type Slot = 'u' | 'r' | 'w';

const KERNELS = {
  matmul: { src: matmul, slots: 'urrwr' },
  reduce: { src: reduce, slots: 'urw' },
  colsum: { src: colsum, slots: 'urw' },
  layernorm: { src: layernorm, slots: 'urrrwww' },
  layernorm_bwd: { src: layernormBwd, slots: 'urrrrrw' },
  layernorm_bwd_params: { src: layernormBwdParams, slots: 'urrrrw' },
  softmax: { src: softmax, slots: 'urw' },
  softmax_bwd: { src: softmaxBwd, slots: 'urw' },
  gelu: { src: gelu, slots: 'urw' },
  gelu_bwd: { src: geluBwd, slots: 'urw' },
  residual: { src: common + residual, slots: 'urrrw' },
  dropout: { src: common + dropout, slots: 'urrw' },
  embed: { src: common + embed, slots: 'urrrrw' },
  embed_bwd_wpe: { src: embedBwdWpe, slots: 'urw' },
  embed_bwd_wte: { src: embedBwdWte, slots: 'urrw' },
  gather: { src: common + gather, slots: 'urrww' },
  cross_entropy: { src: crossEntropy, slots: 'urrww' },
  loss_reduce: { src: lossReduce, slots: 'urrw' },
  grad_norm: { src: gradNorm, slots: 'urw' },
  grad_norm_final: { src: gradNormFinal, slots: 'urrw' },
  adamw: { src: adamw, slots: 'urrrwww' },
  probe_attn: { src: probeAttn, slots: 'urrw' },
  probe_loss: { src: probeLoss, slots: 'urrw' },
  head_edit: { src: headEdit, slots: 'urrw' },
  sae_rows: { src: common + saeRows, slots: 'urrw' },
  sae_relu: { src: saeRelu, slots: 'uw' },
  sae_grad: { src: saeGrad, slots: 'urrw' },
  sae_norm: { src: saeNorm, slots: 'uw' },
  sae_stats: { src: saeStats, slots: 'urwww' },
  sae_err: { src: saeErr, slots: 'urrw' },
} as const;

export type KernelName = keyof typeof KERNELS;
export const KERNEL_NAMES = Object.keys(KERNELS) as KernelName[];
export const KERNEL_SOURCES: Readonly<Record<KernelName, string>> = Object.fromEntries(
  KERNEL_NAMES.map((k) => [k, KERNELS[k].src]),
) as Record<KernelName, string>;

export type Constants = Readonly<Record<string, boolean>>;

/** A buffer, or a 256-byte-aligned sub-range of one. */
export type Ref = GPUBuffer | { readonly buffer: GPUBuffer; readonly offset: number; readonly size: number };

export interface Dispatch {
  readonly label: string;
  readonly kernel: KernelName;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly grid: readonly [number, number, number];
  /** Floating-point operations, for the matmul throughput readout (0 when not meaningful). */
  readonly flops: number;
}

export type MemoryCategory = 'parameters' | 'gradients' | 'optimiser state' | 'activations' | 'scratch' | 'corpus' | 'microscope' | 'other';

/** Uniform fields: a plain number is written as u32, {f} as f32. */
export type UniformField = number | { readonly f: number };

const UNIFORM_SLOT = 256;
const UNIFORM_CHUNK = 256 * UNIFORM_SLOT;

export const MAX_GROUPS_X = 32768;

/** Workgroup grid for `count` workgroups, folded into two dimensions past the 65 535 limit. */
export function grid(count: number): [number, number, number] {
  if (count <= 65535) return [Math.max(1, count), 1, 1];
  return [MAX_GROUPS_X, Math.ceil(count / MAX_GROUPS_X), 1];
}

export function gridThreads(threads: number, perGroup = 256): [number, number, number] {
  return grid(Math.ceil(threads / perGroup));
}

function toBinding(ref: Ref): GPUBufferBinding {
  return ref instanceof GPUBuffer ? { buffer: ref } : { buffer: ref.buffer, offset: ref.offset, size: ref.size };
}

/**
 * Owns the device-side plumbing: shader modules, pipelines (one per kernel and constant set),
 * explicit bind-group layouts, a uniform arena, and a running tally of buffer memory.
 */
export class Gpu {
  private readonly modules = new Map<KernelName, GPUShaderModule>();
  private readonly layouts = new Map<KernelName, GPUBindGroupLayout>();
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private readonly chunks: { buffer: GPUBuffer; data: Uint8Array; used: number; dirty: boolean }[] = [];
  private readonly owned: GPUBuffer[] = [];
  readonly memory = new Map<MemoryCategory, number>();

  constructor(readonly device: GPUDevice) {}

  private module(kernel: KernelName): GPUShaderModule {
    let m = this.modules.get(kernel);
    if (!m) {
      m = this.device.createShaderModule({ label: kernel, code: KERNELS[kernel].src });
      this.modules.set(kernel, m);
    }
    return m;
  }

  private layout(kernel: KernelName): GPUBindGroupLayout {
    let l = this.layouts.get(kernel);
    if (!l) {
      const slots = KERNELS[kernel].slots.split('') as Slot[];
      l = this.device.createBindGroupLayout({
        label: kernel,
        entries: slots.map((s, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: s === 'u' ? 'uniform' : s === 'r' ? 'read-only-storage' : 'storage' },
        })),
      });
      this.layouts.set(kernel, l);
    }
    return l;
  }

  private key(kernel: KernelName, constants: Constants): string {
    const c = Object.keys(constants)
      .sort()
      .map((k) => `${k}=${constants[k] ? 1 : 0}`)
      .join(',');
    return `${kernel}|${c}`;
  }

  private descriptor(kernel: KernelName, constants: Constants): GPUComputePipelineDescriptor {
    const values: Record<string, number> = {};
    for (const [k, v] of Object.entries(constants)) values[k] = v ? 1 : 0;
    return {
      label: this.key(kernel, constants),
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout(kernel)] }),
      compute: { module: this.module(kernel), entryPoint: 'main', constants: values },
    };
  }

  /** Compile pipelines ahead of time, in parallel, so building the graph never stalls. */
  async prepare(variants: readonly (readonly [KernelName, Constants])[]): Promise<void> {
    await Promise.all(
      variants.map(async ([kernel, constants]) => {
        const k = this.key(kernel, constants);
        if (this.pipelines.has(k)) return;
        const p = await this.device.createComputePipelineAsync(this.descriptor(kernel, constants));
        this.pipelines.set(k, p);
      }),
    );
  }

  pipeline(kernel: KernelName, constants: Constants = {}): GPUComputePipeline {
    const k = this.key(kernel, constants);
    let p = this.pipelines.get(k);
    if (!p) {
      p = this.device.createComputePipeline(this.descriptor(kernel, constants));
      this.pipelines.set(k, p);
    }
    return p;
  }

  buffer(label: string, bytes: number, category: MemoryCategory, extraUsage = 0): GPUBuffer {
    const size = Math.max(16, Math.ceil(bytes / 16) * 16);
    const b = this.device.createBuffer({
      label,
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extraUsage,
    });
    this.owned.push(b);
    this.memory.set(category, (this.memory.get(category) ?? 0) + size);
    return b;
  }

  floats(label: string, n: number, category: MemoryCategory): GPUBuffer {
    return this.buffer(label, n * 4, category);
  }

  upload(label: string, data: Float32Array | Uint32Array, category: MemoryCategory): GPUBuffer {
    const b = this.buffer(label, data.byteLength, category);
    this.write(b, data);
    return b;
  }

  write(buffer: GPUBuffer, data: Float32Array | Uint32Array, byteOffset = 0): void {
    this.device.queue.writeBuffer(buffer, byteOffset, data.buffer, data.byteOffset, data.byteLength);
  }

  /** Reserves a 256-byte uniform slot. Call flushUniforms() before the first submit. */
  uniform(fields: readonly UniformField[]): GPUBufferBinding {
    const bytes = Math.ceil((fields.length * 4) / 16) * 16;
    if (bytes > UNIFORM_SLOT) throw new Error('uniform block too large');
    let chunk = this.chunks[this.chunks.length - 1];
    if (!chunk || chunk.used + UNIFORM_SLOT > UNIFORM_CHUNK) {
      const buffer = this.device.createBuffer({ label: 'uniforms', size: UNIFORM_CHUNK, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.owned.push(buffer);
      this.memory.set('other', (this.memory.get('other') ?? 0) + UNIFORM_CHUNK);
      chunk = { buffer, data: new Uint8Array(UNIFORM_CHUNK), used: 0, dirty: false };
      this.chunks.push(chunk);
    }
    const offset = chunk.used;
    const view = new DataView(chunk.data.buffer, offset, bytes);
    fields.forEach((f, i) => {
      if (typeof f === 'number') {
        if (!Number.isInteger(f) || f < 0 || f > 0xffffffff) throw new Error(`uniform field ${i} is not a u32: ${f}`);
        view.setUint32(i * 4, f, true);
      } else view.setFloat32(i * 4, f.f, true);
    });
    chunk.used += UNIFORM_SLOT;
    chunk.dirty = true;
    return { buffer: chunk.buffer, offset, size: bytes };
  }

  flushUniforms(): void {
    for (const c of this.chunks) {
      if (!c.dirty) continue;
      this.device.queue.writeBuffer(c.buffer, 0, c.data, 0, c.used);
      c.dirty = false;
    }
  }

  dispatch(
    label: string,
    kernel: KernelName,
    uniform: readonly UniformField[],
    refs: readonly Ref[],
    workgroups: readonly [number, number, number],
    opts: { constants?: Constants; flops?: number } = {},
  ): Dispatch {
    const slots = KERNELS[kernel].slots;
    if (refs.length !== slots.length - 1) throw new Error(`${label}: ${kernel} takes ${slots.length - 1} buffers, got ${refs.length}`);
    const constants = opts.constants ?? {};
    const pipeline = this.pipeline(kernel, constants);
    const bindGroup = this.device.createBindGroup({
      label,
      layout: this.layout(kernel),
      entries: [{ binding: 0, resource: this.uniform(uniform) }, ...refs.map((r, i) => ({ binding: i + 1, resource: toBinding(r) }))],
    });
    return { label, kernel, pipeline, bindGroup, grid: workgroups, flops: opts.flops ?? 0 };
  }

  encode(pass: GPUComputePassEncoder, program: readonly Dispatch[]): void {
    for (const d of program) {
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bindGroup);
      pass.dispatchWorkgroups(d.grid[0], d.grid[1], d.grid[2]);
    }
  }

  run(program: readonly Dispatch[]): void {
    this.flushUniforms();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.encode(pass, program);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  async read(src: GPUBuffer, byteOffset = 0, byteLength = src.size - byteOffset): Promise<ArrayBuffer> {
    const staging = this.device.createBuffer({ size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, byteOffset, staging, 0, byteLength);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return out;
  }

  async readFloats(src: Ref, count?: number): Promise<Float32Array> {
    const b = src instanceof GPUBuffer ? { buffer: src, offset: 0, size: src.size } : src;
    const len = count === undefined ? b.size : count * 4;
    return new Float32Array(await this.read(b.buffer, b.offset, len));
  }

  totalMemory(): number {
    let t = 0;
    for (const v of this.memory.values()) t += v;
    return t;
  }

  destroy(): void {
    for (const b of this.owned) b.destroy();
    this.owned.length = 0;
    this.memory.clear();
  }
}

/** A sub-range of a buffer, in floats. Offsets must be multiples of 64 floats (256 bytes). */
export function slice(buffer: GPUBuffer, offsetFloats: number, sizeFloats: number): Ref {
  if (offsetFloats % 64 !== 0) throw new Error('sub-range offset must be 256-byte aligned');
  return { buffer, offset: offsetFloats * 4, size: Math.max(4, sizeFloats * 4) };
}

export interface Acquired {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly description: string;
}

/** Ask for the high-performance adapter with the largest buffers it allows. Null if WebGPU is missing. */
export async function acquireDevice(): Promise<Acquired | { readonly error: string }> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu) {
    return { error: 'This browser does not expose WebGPU (navigator.gpu is missing).' };
  }
  let adapter: GPUAdapter | null = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    return { error: `WebGPU refused an adapter: ${String(e)}` };
  }
  if (!adapter) return { error: 'WebGPU is present but no GPU adapter is available (it may be disabled or blocklisted).' };
  const l = adapter.limits;
  try {
    const device = await adapter.requestDevice({
      label: 'gpt-in-a-tab',
      requiredLimits: {
        maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
        maxBufferSize: l.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(8, l.maxStorageBuffersPerShaderStage),
      },
    });
    const info = adapter.info;
    const description = [info.vendor, info.architecture, info.description].filter((s) => s && s.length > 0).join(' · ') || 'unknown GPU';
    return { adapter, device, description };
  } catch (e) {
    return { error: `Could not open the GPU device: ${String(e)}` };
  }
}
