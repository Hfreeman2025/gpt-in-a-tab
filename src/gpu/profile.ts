import type { Dispatch } from './context';
import type { GpuGPT } from './model';

export interface KernelTiming {
  readonly label: string;
  readonly kernel: string;
  /** Mean GPU time of one dispatch, in milliseconds. */
  readonly ms: number;
  readonly flops: number;
}

export interface Profile {
  readonly timings: readonly KernelTiming[];
  /** Sum of the per-dispatch times: the step's kernel time. */
  readonly kernelMs: number;
  /** Wall time of whole steps submitted back to back, for comparison. */
  readonly stepMs: number;
  readonly reps: number;
}

async function timeSubmit(model: GpuGPT, program: readonly Dispatch[], reps: number): Promise<number> {
  const device = model.gpu.device;
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  for (let r = 0; r < reps; r++) model.gpu.encode(pass, program);
  pass.end();
  const t0 = performance.now();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - t0;
}

/**
 * Times every dispatch of the training step by running it `reps` times back to back and taking
 * the wall time of the submission, less the cost of an empty one. The optimiser state is saved
 * first and restored afterwards, so profiling never changes what the model learns.
 */
export async function profileStep(model: GpuGPT, reps = 24): Promise<Profile> {
  const { gpu } = model;
  const device = gpu.device;
  const backup = [model.params, model.m, model.v].map((b) =>
    device.createBuffer({ size: b.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
  );
  const save = device.createCommandEncoder();
  for (const [i, b] of [model.params, model.m, model.v].entries()) save.copyBufferToBuffer(b, 0, backup[i], 0, b.size);
  device.queue.submit([save.finish()]);
  await device.queue.onSubmittedWorkDone();

  try {
    // Warm up, then measure the fixed cost of a submission.
    await timeSubmit(model, model.trainProgram, 1);
    let overhead = Infinity;
    for (let i = 0; i < 5; i++) overhead = Math.min(overhead, await timeSubmit(model, [], 1));

    const stepMs = (await timeSubmit(model, model.trainProgram, 4)) / 4;
    const timings: KernelTiming[] = [];
    for (const d of model.trainProgram) {
      const ms = Math.max(0, (await timeSubmit(model, [d], reps)) - overhead) / reps;
      timings.push({ label: d.label, kernel: d.kernel, ms, flops: d.flops });
    }
    return { timings, kernelMs: timings.reduce((a, t) => a + t.ms, 0), stepMs, reps };
  } finally {
    const restore = device.createCommandEncoder();
    for (const [i, b] of [model.params, model.m, model.v].entries()) restore.copyBufferToBuffer(backup[i], 0, b, 0, b.size);
    device.queue.submit([restore.finish()]);
    await device.queue.onSubmittedWorkDone();
    for (const b of backup) b.destroy();
  }
}

export interface KernelGroup {
  readonly kernel: string;
  readonly calls: number;
  readonly ms: number;
  readonly flops: number;
}

/** Per-kernel totals for one step, largest first. */
export function groupByKernel(p: Profile): KernelGroup[] {
  const m = new Map<string, { calls: number; ms: number; flops: number }>();
  for (const t of p.timings) {
    const g = m.get(t.kernel) ?? { calls: 0, ms: 0, flops: 0 };
    g.calls++;
    g.ms += t.ms;
    g.flops += t.flops;
    m.set(t.kernel, g);
  }
  return [...m.entries()].map(([kernel, g]) => ({ kernel, ...g })).sort((a, b) => b.ms - a.ms);
}
