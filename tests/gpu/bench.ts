/** Dev benchmark: steps per second and a per-kernel profile for the shipped GPU configuration. */
import { GPU_SHAPE, GPU_TRAIN, withVocab } from '../../src/model/config';
import { initParams, paramLayout } from '../../src/model/layout';
import { Rng } from '../../src/model/rng';
import { Gpu, acquireDevice } from '../../src/gpu/context';
import { GpuGPT, PIPELINE_VARIANTS, RING } from '../../src/gpu/model';
import { groupByKernel, profileStep } from '../../src/gpu/profile';

declare global {
  interface Window {
    __bench?: unknown;
  }
}

const out = document.getElementById('out') as HTMLElement;
const log = (s: string) => (out.textContent += s + '\n');

async function main() {
  const params = new URLSearchParams(location.search);
  const B = Number(params.get('B') ?? GPU_TRAIN.batch);
  const acq = await acquireDevice();
  if ('error' in acq) throw new Error(acq.error);
  const g = new Gpu(acq.device);
  await g.prepare(PIPELINE_VARIANTS);
  const cfg = withVocab(GPU_SHAPE, 80);
  const layout = paramLayout(cfg);
  const rng = new Rng(1);
  const corpus = Uint32Array.from({ length: 200_000 }, () => rng.int(80));
  const model = new GpuGPT(g, cfg, { ...GPU_TRAIN, batch: B }, corpus, initParams(layout, 1), B);
  const header = { lr: 1e-3, bc1: 0.1, bc2: 0.01, slot: 0, weightDecay: 0.1, beta1: 0.9, beta2: 0.99, eps: 1e-8, clipNorm: 1 };
  const offsets = () => Uint32Array.from({ length: B }, () => rng.int(corpus.length - cfg.context - 2));

  const steps = 60;
  for (let i = 0; i < 5; i++) {
    model.writeTrainStep({ ...header, slot: i }, offsets());
    model.submit(model.trainProgram);
  }
  await acq.device.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  for (let i = 0; i < steps; i++) {
    model.writeTrainStep({ ...header, slot: i % RING }, offsets());
    model.submit(model.trainProgram);
  }
  await acq.device.queue.onSubmittedWorkDone();
  const ms = (performance.now() - t0) / steps;
  const tokS = (B * cfg.context * 1000) / ms;
  log(`params ${layout.count.toLocaleString()}, B=${B} T=${cfg.context}, dispatches/step ${model.trainProgram.length}`);
  log(`${ms.toFixed(2)} ms/step, ${(1000 / ms).toFixed(1)} steps/s, ${Math.round(tokS).toLocaleString()} tok/s`);
  log(`matmul GFLOP/step ${(model.flopsPerStep / 1e9).toFixed(2)}, effective ${(model.flopsPerStep / ms / 1e6).toFixed(0)} GFLOP/s`);
  log(`buffers ${(g.totalMemory() / 2 ** 20).toFixed(1)} MiB`);

  if (params.has('ab')) {
    const n = 120;
    const plain = async () => {
      const t1 = performance.now();
      for (let i = 0; i < n; i++) {
        model.writeTrainStep({ ...header, slot: i % RING }, offsets());
        model.submit(model.trainProgram);
      }
      await acq.device.queue.onSubmittedWorkDone();
      return (performance.now() - t1) / n;
    };
    const looped = async (depth: number) => {
      const t1 = performance.now();
      const inflight: Promise<unknown>[] = [];
      let waited = 0;
      for (let i = 0; i < n; i++) {
        model.writeTrainStep({ ...header, slot: i % RING }, offsets());
        model.submit(model.trainProgram);
        inflight.push(acq.device.queue.onSubmittedWorkDone());
        if (inflight.length >= depth) {
          const w = performance.now();
          await inflight.shift();
          waited += performance.now() - w;
        }
      }
      await Promise.all(inflight);
      return [(performance.now() - t1) / n, waited / n];
    };
    for (let r = 0; r < 3; r++) {
      const a = await plain();
      const [b, wb] = await looped(3);
      const [c, wc] = await looped(8);
      log(`round ${r}: plain ${a.toFixed(2)} ms/step | loop depth 3 ${b.toFixed(2)} (waiting ${wb.toFixed(2)}) | depth 8 ${c.toFixed(2)} (waiting ${wc.toFixed(2)})`);
    }
    window.__bench = { done: true };
    return;
  }

  // Variants of the app's loop: await pattern, periodic stats reads, weight snapshots.
  if (params.has('loop')) {
    const inflightMax = Number(params.get('inflight') ?? 3);
    const variants: [string, boolean, boolean][] = [
      ['await only', false, false],
      ['+ stats reads every 120 ms', true, false],
      ['+ snapshot every 40 steps', true, true],
    ];
    for (const [name, reads, snaps] of variants) {
      const n = 240;
      const inflight: Promise<unknown>[] = [];
      let lastRead = performance.now();
      const t1 = performance.now();
      for (let i = 0; i < n; i++) {
        model.writeTrainStep({ ...header, slot: i % RING }, offsets());
        model.submit(model.trainProgram);
        inflight.push(acq.device.queue.onSubmittedWorkDone());
        if (inflight.length >= inflightMax) await inflight.shift();
        if (reads && performance.now() - lastRead > 120) {
          lastRead = performance.now();
          void model.readStats();
        }
        if (snaps && i % 40 === 0) void model.readParams();
      }
      await Promise.all(inflight);
      await acq.device.queue.onSubmittedWorkDone();
      const per = (performance.now() - t1) / n;
      log(`loop (${name}, in flight ${inflightMax}): ${per.toFixed(2)} ms/step, ${Math.round((B * cfg.context * 1000) / per).toLocaleString()} tok/s`);
    }
    window.__bench = { done: true };
    return;
  }

  const prof = await profileStep(model);
  log(`\nprofile: kernels ${prof.kernelMs.toFixed(2)} ms, step ${prof.stepMs.toFixed(2)} ms`);
  for (const k of groupByKernel(prof)) {
    const gf = k.flops ? `  ${(k.flops / k.ms / 1e6).toFixed(0)} GFLOP/s` : '';
    log(`${k.kernel.padEnd(22)} ×${String(k.calls).padStart(3)}  ${k.ms.toFixed(3).padStart(7)} ms  ${((100 * k.ms) / prof.kernelMs).toFixed(1).padStart(5)}%${gf}`);
  }
  const top = [...prof.timings].sort((a, b) => b.ms - a.ms).slice(0, 14);
  log('\nslowest dispatches:');
  for (const t of top) log(`${t.label.padEnd(36)} ${t.ms.toFixed(3)} ms${t.flops ? `  ${(t.flops / t.ms / 1e6).toFixed(0)} GFLOP/s` : ''}`);
  window.__bench = { ms, tokS, done: true };
}

main().catch((e) => {
  log(String(e));
  window.__bench = { error: String(e), done: true };
});
