/**
 * GPU ↔ CPU parity suite. Runs in a real browser with WebGPU (see scripts/gpu-parity.mjs).
 * Every kernel is run on random inputs and compared with the plain TypeScript reference in
 * src/cpu/ops.ts; then a whole training step is compared tensor by tensor.
 *
 * Error metric: max |gpu − cpu| divided by max |cpu| over the output tensor. Both sides are
 * float32 and sum in different orders, so agreement to ~1e-6 is typical; the stated
 * tolerance is 1e-4 for single kernels and 2e-3 for a full step.
 */
import * as cpu from '../../src/cpu/ops';
import { CpuGPT } from '../../src/cpu/gpt';
import { CpuTrainer } from '../../src/cpu/trainer';
import { GPU_SHAPE, GPU_TRAIN, withVocab } from '../../src/model/config';
import { gatherBatch, sampleOffsets, splitCorpus } from '../../src/model/data';
import { type DropoutStep, dropoutStep } from '../../src/model/dropout';
import { initParams, paramLayout } from '../../src/model/layout';
import { probeBatch, scoreProbe } from '../../src/model/probe';
import { Rng } from '../../src/model/rng';
import { buildTokenizer } from '../../src/model/tokenizer';
import { Gpu, type Ref, acquireDevice } from '../../src/gpu/context';
import { PIPELINE_VARIANTS, STATS } from '../../src/gpu/model';
import { Microscope } from '../../src/gpu/microscope';
import { GpuSae } from '../../src/gpu/sae';
import { normaliseDecoder, saeForward, saeLayout, saeLoss } from '../../src/cpu/sae';
import { hash32 } from '../../src/model/dropout';
import { GpuTrainer } from '../../src/gpu/trainer';
import * as op from '../../src/gpu/ops';

interface Result {
  name: string;
  error: number;
  tol: number;
  pass: boolean;
  ms: number;
  note?: string;
}

declare global {
  interface Window {
    __parity?: { done: boolean; adapter: string; results: Result[]; fatal?: string };
  }
}

const results: Result[] = [];
const state = { done: false, adapter: '', results };
window.__parity = state;

const out = document.getElementById('out') as HTMLElement;
function log(line: string) {
  out.textContent += line + '\n';
}

function relError(gpu: ArrayLike<number>, ref: ArrayLike<number>): number {
  if (gpu.length < ref.length) return Infinity;
  let worst = 0;
  let scale = 1e-12;
  for (let i = 0; i < ref.length; i++) {
    const d = Math.abs(gpu[i] - ref[i]);
    if (!(d <= worst)) worst = Number.isNaN(d) ? Infinity : Math.max(worst, d);
    scale = Math.max(scale, Math.abs(ref[i]));
  }
  return worst / scale;
}

function rand(n: number, rng: Rng, scale = 1): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = rng.normal() * scale;
  return a;
}

async function test(name: string, fn: () => Promise<[ArrayLike<number>, ArrayLike<number>][] | number>, tol = 1e-4, note?: string) {
  const t0 = performance.now();
  try {
    const r = await fn();
    const error = typeof r === 'number' ? r : Math.max(...r.map(([a, b]) => relError(a, b)));
    const res: Result = { name, error, tol, pass: error <= tol, ms: performance.now() - t0, ...(note ? { note } : {}) };
    results.push(res);
    log(`${res.pass ? 'pass' : 'FAIL'}  ${name.padEnd(48)} ${error.toExponential(2)}  (tol ${tol.toExponential(0)})`);
  } catch (e) {
    results.push({ name, error: Infinity, tol, pass: false, ms: performance.now() - t0, note: String(e) });
    log(`FAIL  ${name}  threw ${String(e)}`);
  }
}

async function main() {
  const acq = await acquireDevice();
  if ('error' in acq) {
    state.adapter = 'none';
    (state as { fatal?: string }).fatal = acq.error;
    log(acq.error);
    state.done = true;
    return;
  }
  state.adapter = acq.description;
  log(`adapter: ${acq.description}\n`);
  const device = acq.device;
  device.addEventListener('uncapturederror', (e) => log(`GPU error: ${(e as GPUUncapturedErrorEvent).error.message}`));
  const g = new Gpu(device);
  await g.prepare(PIPELINE_VARIANTS);
  const dummy = g.floats('dummy', 4, 'scratch');
  const rng = new Rng(2024);
  const up = (a: Float32Array | Uint32Array) => g.upload('t', a, 'other');
  const zeros = (n: number) => g.floats('z', n, 'other');
  const dropBuf = (d: DropoutStep) => {
    const w = new Uint32Array(4);
    w[0] = d.seed;
    w[1] = d.threshold;
    new Float32Array(w.buffer)[2] = d.scale;
    return up(w);
  };
  const run = async (program: Parameters<Gpu['run']>[0], ...reads: Ref[]) => {
    g.run(program);
    return Promise.all(reads.map((r) => g.readFloats(r)));
  };

  // ---------------------------------------------------------------- matmul, all variants
  const [M, K, N] = [67, 45, 70];
  for (const transA of [false, true])
    for (const transB of [false, true])
      for (const bias of [false, true]) {
        await test(`matmul ${transA ? 'T' : 'N'}${transB ? 'T' : 'N'}${bias ? ' + bias' : ''} (67×45×70)`, async () => {
          const a = rand(M * K, rng);
          const b = rand(K * N, rng);
          const bv = rand(N, rng);
          const ref = new Float32Array(M * N);
          for (let i = 0; i < M; i++)
            for (let j = 0; j < N; j++) {
              let s = bias ? bv[j] : 0;
              for (let k = 0; k < K; k++) s += (transA ? a[k * M + i] : a[i * K + k]) * (transB ? b[j * K + k] : b[k * N + j]);
              ref[i * N + j] = s;
            }
          const c = zeros(M * N);
          const d = op.matmul(
            g,
            {
              label: 'mm',
              a: up(a),
              b: up(b),
              c,
              ...(bias ? { bias: up(bv) } : {}),
              M,
              N,
              K,
              lda: transA ? M : K,
              ldb: transB ? K : N,
              ldc: N,
              transA,
              transB,
            },
            dummy,
          );
          const [res] = await run([d], c);
          return [[res, ref]];
        });
      }

  await test('matmul accumulate, alpha, batched strides', async () => {
    // Two batches of 3 heads each, operands interleaved like packed qkv rows.
    const [nb1, nb2, m, n, k] = [2, 3, 20, 9, 13];
    const lda = nb2 * k + 5;
    const a = rand(nb1 * m * lda, rng);
    const b = rand(nb1 * nb2 * k * n, rng);
    const c0 = rand(nb1 * nb2 * m * n, rng);
    const ref = Float32Array.from(c0);
    for (let z1 = 0; z1 < nb1; z1++)
      for (let z2 = 0; z2 < nb2; z2++)
        for (let i = 0; i < m; i++)
          for (let j = 0; j < n; j++) {
            let s = 0;
            for (let kk = 0; kk < k; kk++) s += a[z1 * m * lda + z2 * k + i * lda + kk] * b[(z1 * nb2 + z2) * k * n + kk * n + j];
            ref[(z1 * nb2 + z2) * m * n + i * n + j] += 0.5 * s;
          }
    const c = up(c0);
    const d = op.matmul(
      g,
      {
        label: 'mm',
        a: up(a),
        b: up(b),
        c,
        M: m,
        N: n,
        K: k,
        lda,
        ldb: n,
        ldc: n,
        accum: true,
        alpha: 0.5,
        batch: { nb1, nb2, aS1: m * lda, aS2: k, bS1: nb2 * k * n, bS2: k * n, cS1: nb2 * m * n, cS2: m * n },
      },
      dummy,
    );
    const [res] = await run([d], c);
    return [[res, ref]];
  });

  await test('split-K weight gradient xᵀ·dy (4096×128 → 128×384)', async () => {
    const [m, k, n] = [4096, 128, 384];
    const x = rand(m * k, rng);
    const dy = rand(m * n, rng);
    const ref = new Float32Array(k * n);
    cpu.matmulTN(ref, x, dy, m, k, n);
    const outB = zeros(k * n);
    const parts = zeros(op.scratchForTN(m, k, n));
    const [res] = await run(op.matmulTNSplit(g, 'tn', up(x), up(dy), outB, parts, m, k, n, dummy), outB);
    return [[res, ref]];
  });

  await test('column sums for bias gradients (4096×512)', async () => {
    const [rows, cols] = [4096, 512];
    const x = rand(rows * cols, rng);
    const ref = new Float32Array(cols);
    cpu.colSum(ref, x, rows, cols);
    const outB = zeros(cols);
    const [res] = await run(op.colSum(g, 'cs', up(x), outB, zeros(op.scratchForColSum(rows, cols)), rows, cols), outB);
    return [[res, ref]];
  });

  // ---------------------------------------------------------------- layer norm
  for (const C of [128, 100]) {
    const Nr = 300;
    await test(`layer norm forward (C=${C})`, async () => {
      const x = rand(Nr * C, rng, 3);
      const gm = rand(C, rng);
      const bt = rand(C, rng);
      const y = new Float32Array(Nr * C);
      const mean = new Float32Array(Nr);
      const rstd = new Float32Array(Nr);
      cpu.layerNorm(y, mean, rstd, x, gm, bt, Nr, C);
      const yb = zeros(Nr * C);
      const mb = zeros(Nr);
      const rb = zeros(Nr);
      const [ry, rm, rr] = await run([op.layerNorm(g, 'ln', { x: up(x), g: up(gm), b: up(bt), y: yb, mean: mb, rstd: rb }, Nr, C)], yb, mb, rb);
      return [
        [ry, y],
        [rm, mean],
        [rr, rstd],
      ];
    });
    await test(`layer norm backward, dx accumulates (C=${C})`, async () => {
      const x = rand(Nr * C, rng, 2);
      const gm = rand(C, rng);
      const bt = rand(C, rng);
      const dy = rand(Nr * C, rng);
      const y = new Float32Array(Nr * C);
      const mean = new Float32Array(Nr);
      const rstd = new Float32Array(Nr);
      cpu.layerNorm(y, mean, rstd, x, gm, bt, Nr, C);
      const dx0 = rand(Nr * C, rng);
      const dx = Float32Array.from(dx0);
      const dg = new Float32Array(C);
      const db = new Float32Array(C);
      cpu.layerNormBackward(dx, dg, db, dy, x, mean, rstd, gm, Nr, C);
      const dxb = up(dx0);
      const dgb = zeros(C);
      const dbb = zeros(C);
      const prog = op.layerNormBackward(
        g,
        'lnb',
        { dy: up(dy), x: up(x), mean: up(mean), rstd: up(rstd), g: up(gm), dx: dxb, dg: dgb, db: dbb, scratch: zeros(op.scratchForLayerNorm(Nr, C)) },
        Nr,
        C,
      );
      const [rx, rg, rb] = await run(prog, dxb, dgb, dbb);
      return [
        [rx, dx],
        [rg, dg],
        [rb, db],
      ];
    });
  }

  // ---------------------------------------------------------------- softmax, GELU, embeddings, loss
  for (const T of [128, 37]) {
    await test(`causal softmax forward and backward (T=${T})`, async () => {
      const rows = 8 * T;
      const x = rand(rows * T, rng, 3);
      const dy = rand(rows * T, rng);
      const y = new Float32Array(rows * T);
      cpu.causalSoftmax(y, x, rows, T);
      const dx = new Float32Array(rows * T);
      cpu.causalSoftmaxBackward(dx, dy, y, rows, T);
      const yb = zeros(rows * T);
      const db = up(dy);
      const [ry, rd] = await run([op.softmax(g, 'sm', up(x), yb, rows, T), op.softmaxBackward(g, 'smb', yb, db, rows, T)], yb, db);
      return [
        [ry, y],
        [rd, dx],
      ];
    });
  }

  await test('GELU forward and backward, including |x| = 40', async () => {
    const n = 5000;
    const x = rand(n, rng, 4);
    x[0] = 40;
    x[1] = -40;
    const dy = rand(n, rng);
    const y = new Float32Array(n);
    cpu.gelu(y, x, n);
    const dx = new Float32Array(n);
    cpu.geluBackward(dx, dy, x, n);
    const xb = up(x);
    const yb = zeros(n);
    const db = up(dy);
    const [ry, rd] = await run([op.gelu(g, 'gelu', xb, yb, n), op.geluBackward(g, 'gb', xb, db, n)], yb, db);
    return [
      [ry, y],
      [rd, dx],
    ];
  });

  await test('embeddings with dropout, forward and backward (tied, accumulating)', async () => {
    const [B, T, C, V] = [8, 128, 128, 65];
    const tokens = Uint32Array.from({ length: B * T }, () => rng.int(V));
    const wte = rand(V * C, rng);
    const wpe = rand(T * C, rng);
    const x = new Float32Array(B * T * C);
    const drop = dropoutStep(0.2, 31337);
    cpu.embed(x, tokens, wte, wpe, B, T, C);
    cpu.dropout(x, x, x.length, drop, 0);
    const dout = rand(B * T * C, rng);
    const dwte0 = rand(V * C, rng);
    const dwte = Float32Array.from(dwte0);
    const dwpe = new Float32Array(T * C);
    cpu.embedBackward(dwte, dwpe, dout, tokens, B, T, C);
    const tb = up(tokens);
    const xb = zeros(B * T * C);
    const dwteB = up(dwte0);
    const dwpeB = zeros(T * C);
    const prog = [
      op.embed(g, 'emb', tb, up(wte), up(wpe), dropBuf(drop), xb, B * T, T, C, 0),
      ...op.embedBackward(g, 'embb', tb, up(dout), dwteB, dwpeB, zeros(op.scratchForEmbed(B * T, V, C)), B, T, C, V),
    ];
    const [rx, rte, rpe] = await run(prog, xb, dwteB, dwpeB);
    return [
      [rx, x],
      [rte, dwte],
      [rpe, dwpe],
    ];
  });

  await test('residual with dropout, and the dropout mask kernel', async () => {
    const n = 70000;
    const a = rand(n, rng);
    const b = rand(n, rng);
    const drop = dropoutStep(0.1, 4242);
    const masked = new Float32Array(n);
    cpu.dropout(masked, b, n, drop, 5);
    const y = new Float32Array(n);
    cpu.add(y, a, masked, n);
    const db = dropBuf(drop);
    const yb = zeros(n);
    const mb = zeros(n);
    const [ry, rm] = await run([op.residual(g, 'res', up(a), up(b), db, yb, n, 5), op.dropoutMask(g, 'mask', up(b), db, mb, n, 5)], yb, mb);
    return [
      [ry, y],
      [rm, masked],
    ];
  });

  await test('cross-entropy forward and backward', async () => {
    const [Nn, V] = [500, 65];
    const logits = rand(Nn * V, rng, 3);
    const targets = Uint32Array.from({ length: Nn }, () => rng.int(V));
    const losses = new Float32Array(Nn);
    const d = new Float32Array(Nn * V);
    cpu.crossEntropy(losses, d, logits, targets, Nn, V);
    const db = zeros(Nn * V);
    const lb = zeros(Nn);
    const [rd, rl] = await run([op.crossEntropy(g, 'ce', up(logits), up(targets), db, lb, Nn, V)], db, lb);
    return [
      [rd, d],
      [rl, losses],
    ];
  });

  await test('attention forward and backward (B=3, T=40, C=64, H=4)', async () => attentionParity(3, 40, 64, 4));
  await test('attention forward and backward (B=2, T=128, C=128, H=4)', async () => attentionParity(2, 128, 128, 4));

  async function attentionParity(B: number, T: number, C: number, H: number) {
    const s = { B, T, C, H };
    const qkv = rand(B * T * 3 * C, rng);
    const dy = rand(B * T * C, rng);
    const y = new Float32Array(B * T * C);
    const probs = new Float32Array(B * H * T * T);
    cpu.attention(y, probs, new Float32Array(probs.length), qkv, s);
    const dqkv = new Float32Array(qkv.length);
    cpu.attentionBackward(dqkv, new Float32Array(probs.length), dy, qkv, probs, s);
    const qb = up(qkv);
    const yb = zeros(y.length);
    const pb = zeros(probs.length);
    const sb = zeros(probs.length);
    const dqb = zeros(qkv.length);
    const prog = [
      ...op.attention(g, 'att', { qkv: qb, scores: sb, probs: pb, y: yb }, s, dummy),
      ...op.attentionBackward(g, 'attb', { dy: up(dy), qkv: qb, probs: pb, dScores: sb, dqkv: dqb }, s, dummy),
    ];
    const [ry, rp, rdq] = await run(prog, yb, pb, dqb);
    return [
      [ry, y],
      [rp, probs],
      [rdq, dqkv],
    ] as [ArrayLike<number>, ArrayLike<number>][];
  }

  // ---------------------------------------------------------------- whole model
  const text = 'From fairest creatures we desire increase,\nThat thereby beauty’s rose might never die,\n'.repeat(40);
  const tok = buildTokenizer(text);
  const cfg = withVocab(GPU_SHAPE, tok.size);
  const layout = paramLayout(cfg);
  const corpus = splitCorpus(tok.encode(text), cfg.context);
  const B = 4;
  const train = { ...GPU_TRAIN, batch: B };
  const init = initParams(layout, 11);
  const gt = new GpuTrainer(g, cfg, train, corpus, init, 99, 1000);
  const model = gt.model;
  const ref = new CpuTrainer(cfg, train, corpus, Float32Array.from(init), 99, 1000);

  gt.submitStep();
  const gpuGrads = await model.readGrads();
  const gpuParams = await model.readParams();
  const stats = await model.readStats();
  const refStep = ref.runStep();

  await test(
    `full step with dropout ${train.dropout}: loss`,
    async () => Math.abs(stats[STATS.trainLoss] - refStep.loss) / refStep.loss,
    1e-4,
    `gpu ${stats[STATS.trainLoss].toFixed(6)} cpu ${refStep.loss.toFixed(6)}`,
  );
  await test('full step: gradient norm', async () => Math.abs(stats[STATS.gradNorm] - refStep.gradNorm) / refStep.gradNorm, 1e-3);
  let worstTensor = '';
  await test(`full step: every gradient tensor (${layout.tensors.length})`, async () => {
    let worst = 0;
    for (const t of layout.tensors) {
      const e = relError(gpuGrads.subarray(t.offset, t.offset + t.size), ref.model.grads.subarray(t.offset, t.offset + t.size));
      if (e > worst) {
        worst = e;
        worstTensor = t.name;
      }
    }
    return worst;
  }, 2e-3);
  if (worstTensor) log(`      worst tensor: ${worstTensor}`);
  await test('full step: updated parameters', async () => [[gpuParams, ref.params]], 1e-4);

  // A few more steps through the real trainer: losses should track the reference closely.
  const cpuLosses: number[] = [];
  for (let step = 1; step < 6; step++) {
    gt.submitStep();
    cpuLosses.push(ref.runStep().loss);
  }
  const batch = await gt.readStats();
  const gpuLosses = batch.steps.slice(1).map((x) => x.loss);
  await test('six steps: loss trajectory', async () => [[gpuLosses, cpuLosses]], 2e-3, `gpu ${gpuLosses.map((x) => x.toFixed(4)).join(' ')}`);

  // Forward-only evaluation path, on the same weights as the reference model.
  const evalOffsets = sampleOffsets(new Rng(5), corpus.train, B, cfg.context);
  model.writeEvalStep(7, evalOffsets);
  model.submit(model.evalProgram);
  const s3 = await model.readStats();
  const evalModel = new CpuGPT(cfg, ref.params, B, cfg.context);
  const tokens = corpus.tokens;
  const inputs = new Uint32Array(B * cfg.context);
  const targets = new Uint32Array(B * cfg.context);
  for (let b = 0; b < B; b++) {
    inputs.set(tokens.subarray(evalOffsets[b], evalOffsets[b] + cfg.context), b * cfg.context);
    targets.set(tokens.subarray(evalOffsets[b] + 1, evalOffsets[b] + cfg.context + 1), b * cfg.context);
  }
  const evalRef = evalModel.forward(inputs, targets);
  await test('evaluation pass: held-out loss', async () => Math.abs(s3[STATS.valLoss + 7] - evalRef) / evalRef, 2e-3);

  // ---------------------------------------------------------------- the microscope
  // Batches with repeated random runs: the GPU gather against the CPU batch builder.
  const repOffsets = sampleOffsets(new Rng(8), corpus.train, B, cfg.context, 2);
  model.writeEvalStep(3, repOffsets);
  model.submit([model.evalProgram[0]]);
  const [gIn, gTg] = await Promise.all([g.read(model.inputs), g.read(model.targets)]);
  const repRef = gatherBatch(corpus.tokens, repOffsets, cfg.context, cfg.vocabSize);
  await test(
    'batch with repeated random runs (gather)',
    async () => [
      [new Uint32Array(gIn), repRef.inputs],
      [new Uint32Array(gTg), repRef.targets],
    ],
    0,
  );

  // The induction test on the current weights, against the CPU reference.
  gt.submitProbe();
  const probed = (await gt.readStats()).probes[0];
  const probeRef = probeBatch(cfg, B);
  const pm = new CpuGPT(cfg, ref.params, B, cfg.context);
  pm.forward(probeRef.tokens, probeRef.targets);
  const sc = scoreProbe(cfg, B, probeRef.period, (l) => pm.attentionProbs(l), pm.losses);
  await test(
    'induction test: copy losses',
    async () => [
      [
        [probed.copyFirst, probed.copySecond],
        [sc.copyFirst, sc.copySecond],
      ],
    ],
    2e-3,
    `gpu ${probed.copySecond.toFixed(4)} cpu ${sc.copySecond.toFixed(4)}`,
  );
  await test(
    `induction test: prefix and previous scores (${sc.prefix.length} heads)`,
    async () => [
      [probed.prefix, sc.prefix],
      [probed.prev, sc.prev],
    ],
    2e-3,
  );

  // Knockouts and activation patching: the edited forward pass against the CPU hook.
  const scope = new Microscope(model);
  const ko = await scope.knockouts(sampleOffsets(new Rng(6), corpus.val ?? corpus.train, B, cfg.context));
  const head = 5;
  const H = cfg.nHead;
  const kc = new CpuGPT(cfg, ref.params, B, cfg.context);
  const base = kc.forward(probeRef.tokens, probeRef.targets);
  const copyOf = (m: CpuGPT) => scoreProbe(cfg, B, probeRef.period, (l) => m.attentionProbs(l), m.losses).copySecond;
  const baseCopy = copyOf(kc);
  kc.editAttention = (l, y) => {
    if (l !== Math.floor(head / H)) return;
    const hs = cfg.dModel / H;
    for (let n = 0; n < B * cfg.context; n++) y.fill(0, n * cfg.dModel + (head % H) * hs, n * cfg.dModel + ((head % H) + 1) * hs);
  };
  kc.forward(probeRef.tokens, probeRef.targets);
  const koRef = copyOf(kc) - baseCopy;
  await test(
    'knockout of one head: change in copy loss',
    async () => Math.abs(ko.copy[head] - koRef) / Math.max(1e-3, Math.abs(base)),
    2e-3,
    `gpu ${ko.copy[head].toFixed(5)} cpu ${koRef.toFixed(5)}`,
  );

  const pt = await scope.patching();
  const clean = new CpuGPT(cfg, ref.params, B, cfg.context);
  clean.forward(probeRef.tokens, probeRef.targets);
  const savedY = Array.from({ length: cfg.nLayer }, (_, l) => Float32Array.from(clean.attentionOutput(l)));
  const pc = new CpuGPT(cfg, ref.params, B, cfg.context);
  pc.editAttention = (l, y) => {
    if (l !== Math.floor(head / H)) return;
    const hs = cfg.dModel / H;
    for (let n = 0; n < B * cfg.context; n++) {
      const o = n * cfg.dModel + (head % H) * hs;
      y.set(savedY[l].subarray(o, o + hs), o);
    }
  };
  pc.forward(probeRef.corrupt, probeRef.corruptTargets);
  const patchedRef = copyOf(pc);
  await test(
    'activation patching of one head: copy loss',
    async () => Math.abs(pt.patched[head] - patchedRef) / patchedRef,
    2e-3,
    `gpu ${pt.patched[head].toFixed(5)} cpu ${patchedRef.toFixed(5)}`,
  );

  // The sparse autoencoder: one optimiser step, and the analysis pass, against the CPU reference.
  const saeOpts = { features: 64, batch: 256, steps: 1, lambda: 2, lr: 1e-3, site: 3, topK: 4 };
  const sae = new GpuSae(model, saeOpts);
  const rows = await sae.collect(corpus);
  const p0 = await sae.readParams();
  sae.submitStep(false);
  const p1 = await sae.readParams();
  // The same positions on the CPU: every training window in order, residual stream entering block 3.
  const C = cfg.dModel;
  const actsRef = new Float32Array(rows * C);
  const cm = new CpuGPT(cfg, ref.params, B, cfg.context);
  for (let k = 0; k < rows / (B * cfg.context); k++) {
    const ins = new Uint32Array(B * cfg.context);
    for (let b = 0; b < B; b++)
      ins.set(corpus.tokens.subarray(corpus.train.start + (k * B + b) * cfg.context, corpus.train.start + (k * B + b + 1) * cfg.context), b * cfg.context);
    cm.forward(ins);
    actsRef.set(cm.residual(saeOpts.site), k * B * cfg.context * C);
  }
  const lay = saeLayout(C, saeOpts.features);
  const xb = new Float32Array(saeOpts.batch * C);
  for (let b = 0; b < saeOpts.batch; b++) {
    const row = hash32((hash32((0x5ae ^ 1) >>> 0) ^ Math.imul(b, 0x9e3779b9)) >>> 0) % rows;
    for (let c = 0; c < C; c++) xb[b * C + c] = actsRef[row * C + c] * sae.scale;
  }
  const { grads } = saeLoss(p0, lay, xb, saeOpts.batch, saeOpts.lambda);
  const p1Ref = Float32Array.from(p0);
  const lr0 = saeOpts.lr / 100;
  for (let i = 0; i < lay.total; i++) {
    const m = 0.1 * grads[i];
    const v = 0.001 * grads[i] * grads[i];
    p1Ref[i] -= (lr0 * (m / 0.1)) / (Math.sqrt(v / 0.001) + 1e-8);
  }
  normaliseDecoder(p1Ref, lay);
  await test(`sparse autoencoder: one training step (${saeOpts.features} features, ${saeOpts.batch} rows)`, async () => [[p1, p1Ref]], 1e-4);
  const res = await sae.analyse();
  const all = saeForward(
    p1,
    lay,
    Float32Array.from(actsRef, (v) => v * sae.scale),
    rows,
  );
  const counts = new Float64Array(saeOpts.features);
  for (let r = 0; r < rows - (rows % saeOpts.batch); r++) for (let j = 0; j < saeOpts.features; j++) if (all.f[r * saeOpts.features + j] > 0) counts[j]++;
  await test(
    'sparse autoencoder: firing counts over every position',
    async () => [[res.features.map((f) => f.density * res.positions), Array.from(counts)]],
    2e-3,
  );

  const failed = results.filter((r) => !r.pass).length;
  log(`\n${results.length - failed}/${results.length} passed`);
  state.done = true;
}

main().catch((e) => {
  (state as { fatal?: string }).fatal = String(e);
  log(`fatal: ${String(e)}`);
  state.done = true;
});
