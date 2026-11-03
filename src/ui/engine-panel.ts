import type { EngineInfo, Progress } from '../app/engine';
import { type Profile, groupByKernel } from '../gpu/profile';
import type { StepStats } from '../gpu/trainer';
import { bytes, h, int, seconds, sig3 } from './dom';

export interface EngineView {
  readonly info: EngineInfo;
  readonly progress: Progress | null;
  readonly last: StepStats | null;
  readonly heldOut: number | null;
  readonly sixty: { readonly step: number; readonly loss: number; readonly heldOut: number | null; readonly tps: number } | null;
  readonly profile: Profile | null;
  readonly profiling: boolean;
  readonly profileError: string | null;
}

const KERNEL_NAMES: Record<string, string> = {
  matmul: 'tiled matrix multiply',
  reduce: 'sum partial results',
  colsum: 'column sums (bias gradients)',
  layernorm: 'layer norm',
  layernorm_bwd: 'layer norm, backward',
  layernorm_bwd_params: 'layer norm gain and bias gradients',
  softmax: 'causal softmax',
  softmax_bwd: 'causal softmax, backward',
  gelu: 'GELU',
  gelu_bwd: 'GELU, backward',
  residual: 'residual add with dropout',
  dropout: 'dropout mask, backward',
  embed: 'token and position embedding',
  embed_bwd_wpe: 'position embedding gradient',
  embed_bwd_wte: 'token embedding gradient',
  gather: 'gather the batch',
  cross_entropy: 'cross-entropy, forward and backward',
  loss_reduce: 'mean loss',
  grad_norm: 'gradient norm, partial',
  grad_norm_final: 'gradient norm and clipping',
  adamw: 'AdamW update',
};

function spec(rows: readonly (readonly [string, string])[]): HTMLDListElement {
  const dl = h('dl', { class: 'spec' });
  for (const [k, v] of rows) dl.append(h('dt', {}, k), h('dd', {}, v));
  return dl;
}

function col(title: string, ...body: Node[]): HTMLElement {
  return h('div', { class: 'engine-col' }, h('h3', {}, title), ...body);
}

export function renderEngine(root: HTMLElement, v: EngineView, measure: () => void): void {
  const { info, progress: p, last } = v;
  const c = info.cfg;
  const t = info.train;
  const tokensPerStep = t.batch * c.context;
  root.textContent = '';

  root.append(
    col(
      'The model',
      spec([
        ['layers', String(c.nLayer)],
        ['attention heads', `${c.nHead} of ${c.dModel / c.nHead} wide`],
        ['width', String(c.dModel)],
        ['context', `${c.context} characters`],
        ['vocabulary', `${c.vocabSize} characters`],
        ['parameters', int(info.paramCount)],
        ['output head', 'tied to the token embedding'],
        ['dropout', String(t.dropout)],
      ]),
    ),
  );

  const [bc1, bc2] = p ? [1 - t.beta1 ** p.completed, 1 - t.beta2 ** p.completed] : [0, 0];
  root.append(
    col(
      'The run',
      spec([
        ['batch', `${t.batch} × ${c.context} = ${int(tokensPerStep)} tokens`],
        ['steps planned', p ? `${int(p.totalSteps)}, set by ${p.plannedBy === 'data' ? 'the text’s length' : 'the clock'}` : '–'],
        ['learning rate', last ? last.lr.toExponential(2) : '–'],
        ['schedule', `${t.peakLr.toExponential(0)} → ${t.minLr.toExponential(0)}, cosine`],
        ['warm-up', `${t.warmupSteps} steps`],
        ['gradient norm', last ? last.gradNorm.toFixed(3) : '–'],
        ['clipped at', `${t.clipNorm}${last && last.gradNorm > t.clipNorm ? ' (clipping now)' : ''}`],
        ['Adam bias corrections', p ? `${bc1.toFixed(3)}, ${bc2.toFixed(3)}` : '–'],
        ['tokens seen', p ? int(p.tokensSeen) : '–'],
        ['passes over the text', p ? p.epochs.toFixed(1) : '–'],
      ]),
    ),
  );

  const total = info.memory.reduce((a, m) => a + m.bytes, 0);
  root.append(col(info.kind === 'gpu' ? 'GPU buffers' : 'Memory', spec([...info.memory.map((m) => [m.category, bytes(m.bytes)] as const), ['total', bytes(total)]])));

  const gflops = p && p.tokensPerSecond > 0 ? (info.flopsPerStep / tokensPerStep) * p.tokensPerSecond : 0;
  const deviceRows: [string, string][] = [
    ['running on', info.device],
    ...(info.kind === 'gpu' ? ([['dispatches a step', String(info.dispatchesPerStep)]] as [string, string][]) : []),
    [info.kind === 'gpu' ? 'matmul work a step' : 'work a step', `${(info.flopsPerStep / 1e9).toFixed(2)} GFLOP`],
    ['throughput', p && p.tokensPerSecond > 0 ? `${sig3(p.tokensPerSecond)} tokens/s, ${sig3(gflops / 1e9)} GFLOP/s` : '–'],
    ['time', p ? seconds(p.elapsed) : '–'],
  ];
  if (v.sixty) {
    deviceRows.push(['at 60 seconds', `step ${int(v.sixty.step)}, loss ${v.sixty.loss.toFixed(3)}${v.sixty.heldOut !== null ? `, held out ${v.sixty.heldOut.toFixed(3)}` : ''}, ${sig3(v.sixty.tps)} tokens/s`]);
  }
  root.append(col('Device', spec(deviceRows)));

  // Kernel timings.
  const kernels = h('div', { class: 'engine-col kernels' }, h('h3', {}, 'Kernels, one training step'));
  if (info.kind !== 'gpu') {
    kernels.append(h('p', { class: 'kernel-note' }, 'This run uses the CPU reference implementation in a worker, so there are no GPU kernels to time.'));
  } else if (v.profileError) {
    kernels.append(h('p', { class: 'kernel-note' }, `Could not time the kernels: ${v.profileError}`));
  } else if (!v.profile) {
    const btn = h('button', { type: 'button', class: 'key' }, v.profiling ? 'Timing every kernel…' : 'Time every kernel');
    btn.disabled = v.profiling || !p;
    btn.addEventListener('click', measure);
    kernels.append(
      h('p', { class: 'kernel-note' }, p ? 'Runs each dispatch of one step 24 times back to back and takes the average. Training pauses for about a second and carries on exactly where it was.' : 'Start training to time the kernels.'),
      btn,
    );
  } else {
    const prof = v.profile;
    const groups = groupByKernel(prof);
    const max = Math.max(...groups.map((g) => g.ms));
    const table = h('table', { class: 'kernel-table' });
    table.append(h('thead', {}, h('tr', {}, h('th', {}, 'kernel'), h('th', {}, 'dispatches'), h('th', {}, 'ms a step'), h('th', {}, 'share'), h('th', { class: 'bar-head' }, ''), h('th', {}, 'GFLOP/s'))));
    const body = h('tbody');
    for (const g of groups) {
      body.append(
        h(
          'tr',
          {},
          h('td', {}, h('span', { class: 'kernel-name' }, KERNEL_NAMES[g.kernel] ?? g.kernel), h('span', { class: 'file' }, ` ${g.kernel}.wgsl`)),
          h('td', {}, String(g.calls)),
          h('td', {}, g.ms.toFixed(3)),
          h('td', {}, `${((100 * g.ms) / prof.kernelMs).toFixed(1)}%`),
          h('td', { class: 'bar-cell' }, h('span', { class: 'bar', style: `width:${((100 * g.ms) / max).toFixed(1)}%` })),
          h('td', {}, g.flops ? sig3(g.flops / g.ms / 1e6) : ''),
        ),
      );
    }
    table.append(body);
    const again = h('button', { type: 'button', class: 'key' }, v.profiling ? 'Timing…' : 'Time them again');
    again.disabled = v.profiling;
    again.addEventListener('click', measure);
    kernels.append(
      h('div', { class: 'table-scroll' }, table),
      h(
        'p',
        { class: 'kernel-note' },
        `Kernels add up to ${prof.kernelMs.toFixed(2)} ms; a whole step, submitted back to back, takes ${prof.stepMs.toFixed(2)} ms. Measured by running each of the ${prof.timings.length} dispatches ${prof.reps} times. `,
        again,
      ),
    );
  }
  root.append(kernels);
}
