import { CPU_SHAPE, GPU_SHAPE } from '../model/config';
import { $, h, int, seconds, sig3 } from '../ui/dom';
import { renderEngine } from '../ui/engine-panel';
import type { Session } from './session';

/** What the page exposes for its own browser tests and measurements (scripts/*.mjs). */
export interface Metrics {
  readonly state: string;
  readonly kind: 'gpu' | 'cpu';
  readonly completed: number;
  readonly totalSteps: number;
  readonly elapsed: number;
  readonly tokensPerSecond: number;
  readonly tokensSeen: number;
  readonly epochs: number;
  readonly loss: number;
  readonly heldOut: number | null;
  readonly sixty: Session['sixty'];
  readonly lr: number | null;
  readonly checkpoints: number;
  readonly drawn: number;
  readonly sessions: number;
  readonly params: number;
}

declare global {
  interface Window {
    __metrics?: Metrics;
  }
}

/** The panel's readouts for a run (or their resting state), and the metrics behind them. */
export function paintReadouts(s: Session | null, latestLoss: number, sessions: number): void {
  const p = s?.progress ?? null;
  const set = (id: string, text: string) => {
    $(id).textContent = text;
  };
  if (!s || !p) {
    for (const id of ['r-tps', 'r-loss', 'r-val', 'r-lr', 'r-grad']) set(id, '–');
    set('r-step', '0');
    set('r-time', '0.0 s');
    set('r-passes', '0.0');
    return;
  }
  const step = $('r-step');
  step.textContent = int(p.completed);
  step.append(h('small', {}, ` / ${int(p.totalSteps)}`));
  set('r-time', seconds(p.elapsed));
  set('r-tps', p.tokensPerSecond > 0 ? sig3(p.tokensPerSecond) : '–');
  set('r-passes', p.epochs.toFixed(1));
  set('r-loss', Number.isFinite(latestLoss) ? latestLoss.toFixed(3) : '–');
  set('r-val', s.corpus.val === null ? 'n/a' : s.heldOut !== null ? s.heldOut.toFixed(3) : '–');
  set('r-lr', s.last ? s.last.lr.toExponential(2) : '–');
  set('r-grad', s.last ? s.last.gradNorm.toFixed(3) : '–');
  window.__metrics = {
    state: s.engine.state,
    kind: s.engine.info.kind,
    completed: p.completed,
    totalSteps: p.totalSteps,
    elapsed: p.elapsed,
    tokensPerSecond: p.tokensPerSecond,
    tokensSeen: p.tokensSeen,
    epochs: p.epochs,
    loss: latestLoss,
    heldOut: s.heldOut,
    sixty: s.sixty,
    lr: s.last?.lr ?? null,
    checkpoints: s.checkpoints.length,
    drawn: s.checkpoints.filter((c) => c.sample && c.epoch === s.epoch).length,
    sessions,
    params: s.engine.info.paramCount,
  };
}

/** The system page: the model, its buffers and kernels for a run, or what it will show. */
export function renderSystem(s: Session | null, useCpu: boolean, onMeasure: () => void): void {
  const body = $('system-body');
  if (!s) {
    const shape = useCpu ? CPU_SHAPE : GPU_SHAPE;
    body.replaceChildren(
      h(
        'p',
        { class: 'kernel-note' },
        `Nothing is running. During a run this shows the model (${shape.nLayer} layers, ${shape.nHead} heads, ${shape.dModel} wide, ${shape.context} characters of context), every buffer on the ${useCpu ? 'CPU' : 'GPU'}, the optimiser’s state and the time each kernel takes.`,
      ),
    );
    return;
  }
  renderEngine(
    body,
    {
      info: s.engine.info,
      progress: s.progress,
      last: s.last,
      heldOut: s.heldOut,
      sixty: s.sixty,
      profile: s.profile,
      profiling: s.profiling,
      profileError: s.profileError,
    },
    onMeasure,
  );
}
