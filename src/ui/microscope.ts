import { type DissectProgress, type Dissection, PROBE_EVERY } from '../app/engine';
import { type DownloadState, type RecordedRun, type RecordedWeights, WeightsDownload, decodePattern } from '../app/recorded';
import type { SaeFeature } from '../gpu/sae';
import type { ModelConfig } from '../model/config';
import { INDUCTION, type Moment, type ProbeReading, headName } from '../model/probe';
import { $, bytes, h, int, reducedMotion } from './dom';
import { FeaturesPane } from './panes/features';
import { HeadsPane } from './panes/heads';
import type { MicroscopeHost, Mode, Pane, PaneContext, Subject, Which } from './panes/types';

export type { MicroscopeHost, Mode } from './panes/types';

/**
 * The microscope: the mode keys over the screen, the induction box in the panel, and the HEADS
 * and FEATURES panes, each showing the live run when it has something and the recorded run
 * otherwise, always labelled with which it is.
 */
export class Microscope {
  mode: Mode = 'train';
  private live: Subject | null = null;
  private recorded: Subject | null = null;
  private run: RecordedRun | null = null;
  private download: WeightsDownload | null = null;
  private recordedWeights: RecordedWeights | null = null;
  private view: Record<Pane, Which> = { heads: 'recorded', features: 'recorded' };
  /** False for the CPU model, which has no autoencoder to take apart. */
  private liveDissects = true;
  private writeTimer = 0;
  private writeEpoch = 0;
  private readonly heads: HeadsPane;
  private readonly features: FeaturesPane;

  constructor(host: MicroscopeHost) {
    const context: PaneContext = {
      host,
      mode: () => this.mode,
      shown: (pane) => this.subject(this.view[pane]) ?? this.recorded ?? this.live,
      subject: (which) => this.subject(which),
      view: (pane) => this.view[pane],
      live: () => this.live,
      run: () => this.run,
      recordedWeights: () => this.recordedWeights,
      viewSwitch: (pane) => this.viewSwitch(pane),
      weightsGate: (box, which) => this.weightsGate(box, which),
      scheduleWrite: () => this.scheduleWrite(),
    };
    this.heads = new HeadsPane(context);
    this.features = new FeaturesPane(context);
    for (const m of ['train', 'heads', 'features'] as Mode[]) {
      $(`mode-${m}`).addEventListener('click', () => this.setMode(m));
    }
    $('modes').addEventListener('keydown', (e) => {
      const order: Mode[] = ['train', 'heads', 'features'];
      const i = order.indexOf(this.mode);
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        const next = order[(i + (e.key === 'ArrowRight' ? 1 : 2)) % 3];
        this.setMode(next);
        $(`mode-${next}`).focus();
      }
    });
    this.renderMoment();
  }

  // ------------------------------------------------------------------ modes

  setMode(m: Mode): void {
    this.mode = m;
    for (const k of ['train', 'heads', 'features'] as Mode[]) {
      const on = k === m;
      $(`mode-${k}`).setAttribute('aria-selected', String(on));
      $(`mode-${k}`).tabIndex = on ? 0 : -1;
      $(`pane-${k}`).hidden = !on;
    }
    document.documentElement.dataset.mode = m;
    if (m === 'heads') {
      this.heads.render();
      void this.heads.refreshLivePatterns();
    }
    if (m === 'features') this.features.render();
    if (m !== 'train') this.scheduleWrite();
  }

  // ------------------------------------------------------------------ recorded run

  setRecorded(run: RecordedRun, tokens: Uint32Array, chars: readonly string[], features: Promise<SaeFeature[]>): void {
    this.run = run;
    const d = run.dissection;
    const patterns = new Map<number, Float32Array>();
    for (const [k, v] of Object.entries(d.patterns)) patterns.set(Number(k), decodePattern(v));
    this.recorded = {
      which: 'recorded',
      cfg: run.cfg,
      tokens,
      chars,
      latest: run.probes[run.probes.length - 1] ?? null,
      moment: run.moment,
      knockouts: d.knockouts,
      patching: d.patching,
      patterns,
      sae: {
        opts: d.sae.opts,
        curve: [...d.sae.curve],
        fvu: d.sae.fvu,
        l0: d.sae.l0,
        dead: d.sae.dead,
        positions: d.sae.positions,
        seconds: d.sae.seconds,
        scale: d.sae.scale,
        features: null,
        params: null,
        progress: null,
      },
      stage: null,
    };
    this.download = new WeightsDownload(run, (s) => this.onDownload(s));
    void features
      .then((f) => {
        if (this.recorded?.sae) this.recorded.sae.features = f;
        this.features.render();
      })
      .catch(() => {
        if (this.recorded) this.recorded.stage = 'The recorded features did not load.';
        this.features.render();
      });
    if (this.heads.head === null && run.moment) this.heads.head = run.moment.head;
    // A live run that has no autoencoder yet shows the recorded one meanwhile.
    if (!this.live?.sae) this.view.features = 'recorded';
    this.renderMoment();
    this.heads.render();
    this.features.render();
  }

  private onDownload(s: DownloadState): void {
    if (s.kind === 'ready') {
      this.recordedWeights = s.weights;
      if (this.recorded?.sae) this.recorded.sae.params = s.weights.sae;
    }
    this.heads.render();
    this.features.render();
    this.scheduleWrite();
  }

  // ------------------------------------------------------------------ live run

  /** Whether the live engine takes its model apart (the GPU does; the CPU fallback does not). */
  startLive(cfg: ModelConfig, tokens: Uint32Array, chars: readonly string[], dissects: boolean): void {
    this.liveDissects = dissects;
    this.live = {
      which: 'live',
      cfg,
      tokens,
      chars,
      latest: null,
      moment: null,
      knockouts: null,
      patching: null,
      patterns: new Map(),
      sae: null,
      stage: dissects
        ? null
        : 'This device trains the small CPU model, which is not taken apart here. Switch to Recorded for the GPU run’s knockouts, patching and features.',
    };
    this.heads.knocked.clear();
    this.view = { heads: 'live', features: this.recorded || !dissects ? 'recorded' : 'live' };
    this.renderMoment();
    this.heads.render();
    this.features.render();
  }

  endLive(): void {
    this.live = null;
    this.view = { heads: 'recorded', features: 'recorded' };
    this.renderMoment();
    this.heads.render();
    this.features.render();
  }

  probes(readings: readonly ProbeReading[]): void {
    if (!this.live || readings.length === 0) return;
    this.live.latest = readings[readings.length - 1];
    if (this.mode === 'heads') {
      this.heads.renderGrid();
      this.heads.schedulePatterns();
    }
    this.renderMoment();
  }

  setMoment(m: Moment): void {
    if (!this.live) return;
    this.live.moment = m;
    if (this.heads.head === null || this.heads.head === this.recorded?.moment?.head) this.heads.head = m.head;
    this.renderMoment();
    this.heads.render();
  }

  dissectProgress(p: DissectProgress): void {
    const s = this.live;
    if (!s) return;
    if (p.stage === 'heads') s.stage = 'Knocking out each head, and patching each one in…';
    else if (p.stage === 'heads-done') {
      s.stage = null;
      s.knockouts = p.knockouts;
      s.patching = p.patching;
      s.patterns = new Map(p.patterns.map((x, i) => [i, x]));
      this.heads.render();
      this.scheduleWrite();
      return;
    } else if (p.stage === 'collect') s.stage = 'Collecting the residual stream at every position of the text…';
    else if (p.stage === 'sae') {
      s.stage = null;
      if (!s.sae) {
        s.sae = {
          opts: { features: 0, batch: 0, steps: p.total, lambda: 0, lr: 0, site: 0, topK: 0 },
          curve: [],
          fvu: null,
          l0: null,
          dead: null,
          positions: 0,
          seconds: null,
          scale: 1,
          features: null,
          params: null,
          progress: null,
        };
        this.view.features = 'live';
      }
      s.sae.progress = { step: p.step, total: p.total };
      if (p.at) s.sae.curve.push(p.at);
    } else s.stage = 'Finding each feature’s strongest contexts…';
    this.features.render();
    this.heads.render();
  }

  dissected(d: Dissection): void {
    const s = this.live;
    if (!s) return;
    s.stage = null;
    s.knockouts = d.knockouts;
    s.patching = d.patching;
    s.patterns = new Map(d.patterns.map((p, i) => [i, p]));
    const r = d.sae.result;
    s.sae = {
      opts: d.sae.opts,
      curve: [...d.sae.curve],
      fvu: r.fvu,
      l0: r.l0,
      dead: r.dead,
      positions: r.positions,
      seconds: d.sae.seconds,
      scale: r.scale,
      features: [...r.features],
      params: d.sae.params,
      progress: null,
    };
    this.view = { heads: 'live', features: 'live' };
    this.features.feature = null;
    this.heads.render();
    this.features.render();
    this.scheduleWrite();
  }

  dissectFailed(message: string): void {
    if (!this.live) return;
    this.live.stage = `Could not take the model apart: ${message}`;
    this.heads.render();
    this.features.render();
  }

  // ------------------------------------------------------------------ the induction box

  private renderMoment(): void {
    const box = $('moment');
    const live = this.live;
    const ref = this.recorded;
    const H = (live ?? ref)?.cfg.nHead ?? 4;
    const parts: Node[] = [];
    if (live?.moment) {
      const m = live.moment;
      const now = live.latest && live.latest.step > m.step ? live.latest : null;
      parts.push(
        h('p', { class: 'moment-head' }, h('b', {}, headName(m.head, H)), ' learned to copy at step ', h('b', {}, int(m.step))),
        h(
          'p',
          { class: 'moment-why' },
          `On random characters shown twice, its attention went to the character after each earlier occurrence (score ${m.prefix.toFixed(2)}). The loss on the second copy fell from ${(m.before?.copySecond ?? m.copyFirst).toFixed(2)} nats${m.before ? ` at step ${int(m.before.step)}` : ''} to ${m.copySecond.toFixed(2)} at the moment${now ? `, and ${now.copySecond.toFixed(2)} by step ${int(now.step)}` : ''}.`,
          m.partner !== null ? ` It works with ${headName(m.partner, H)}, which attends to the previous character (${m.partnerPrev.toFixed(2)}).` : '',
        ),
      );
    } else if (live) {
      const r = live.latest;
      const best = r ? r.prefix.reduce((bi, v, i, a) => (v > a[bi] ? i : bi), 0) : -1;
      parts.push(
        h('p', { class: 'moment-head moment-wait' }, 'No induction head yet'),
        h(
          'p',
          { class: 'moment-why' },
          r
            ? `Every ${PROBE_EVERY} steps: random characters, shown twice. Copy loss ${r.copyFirst.toFixed(2)} → ${r.copySecond.toFixed(2)} nats; strongest prefix match ${headName(best, H)} at ${r.prefix[best].toFixed(2)} (a head counts at ${INDUCTION.prefix}).`
            : `Every ${PROBE_EVERY} steps the model reads random characters shown twice. A head that has learned to copy looks back to the character after the previous occurrence.`,
        ),
      );
    } else {
      parts.push(h('p', { class: 'moment-head moment-wait' }, 'Waiting for a run'));
    }
    if (ref?.moment && this.run) {
      parts.push(
        h(
          'p',
          { class: 'moment-ref' },
          h('span', { class: 'badge badge--rec' }, 'REC'),
          ` Recorded run: ${headName(ref.moment.head, ref.cfg.nHead)} at step ${int(ref.moment.step)}, ${(this.run.seconds * (ref.moment.step / this.run.totalSteps)).toFixed(0)} s in on ${this.run.device}.`,
        ),
      );
    }
    box.replaceChildren(...parts);
  }

  // ------------------------------------------------------------------ shared by the panes

  private subject(which: Which): Subject | null {
    return which === 'live' ? this.live : this.recorded;
  }

  private viewSwitch(pane: Pane): HTMLElement {
    const wrap = h('span', { class: 'source-switch', role: 'group', 'aria-label': 'Which model' });
    for (const w of ['live', 'recorded'] as Which[]) {
      // The CPU model has no autoencoder: its Live features would only ever be empty.
      const has = this.subject(w) !== null && !(pane === 'features' && w === 'live' && !this.liveDissects);
      const b = h(
        'button',
        { type: 'button', class: `switch switch--${w}`, 'aria-pressed': String(this.view[pane] === w), disabled: !has },
        w === 'live' ? 'Live' : 'Recorded',
      );
      b.addEventListener('click', () => {
        this.view[pane] = w;
        this.features.feature = null;
        if (pane === 'heads') {
          this.heads.render();
          void this.heads.refreshLivePatterns();
        } else this.features.render();
        this.scheduleWrite();
      });
      wrap.append(b);
    }
    return wrap;
  }

  /** Writing with the recorded model needs its weights: offer the download, state its size. */
  private weightsGate(box: HTMLElement, which: Which): void {
    if (which === 'live' || !this.run || !this.download) {
      box.replaceChildren();
      return;
    }
    const st = this.download.state;
    const size = bytes(this.run.weights.bytes);
    if (st.kind === 'ready') {
      box.replaceChildren(h('p', { class: 'gate-note' }, `Recorded model loaded (${size}${st.fromCache ? ', from this browser’s cache' : ''}).`));
      return;
    }
    if (st.kind === 'loading') {
      const cancel = h('button', { type: 'button', class: 'key key--small' }, 'Cancel');
      cancel.addEventListener('click', () => this.download?.cancel());
      box.replaceChildren(
        h(
          'div',
          { class: 'gate' },
          h('progress', { max: String(st.total), value: String(st.received), 'aria-label': 'Downloading the recorded model' }),
          h('span', { class: 'gate-note' }, `${bytes(st.received)} of ${size}${st.cached ? ' from cache' : ''}`),
          cancel,
        ),
      );
      return;
    }
    const go = h('button', { type: 'button', class: 'key key--small' }, st.kind === 'failed' || st.kind === 'cancelled' ? 'Try again' : `Load it (${size})`);
    go.addEventListener('click', () => void this.download?.start());
    const why =
      st.kind === 'failed'
        ? `The recorded model did not download: ${st.message}. Everything else here still works.`
        : st.kind === 'cancelled'
          ? 'Download cancelled. Everything else here still works.'
          : `To write with the recorded model, load its weights: ${size}, half precision, kept in this browser’s cache afterwards.`;
    box.replaceChildren(h('div', { class: `gate${st.kind === 'failed' ? ' gate--failed' : ''}` }, h('span', { class: 'gate-note' }, why), go));
  }

  // ------------------------------------------------------------------ writing with an intervention

  /** Writing waits for a pause in the controls, and only the latest request's text is shown. */
  private scheduleWrite(): void {
    window.clearTimeout(this.writeTimer);
    this.writeTimer = window.setTimeout(() => void this.write(), reducedMotion() ? 0 : 160);
  }

  private async write(): Promise<void> {
    const epoch = ++this.writeEpoch;
    const stillCurrent = () => epoch === this.writeEpoch;
    if (this.mode === 'heads') await this.heads.write(stillCurrent);
    else if (this.mode === 'features') await this.features.write(stillCurrent);
  }
}
