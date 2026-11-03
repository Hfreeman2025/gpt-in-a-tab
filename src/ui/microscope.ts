import type { DissectProgress, Dissection } from '../app/engine';
import { type DownloadState, type RecordedRun, WeightsDownload, decodePattern } from '../app/recorded';
import type { FeatureClamp } from '../cpu/sae';
import { saeLayout } from '../cpu/sae';
import type { SaeFeature, SaeOptions, SaeStep } from '../gpu/sae';
import type { Knockouts, Patching } from '../gpu/microscope';
import type { ModelConfig } from '../model/config';
import type { GenerationEdit } from '../model/generate';
import { type FeatureLabel, labelFeature, sayChar } from '../model/labels';
import { INDUCTION, type Moment, type ProbeReading, headName } from '../model/probe';
import { $, bytes, cssVar, h, int, reducedMotion } from './dom';

export type Mode = 'train' | 'heads' | 'features';
type Which = 'live' | 'recorded';

/** What the panes can show about one model, live or recorded. */
interface Subject {
  readonly which: Which;
  readonly cfg: ModelConfig;
  /** The text, as tokens, and how to spell them, for feature contexts. */
  readonly tokens: Uint32Array;
  readonly chars: readonly string[];
  latest: ProbeReading | null;
  moment: Moment | null;
  knockouts: Knockouts | null;
  patching: Patching | null;
  patterns: Map<number, Float32Array>;
  sae: {
    readonly opts: SaeOptions;
    curve: SaeStep[];
    fvu: number | null;
    l0: number | null;
    dead: number | null;
    positions: number;
    seconds: number | null;
    scale: number;
    features: SaeFeature[] | null;
    params: Float32Array | null;
    progress: { step: number; total: number } | null;
  } | null;
  stage: string | null;
}

/** What the page's controller does for the microscope. */
export interface MicroscopeHost {
  /**
   * Writes text from the live model (its latest weights) or the recorded one, with an
   * intervention. 'free': from the prompt, sampled. 'copy': greedily continuing COPY_PROMPT, a run
   * of random letters shown twice; the answer is how the run goes on.
   */
  write(which: Which, edit: GenerationEdit | null, weights: Float32Array | null, kind: 'free' | 'copy'): Promise<string | null>;
  /** The copying demonstration's prompt and the continuation a perfect copier would write. */
  copyPrompt(which: Which): { prompt: string; expected: string } | null;
  /** The head whose attention lights the text when a character is inspected (null: all of them). */
  selectHead(head: number | null): void;
  /** Attention of every head on the latest induction test, for the live model. */
  livePatterns(): Promise<Float32Array | null>;
}

const say = (c: string) => (c === '\n' ? '↵' : c);

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
  private recordedWeights: { model: Float32Array; sae: Float32Array } | null = null;
  private view: { heads: Which; features: Which } = { heads: 'recorded', features: 'recorded' };
  private head: number | null = null;
  private knocked = new Set<number>();
  private feature: number | null = null;
  private clampValue: number | null = null;
  private filter = '';
  private writeTimer = 0;
  private writeEpoch = 0;
  private patternTimer = 0;

  constructor(private readonly host: MicroscopeHost) {
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
    $<HTMLInputElement>('feature-filter').addEventListener('input', (e) => {
      this.filter = (e.target as HTMLInputElement).value;
      this.renderFeatureList();
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
      this.renderHeads();
      void this.refreshLivePatterns();
    }
    if (m === 'features') this.renderFeatures();
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
      sae: { opts: d.sae.opts, curve: [...d.sae.curve], fvu: d.sae.fvu, l0: d.sae.l0, dead: d.sae.dead, positions: d.sae.positions, seconds: d.sae.seconds, scale: d.sae.scale, features: null, params: null, progress: null },
      stage: null,
    };
    this.download = new WeightsDownload(run, (s) => this.onDownload(s));
    void features
      .then((f) => {
        if (this.recorded?.sae) this.recorded.sae.features = f;
        this.renderFeatures();
      })
      .catch(() => {
        if (this.recorded) this.recorded.stage = 'The recorded features did not load.';
        this.renderFeatures();
      });
    if (this.head === null && run.moment) this.head = run.moment.head;
    // A live run that has no autoencoder yet shows the recorded one meanwhile.
    if (!this.live?.sae) this.view.features = 'recorded';
    this.renderMoment();
    this.renderHeads();
    this.renderFeatures();
  }

  private onDownload(s: DownloadState): void {
    if (s.kind === 'ready') {
      this.recordedWeights = s.weights;
      if (this.recorded?.sae) this.recorded.sae.params = s.weights.sae;
    }
    this.renderHeads();
    this.renderFeatures();
    this.scheduleWrite();
  }

  // ------------------------------------------------------------------ live run

  /** Whether the live engine takes its model apart (the GPU does; the CPU fallback does not). */
  private liveDissects = true;

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
      stage: dissects ? null : 'This device trains the small CPU model, which is not taken apart here. Switch to Recorded for the GPU run’s knockouts, patching and features.',
    };
    this.knocked.clear();
    this.view = { heads: 'live', features: this.recorded || !dissects ? 'recorded' : 'live' };
    this.renderMoment();
    this.renderHeads();
    this.renderFeatures();
  }

  endLive(): void {
    this.live = null;
    this.view = { heads: 'recorded', features: 'recorded' };
    this.renderMoment();
    this.renderHeads();
    this.renderFeatures();
  }

  probes(readings: readonly ProbeReading[]): void {
    if (!this.live || readings.length === 0) return;
    this.live.latest = readings[readings.length - 1];
    if (this.mode === 'heads') {
      this.renderGrid();
      this.schedulePatterns();
    }
    this.renderMoment();
  }

  setMoment(m: Moment): void {
    if (!this.live) return;
    this.live.moment = m;
    if (this.head === null || this.head === this.recorded?.moment?.head) this.head = m.head;
    this.renderMoment();
    this.renderHeads();
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
      this.renderHeads();
      this.scheduleWrite();
      return;
    }
    else if (p.stage === 'collect') s.stage = 'Collecting the residual stream at every position of the text…';
    else if (p.stage === 'sae') {
      s.stage = null;
      if (!s.sae) {
        s.sae = { opts: { features: 0, batch: 0, steps: p.total, lambda: 0, lr: 0, site: 0, topK: 0 }, curve: [], fvu: null, l0: null, dead: null, positions: 0, seconds: null, scale: 1, features: null, params: null, progress: null };
        this.view.features = 'live';
      }
      s.sae.progress = { step: p.step, total: p.total };
      if (p.at) s.sae.curve.push(p.at);
    } else s.stage = 'Finding each feature’s strongest contexts…';
    this.renderFeatures();
    this.renderHeads();
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
    this.feature = null;
    this.renderHeads();
    this.renderFeatures();
    this.scheduleWrite();
  }

  dissectFailed(message: string): void {
    if (!this.live) return;
    this.live.stage = `Could not take the model apart: ${message}`;
    this.renderHeads();
    this.renderFeatures();
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
            ? `Every ${20} steps: random characters, shown twice. Copy loss ${r.copyFirst.toFixed(2)} → ${r.copySecond.toFixed(2)} nats; strongest prefix match ${headName(best, H)} at ${r.prefix[best].toFixed(2)} (a head counts at ${INDUCTION.prefix}).`
            : 'Every 20 steps the model reads random characters shown twice. A head that has learned to copy looks back to the character after the previous occurrence.',
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

  // ------------------------------------------------------------------ HEADS

  private subject(which: Which): Subject | null {
    return which === 'live' ? this.live : this.recorded;
  }

  private viewSwitch(pane: 'heads' | 'features'): HTMLElement {
    const wrap = h('span', { class: 'source-switch', role: 'group', 'aria-label': 'Which model' });
    for (const w of ['live', 'recorded'] as Which[]) {
      // The CPU model has no autoencoder: its Live features would only ever be empty.
      const has = this.subject(w) !== null && !(pane === 'features' && w === 'live' && !this.liveDissects);
      const b = h('button', { type: 'button', class: `switch switch--${w}`, 'aria-pressed': String(this.view[pane] === w), disabled: !has }, w === 'live' ? 'Live' : 'Recorded');
      b.addEventListener('click', () => {
        this.view[pane] = w;
        this.feature = null;
        if (pane === 'heads') {
          this.renderHeads();
          void this.refreshLivePatterns();
        } else this.renderFeatures();
        this.scheduleWrite();
      });
      wrap.append(b);
    }
    return wrap;
  }

  private renderHeads(): void {
    if (this.mode !== 'heads') return;
    const s = this.subject(this.view.heads) ?? this.recorded ?? this.live;
    $('heads-switch').replaceChildren(this.viewSwitch('heads'));
    $('heads-note').textContent = s
      ? s.which === 'recorded'
        ? `recorded run, ${this.run?.recorded.slice(0, 10) ?? ''}, ${this.run?.device ?? ''}`
        : s.knockouts
          ? `this run, at step ${int(s.latest?.step ?? 0)}`
          : 'this run: knockouts and patching run when training ends'
      : '';
    this.renderGrid();
    this.renderPattern();
    this.renderKnockouts();
    this.renderKnockWrite();
  }

  private renderGrid(): void {
    const s = this.subject(this.view.heads) ?? this.recorded ?? this.live;
    const grid = $('head-grid');
    if (!s) {
      grid.replaceChildren(h('p', { class: 'pane-empty' }, 'Nothing to show yet.'));
      return;
    }
    const { nLayer: L, nHead: H } = s.cfg;
    grid.style.setProperty('--heads', String(H));
    const cells: Node[] = [];
    for (let k = 0; k < L * H; k++) {
      const copy = s.latest?.prefix[k] ?? 0;
      const prev = s.latest?.prev[k] ?? 0;
      const role = copy >= INDUCTION.prefix ? 'copies' : prev >= 0.5 ? 'previous' : '';
      const ko = s.knockouts;
      const pt = s.patching;
      const cell = h(
        'button',
        {
          type: 'button',
          class: `head-cell${role ? ` head-cell--${role}` : ''}${this.knocked.has(k) ? ' head-cell--off' : ''}`,
          'aria-pressed': String(this.head === k),
          'aria-label': `${headName(k, H)}: copy score ${copy.toFixed(2)}, previous-character score ${prev.toFixed(2)}${ko ? `, knocking it out changes the copy loss by ${ko.copy[k].toFixed(2)} nats` : ''}`,
        },
        h('span', { class: 'head-name' }, headName(k, H)),
        h('span', { class: 'meter meter--copy' }, h('i', { style: `--v:${Math.min(1, copy).toFixed(3)}` }), h('span', {}, `copy ${copy.toFixed(2)}`)),
        h('span', { class: 'meter meter--prev' }, h('i', { style: `--v:${Math.min(1, prev).toFixed(3)}` }), h('span', {}, `prev ${prev.toFixed(2)}`)),
        ko ? h('span', { class: 'head-ko' }, `off: copy ${signed(ko.copy[k])}`, h('br'), `text ${signed(ko.text[k])}`) : null,
        pt ? h('span', { class: 'head-pt' }, `patched: ${Math.round(Math.max(0, pt.recovered[k]) * 100)}%`) : null,
      );
      cell.addEventListener('click', (e) => {
        if (e.shiftKey || e.altKey) this.toggleKnock(k);
        else {
          this.head = this.head === k ? null : k;
          this.host.selectHead(this.head);
          this.renderGrid();
          this.renderPattern();
        }
      });
      cells.push(cell);
    }
    grid.replaceChildren(...cells);
  }

  private toggleKnock(k: number): void {
    if (this.knocked.has(k)) this.knocked.delete(k);
    else this.knocked.add(k);
    this.renderGrid();
    this.renderKnockWrite();
    this.scheduleWrite();
  }

  private schedulePatterns(): void {
    if (this.view.heads !== 'live' || this.patternTimer) return;
    this.patternTimer = window.setTimeout(() => {
      this.patternTimer = 0;
      void this.refreshLivePatterns();
    }, 400);
  }

  private async refreshLivePatterns(): Promise<void> {
    const s = this.live;
    if (!s || this.view.heads !== 'live' || this.mode !== 'heads' || s.knockouts) return;
    const all = await this.host.livePatterns();
    if (!all || this.live !== s) return;
    const T = s.cfg.context;
    for (let k = 0; k < s.cfg.nLayer * s.cfg.nHead; k++) s.patterns.set(k, all.subarray(k * T * T, (k + 1) * T * T));
    this.renderPattern();
  }

  private renderPattern(): void {
    const s = this.subject(this.view.heads) ?? this.recorded ?? this.live;
    const canvas = $<HTMLCanvasElement>('pattern');
    const cap = $('pattern-caption');
    const k = this.head;
    const ctx = canvas.getContext('2d');
    if (!s || !ctx) return;
    const T = s.cfg.context;
    const H = s.cfg.nHead;
    canvas.width = T;
    canvas.height = T;
    const img = ctx.createImageData(T, T);
    const p = k !== null ? s.patterns.get(k) : undefined;
    const bg = [8, 9, 10];
    const att = (cssVar('--att-rgb') || '255 155 61').split(/\s+/).map(Number);
    for (let i = 0; i < T * T; i++) {
      const v = p ? Math.min(1, Math.sqrt(p[i])) : 0;
      img.data[i * 4] = bg[0] + (att[0] - bg[0]) * v;
      img.data[i * 4 + 1] = bg[1] + (att[1] - bg[1]) * v;
      img.data[i * 4 + 2] = bg[2] + (att[2] - bg[2]) * v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    if (k === null) {
      cap.textContent = 'Pick a head to see where it looks on the induction test.';
      return;
    }
    const name = headName(k, H);
    const copy = s.latest?.prefix[k] ?? 0;
    const prev = s.latest?.prev[k] ?? 0;
    if (!p) {
      cap.textContent = s.which === 'recorded' ? `${name}: the recorded run keeps patterns only for its induction head and partner. Your own run shows every head.` : `${name}: waiting for the next test.`;
      return;
    }
    const role =
      copy >= INDUCTION.prefix
        ? `an induction head: in the second half, every row lights up the diagonal stripe half a window back, the character after the earlier occurrence of its own (copy score ${copy.toFixed(2)})`
        : prev >= 0.5
          ? `a previous-character head: each row looks one step back, the bright line just under the diagonal (score ${prev.toFixed(2)})`
          : `neither: copy ${copy.toFixed(2)}, previous character ${prev.toFixed(2)}`;
    cap.textContent = `${name} on ${T} random characters, the second ${T / 2} repeating the first. Row: where it is reading; column: where it looks. It is ${role}.`;
  }

  private renderKnockouts(): void {
    const s = this.subject(this.view.heads) ?? this.recorded ?? this.live;
    const box = $('ko-table');
    if (!s) return;
    const ko = s.knockouts;
    const pt = s.patching;
    if (!ko || !pt) {
      box.replaceChildren(h('p', { class: 'pane-empty' }, s.stage ?? 'Knockouts and patching run when training ends (they take under a second).'));
      return;
    }
    const H = s.cfg.nHead;
    const maxCopy = Math.max(0.05, ...ko.copy.map(Math.abs));
    const maxText = Math.max(0.02, ...ko.text.map(Math.abs));
    const rows = ko.copy.map((_, k) =>
      h(
        'tr',
        { class: this.head === k ? 'sel' : '' },
        h('th', { scope: 'row' }, headName(k, H)),
        h('td', {}, bar(ko.copy[k] / maxCopy, 'copy'), h('span', { class: 'num' }, signed(ko.copy[k]))),
        h('td', {}, bar(ko.text[k] / maxText, 'text'), h('span', { class: 'num' }, signed(ko.text[k]))),
        h('td', {}, bar(Math.max(0, pt.recovered[k]), 'patch'), h('span', { class: 'num' }, `${Math.round(pt.recovered[k] * 100)}%`)),
      ),
    );
    box.replaceChildren(
      h(
        'table',
        { class: 'ko' },
        h(
          'thead',
          {},
          h(
            'tr',
            {},
            h('th', {}, 'Head'),
            h('th', { title: `Change in loss on the test's second copy (base ${ko.baseCopy.toFixed(2)} nats) with the head's output set to zero` }, 'Knocked out: copy loss'),
            h('th', { title: `Change in loss on held-out text (base ${ko.baseText.toFixed(2)} nats)` }, 'Knocked out: text loss'),
            h('th', { title: `Corrupted test (first copy replaced): second-copy loss ${pt.corrupt.toFixed(2)}; clean ${pt.clean.toFixed(2)}. Patching one head's output in from the clean run restores this share.` }, 'Patched in from clean'),
          ),
        ),
        h('tbody', {}, ...rows),
      ),
      h(
        'p',
        { class: 'ko-note' },
        `Zero-ablation of each head's output, in nats per character: on the induction test's second copy (base ${ko.baseCopy.toFixed(2)}) and on held-out text (base ${ko.baseText.toFixed(2)}). Activation patching: with the first copy replaced by other random characters the second copy costs ${pt.corrupt.toFixed(2)} nats instead of ${pt.clean.toFixed(2)}; each row restores one head's output from the clean run. A previous-character head restores little on its own: the head that copies still reads the corrupted first copy.`,
      ),
    );
  }

  private renderKnockWrite(): void {
    const s = this.subject(this.view.heads) ?? this.recorded ?? this.live;
    const label = $('ko-write-label');
    const H = s?.cfg.nHead ?? 4;
    const names = [...this.knocked]
      .sort((a, b) => a - b)
      .map((k) => headName(k, H))
      .join(', ');
    label.textContent = this.knocked.size ? `Writing, with ${names} knocked out` : 'Shift-click heads above to knock them out';
    $('copy-off-label').textContent = this.knocked.size ? `${names} off` : 'knocked out';
    this.weightsGate($('ko-gate'), this.view.heads);
  }

  // ------------------------------------------------------------------ FEATURES

  private renderFeatures(): void {
    if (this.mode !== 'features') return;
    const s = this.subject(this.view.features) ?? this.recorded ?? this.live;
    $('features-switch').replaceChildren(this.viewSwitch('features'));
    const status = $('sae-status');
    const sae = s?.sae;
    if (!s || !sae) {
      status.replaceChildren(h('p', { class: 'pane-empty' }, s?.stage ?? 'The sparse autoencoder trains when the run ends, on the trained model’s residual stream.'));
      $('feature-list').replaceChildren();
      $('feature-detail').replaceChildren();
      return;
    }
    const facts: string[] = [];
    if (sae.opts.features) facts.push(`${int(sae.opts.features)} features on the residual stream entering block ${sae.opts.site + 1}`);
    if (sae.progress) facts.push(`training: step ${int(sae.progress.step)} of ${int(sae.progress.total)}`);
    else if (sae.seconds !== null) facts.push(`${int(sae.opts.steps)} steps of ${int(sae.opts.batch)} positions in ${sae.seconds.toFixed(1)} s on the GPU`);
    const last = sae.curve[sae.curve.length - 1];
    const fvu = sae.fvu ?? last?.mse ?? null;
    const l0 = sae.l0 ?? last?.l0 ?? null;
    const metrics = h(
      'dl',
      { class: 'sae-metrics' },
      metric('Unexplained', fvu === null ? '–' : `${(fvu * 100).toFixed(1)}%`, 'Fraction of the residual stream’s variance the reconstruction misses'),
      metric('Active', l0 === null ? '–' : l0.toFixed(1), 'Features firing per character, on average (L0)'),
      metric('Dead', sae.dead === null ? '–' : `${(sae.dead * 100).toFixed(1)}%`, 'Features that never fire on any position of the text'),
      metric('Positions', sae.positions ? int(sae.positions) : '–', 'Characters of the text it was trained and measured on'),
    );
    const curve = h('canvas', { class: 'sae-curve', width: '240', height: '56', 'aria-label': 'Training curve: unexplained variance and active features' });
    status.replaceChildren(h('p', { class: 'sae-facts' }, facts.join(' · ')), h('div', { class: 'sae-row' }, metrics, curve, s.stage ? h('p', { class: 'pane-empty' }, s.stage) : null));
    drawCurve(curve, sae.curve);
    this.renderFeatureList();
    this.renderFeatureDetail();
  }

  private labelled = new WeakMap<SaeFeature, FeatureLabel | null>();

  private labelOf(s: Subject, f: SaeFeature): FeatureLabel | null {
    if (!this.labelled.has(f)) this.labelled.set(f, labelFeature(f.top.map((t) => this.context(s, t.pos, 20, 6))));
    return this.labelled.get(f) ?? null;
  }

  private context(s: Subject, pos: number, before: number, after: number): { text: string; at: number } {
    const a = Math.max(0, pos - before);
    const b = Math.min(s.tokens.length, pos + after + 1);
    let text = '';
    for (let i = a; i < b; i++) text += s.chars[s.tokens[i]] ?? '';
    return { text, at: pos - a };
  }

  private sortedFeatures(s: Subject): SaeFeature[] {
    const fs = s.sae?.features ?? [];
    const q = this.filter.trim().toLowerCase();
    const scored = fs.map((f) => ({ f, l: this.labelOf(s, f) }));
    const shown = q ? scored.filter(({ f, l }) => String(f.id) === q.replace('#', '') || (l?.text.toLowerCase().includes(q) ?? false)) : scored;
    return shown
      .sort((a, b) => {
        const la = a.l ? a.l.covered / a.l.of : 0;
        const lb = b.l ? b.l.covered / b.l.of : 0;
        if (a.f.top.length === 0 || b.f.top.length === 0) return b.f.top.length - a.f.top.length;
        return lb - la || (b.f.top[0]?.value ?? 0) - (a.f.top[0]?.value ?? 0);
      })
      .map(({ f }) => f);
  }

  private renderFeatureList(): void {
    const s = this.subject(this.view.features) ?? this.recorded ?? this.live;
    const list = $('feature-list');
    if (!s?.sae) return;
    if (!s.sae.features) {
      list.replaceChildren(h('p', { class: 'pane-empty' }, s.sae.progress ? 'Features appear when training ends.' : 'Loading the features…'));
      return;
    }
    const fs = this.sortedFeatures(s);
    if (this.feature === null && fs.length) {
      this.feature = fs[0].id;
      this.scheduleWrite();
    }
    const labelled = fs.filter((f) => this.labelOf(s, f)).length;
    $('feature-count').textContent = `${int(fs.length)} shown · ${int(labelled)} with a rule`;
    const rows = fs.slice(0, 400).map((f) => {
      const l = this.labelOf(s, f);
      const b = h(
        'button',
        { type: 'button', class: 'feature-row', role: 'option', 'aria-selected': String(this.feature === f.id) },
        h('span', { class: 'fid' }, `#${f.id}`),
        h('span', { class: `flabel${l ? '' : ' flabel--none'}` }, l ? l.text : f.top.length ? 'no rule fits' : 'never fires'),
        h('span', { class: 'fdens', title: `fires on ${(f.density * 100).toFixed(2)}% of characters` }, density(f.density)),
      );
      b.addEventListener('click', () => {
        this.feature = f.id;
        this.clampValue = null;
        for (const r of list.querySelectorAll('.feature-row')) r.setAttribute('aria-selected', String(r === b));
        this.renderFeatureDetail();
        this.scheduleWrite();
      });
      return b;
    });
    list.replaceChildren(...rows);
  }

  private renderFeatureDetail(): void {
    const s = this.subject(this.view.features) ?? this.recorded ?? this.live;
    const box = $('feature-detail');
    const f = s?.sae?.features?.find((x) => x.id === this.feature);
    if (!s || !f) {
      box.replaceChildren();
      return;
    }
    const l = this.labelOf(s, f);
    const max = f.top[0]?.value ?? 0;
    const contexts = f.top.map((t) => {
      const c = this.context(s, t.pos, 26, 8);
      const before = c.text.slice(0, c.at).replace(/\n/g, '↵');
      const after = c.text.slice(c.at + 1).replace(/\n/g, '↵');
      return h(
        'li',
        {},
        h('span', { class: 'ctx' }, h('span', { class: 'ctx-pre' }, before), h('mark', { style: `--w:${(0.25 + 0.75 * (t.value / Math.max(max, 1e-9))).toFixed(3)}` }, say(c.text[c.at] ?? '')), h('span', { class: 'ctx-post' }, after)),
        h('span', { class: 'num' }, t.value.toFixed(2)),
      );
    });
    const value = this.clampValue ?? max;
    const slider = h('input', { type: 'range', id: 'clamp', min: '0', max: (Math.max(max, 0.1) * 3).toFixed(3), step: (Math.max(max, 0.1) / 50).toFixed(4), value: value.toFixed(4), 'aria-describedby': 'clamp-out' }) as HTMLInputElement;
    const out = h('output', { id: 'clamp-out', for: 'clamp' }, value.toFixed(2));
    slider.addEventListener('input', () => {
      this.clampValue = Number(slider.value);
      out.textContent = this.clampValue.toFixed(2);
      this.scheduleWrite();
    });
    const preset = (name: string, v: number) => {
      const b = h('button', { type: 'button', class: 'key key--small' }, name);
      b.addEventListener('click', () => {
        this.clampValue = v;
        slider.value = String(v);
        out.textContent = v.toFixed(2);
        this.scheduleWrite();
      });
      return b;
    };
    box.replaceChildren(
      h('p', { class: 'fd-head' }, h('b', {}, `#${f.id}`), ' ', l ? h('span', { class: 'fd-label' }, l.text) : h('span', { class: 'fd-label fd-label--none' }, 'no rule fits these contexts')),
      h(
        'p',
        { class: 'fd-facts' },
        l ? `Label by rule: holds for ${l.covered} of the ${l.of} strongest contexts. ` : '',
        `Fires on ${(f.density * 100).toFixed(2)}% of characters, mean ${f.mean.toFixed(2)} when it does, strongest ${max.toFixed(2)}.`,
      ),
      h('ol', { class: 'contexts' }, ...contexts),
      h(
        'div',
        { class: 'clamp' },
        h('label', { for: 'clamp' }, 'Pin it at'),
        slider,
        out,
        preset('Off', 0),
        preset('Strongest', max),
        preset('Triple', max * 3),
      ),
      h('div', { id: 'fd-gate' }),
    );
    this.weightsGate($('fd-gate'), this.view.features);
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
        h('div', { class: 'gate' }, h('progress', { max: String(st.total), value: String(st.received), 'aria-label': 'Downloading the recorded model' }), h('span', { class: 'gate-note' }, `${bytes(st.received)} of ${size}${st.cached ? ' from cache' : ''}`), cancel),
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

  private scheduleWrite(): void {
    window.clearTimeout(this.writeTimer);
    this.writeTimer = window.setTimeout(() => void this.write(), reducedMotion() ? 0 : 160);
  }

  /** Called by the controller when the model being written from changes (a new checkpoint). */
  refreshWriting(): void {
    if (this.mode !== 'train') this.scheduleWrite();
  }

  private async write(): Promise<void> {
    const epoch = ++this.writeEpoch;
    if (this.mode === 'heads') {
      const which = this.view.heads;
      const target = $('ko-write');
      const base = $('ko-base');
      const s = this.subject(which);
      if (!s) return;
      const weights = which === 'recorded' ? (this.recordedWeights?.model ?? null) : null;
      if (which === 'recorded' && !weights) {
        base.textContent = '';
        target.textContent = '';
        return;
      }
      const off = this.knocked.size ? { knockout: [...this.knocked] } : null;
      const [a, b, ca, cb] = await Promise.all([
        this.host.write(which, null, weights, 'free'),
        off ? this.host.write(which, off, weights, 'free') : Promise.resolve(null),
        this.host.write(which, null, weights, 'copy'),
        off ? this.host.write(which, off, weights, 'copy') : Promise.resolve(null),
      ]);
      if (epoch !== this.writeEpoch) return;
      showDiff(base, a ?? '', null);
      showDiff(target, b ?? '', b ? (a ?? '') : null);
      const cp = this.host.copyPrompt(which);
      if (cp) {
        $('copy-prompt').textContent = cp.prompt.replace(/\n/g, '↵');
        showCopy($('copy-base'), ca ?? '', cp.expected);
        showCopy($('copy-off'), cb ?? '', cb === null ? null : cp.expected);
      }
      return;
    }
    if (this.mode === 'features') {
      const which = this.view.features;
      const s = this.subject(which);
      const f = s?.sae?.features?.find((x) => x.id === this.feature);
      const params = s?.sae?.params;
      const box = $('feature-detail');
      if (!s || !f || !s.sae || !params) return;
      const weights = which === 'recorded' ? (this.recordedWeights?.model ?? null) : null;
      if (which === 'recorded' && !weights) return;
      const lay = saeLayout(s.cfg.dModel, s.sae.opts.features);
      const enc = new Float32Array(lay.C);
      for (let c = 0; c < lay.C; c++) enc[c] = params[lay.wEnc + c * lay.F + f.id];
      const clamp: FeatureClamp & { site: number } = {
        enc,
        bias: params[lay.bEnc + f.id],
        dec: params.slice(lay.wDec + f.id * lay.C, lay.wDec + (f.id + 1) * lay.C),
        scale: s.sae.scale,
        value: this.clampValue ?? f.top[0]?.value ?? 0,
        site: s.sae.opts.site,
      };
      const [a, b] = await Promise.all([this.host.write(which, null, weights, 'free'), this.host.write(which, { clamp }, weights, 'free')]);
      if (epoch !== this.writeEpoch) return;
      let cmp = box.querySelector('.compare');
      if (!cmp) {
        cmp = h(
          'div',
          { class: 'compare' },
          h('div', {}, h('p', { class: 'compare-label' }, 'As trained'), h('pre', { class: 'compare-text', id: 'fd-base' })),
          h('div', {}, h('p', { class: 'compare-label', id: 'fd-pinned-label' }), h('pre', { class: 'compare-text', id: 'fd-pinned' })),
        );
        box.append(cmp);
      }
      $('fd-pinned-label').textContent = `#${f.id} pinned at ${clamp.value.toFixed(2)}`;
      showDiff($('fd-base'), a ?? '', null);
      showDiff($('fd-pinned'), b ?? '', a ?? '');
    }
  }
}

function signed(v: number): string {
  return `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`;
}

function bar(v: number, kind: string): HTMLElement {
  const w = Math.max(0, Math.min(1, Math.abs(v)));
  return h('span', { class: `bar bar--${kind}${v < 0 ? ' bar--neg' : ''}`, style: `--v:${w.toFixed(3)}` });
}

function metric(name: string, value: string, title: string): HTMLElement {
  return h('div', { title }, h('dt', {}, name), h('dd', {}, value));
}

function density(d: number): string {
  if (d === 0) return '0';
  if (d >= 0.01) return `${(d * 100).toFixed(1)}%`;
  return `${(d * 100).toFixed(2)}%`;
}

/**
 * Text, with the first character that differs from `against` marked: same random numbers, so
 * everything before it is shared, and everything after follows from it.
 */
function showDiff(el: HTMLElement, text: string, against: string | null): void {
  if (!text) {
    el.textContent = '';
    return;
  }
  let i = 0;
  if (against !== null) while (i < text.length && text[i] === against[i]) i++;
  if (against === null || i >= text.length) {
    el.textContent = text;
    return;
  }
  el.replaceChildren(h('span', { class: 'same' }, text.slice(0, i)), h('mark', { title: `differs from character ${i + 1} on` }, text[i]), text.slice(i + 1));
}

/** A copying attempt: each character right (it matches the run) or wrong. */
function showCopy(el: HTMLElement, text: string, expected: string | null): void {
  if (expected === null) {
    el.textContent = '';
    return;
  }
  let right = 0;
  const spans = Array.from(text, (c, i) => {
    const ok = c === expected[i];
    if (ok) right++;
    return h('span', { class: ok ? 'ok' : 'bad' }, c === '\n' ? '↵' : c);
  });
  el.replaceChildren(...spans, h('span', { class: 'score' }, ` ${right}/${text.length}`));
}

function drawCurve(canvas: HTMLCanvasElement, curve: readonly SaeStep[]): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const W = 240;
  const H = 56;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = `${W}px`;
  canvas.style.height = `${H}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = cssVar('--grid-2');
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
  if (curve.length < 2) return;
  const last = curve[curve.length - 1].step;
  const maxL0 = Math.max(...curve.map((c) => c.l0), 1);
  const plot = (get: (c: SaeStep) => number, max: number, colour: string) => {
    ctx.strokeStyle = colour;
    ctx.beginPath();
    curve.forEach((c, i) => {
      const x = 2 + ((W - 4) * c.step) / Math.max(last, 1);
      const y = H - 3 - ((H - 6) * Math.min(1, get(c) / max));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  plot((c) => c.mse, 1, cssVar('--ch1'));
  plot((c) => c.l0, maxL0, cssVar('--ch2'));
}

export { sayChar };
