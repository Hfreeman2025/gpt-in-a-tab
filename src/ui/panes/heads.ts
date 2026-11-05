import { INDUCTION, headName } from '../../model/probe';
import { $, cssVar, h, int } from '../dom';
import { bar, showCopy, showDiff, signed } from './parts';
import type { PaneContext } from './types';

/**
 * HEADS: every head's job on the induction test, its attention pattern, what knocking it out or
 * patching it in does, and the model writing with chosen heads knocked out.
 */
export class HeadsPane {
  /** The head looked at (pattern, highlighted row), and the heads knocked out while writing. */
  head: number | null = null;
  readonly knocked = new Set<number>();
  private patternTimer = 0;

  constructor(private readonly c: PaneContext) {}

  render(): void {
    if (this.c.mode() !== 'heads') return;
    const s = this.c.shown('heads');
    const run = this.c.run();
    $('heads-switch').replaceChildren(this.c.viewSwitch('heads'));
    $('heads-note').textContent = s
      ? s.which === 'recorded'
        ? `recorded run, ${run?.recorded.slice(0, 10) ?? ''}, ${run?.device ?? ''}`
        : s.knockouts
          ? `this run, at step ${int(s.latest?.step ?? 0)}`
          : 'this run: knockouts and patching run when training ends'
      : '';
    this.renderGrid();
    this.renderPattern();
    this.renderKnockouts();
    this.renderKnockWrite();
  }

  renderGrid(): void {
    const s = this.c.shown('heads');
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
          this.c.host.selectHead(this.head);
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
    this.c.scheduleWrite();
  }

  /** While training, the live patterns are re-read from the GPU now and then, not on every test. */
  schedulePatterns(): void {
    if (this.c.view('heads') !== 'live' || this.patternTimer) return;
    this.patternTimer = window.setTimeout(() => {
      this.patternTimer = 0;
      void this.refreshLivePatterns();
    }, 400);
  }

  async refreshLivePatterns(): Promise<void> {
    const s = this.c.live();
    if (!s || this.c.view('heads') !== 'live' || this.c.mode() !== 'heads' || s.knockouts) return;
    const all = await this.c.host.livePatterns();
    if (!all || this.c.live() !== s) return;
    const T = s.cfg.context;
    for (let k = 0; k < s.cfg.nLayer * s.cfg.nHead; k++) s.patterns.set(k, all.subarray(k * T * T, (k + 1) * T * T));
    this.renderPattern();
  }

  /** The chosen head's attention on the induction test, a pixel per query and key position. */
  private renderPattern(): void {
    const s = this.c.shown('heads');
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
      // Square root, so a head that spreads its attention still shows where it spreads it.
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
      cap.textContent =
        s.which === 'recorded'
          ? `${name}: the recorded run keeps patterns only for its induction head and partner. Your own run shows every head.`
          : `${name}: waiting for the next test.`;
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
    const s = this.c.shown('heads');
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
            h(
              'th',
              { title: `Change in loss on the test's second copy (base ${ko.baseCopy.toFixed(2)} nats) with the head's output set to zero` },
              'Knocked out: copy loss',
            ),
            h('th', { title: `Change in loss on held-out text (base ${ko.baseText.toFixed(2)} nats)` }, 'Knocked out: text loss'),
            h(
              'th',
              {
                title: `Corrupted test (first copy replaced): second-copy loss ${pt.corrupt.toFixed(2)}; clean ${pt.clean.toFixed(2)}. Patching one head's output in from the clean run restores this share.`,
              },
              'Patched in from clean',
            ),
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
    const s = this.c.shown('heads');
    const H = s?.cfg.nHead ?? 4;
    const names = [...this.knocked]
      .sort((a, b) => a - b)
      .map((k) => headName(k, H))
      .join(', ');
    $('ko-write-label').textContent = this.knocked.size ? `Writing, with ${names} knocked out` : 'Shift-click heads above to knock them out';
    $('copy-off-label').textContent = this.knocked.size ? `${names} off` : 'knocked out';
    this.c.weightsGate($('ko-gate'), this.c.view('heads'));
  }

  /** Writes as trained and with the chosen heads knocked out, freely and on the copying demonstration. */
  async write(stillCurrent: () => boolean): Promise<void> {
    const which = this.c.view('heads');
    const target = $('ko-write');
    const base = $('ko-base');
    const s = this.c.subject(which);
    if (!s) return;
    const weights = which === 'recorded' ? (this.c.recordedWeights()?.model ?? null) : null;
    if (which === 'recorded' && !weights) {
      base.textContent = '';
      target.textContent = '';
      return;
    }
    const host = this.c.host;
    const off = this.knocked.size ? { knockout: [...this.knocked] } : null;
    const [a, b, ca, cb] = await Promise.all([
      host.write(which, null, weights, 'free'),
      off ? host.write(which, off, weights, 'free') : Promise.resolve(null),
      host.write(which, null, weights, 'copy'),
      off ? host.write(which, off, weights, 'copy') : Promise.resolve(null),
    ]);
    if (!stillCurrent()) return;
    showDiff(base, a ?? '', null);
    showDiff(target, b ?? '', b ? (a ?? '') : null);
    const cp = host.copyPrompt(which);
    if (cp) {
      $('copy-prompt').textContent = cp.prompt.replace(/\n/g, '↵');
      showCopy($('copy-base'), ca ?? '', cp.expected);
      showCopy($('copy-off'), cb ?? '', cb === null ? null : cp.expected);
    }
  }
}
