import { type FeatureClamp, saeLayout } from '../../cpu/sae';
import type { SaeFeature } from '../../gpu/sae';
import { type FeatureLabel, labelFeature } from '../../model/labels';
import { $, h, int } from '../dom';
import { density, drawCurve, metric, say, showDiff } from './parts';
import type { PaneContext, Subject } from './types';

/**
 * FEATURES: the sparse autoencoder's fidelity, its features listed by how well a rule labels
 * them, one feature's strongest contexts, and the model writing with that feature pinned.
 */
export class FeaturesPane {
  /** The feature in view, the value it is pinned at (null: its strongest), the list's filter. */
  feature: number | null = null;
  private clampValue: number | null = null;
  private filter = '';
  private readonly labelled = new WeakMap<SaeFeature, FeatureLabel | null>();

  constructor(private readonly c: PaneContext) {
    $<HTMLInputElement>('feature-filter').addEventListener('input', (e) => {
      this.filter = (e.target as HTMLInputElement).value;
      this.renderList();
    });
  }

  render(): void {
    if (this.c.mode() !== 'features') return;
    const s = this.c.shown('features');
    $('features-switch').replaceChildren(this.c.viewSwitch('features'));
    const status = $('sae-status');
    const sae = s?.sae;
    if (!s || !sae) {
      status.replaceChildren(
        h('p', { class: 'pane-empty' }, s?.stage ?? 'The sparse autoencoder trains when the run ends, on the trained model’s residual stream.'),
      );
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
    status.replaceChildren(
      h('p', { class: 'sae-facts' }, facts.join(' · ')),
      h('div', { class: 'sae-row' }, metrics, curve, s.stage ? h('p', { class: 'pane-empty' }, s.stage) : null),
    );
    drawCurve(curve, sae.curve);
    this.renderList();
    this.renderDetail();
  }

  /** Labels are worked out once per feature: the rules read the text around its top contexts. */
  private labelOf(s: Subject, f: SaeFeature): FeatureLabel | null {
    if (!this.labelled.has(f)) this.labelled.set(f, labelFeature(f.top.map((t) => context(s, t.pos, 20, 6))));
    return this.labelled.get(f) ?? null;
  }

  /** Best-labelled first (by how many contexts the rule covers), then the strongest. */
  private sorted(s: Subject): SaeFeature[] {
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

  private renderList(): void {
    const s = this.c.shown('features');
    const list = $('feature-list');
    if (!s?.sae) return;
    if (!s.sae.features) {
      list.replaceChildren(h('p', { class: 'pane-empty' }, s.sae.progress ? 'Features appear when training ends.' : 'Loading the features…'));
      return;
    }
    const fs = this.sorted(s);
    if (this.feature === null && fs.length) {
      this.feature = fs[0].id;
      this.c.scheduleWrite();
    }
    const labelled = fs.filter((f) => this.labelOf(s, f)).length;
    $('feature-count').textContent = `${int(fs.length)} shown · ${int(labelled)} with a rule`;
    // The first 400 are plenty to browse; the filter reaches the rest.
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
        this.renderDetail();
        this.c.scheduleWrite();
      });
      return b;
    });
    list.replaceChildren(...rows);
  }

  private renderDetail(): void {
    const s = this.c.shown('features');
    const box = $('feature-detail');
    const f = s?.sae?.features?.find((x) => x.id === this.feature);
    if (!s || !f) {
      box.replaceChildren();
      return;
    }
    const l = this.labelOf(s, f);
    const max = f.top[0]?.value ?? 0;
    const contexts = f.top.map((t) => {
      const c = context(s, t.pos, 26, 8);
      const before = c.text.slice(0, c.at).replace(/\n/g, '↵');
      const after = c.text.slice(c.at + 1).replace(/\n/g, '↵');
      return h(
        'li',
        {},
        h(
          'span',
          { class: 'ctx' },
          h('span', { class: 'ctx-pre' }, before),
          h('mark', { style: `--w:${(0.25 + 0.75 * (t.value / Math.max(max, 1e-9))).toFixed(3)}` }, say(c.text[c.at] ?? '')),
          h('span', { class: 'ctx-post' }, after),
        ),
        h('span', { class: 'num' }, t.value.toFixed(2)),
      );
    });
    const value = this.clampValue ?? max;
    const slider = h('input', {
      type: 'range',
      id: 'clamp',
      min: '0',
      max: (Math.max(max, 0.1) * 3).toFixed(3),
      step: (Math.max(max, 0.1) / 50).toFixed(4),
      value: value.toFixed(4),
      'aria-describedby': 'clamp-out',
    }) as HTMLInputElement;
    const out = h('output', { id: 'clamp-out', for: 'clamp' }, value.toFixed(2));
    slider.addEventListener('input', () => {
      this.clampValue = Number(slider.value);
      out.textContent = this.clampValue.toFixed(2);
      this.c.scheduleWrite();
    });
    const preset = (name: string, v: number) => {
      const b = h('button', { type: 'button', class: 'key key--small' }, name);
      b.addEventListener('click', () => {
        this.clampValue = v;
        slider.value = String(v);
        out.textContent = v.toFixed(2);
        this.c.scheduleWrite();
      });
      return b;
    };
    box.replaceChildren(
      h(
        'p',
        { class: 'fd-head' },
        h('b', {}, `#${f.id}`),
        ' ',
        l ? h('span', { class: 'fd-label' }, l.text) : h('span', { class: 'fd-label fd-label--none' }, 'no rule fits these contexts'),
      ),
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
    this.c.weightsGate($('fd-gate'), this.c.view('features'));
  }

  /** Writes as trained and with the feature in view pinned, side by side. */
  async write(stillCurrent: () => boolean): Promise<void> {
    const which = this.c.view('features');
    const s = this.c.subject(which);
    const f = s?.sae?.features?.find((x) => x.id === this.feature);
    const params = s?.sae?.params;
    const box = $('feature-detail');
    if (!s || !f || !s.sae || !params) return;
    const weights = which === 'recorded' ? (this.c.recordedWeights()?.model ?? null) : null;
    if (which === 'recorded' && !weights) return;
    // Only feature f's encoder column and decoder row are needed to pin it (see clampFeature).
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
    const host = this.c.host;
    const [a, b] = await Promise.all([host.write(which, null, weights, 'free'), host.write(which, { clamp }, weights, 'free')]);
    if (!stillCurrent()) return;
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

/** The text around a corpus position, and where in it that position falls. */
function context(s: Subject, pos: number, before: number, after: number): { text: string; at: number } {
  const a = Math.max(0, pos - before);
  const b = Math.min(s.tokens.length, pos + after + 1);
  let text = '';
  for (let i = a; i < b; i++) text += s.chars[s.tokens[i]] ?? '';
  return { text, at: pos - a };
}
