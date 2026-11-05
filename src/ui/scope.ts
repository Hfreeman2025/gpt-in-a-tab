import { type Moment, type ProbeReading, headName } from '../model/probe';
import { cssVar, reducedMotion } from './dom';
import { type Geometry, drawLanes, drawMoment } from './scope-lanes';

export interface Point {
  readonly step: number;
  readonly value: number;
}

export interface Mark {
  readonly step: number;
  /** Its sample has been drawn and can be shown. */
  readonly ready: boolean;
}

/** A recorded run drawn behind the live one, as a scope's reference waveform. */
export interface Reference {
  readonly train: readonly Point[];
  readonly copy: readonly Point[];
  readonly probes: readonly ProbeReading[];
  readonly moment: Moment | null;
  readonly total: number;
  readonly heads: { readonly L: number; readonly H: number };
}

/** A value in [10ⁿ × {1, 2, 2.5, 5}] at least `x`. */
function nice(x: number): number {
  const p = 10 ** Math.floor(Math.log10(Math.max(x, 1e-9)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= x) return m * p;
  return 10 * p;
}

/**
 * The trace: a graticule, four channels (training loss, held-out loss, learning rate, and the
 * loss on the second copy of the induction test), the chance line, a tick for every checkpoint
 * and the cursor, which is the scrubber. Under it, on the same time base, one lane per attention
 * head, like a logic analyser's digital channels: orange where the head matches prefixes (the
 * induction test), phosphor where it attends to the previous character. A recorded run can sit
 * behind it all as a dim reference. Pointer, touch and keyboard move the cursor; the owner
 * decides which checkpoint that means.
 */
export class Scope {
  private train: Point[] = [];
  private smooth: Point[] = [];
  private evals: Point[] = [];
  private lr: Point[] = [];
  private marks: Mark[] = [];
  private copy: Point[] = [];
  private probes: ProbeReading[] = [];
  private shape = { L: 0, H: 0 };
  private moment: Moment | null = null;
  private momentAt = 0;
  private ref: Reference | null = null;
  private total = 1;
  private chance = 0;
  private vocab = 0;
  private peakLr = 1;
  private ema = 0;
  private emaN = 0;
  private cursor: number | null = null;
  private live = true;
  private queued = false;
  private dragging = false;
  readonly divisions = 10;
  /** Nice scale values from the last draw, for the readout beside the trace. */
  scale = { natsPerDiv: 0.5, stepsPerDiv: 100 };

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly handlers: { scrub(step: number): void; step(delta: number): void; live(): void; scaled(text: string): void },
  ) {
    new ResizeObserver(() => this.draw()).observe(canvas);
    // Canvas text is drawn in whatever face is ready; draw again once Plex has arrived.
    void document.fonts?.ready.then(() => this.request());
    canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      this.dragging = true;
      canvas.setPointerCapture(e.pointerId);
      this.handlers.scrub(this.stepAt(e.clientX));
    });
    canvas.addEventListener('pointermove', (e) => {
      if (this.dragging) this.handlers.scrub(this.stepAt(e.clientX));
    });
    const stop = () => {
      this.dragging = false;
    };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
    canvas.addEventListener('keydown', (e) => {
      const keys: Record<string, () => void> = {
        ArrowLeft: () => this.handlers.step(-1),
        ArrowDown: () => this.handlers.step(-1),
        ArrowRight: () => this.handlers.step(1),
        ArrowUp: () => this.handlers.step(1),
        PageDown: () => this.handlers.step(-5),
        PageUp: () => this.handlers.step(5),
        Home: () => this.handlers.step(-1e9),
        End: () => this.handlers.live(),
      };
      const f = keys[e.key];
      if (f) {
        e.preventDefault();
        f();
      }
    });
  }

  reset(vocab: number, total: number, peakLr: number): void {
    this.train = [];
    this.smooth = [];
    this.evals = [];
    this.lr = [];
    this.marks = [];
    this.copy = [];
    this.probes = [];
    this.moment = null;
    this.vocab = vocab;
    this.chance = vocab > 0 ? Math.log(vocab) : 0;
    this.total = Math.max(1, total);
    this.peakLr = peakLr;
    this.ema = 0;
    this.emaN = 0;
    this.cursor = null;
    this.live = true;
    this.request();
  }

  setTotal(total: number): void {
    if (total !== this.total) {
      this.total = Math.max(1, total);
      this.request();
    }
  }

  addSteps(points: readonly { step: number; loss: number; lr: number }[]): void {
    for (const p of points) {
      if (!Number.isFinite(p.loss)) continue;
      this.train.push({ step: p.step, value: p.loss });
      this.lr.push({ step: p.step, value: p.lr });
      this.emaN++;
      this.ema = 0.96 * this.ema + 0.04 * p.loss;
      this.smooth.push({ step: p.step, value: this.ema / (1 - 0.96 ** this.emaN) });
    }
    this.request();
  }

  addEvals(points: readonly { step: number; loss: number }[]): void {
    for (const p of points) if (Number.isFinite(p.loss)) this.evals.push({ step: p.step, value: p.loss });
    this.request();
  }

  /** Layers and heads, for the lanes. */
  setShape(L: number, H: number): void {
    this.shape = { L, H };
    this.request();
  }

  addProbes(readings: readonly ProbeReading[]): void {
    for (const r of readings) {
      this.probes.push(r);
      if (Number.isFinite(r.copySecond)) this.copy.push({ step: r.step, value: r.copySecond });
    }
    this.request();
  }

  /** The induction moment; `reveal` draws it in (unless the reader prefers reduced motion). */
  setMoment(m: Moment | null, reveal = false): void {
    this.moment = m;
    this.momentAt = reveal && !reducedMotion() ? performance.now() : 0;
    this.request();
  }

  setReference(ref: Reference | null): void {
    this.ref = ref;
    this.request();
  }

  setMarks(marks: readonly Mark[]): void {
    this.marks = [...marks];
    this.request();
  }

  /** The step shown, and whether it is following the run. */
  setCursor(step: number | null, live: boolean): void {
    this.cursor = step;
    this.live = live;
    this.request();
  }

  get latestLoss(): number {
    return this.smooth.length ? this.smooth[this.smooth.length - 1].value : NaN;
  }

  lossAt(step: number): number {
    return this.valueAt(this.smooth, step);
  }

  heldOutAt(step: number): number {
    return this.valueAt(this.evals, step);
  }

  private valueAt(series: readonly Point[], step: number): number {
    let lo = 0;
    let hi = series.length - 1;
    let best = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (series[mid].step <= step) {
        best = mid;
        lo = mid + 1;
      } else hi = mid - 1;
    }
    return best >= 0 ? series[best].value : NaN;
  }

  private geometry(): Geometry {
    const W = this.canvas.clientWidth;
    const H = this.canvas.clientHeight;
    const n = this.shape.L * this.shape.H;
    const left = n ? 48 : 34;
    const right = 14;
    const top = 12;
    const axis = 18;
    const laneH = n ? Math.max(4, Math.min(10, Math.floor((H * 0.32) / n))) : 0;
    const layerGap = 3;
    const lanes = n ? n * laneH + (this.shape.L - 1) * layerGap : 0;
    const bottom = axis + (n ? lanes + 10 : 2);
    const h = H - top - bottom;
    return { W, H, left, right, top, bottom, w: W - left - right, h, laneH, layerGap, lanesTop: top + h + axis + 6, lanes };
  }

  private stepAt(clientX: number): number {
    const r = this.canvas.getBoundingClientRect();
    const g = this.geometry();
    const span = this.scale.stepsPerDiv * this.divisions;
    const f = (clientX - r.left - g.left) / g.w;
    return Math.max(0, Math.round(f * span));
  }

  private request(): void {
    if (this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => {
      this.queued = false;
      this.draw();
    });
  }

  draw(): void {
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const g = this.geometry();
    if (g.W <= 0 || g.H <= 0) return;
    if (c.width !== Math.round(g.W * dpr) || c.height !== Math.round(g.H * dpr)) {
      c.width = Math.round(g.W * dpr);
      c.height = Math.round(g.H * dpr);
    }
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, g.W, g.H);
    const css = (n: string) => cssVar(n);
    // Numbers are records (Plex Mono); names of things are signs (Plex Sans Condensed).
    const label = `10.5px ${css('--mono') || 'monospace'}`;

    // Scales: ten divisions across, as many 50-pixel divisions down as fit.
    const target = Math.max(4, Math.min(8, Math.round(g.h / 52)));
    let top = Math.max(this.chance, 1);
    for (const p of this.evals) top = Math.max(top, p.value);
    for (const p of this.smooth) top = Math.max(top, p.value);
    for (const p of this.copy) top = Math.max(top, p.value);
    const ref = this.ref;
    if (ref) for (const p of [...ref.train, ...ref.copy]) top = Math.max(top, p.value);
    const natsPerDiv = nice((top * 1.05) / target);
    const vdiv = Math.max(3, Math.ceil((top * 1.05) / natsPerDiv));
    const stepsPerDiv = nice(Math.max(this.total, ref?.total ?? 0) / this.divisions);
    this.scale = { natsPerDiv, stepsPerDiv };
    const yMax = natsPerDiv * vdiv;
    const xMax = stepsPerDiv * this.divisions;
    const x = (s: number) => g.left + (s / xMax) * g.w;
    const y = (v: number) => g.top + (1 - v / yMax) * g.h;
    this.handlers.scaled(`${natsPerDiv} nat/div  ${stepsPerDiv.toLocaleString('en-GB')} steps/div`);

    // Graticule: dotted divisions, a solid frame, minor ticks on the centre lines.
    ctx.lineWidth = 1;
    ctx.strokeStyle = css('--grid-2');
    ctx.setLineDash([1, 3]);
    for (let i = 1; i < this.divisions; i++) {
      const px = Math.round(g.left + (i * g.w) / this.divisions) + 0.5;
      ctx.beginPath();
      ctx.moveTo(px, g.top);
      ctx.lineTo(px, g.top + g.h);
      ctx.stroke();
    }
    for (let j = 1; j < vdiv; j++) {
      const py = Math.round(g.top + (j * g.h) / vdiv) + 0.5;
      ctx.beginPath();
      ctx.moveTo(g.left, py);
      ctx.lineTo(g.left + g.w, py);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.strokeStyle = css('--edge');
    ctx.strokeRect(Math.round(g.left) + 0.5, Math.round(g.top) + 0.5, Math.round(g.w), Math.round(g.h));
    ctx.strokeStyle = css('--grid-2');
    const cx = Math.round(g.left + g.w / 2) + 0.5;
    const cy = Math.round(g.top + (Math.floor(vdiv / 2) * g.h) / vdiv) + 0.5;
    for (let i = 0; i <= this.divisions * 5; i++) {
      const px = Math.round(g.left + (i * g.w) / (this.divisions * 5)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(px, cy - (i % 5 === 0 ? 4 : 2));
      ctx.lineTo(px, cy + (i % 5 === 0 ? 4 : 2));
      ctx.stroke();
    }
    for (let j = 0; j <= vdiv * 5; j++) {
      const py = Math.round(g.top + (j * g.h) / (vdiv * 5)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(cx - (j % 5 === 0 ? 4 : 2), py);
      ctx.lineTo(cx + (j % 5 === 0 ? 4 : 2), py);
      ctx.stroke();
    }

    // Axis values.
    ctx.font = label;
    ctx.fillStyle = css('--faint');
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let j = 0; j <= vdiv; j += vdiv > 5 ? 2 : 1) ctx.fillText(String(+(natsPerDiv * j).toFixed(2)), g.left - 6, y(natsPerDiv * j));
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    ctx.fillText('0', g.left, g.top + g.h + 5);
    ctx.textAlign = 'right';
    const stepsLabel = `${xMax.toLocaleString('en-GB')} steps`;
    ctx.fillText(stepsLabel, g.left + g.w, g.top + g.h + 5);

    // The planned end of the run: a dashed line, numbered on the axis.
    if (this.total < xMax) {
      const ex = Math.round(x(this.total)) + 0.5;
      ctx.strokeStyle = css('--dim');
      ctx.globalAlpha = 0.7;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(ex, g.top);
      ctx.lineTo(ex, g.top + g.h);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      const end = `end ${this.total.toLocaleString('en-GB')}`;
      const room = g.left + g.w - ctx.measureText(stepsLabel).width - 12;
      const w = ctx.measureText(end).width;
      ctx.fillStyle = css('--dim');
      ctx.textAlign = 'center';
      if (ex + w / 2 < room) ctx.fillText(end, ex, g.top + g.h + 5);
      else if (ex - w > g.left + 20) {
        ctx.textAlign = 'right';
        ctx.fillText(end, Math.min(ex, room), g.top + g.h + 5);
      }
    }

    // Chance: a uniform guess over the vocabulary.
    if (this.chance > 0) {
      ctx.strokeStyle = css('--dim');
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(g.left, y(this.chance));
      ctx.lineTo(g.left + g.w, y(this.chance));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = css('--dim');
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`chance ${this.chance.toFixed(2)} = ln ${this.vocab}`, g.left + g.w - 4, y(this.chance) - 3);
    }

    const line = (pts: readonly Point[], map: (v: number) => number, stroke: string, width: number, alpha: number) => {
      if (pts.length < 2) return;
      ctx.save();
      ctx.beginPath();
      ctx.rect(g.left, g.top, g.w, g.h);
      ctx.clip();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      const stride = Math.max(1, Math.floor(pts.length / (g.w * 1.5)));
      for (let i = 0; i < pts.length; i += stride) {
        const p = pts[i];
        if (i === 0) ctx.moveTo(x(p.step), map(p.value));
        else ctx.lineTo(x(p.step), map(p.value));
      }
      const last = pts[pts.length - 1];
      ctx.lineTo(x(last.step), map(last.value));
      ctx.stroke();
      ctx.restore();
    };

    // CH3: the learning rate on its own scale (peak at three quarters of the height).
    const lrY = (v: number) => g.top + g.h - (v / (this.peakLr / 0.75)) * g.h;
    line(this.lr, lrY, css('--ch3'), 1, 0.75);
    // Its scale, read where it peaks (drawn last, clear of the cursor's readout).
    let peak: Point | null = null;
    for (const p of this.lr) if (!peak || p.value > peak.value) peak = p;
    const peakLabel =
      peak && peak.value > 0 && this.lr[this.lr.length - 1].step > peak.step
        ? { x: x(peak.step) + 6, y: lrY(peak.value) - 4, text: `peak ${peak.value.toExponential(1)}` }
        : null;
    let readout: { left: number; right: number; bottom: number } | null = null;
    // REF: the recorded run, dim, behind everything live.
    if (ref) {
      line(ref.train, y, css('--phos'), 1.25, 0.2);
      ctx.save();
      ctx.setLineDash([3, 3]);
      line(ref.copy, y, css('--ch4'), 1.25, 0.32);
      ctx.restore();
      if (ref.moment) {
        const rx = Math.round(x(ref.moment.step)) + 0.5;
        ctx.save();
        ctx.globalAlpha = 0.5;
        ctx.strokeStyle = css('--att');
        ctx.setLineDash([1, 3]);
        ctx.beginPath();
        ctx.moveTo(rx, g.top);
        ctx.lineTo(rx, g.top + g.h + (g.lanes ? g.lanes + 24 : 0));
        ctx.stroke();
        ctx.restore();
        // Said in words, low on the graticule, out of the live annotation's way.
        if (!this.moment || Math.abs(this.moment.step - ref.moment.step) > 0) {
          ctx.font = `11px ${css('--sans') || 'sans-serif'}`;
          const long = `recorded: ${headName(ref.moment.head, ref.heads.H)} learned to copy at step ${ref.moment.step.toLocaleString('en-GB')}`;
          const short = `rec. ${headName(ref.moment.head, ref.heads.H)} copies at ${ref.moment.step.toLocaleString('en-GB')}`;
          const fits = (t: string) => {
            const w = ctx.measureText(t).width;
            if (rx + 6 + w < g.left + g.w) return rx + 6;
            if (rx - 6 - w > g.left + 2) return rx - 6 - w;
            return null;
          };
          const text = fits(long) !== null ? long : short;
          const w = ctx.measureText(text).width;
          const lx = fits(text) ?? Math.max(g.left + 2, Math.min(rx + 6, g.left + g.w - w - 2));
          ctx.fillStyle = css('--screen');
          ctx.fillRect(lx - 3, g.top + g.h - 40, w + 6, 16);
          ctx.fillStyle = css('--dim');
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(text, lx, g.top + g.h - 32);
        }
      }
    }
    // CH4: the loss on the second copy of the induction test.
    line(this.copy, y, css('--ch4'), 1.75, 0.95);
    // CH1: training loss, raw and averaged. CH2: held-out loss.
    line(this.train, y, css('--ch1'), 1, 0.28);
    line(this.smooth, y, css('--ch1'), 2, 1);
    line(this.evals, y, css('--ch2'), 1.5, 0.9);
    ctx.fillStyle = css('--ch2');
    for (const p of this.evals) ctx.fillRect(Math.round(x(p.step)) - 2, Math.round(y(p.value)) - 2, 4, 4);

    // The beam: where the run is now.
    if (this.smooth.length) {
      const p = this.smooth[this.smooth.length - 1];
      ctx.fillStyle = css('--ch1');
      ctx.beginPath();
      ctx.arc(x(p.step), y(p.value), 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // A tick under the trace for every checkpoint: bright once its text is ready.
    for (const m of this.marks) {
      ctx.fillStyle = m.ready ? css('--phos') : css('--faint');
      ctx.fillRect(Math.round(x(m.step)) - 0.5, g.top + g.h - (m.ready ? 6 : 3), 1, m.ready ? 6 : 3);
    }

    drawLanes(ctx, g, x, css, xMax, { shape: this.shape, moment: this.moment, ref: this.ref, probes: this.probes });
    if (this.moment) {
      const p = Math.min(1, this.momentAt ? (performance.now() - this.momentAt) / 900 : 1);
      if (p < 1) this.request();
      drawMoment(ctx, g, x, y, css, this.moment, this.shape.H || 1, this.copy, 1 - (1 - p) ** 3);
    }

    // The cursor: green while it follows the run, white when replaying.
    if (this.cursor !== null) {
      const px = Math.round(x(this.cursor)) + 0.5;
      const colour = this.live ? css('--run') : css('--cursor');
      ctx.strokeStyle = colour;
      ctx.setLineDash(this.live ? [] : [5, 3]);
      ctx.beginPath();
      ctx.moveTo(px, g.top);
      ctx.lineTo(px, g.top + g.h);
      if (g.lanes) {
        ctx.moveTo(px, g.lanesTop - 2);
        ctx.lineTo(px, g.lanesTop + g.lanes + 2);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      // handle
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.moveTo(px - 5, g.top + g.h + 1);
      ctx.lineTo(px + 5, g.top + g.h + 1);
      ctx.lineTo(px, g.top + g.h - 6);
      ctx.closePath();
      ctx.fill();
      // readout
      const loss = this.lossAt(this.cursor);
      const held = this.heldOutAt(this.cursor);
      const label = `step ${this.cursor.toLocaleString('en-GB')}${Number.isFinite(loss) ? `  loss ${loss.toFixed(3)}` : ''}${Number.isFinite(held) ? `  held-out ${held.toFixed(3)}` : ''}`;
      ctx.font = `11px ${css('--mono') || 'monospace'}`;
      const tw = ctx.measureText(label).width + 12;
      const bx = Math.min(Math.max(px - tw / 2, g.left + 2), g.left + g.w - tw - 2);
      ctx.fillStyle = css('--screen');
      ctx.fillRect(bx, g.top + 3, tw, 18);
      ctx.strokeStyle = colour;
      ctx.strokeRect(Math.round(bx) + 0.5, g.top + 3.5, Math.round(tw), 17);
      ctx.fillStyle = colour;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, bx + 6, g.top + 12.5);
      readout = { left: bx, right: bx + tw, bottom: g.top + 21 };
    }

    if (peakLabel) {
      ctx.font = label;
      const w = ctx.measureText(peakLabel.text).width;
      const lx = peakLabel.x;
      const hidden = readout && peakLabel.y - 12 < readout.bottom && lx < readout.right + 4 && lx + w > readout.left - 4;
      if (!hidden && lx + w < g.left + g.w - 4) {
        ctx.fillStyle = css('--ch3');
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText(peakLabel.text, lx, peakLabel.y);
      }
    }
  }
}
