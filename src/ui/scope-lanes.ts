import { type Moment, type ProbeReading, headName } from '../model/probe';
import type { Point, Reference } from './scope';

/** Where the trace and the head lanes sit on the scope's canvas, in CSS pixels. */
export interface Geometry {
  readonly W: number;
  readonly H: number;
  readonly left: number;
  readonly right: number;
  readonly top: number;
  readonly bottom: number;
  readonly w: number;
  readonly h: number;
  readonly laneH: number;
  readonly layerGap: number;
  readonly lanesTop: number;
  readonly lanes: number;
}

/** What the head lanes show: the live run's tests, the moment, and the reference run ahead of it. */
export interface Lanes {
  readonly shape: { readonly L: number; readonly H: number };
  readonly moment: Moment | null;
  readonly ref: Reference | null;
  readonly probes: readonly ProbeReading[];
}

/** One lane per head under the trace: orange for prefix matching, phosphor for the previous character. */
export function drawLanes(ctx: CanvasRenderingContext2D, g: Geometry, x: (s: number) => number, css: (n: string) => string, xMax: number, d: Lanes): void {
  const { L, H } = d.shape;
  if (!g.lanes) return;
  const laneY = (k: number) => g.lanesTop + k * g.laneH + Math.floor(k / H) * g.layerGap;
  ctx.font = `500 ${Math.min(10.5, g.laneH + 2)}px ${css('--cond') || 'sans-serif'}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const att = css('--att-rgb').split(/\s+/).join(',');
  const phos = css('--phos-rgb').split(/\s+/).join(',');
  const m = d.moment;
  for (let k = 0; k < L * H; k++) {
    const y0 = laneY(k);
    const special = m && (k === m.head || k === m.partner);
    // "L2 H1 H2 H3 H4": the layer once, at its first lane; each lane its head.
    ctx.fillStyle = special ? css('--att') : css('--faint');
    if (g.laneH >= 6) ctx.fillText(`H${(k % H) + 1}`, g.left - 5, y0 + g.laneH / 2);
    if (k % H === 0) {
      ctx.fillStyle = css('--dim');
      ctx.fillText(`L${Math.floor(k / H) + 1}`, g.left - (g.laneH >= 6 ? 24 : 5), y0 + g.laneH / 2);
    }
    ctx.fillStyle = css('--bezel');
    ctx.fillRect(g.left, y0, g.w, g.laneH - 1);
  }
  // Ahead of the live run, the recorded run's lanes, dim: the reference the live run overwrites.
  const ref = d.ref && d.ref.heads.L === L && d.ref.heads.H === H ? d.ref : null;
  const liveEnd = d.probes.length ? d.probes[d.probes.length - 1].step : -1;
  const ahead = ref ? ref.probes.filter((p) => p.step > liveEnd) : [];
  const ps = [...ahead, ...d.probes];
  for (let i = 0; i < ps.length; i++) {
    ctx.globalAlpha = i < ahead.length ? 0.3 : 1;
    const p = ps[i];
    const x0 = x(p.step);
    const next = i + 1 < ps.length && (i + 1 < ahead.length || i >= ahead.length) ? ps[i + 1] : null;
    const x1 = next ? x(next.step) : Math.min(x(xMax), x0 + 3);
    const w = Math.max(1, x1 - x0);
    for (let k = 0; k < L * H; k++) {
      const y0 = laneY(k);
      const a = Math.min(1, Math.max(0, p.prefix[k]));
      if (a > 0.02) {
        ctx.fillStyle = `rgba(${att},${Math.min(1, 0.15 + Math.sqrt(a)).toFixed(3)})`;
        ctx.fillRect(x0, y0, w, g.laneH - 1);
      }
      const b = Math.min(1, Math.max(0, p.prev[k]));
      if (b > 0.1) {
        ctx.fillStyle = `rgba(${phos},${(b * 0.85).toFixed(3)})`;
        ctx.fillRect(x0, y0 + g.laneH - 3, w, 2);
      }
    }
  }
  ctx.globalAlpha = 1;
  if (m) {
    for (const k of [m.head, m.partner]) {
      if (k === null) continue;
      ctx.strokeStyle = k === m.head ? css('--att') : css('--phos');
      ctx.lineWidth = 1;
      ctx.strokeRect(g.left - 0.5, laneY(k) - 0.5, g.w + 1, g.laneH);
    }
  }
}

/** The induction moment: an upright through trace and lanes, and what happened, in words. */
/** `ease` runs from 0 to 1 as the moment is revealed; the upright grows and the words fade in. */
export function drawMoment(
  ctx: CanvasRenderingContext2D,
  g: Geometry,
  x: (s: number) => number,
  y: (v: number) => number,
  css: (n: string) => string,
  m: Moment,
  H: number,
  copy: readonly Point[],
  ease: number,
): void {
  const px = Math.round(x(m.step)) + 0.5;
  const bottom = g.lanes ? g.lanesTop + g.lanes + 2 : g.top + g.h;
  ctx.strokeStyle = css('--att');
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(px, g.top);
  ctx.lineTo(px, g.top + (bottom - g.top) * ease);
  ctx.stroke();
  ctx.lineWidth = 1;
  // The cliff: the copy loss before the head, and at the moment.
  if (m.before) {
    const x0 = x(m.before.step);
    ctx.strokeStyle = css('--ch4');
    ctx.globalAlpha = ease;
    ctx.beginPath();
    ctx.moveTo(x0, y(m.before.copySecond));
    ctx.lineTo(px, y(m.before.copySecond));
    ctx.lineTo(px, y(m.copySecond));
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  const cond = css('--cond') || 'sans-serif';
  const mono = css('--mono') || 'monospace';
  const line1 = `${headName(m.head, H)} learned to copy at step ${m.step.toLocaleString('en-GB')}`;
  const drop = m.before ? m.before.copySecond : m.copyFirst;
  const latest = copy.length ? copy[copy.length - 1] : null;
  const now = latest && latest.step > m.step ? latest.value : m.copySecond;
  const line2 = `copy loss ${drop.toFixed(2)} → ${now.toFixed(2)} nats`;
  ctx.font = `600 13px ${cond}`;
  const w1 = ctx.measureText(line1).width;
  ctx.font = `11px ${mono}`;
  const w2 = ctx.measureText(line2).width;
  const tw = Math.max(w1, w2) + 14;
  const right = px + 8 + tw < g.left + g.w;
  const bx = right ? px + 8 : px - 8 - tw;
  const by = g.top + Math.round(g.h * 0.36);
  ctx.globalAlpha = ease;
  ctx.fillStyle = css('--screen');
  ctx.fillRect(bx, by, tw, 38);
  ctx.strokeStyle = css('--att');
  ctx.strokeRect(Math.round(bx) + 0.5, by + 0.5, Math.round(tw), 37);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.font = `600 13px ${cond}`;
  ctx.fillStyle = css('--att');
  ctx.fillText(line1, bx + 7, by + 12);
  ctx.font = `11px ${mono}`;
  ctx.fillStyle = css('--ch4');
  ctx.fillText(line2, bx + 7, by + 27);
  ctx.globalAlpha = 1;
}
