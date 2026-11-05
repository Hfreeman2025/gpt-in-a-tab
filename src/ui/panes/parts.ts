import type { SaeStep } from '../../gpu/sae';
import { cssVar, h } from '../dom';

/** A character as shown inside a line of text: line breaks become visible. */
export const say = (c: string): string => (c === '\n' ? '↵' : c);

export function signed(v: number): string {
  return `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`;
}

export function bar(v: number, kind: string): HTMLElement {
  const w = Math.max(0, Math.min(1, Math.abs(v)));
  return h('span', { class: `bar bar--${kind}${v < 0 ? ' bar--neg' : ''}`, style: `--v:${w.toFixed(3)}` });
}

export function metric(name: string, value: string, title: string): HTMLElement {
  return h('div', { title }, h('dt', {}, name), h('dd', {}, value));
}

export function density(d: number): string {
  if (d === 0) return '0';
  if (d >= 0.01) return `${(d * 100).toFixed(1)}%`;
  return `${(d * 100).toFixed(2)}%`;
}

/**
 * Text, with the first character that differs from `against` marked: same random numbers, so
 * everything before it is shared, and everything after follows from it.
 */
export function showDiff(el: HTMLElement, text: string, against: string | null): void {
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
export function showCopy(el: HTMLElement, text: string, expected: string | null): void {
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

/** The autoencoder's training curve: unexplained variance (CH1 colour) and active features (CH2). */
export function drawCurve(canvas: HTMLCanvasElement, curve: readonly SaeStep[]): void {
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
      const y = H - 3 - (H - 6) * Math.min(1, get(c) / max);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  plot((c) => c.mse, 1, cssVar('--ch1'));
  plot((c) => c.l0, maxL0, cssVar('--ch2'));
}
