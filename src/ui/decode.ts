import { h, reducedMotion } from './dom';

export interface Reading {
  /** Identifies what is shown (checkpoint, sampling settings), so changes can be flashed. */
  readonly key: string;
  readonly settings: string;
  readonly prompt: readonly string[];
  readonly chars: readonly string[];
  readonly probs: ArrayLike<number>;
}

/**
 * Brightness from confidence. The floor keeps the faintest glyph at 4.6:1 against the screen,
 * so the encoding is extra information, never a cost to reading.
 */
export function intensity(p: number): number {
  const q = Math.max(0, Math.min(1, p));
  return Math.round((0.52 + 0.48 * Math.sqrt(q)) * 100) / 100;
}

/**
 * The decode readout: what the model wrote at the cursor's checkpoint. Glyphs that differ from
 * the previous reading (with the same prompt and temperature) strike bright and decay. Pointer
 * or arrow keys pick a character to inspect.
 */
export class Decode {
  private shown: Reading | null = null;
  private glyphs: HTMLSpanElement[] = [];
  private promptGlyphs: HTMLSpanElement[] = [];
  private focus: number | null = null;
  /** How many glyphs fit on the screen; the rest of the sample is cut, not shown half-drawn. */
  private visible = 0;
  /** The auto-range: type scale, stepped down when a reading overflows, never back up mid-run. */
  private range = 1;

  constructor(
    private readonly body: HTMLElement,
    private readonly onInspect: (index: number | null) => void,
  ) {
    // A new screen size, or the face arriving after a fallback, re-ranges the type.
    let pending = 0;
    new ResizeObserver(() => {
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(() => this.resetRange());
    }).observe(body.parentElement ?? body);
    void document.fonts?.ready.then(() => this.resetRange());
    body.tabIndex = 0;
    body.setAttribute('role', 'group');
    body.setAttribute('aria-label', 'The model’s output. Arrow keys inspect one character at a time.');
    const glyphAt = (e: Event) => (e.target as HTMLElement).closest('.g[data-i]:not(.cut)') as HTMLElement | null;
    // A mouse inspects what it points at; a finger has no hover, so a tap pins a character and a
    // tap anywhere else in the text lets it go.
    body.addEventListener('pointerover', (e) => {
      const g = glyphAt(e);
      if (g && e.pointerType !== 'touch') this.inspect(Number(g.dataset.i));
    });
    body.addEventListener('pointerleave', (e) => {
      if (e.pointerType !== 'touch') this.inspect(null);
    });
    body.addEventListener('click', (e) => {
      const g = glyphAt(e);
      this.inspect(g ? Number(g.dataset.i) : null);
    });
    body.addEventListener('keydown', (e) => {
      const n = this.visible;
      if (!n) return;
      const moves: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
      if (e.key in moves) {
        e.preventDefault();
        const at = this.focus === null ? (moves[e.key] > 0 ? 0 : n - 1) : Math.max(0, Math.min(n - 1, this.focus + moves[e.key]));
        this.inspect(at);
      } else if (e.key === 'Escape') this.inspect(null);
    });
  }

  /** The number of characters on screen, which may be fewer than the sample when it is long. */
  get length(): number {
    return this.visible;
  }

  get inspected(): number | null {
    return this.focus;
  }

  private inspect(i: number | null): void {
    if (i === this.focus) return;
    if (this.focus !== null) this.glyphs[this.focus]?.classList.remove('at');
    this.focus = i;
    if (i !== null) this.glyphs[i]?.classList.add('at');
    this.onInspect(i);
  }

  message(text: string): void {
    this.shown = null;
    this.glyphs = [];
    this.promptGlyphs = [];
    this.focus = null;
    this.visible = 0;
    this.body.replaceChildren(h('span', { class: 'text-empty' }, text));
  }

  show(r: Reading): void {
    const prev = this.shown;
    const flash = prev !== null && prev.settings === r.settings && prev.key !== r.key && !reducedMotion();
    const frag = document.createDocumentFragment();
    this.promptGlyphs = r.prompt.map((c) => {
      const s = h('span', { class: 'g p' }, c);
      frag.append(s);
      return s;
    });
    this.glyphs = r.chars.map((c, i) => {
      const s = h('span', { class: 'g', style: `--a:${intensity(r.probs[i])}` }, c);
      s.dataset.i = String(i);
      if (flash && prev.chars[i] !== c) s.classList.add('n');
      frag.append(s);
      return s;
    });
    this.body.replaceChildren(frag);
    this.shown = r;
    this.fit();
    if (this.focus !== null) {
      if (this.focus >= this.visible) this.inspect(null);
      else this.glyphs[this.focus].classList.add('at');
    }
  }

  /** Forget the auto-range, for a new run or a new screen size. */
  resetRange(): void {
    this.range = 1;
    this.body.style.removeProperty('--range');
    if (this.shown) this.fit();
  }

  /**
   * Auto-range, as a scope does: step the type down until the reading fits, to a floor; then cut
   * at the last whole line that fits. The scale only steps down during a run, so scrubbing
   * compares readings at one size instead of the text breathing in and out.
   */
  private fit(): void {
    const b = this.body;
    for (const g of b.querySelectorAll('.g.cut')) g.classList.remove('cut');
    const over = () => b.scrollWidth > b.clientWidth + 1 || b.scrollHeight > b.clientHeight + 1;
    while (over() && this.range > 0.8) {
      this.range = Math.max(0.8, Math.round(this.range * 92) / 100);
      b.style.setProperty('--range', String(this.range));
    }
    const box = b.getBoundingClientRect();
    const fits = (g: HTMLElement) => {
      const r = g.getBoundingClientRect();
      return r.right <= box.right + 1 && r.bottom <= box.bottom + 1;
    };
    // Glyphs flow in order through the columns, so the ones that fit are a prefix.
    let lo = 0;
    let hi = this.glyphs.length;
    if (hi && !fits(this.glyphs[hi - 1])) {
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (fits(this.glyphs[mid])) lo = mid + 1;
        else hi = mid;
      }
    } else lo = hi;
    this.visible = lo;
    for (let i = lo; i < this.glyphs.length; i++) this.glyphs[i].classList.add('cut');
  }

  /** Light the characters the model attended to: `weights` by position in [prompt, sample]. */
  light(promptSeen: number, start: number, weights: ArrayLike<number>): void {
    this.unlight();
    const P = this.promptGlyphs.length;
    const even = 1 / Math.max(1, weights.length);
    let max = even;
    for (let j = 0; j < weights.length; j++) max = Math.max(max, weights[j]);
    for (let j = 0; j < weights.length; j++) {
      // As in the heads: what rises above an even spread, relative to the strongest.
      const w = max > even * 1.05 ? Math.max(0, (weights[j] - even) / (max - even)) : 0;
      if (w < 0.05) continue;
      const idx = start + j; // index into [trimmed prompt, sample]
      const g = idx < promptSeen ? this.promptGlyphs[P - promptSeen + idx] : this.glyphs[idx - promptSeen];
      if (!g) continue;
      g.classList.add('att');
      g.style.setProperty('--w', Math.min(0.8, w ** 1.2 * 0.8).toFixed(3));
    }
  }

  unlight(): void {
    for (const g of this.body.querySelectorAll('.g.att')) {
      g.classList.remove('att');
      (g as HTMLElement).style.removeProperty('--w');
    }
  }
}
