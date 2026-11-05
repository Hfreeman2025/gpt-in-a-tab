import { type Tokenizer, buildTokenizer, normaliseText } from '../model/tokenizer';
import { SAMPLES, type SampleId, sampleById } from '../samples';
import { $, h, int } from '../ui/dom';

export const MIN_CHARS = 600;
export const MAX_CHARS = 5_000_000;

export type Choice = SampleId | 'own';

/** What the source panel asks of the page around it. */
export interface SourceHost {
  /** Mid-run, or while a run starts, the text cannot change. */
  locked(): boolean;
  /** A key was pressed: pick that text (the page may start a run on it). */
  pick(id: Choice): Promise<void>;
  /** A sample text is loading, or failed to: the run key says so. */
  status(state: 'arming' | 'armed', note: string): void;
  /** The visitor typed or pasted into the source. */
  edited(): void;
}

/** The source panel: a key per sample text and one for pasting, the text itself, and its credit. */
export class Source {
  choice: Choice = 'sonnets';
  private loading = false;
  private ownText = '';
  private readonly texts = new Map<SampleId, string>();

  constructor(private readonly host: SourceHost) {
    this.buildKeys();
    this.textarea.addEventListener('input', () => {
      if (host.locked()) return;
      if (this.choice !== 'own') {
        this.choice = 'own';
        this.check('own');
      }
      this.ownText = this.textarea.value;
      host.edited();
    });
  }

  get textarea(): HTMLTextAreaElement {
    return $<HTMLTextAreaElement>('source-text');
  }

  get busy(): boolean {
    return this.loading;
  }

  /** The text as it will be trained on. */
  text(): string {
    return normaliseText(this.textarea.value).slice(0, MAX_CHARS);
  }

  get ready(): boolean {
    return [...normaliseText(this.textarea.value)].length >= MIN_CHARS;
  }

  /** The keys and the text follow the run: locked while it trains. */
  lock(locked: boolean): void {
    for (const k of $('softkeys').children) (k as HTMLButtonElement).disabled = locked;
    this.textarea.readOnly = locked;
  }

  /** Shows a text: false when it did not load or another was picked meanwhile. */
  async select(id: Choice, user: boolean): Promise<boolean> {
    if (this.loading) return false;
    if (this.choice === 'own') this.ownText = this.textarea.value;
    this.choice = id;
    this.check(id);
    const ta = this.textarea;
    if (id === 'own') {
      ta.value = this.ownText;
      ta.readOnly = false;
      ta.placeholder = `Paste any text: a novel, letters, lyrics, code. At least ${int(MIN_CHARS)} characters; the more the better.`;
      if (user) ta.focus();
      return true;
    }
    const sample = sampleById(id);
    if (!sample) return false;
    let text = this.texts.get(id);
    if (text === undefined) {
      this.loading = true;
      $('source-credit').textContent = 'Loading the text…';
      this.host.status('arming', 'loading text');
      try {
        text = normaliseText(await sample.load());
        this.texts.set(id, text);
      } catch (e) {
        $('source-credit').replaceChildren(h('span', { class: 'warn' }, `Could not load the text (${String(e)}). Check the connection and pick it again.`));
        this.host.status('armed', 'no text');
        return false;
      } finally {
        this.loading = false;
      }
    }
    if (this.choice !== id) return false;
    ta.value = text;
    ta.readOnly = false;
    this.scrollToText(text);
    return true;
  }

  /** The count and credit under the text; returns its length in characters and its tokenizer. */
  describe(): { chars: number; tok: Tokenizer } {
    const text = normaliseText(this.textarea.value);
    const n = [...text].length;
    const tok = buildTokenizer(text.slice(0, MAX_CHARS));
    $('source-count').textContent = `${int(n)} chars · ${tok.size} distinct`;
    const credit = $('source-credit');
    if (this.choice === 'own') {
      credit.textContent = 'Your text. It stays in this tab.';
      if (n < MIN_CHARS) credit.append(' ', h('span', { class: 'warn' }, `It needs at least ${int(MIN_CHARS)} characters.`));
      if (n > MAX_CHARS) credit.append(' ', h('span', { class: 'warn' }, `Only the first ${int(MAX_CHARS)} are used.`));
    } else credit.textContent = sampleById(this.choice)?.credit ?? '';
    if (tok.unknown !== null) credit.append(' The rarest characters share one token.');
    return { chars: n, tok };
  }

  private buildKeys(): void {
    const box = $('softkeys');
    const keys: [Choice, string][] = [...SAMPLES.map((s) => [s.id, s.key] as [Choice, string]), ['own', 'Paste']];
    for (const [id, label] of keys) {
      const b = h('button', { type: 'button', class: 'key', role: 'radio', 'aria-checked': String(id === this.choice), 'data-id': id }, label);
      b.addEventListener('click', () => void this.host.pick(id));
      box.append(b);
    }
    box.addEventListener('keydown', (e) => {
      const ids = keys.map(([id]) => id);
      const at = ids.indexOf(this.choice);
      const d = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0;
      if (!d || this.host.locked()) return;
      e.preventDefault();
      const next = ids[(at + d + ids.length) % ids.length];
      void this.host.pick(next).then(() => (box.querySelector(`[data-id="${next}"]`) as HTMLElement | null)?.focus());
    });
  }

  private check(id: Choice): void {
    for (const b of $('softkeys').children) b.setAttribute('aria-checked', String((b as HTMLElement).dataset.id === id));
  }

  /** Show the source from its first real line of text, past any title page. */
  private scrollToText(text: string): void {
    const ta = this.textarea;
    const lines = text.split('\n');
    const first = lines.findIndex((l) => l.trim().length >= 24);
    requestAnimationFrame(() => {
      const css = getComputedStyle(ta);
      const lh = parseFloat(css.lineHeight) || 16;
      // Put a blank line (or the text itself) flush at the top, so no half-line peeks in above it.
      const top = first > 0 && lines[first - 1].trim() === '' ? first - 1 : Math.max(0, first);
      ta.scrollTop = top === 0 ? 0 : parseFloat(css.paddingTop) + top * lh;
    });
  }
}
