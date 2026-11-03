/** Character-level tokenizer: one token per Unicode code point, vocabulary sorted by code point. */
export interface Tokenizer {
  readonly chars: readonly string[];
  readonly size: number;
  encode(text: string): Uint32Array;
  decode(tokens: ArrayLike<number>): string;
  /** Characters not in the vocabulary (when it was capped) map to this token. */
  readonly unknown: number | null;
}

export const MAX_VOCAB = 256;
const UNKNOWN_GLYPH = '·';

export function normaliseText(text: string): string {
  return text.replace(/\r\n?/g, '\n').normalize('NFC');
}

export function buildTokenizer(text: string, maxVocab = MAX_VOCAB): Tokenizer {
  const counts = new Map<string, number>();
  for (const ch of text) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  let chars = [...counts.keys()];
  let unknownChar: string | null = null;
  if (chars.length > maxVocab) {
    // Keep the most frequent characters; everything else becomes a single "unknown" token.
    chars.sort((a, b) => (counts.get(b) ?? 0) - (counts.get(a) ?? 0));
    chars = chars.slice(0, maxVocab - 1);
    // A glyph for "everything else" that is not itself one of the kept characters.
    unknownChar = [UNKNOWN_GLYPH, '\ufffd', '\u00a4', '\u25a1'].find((c) => !chars.includes(c)) ?? '\u0000';
    chars.push(unknownChar);
  }
  chars.sort((a, b) => (a.codePointAt(0) ?? 0) - (b.codePointAt(0) ?? 0));
  const index = new Map(chars.map((c, i) => [c, i]));
  const unknown = unknownChar === null ? null : (index.get(unknownChar) ?? null);
  return {
    chars,
    size: chars.length,
    unknown,
    encode(s: string) {
      const out: number[] = [];
      for (const ch of s) {
        const id = index.get(ch);
        if (id !== undefined) out.push(id);
        else if (unknown !== null) out.push(unknown);
      }
      return Uint32Array.from(out);
    },
    decode(tokens: ArrayLike<number>) {
      let s = '';
      for (let i = 0; i < tokens.length; i++) s += chars[tokens[i]] ?? '';
      return s;
    },
  };
}
