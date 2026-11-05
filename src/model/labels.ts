/**
 * Labels for sparse-autoencoder features, by rule, from their top-activating contexts: no model
 * writes them. Each candidate rule is a test on one context (the character the feature fired on,
 * its neighbours, the word around it); a label is the most specific rule, or pair of rules, that
 * holds for at least LABEL_MIN of the contexts (specificity discounted by coverage), reported with
 * how many it covers. When nothing
 * holds that often, the feature has no label and the contexts speak for themselves.
 */

export interface Context {
  /** The text around the position, and the index in it of the character the feature fired on. */
  readonly text: string;
  readonly at: number;
}

export interface FeatureLabel {
  readonly text: string;
  readonly covered: number;
  readonly of: number;
}

export const LABEL_MIN = 0.75;

const isLetter = (c: string | undefined) => c !== undefined && /\p{L}/u.test(c);
const isWordChar = (c: string | undefined) => c !== undefined && (isLetter(c) || c === '’' || c === "'");

/** How a character is said in a label. */
export function sayChar(c: string): string {
  if (c === ' ') return 'a space';
  if (c === '\n') return 'a line break';
  return `“${c}”`;
}

function wordAround(text: string, at: number): { word: string; index: number } | null {
  if (!isLetter(text[at])) return null;
  let a = at;
  let b = at;
  while (a > 0 && isWordChar(text[a - 1])) a--;
  while (b + 1 < text.length && isWordChar(text[b + 1])) b++;
  // A word cut off by the edge of the context is not a word we can name.
  if (a === 0 || b === text.length - 1) return null;
  return { word: text.slice(a, b + 1).toLowerCase(), index: at - a };
}

/** The word that ends just before a non-letter at `at` (for spaces and punctuation after a word). */
function wordBefore(text: string, at: number): string | null {
  if (isLetter(text[at]) || !isLetter(text[at - 1])) return null;
  let a = at - 1;
  while (a > 0 && isWordChar(text[a - 1])) a--;
  if (a === 0) return null;
  return text.slice(a, at).toLowerCase();
}

interface Rule {
  readonly key: string;
  readonly text: string;
  /** Higher is more specific; a pair adds its parts. */
  readonly weight: number;
  readonly test: (c: Context) => boolean;
}

function rulesFor(contexts: readonly Context[]): Rule[] {
  const rules: Rule[] = [];
  const seen = new Set<string>();
  const add = (r: Rule) => {
    if (seen.has(r.key)) return;
    seen.add(r.key);
    rules.push(r);
  };
  for (const c of contexts) {
    const ch = c.text[c.at];
    const prev = c.text[c.at - 1];
    const next = c.text[c.at + 1];
    if (ch !== undefined) add({ key: `char:${ch}`, text: sayChar(ch), weight: 3, test: (x) => x.text[x.at] === ch });
    if (prev !== undefined) add({ key: `prev:${prev}`, text: `after ${sayChar(prev)}`, weight: 2, test: (x) => x.text[x.at - 1] === prev });
    if (next !== undefined) add({ key: `next:${next}`, text: `before ${sayChar(next)}`, weight: 2, test: (x) => x.text[x.at + 1] === next });
    const w = wordAround(c.text, c.at);
    if (w) add({ key: `word:${w.word}`, text: `in “${w.word}”`, weight: 6, test: (x) => wordAround(x.text, x.at)?.word === w.word });
    const pw = wordBefore(c.text, c.at);
    if (pw)
      add({
        key: `after-word:${pw}`,
        text: `${ch === ' ' ? 'the space' : sayChar(ch ?? '')} after “${pw}”`,
        weight: 6,
        test: (x) => wordBefore(x.text, x.at) === pw && x.text[x.at] === ch,
      });
  }
  add({ key: 'word-start', text: 'starting a word', weight: 1.5, test: (x) => isLetter(x.text[x.at]) && !isWordChar(x.text[x.at - 1]) });
  add({
    key: 'word-end',
    text: 'ending a word',
    weight: 1.5,
    test: (x) => isLetter(x.text[x.at]) && x.text[x.at + 1] !== undefined && !isWordChar(x.text[x.at + 1]),
  });
  add({ key: 'line-start', text: 'starting a line', weight: 2, test: (x) => x.text[x.at - 1] === '\n' });
  add({ key: 'capital', text: 'a capital letter', weight: 1, test: (x) => /\p{Lu}/u.test(x.text[x.at] ?? '') });
  add({ key: 'vowel', text: 'a vowel', weight: 1, test: (x) => /[aeiouy]/i.test(x.text[x.at] ?? '') });
  add({ key: 'punct', text: 'punctuation', weight: 1, test: (x) => /[.,;:!?’'‘—-]/u.test(x.text[x.at] ?? '') });
  add({ key: 'after-punct', text: 'after punctuation', weight: 1, test: (x) => /[.,;:!?]/.test(x.text[x.at - 1] ?? '') });
  return rules;
}

/** The label for a feature from its contexts, or null when no rule covers enough of them. */
export function labelFeature(contexts: readonly Context[]): FeatureLabel | null {
  const n = contexts.length;
  if (n < 4) return null;
  const need = Math.ceil(LABEL_MIN * n);
  const rules = rulesFor(contexts);
  const hits = rules.map((r) => contexts.map(r.test));
  const count = (h: readonly boolean[]) => h.reduce((a, b) => a + (b ? 1 : 0), 0);
  let best: { text: string; covered: number; weight: number } | null = null;
  // Specificity, discounted by coverage: a rule that holds everywhere beats a sharper one that misses.
  const consider = (text: string, covered: number, weight: number) => {
    if (covered < need) return;
    const score = (weight * covered) / n;
    if (!best || score > best.weight || (score === best.weight && covered > best.covered)) best = { text, covered, weight: score };
  };
  for (let i = 0; i < rules.length; i++) {
    const ci = count(hits[i]);
    if (ci < need) continue;
    consider(rules[i].text, ci, rules[i].weight);
    for (let j = i + 1; j < rules.length; j++) {
      if (/^(after-)?word:/.test(rules[i].key) || /^(after-)?word:/.test(rules[j].key)) continue;
      const both = hits[i].map((h, k) => h && hits[j][k]);
      const cb = count(both);
      if (cb < need) continue;
      // Say the character first: "“t” ending a word", "“t” before a space".
      const [a, b] = rules[j].key.startsWith('char:') ? [rules[j], rules[i]] : [rules[i], rules[j]];
      consider(`${a.text} ${b.text}`, cb, a.weight + b.weight);
    }
  }
  if (!best) return null;
  const chosen = best as { text: string; covered: number };
  return { text: chosen.text, covered: chosen.covered, of: n };
}
