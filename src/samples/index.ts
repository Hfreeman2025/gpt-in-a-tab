/** Bundled texts: public domain from Project Gutenberg, a public-domain word list, and this page's own shaders. */
export type SampleId = 'sonnets' | 'alice' | 'tragedies' | 'words' | 'kernels';

export interface Sample {
  readonly id: SampleId;
  /** Softkey label. */
  readonly key: string;
  readonly title: string;
  readonly credit: string;
  load(): Promise<string>;
}

// The shaders ship in the main bundle anyway (they run the model), so read them eagerly.
const wgsl = import.meta.glob<string>('../gpu/wgsl/*.wgsl', { query: '?raw', import: 'default', eager: true });

export const SAMPLES: readonly Sample[] = [
  {
    id: 'sonnets',
    key: 'Sonnets',
    title: 'Shakespeare’s Sonnets',
    credit: 'Shakespeare’s Sonnets (1609). Project Gutenberg ebook 1041, public domain.',
    load: () => import('./sonnets.txt?raw').then((m) => m.default),
  },
  {
    id: 'alice',
    key: 'Alice',
    title: 'Alice’s Adventures in Wonderland',
    credit: 'Lewis Carroll, Alice’s Adventures in Wonderland (1865). Project Gutenberg ebook 11, public domain.',
    load: () => import('./alice.txt?raw').then((m) => m.default),
  },
  {
    id: 'tragedies',
    key: 'Tragedies',
    title: 'Five tragedies',
    credit: 'Shakespeare: Hamlet, Macbeth, Romeo and Juliet, Othello, King Lear. Project Gutenberg ebooks 1524, 1533, 1513, 1531, 1532, public domain.',
    load: () => import('./tragedies.txt?raw').then((m) => m.default),
  },
  {
    id: 'words',
    key: 'Words',
    title: 'Webster’s word list',
    credit: '30,000 entries from Webster’s Second New International Dictionary (1934), as shipped with BSD Unix; copyright lapsed. It learns to invent words.',
    load: () => import('./words.txt?raw').then((m) => m.default),
  },
  {
    id: 'kernels',
    key: 'Kernels',
    title: 'Its own GPU kernels',
    credit: 'The WGSL source of this page’s own compute shaders (MIT). It learns to write GPU code, badly.',
    load: async () =>
      Object.keys(wgsl)
        .sort()
        .map((n) => `// ${n.split('/').pop()}\n\n${wgsl[n]}`)
        .join('\n\n'),
  },
];

export function sampleById(id: string): Sample | undefined {
  return SAMPLES.find((s) => s.id === id);
}
