// Fetches the bundled public-domain texts from Project Gutenberg, once, at build time, and writes
// them to src/samples/ with the Project Gutenberg header and footer removed. Records where each
// came from in src/samples/sources.json. The page itself never contacts Project Gutenberg.
//   node scripts/fetch-texts.mjs [--force]
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { root } from './browser.mjs';

// Each sample is one or more ebooks, checked by title so a renumbering can never slip in the wrong book.
const SAMPLES = [
  { id: 'sonnets', books: [[1041, "Shakespeare's Sonnets"]] },
  { id: 'alice', books: [[11, "Alice's Adventures in Wonderland"]] },
  {
    id: 'tragedies',
    books: [
      [1524, 'Hamlet'],
      [1533, 'Macbeth'],
      [1513, 'Romeo and Juliet'],
      [1531, 'Othello'],
      [1532, 'King Lear'],
    ],
  },
];

const force = process.argv.includes('--force');
const out = path.join(root, 'src/samples');
const sourcesPath = path.join(out, 'sources.json');
const sources = fs.existsSync(sourcesPath) ? JSON.parse(fs.readFileSync(sourcesPath, 'utf8')) : {};

/**
 * The text between the "*** START OF ..." and "*** END OF ..." markers that every Project
 * Gutenberg plain-text file carries, with line endings normalised.
 */
export function stripGutenberg(raw) {
  const text = raw.replace(/^﻿/, '').replace(/\r\n?/g, '\n');
  const start = text.search(/^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG E-?BOOK[^\n]*\*\*\*[ \t]*$/m);
  const end = text.search(/^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG E-?BOOK[^\n]*\*\*\*[ \t]*$/m);
  if (start < 0 || end < 0 || end <= start) throw new Error('Project Gutenberg start/end markers not found');
  const body = text.slice(text.indexOf('\n', start) + 1, end);
  return body.replace(/^\s*\n/, '').replace(/\s+$/, '\n');
}

/**
 * Light, documented clean-up after the header and footer are gone: drop "[Illustration]" markers
 * and a book's table of contents (they are typesetting, not text). Prose and verse are untouched.
 */
export function tidy(text) {
  let t = text.replace(/^[ \t]*\[Illustration[^\]]*\][ \t]*\n/gm, '');
  // A contents list: the line "Contents", then indented entries (and blank lines), ending at the
  // first line that starts in the first column.
  const lines = t.split('\n');
  const at = lines.findIndex((l, i) => i < 200 && /^\s*Contents\s*$/.test(l));
  if (at >= 0) {
    let stop = at + 1;
    while (stop < lines.length && (lines[stop].trim() === '' || /^\s/.test(lines[stop]))) stop++;
    lines.splice(at, stop - at);
    t = lines.join('\n');
  }
  return t.replace(/\n{4,}/g, '\n\n\n').replace(/^\s+/, '');
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

for (const sample of SAMPLES) {
  const file = path.join(out, `${sample.id}.txt`);
  const wanted = sample.books.map(([n]) => n).join(',');
  if (!force && sources[sample.id] && sources[sample.id].books.map((b) => b.ebook).join(',') === wanted && fs.existsSync(file)) {
    console.log(`${sample.id}: already fetched (ebooks ${wanted})`);
    continue;
  }
  const parts = [];
  const books = [];
  for (const [ebook, expected] of sample.books) {
    const url = `https://www.gutenberg.org/cache/epub/${ebook}/pg${ebook}.txt`;
    // Raw downloads are cached (gitignored) so re-running the clean-up never touches the network.
    const cached = path.join(root, '.tmp/gutenberg', `pg${ebook}.txt`);
    let raw;
    if (fs.existsSync(cached)) raw = fs.readFileSync(cached, 'utf8');
    else {
      const res = await fetch(url, { headers: { 'user-agent': 'gpt-in-a-tab build script (one-off fetch of public-domain texts)' } });
      if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
      raw = await res.text();
      fs.mkdirSync(path.dirname(cached), { recursive: true });
      fs.writeFileSync(cached, raw);
      await sleep(1500); // be gentle with the server
    }
    const title = raw.match(/^Title:\s*(.+)$/m)?.[1]?.trim() ?? '';
    const author = raw.match(/^Author:\s*(.+)$/m)?.[1]?.trim() ?? '';
    if (title.replace(/[\u2019']/g, "'") !== expected) throw new Error(`ebook ${ebook} is "${title}", expected "${expected}"`);
    const body = tidy(stripGutenberg(raw));
    parts.push(body);
    books.push({ ebook, title, author, url, page: `https://www.gutenberg.org/ebooks/${ebook}`, characters: [...body].length });
    console.log(`${sample.id}: ebook ${ebook}, "${title}" by ${author}, ${[...body].length} characters`);
  }
  const text = parts.join('\n\n\n');
  fs.writeFileSync(file, text);
  sources[sample.id] = {
    books,
    fetched: new Date().toISOString().slice(0, 10),
    characters: [...text].length,
    sha256: crypto.createHash('sha256').update(text).digest('hex'),
  };
}
for (const id of Object.keys(sources)) if (!SAMPLES.some((s) => s.id === id)) delete sources[id];
fs.writeFileSync(sourcesPath, JSON.stringify(sources, null, 2) + '\n');
