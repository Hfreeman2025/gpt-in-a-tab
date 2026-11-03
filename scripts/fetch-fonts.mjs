// Downloads IBM Plex (SIL Open Font Licence 1.1), latin subset, from @fontsource 5.3.0 once, so
// the page serves its own fonts and makes no third-party requests. Sans reads, Mono records, Sans
// Condensed signs.   node scripts/fetch-fonts.mjs
import fs from 'node:fs';
import path from 'node:path';
import { root } from './browser.mjs';

const VERSION = '5.3.0';
const FILES = [
  ['ibm-plex-sans', '400-normal'],
  ['ibm-plex-sans', '400-italic'],
  ['ibm-plex-sans', '500-normal'],
  ['ibm-plex-sans', '600-normal'],
  ['ibm-plex-mono', '400-normal'],
  ['ibm-plex-mono', '600-normal'],
  ['ibm-plex-sans-condensed', '500-normal'],
  ['ibm-plex-sans-condensed', '600-normal'],
];
const dir = path.join(root, 'public/fonts');
fs.mkdirSync(dir, { recursive: true });
const get = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return Buffer.from(await r.arrayBuffer());
};
for (const [family, face] of FILES) {
  const name = `${family}-latin-${face}.woff2`;
  const buf = await get(`https://cdn.jsdelivr.net/npm/@fontsource/${family}@${VERSION}/files/${name}`);
  if (buf.subarray(0, 4).toString('latin1') !== 'wOF2') throw new Error(`${name} is not a woff2 file`);
  fs.writeFileSync(path.join(dir, name), buf);
  console.log(`${name}: ${buf.length} bytes`);
}
const licence = await get(`https://cdn.jsdelivr.net/npm/@fontsource/ibm-plex-sans@${VERSION}/LICENSE`);
fs.writeFileSync(path.join(dir, 'OFL.txt'), licence);
console.log(`OFL.txt: ${licence.length} bytes`);
