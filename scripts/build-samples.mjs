// Builds the one bundled sample that is not from Project Gutenberg (those come from
// scripts/fetch-texts.mjs): the word list. Run once; the output lives in src/samples/.
//   node scripts/build-samples.mjs
import fs from 'node:fs';
import path from 'node:path';
import { root } from './browser.mjs';

// Webster's Second New International Dictionary (1934) word list, as shipped with BSD Unix.
// The README that ships with it notes the copyright has lapsed. A seeded shuffle of 30,000
// lowercase entries keeps the download small while leaving far more text than a run can memorise.
const dict = '/usr/share/dict/web2';
if (fs.existsSync(dict)) {
  const words = fs.readFileSync(dict, 'utf8').split('\n').filter((w) => /^[a-z]{3,18}$/.test(w));
  let s = 20261101 >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), s | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = words.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [words[i], words[j]] = [words[j], words[i]];
  }
  const picked = words.slice(0, 30000).join('\n') + '\n';
  fs.writeFileSync(path.join(root, 'src/samples/words.txt'), picked);
  console.log(`words: ${words.length} eligible, kept 30000, ${picked.length} chars`);
} else {
  console.log('words: /usr/share/dict/web2 not found, keeping the existing src/samples/words.txt');
}
