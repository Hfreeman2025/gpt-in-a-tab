// Downloads the two typefaces (SIL Open Font Licence) from Google Fonts once, so the page serves
// them itself and makes no third-party requests.   node scripts/fetch-fonts.mjs
import fs from 'node:fs';
import path from 'node:path';
import { root } from './browser.mjs';

const css = 'https://fonts.googleapis.com/css2?family=B612:wght@400;700&family=B612+Mono:wght@400;700&display=swap';
const ua = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36';
const dir = path.join(root, "public/fonts");
fs.mkdirSync(dir, { recursive: true });
const sheet = await (await fetch(css, { headers: { 'user-agent': ua } })).text();
const faces = [...sheet.matchAll(/font-family: '([^']+)';\s*font-style: (\w+);\s*font-weight: (\d+);[^}]*?src: url\(([^)]+)\)/g)];
for (const [, family, style, weight, url] of faces) {
  const name = `${family.toLowerCase().replace(/\s+/g, '-')}-${weight}${style === 'italic' ? 'i' : ''}.woff2`;
  const buf = Buffer.from(await (await fetch(url)).arrayBuffer());
  fs.writeFileSync(path.join(dir, name), buf);
  console.log(`${name}: ${buf.length} bytes`);
}
