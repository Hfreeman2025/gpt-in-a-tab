// Every installed package must carry a permissive licence. Nothing ships at runtime (the page
// has no dependencies), but the toolchain is checked too.
import fs from 'node:fs';
import path from 'node:path';
import { root } from './browser.mjs';

// Permissive licences only: MIT, Apache-2.0, BSD (any clause count) and ISC.
const ALLOWED = new Set(['MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', '0BSD', 'ISC']);
const lock = JSON.parse(fs.readFileSync(path.join(root, 'package-lock.json'), 'utf8'));
const counts = new Map();
const bad = [];

// Accept SPDX expressions made only of allowed licences, e.g. "(MIT OR Apache-2.0)".
const ok = (expr) =>
  expr
    .replace(/[()]/g, ' ')
    .split(/\s+(?:OR|AND)\s+|\s+/)
    .filter(Boolean)
    .every((id) => ALLOWED.has(id));

for (const [key, meta] of Object.entries(lock.packages ?? {})) {
  if (key === '') continue;
  let lic = meta.license;
  if (!lic) {
    try {
      lic = JSON.parse(fs.readFileSync(path.join(root, key, 'package.json'), 'utf8')).license;
    } catch {
      lic = undefined;
    }
  }
  const id = typeof lic === 'string' ? lic : (lic?.type ?? 'UNKNOWN');
  counts.set(id, (counts.get(id) ?? 0) + 1);
  if (!ok(id)) bad.push(`${key.replace(/^node_modules\//, '')}: ${id}`);
}

console.log(
  [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([l, n]) => `${l} ${n}`)
    .join(', '),
);
if (bad.length) {
  console.error(`\nNot on the allow-list:\n${bad.join('\n')}`);
  process.exit(1);
}
console.log('All dependency licences are on the allow-list.');
