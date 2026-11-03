// Opens a development page (tests/gpu/*.html) in Chromium with WebGPU, streams its console, and
// waits for window.__exp.done. Used for the experiments behind the run's settings.
//   node scripts/run-page.mjs tests/gpu/induction.html "?steps=1700" [--out file.json]
import fs from 'node:fs';
import { launchWithWebGPU, startServer } from './browser.mjs';

const [page, query = ''] = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const outArg = process.argv.indexOf('--out');
const { server, url } = await startServer({ dist: false });
const { browser, page: p, adapter } = await launchWithWebGPU(url);
console.log(`adapter: ${adapter}`);
p.on('console', (m) => console.log(m.text()));
p.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
await p.goto(`${url}/${page}${query}`);
await p.waitForFunction(() => window.__exp?.done === true, null, { timeout: 30 * 60_000 });
if (outArg > 0) fs.writeFileSync(process.argv[outArg + 1], JSON.stringify(await p.evaluate(() => window.__exp), null, 1));
await browser.close();
await server.close();
