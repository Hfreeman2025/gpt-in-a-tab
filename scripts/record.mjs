// Records a run of this page's own code for the instant opener and the fallback: the production
// build in headless Chromium with WebGPU, ?record, trained on the sonnets, dissected. Writes
// public/recorded/run.json, features.json and weights.bin.
//   node scripts/record.mjs
import fs from 'node:fs';
import path from 'node:path';
import { launchWithWebGPU, root, startServer } from './browser.mjs';

const dir = path.join(root, 'public/recorded');
fs.mkdirSync(dir, { recursive: true });
const { server, url } = await startServer({ dist: true });
const { browser, page, adapter } = await launchWithWebGPU(url, { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
const problems = [];
page.on('console', (m) => {
  if (m.type() === 'error' || m.type() === 'warning') problems.push(`[${m.type()}] ${m.text()}`);
});
page.on('pageerror', (e) => problems.push(`[pageerror] ${e.message}`));
try {
  console.log(`adapter: ${adapter}`);
  await page.goto(`${url}/?record&text=sonnets`);
  await page.waitForFunction(() => window.__record !== undefined, null, { timeout: 10 * 60_000, polling: 1000 });
  const rec = await page.evaluate(() => window.__record);
  const weights = Buffer.from(rec.weights, 'base64');
  if (weights.length !== rec.run.weights.bytes) throw new Error(`weights: ${weights.length} bytes, expected ${rec.run.weights.bytes}`);
  fs.writeFileSync(path.join(dir, 'run.json'), JSON.stringify(rec.run));
  fs.writeFileSync(path.join(dir, 'features.json'), JSON.stringify(rec.features));
  fs.writeFileSync(path.join(dir, 'weights.bin'), weights);
  const r = rec.run;
  const m = r.moment;
  console.log(`recorded on ${r.device}: ${r.totalSteps} steps in ${r.seconds.toFixed(1)} s`);
  console.log(
    m
      ? `moment: head ${m.head} at step ${m.step}, prefix ${m.prefix.toFixed(2)}, copy ${m.before?.copySecond.toFixed(2)} → ${m.copySecond.toFixed(2)}`
      : 'no induction head',
  );
  const s = r.dissection.sae;
  console.log(`sae: fvu ${s.fvu} L0 ${s.l0} dead ${s.dead} in ${s.seconds} s`);
  for (const f of ['run.json', 'features.json', 'weights.bin'])
    console.log(`public/recorded/${f}: ${fs.statSync(path.join(dir, f)).size.toLocaleString('en-GB')} bytes`);
} catch (e) {
  problems.push(String(e).split('\n')[0]);
} finally {
  await browser.close();
  await server.close();
}
if (problems.length) {
  console.log(problems.join('\n'));
  process.exit(1);
}
