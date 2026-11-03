// Trains in Chromium with WebGPU, no screenshots, and reports throughput and loss over time,
// including the loss at 60 seconds. Usage: node scripts/measure.mjs [sonnets|alice|tragedies|words|kernels] [--keep]
//   --keep  keeps training past the planned run (at the final learning rate) to reach 60 s.
import { launchWithWebGPU, startServer } from './browser.mjs';

const text = process.argv.find((a) => /^(sonnets|alice|tragedies|words|kernels)$/.test(a)) ?? 'sonnets';
const keep = process.argv.includes('--keep');
const { server, url } = await startServer();
// The page starts training on its own unless the reader prefers reduced motion.
const { browser, page, mode, adapter } = await launchWithWebGPU(url);
console.log(`Chromium ${mode}, adapter ${adapter}, text ${text}`);
page.on('console', (m) => console.log(`[console.${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
page.on('crash', () => console.log('[crash]'));
page.on('framenavigated', (f) => { if (f === page.mainFrame()) console.log(`[navigated] ${f.url()}`); });
await page.addInitScript(() => {
  new PerformanceObserver((list) => {
    for (const e of list.getEntries()) if (e.duration > 200) console.log(`long task ${Math.round(e.duration)} ms at ${Math.round(e.startTime)}`);
  }).observe({ type: 'longtask', buffered: true });
});
await page.goto(`${url}/?text=${text}`);
const rows = [];
const t0 = Date.now();
let last = null;
const limit = Number(process.env.MEASURE_SECONDS ?? 150) * 1000;
while (Date.now() - t0 < limit) {
  await page.waitForTimeout(1000);
  const m = await page.evaluate(() => window.__metrics ?? null).catch(() => null);
  if (!m) continue;
  last = m;
  rows.push(m);
  console.log(`${m.elapsed.toFixed(1).padStart(6)} s  step ${String(m.completed).padStart(5)}/${m.totalSteps}  ${Math.round(m.tokensPerSecond).toLocaleString('en-GB').padStart(9)} tok/s  loss ${m.loss.toFixed(3)}  held-out ${m.heldOut === null ? '-' : m.heldOut.toFixed(3)}`);
  if (m.state === 'finished') {
    if (keep && m.elapsed < 61) {
      await page.click('#runstop');
      continue;
    }
    break;
  }
}
const steady = rows.filter((r) => r.elapsed > 3 && r.state === 'training').map((r) => r.tokensPerSecond);
const median = steady.sort((a, b) => a - b)[Math.floor(steady.length / 2)] ?? 0;
console.log('\nsummary');
console.log(JSON.stringify({ text, final: last, medianTokensPerSecond: Math.round(median), averageTokensPerSecond: Math.round(last.tokensSeen / last.elapsed) }, null, 2));
await browser.close();
await server.close();
