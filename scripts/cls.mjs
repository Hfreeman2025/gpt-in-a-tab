// Layout shift while the page loads: every layout-shift entry (without recent input) in the first
// six seconds, at 1440×900 and 390×844, on the production build. Fonts swap in, the recorded run
// arrives and the first text is drawn in that window.
//   node scripts/cls.mjs
import { launchWithWebGPU, startServer } from './browser.mjs';

const { server, url } = await startServer({ dist: true });
let worst = 0;
for (const [name, viewport] of [
  ['desktop', { width: 1440, height: 900 }],
  ['mobile', { width: 390, height: 844 }],
]) {
  const { browser, page } = await launchWithWebGPU(url, { viewport, deviceScaleFactor: 2 });
  await page.addInitScript(() => {
    window.__cls = { total: 0, entries: [] };
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) {
        if (e.hadRecentInput) continue;
        window.__cls.total += e.value;
        window.__cls.entries.push({
          t: Math.round(e.startTime),
          v: e.value,
          from: (e.sources ?? []).map((s) => s.node?.id || s.node?.className || s.node?.nodeName || '?').join(' '),
        });
      }
    }).observe({ type: 'layout-shift', buffered: true });
  });
  await page.goto(`${url}/`);
  await page.waitForTimeout(6000);
  const cls = await page.evaluate(() => ({
    ...window.__cls,
    fonts: [...document.fonts].filter((f) => f.status === 'loaded').map((f) => `${f.family} ${f.weight}${f.style === 'italic' ? 'i' : ''}`),
  }));
  worst = Math.max(worst, cls.total);
  console.log(`${name}: CLS ${cls.total.toFixed(4)} over ${cls.entries.length} shifts; fonts loaded: ${cls.fonts.join(', ')}`);
  for (const e of cls.entries.slice(0, 6)) console.log(`   ${e.t} ms  ${e.v.toFixed(4)}  ${e.from}`);
  await browser.close();
}
await server.close();
process.exit(worst > 0 ? 1 : 0);
