// Performance of the production build in Chromium with WebGPU: first and largest contentful
// paint, time until the model's text is live, JavaScript shipped (gzipped), and frame times
// while scrubbing through training as fast as a hand can drag, with training running.
//   node scripts/perf.mjs
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { launchWithWebGPU, root, startServer } from './browser.mjs';

const { server, url } = await startServer({ dist: true });
const dist = path.join(root, 'dist');
const gz = (f) => zlib.gzipSync(fs.readFileSync(f), { level: 9 }).length;
const kb = (b) => `${(b / 1024).toFixed(1)} KB`;

const { browser, page } = await launchWithWebGPU(url, { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
const loaded = new Map();
page.on('response', async (r) => {
  const u = new URL(r.url());
  if (u.pathname.endsWith('.js')) loaded.set(u.pathname, path.join(dist, u.pathname));
});
await page.addInitScript(() => {
  window.__perf = { fcp: 0, lcp: 0, firstText: 0, ref: 0, moment: 0 };
  const watchRef = () => {
    const el = document.getElementById('ref-legend');
    if (el && !el.hidden) window.__perf.ref = performance.now();
    else requestAnimationFrame(watchRef);
  };
  requestAnimationFrame(watchRef);
  new PerformanceObserver((l) => {
    for (const e of l.getEntries()) if (e.name === 'first-contentful-paint') window.__perf.fcp = e.startTime;
  }).observe({ type: 'paint', buffered: true });
  new PerformanceObserver((l) => {
    for (const e of l.getEntries()) window.__perf.lcp = e.startTime;
  }).observe({ type: 'largest-contentful-paint', buffered: true });
  const watch = () => {
    if (!window.__perf.firstText && document.querySelector('#text-body .g')) window.__perf.firstText = performance.now();
    if (!window.__perf.firstText) requestAnimationFrame(watch);
  };
  requestAnimationFrame(watch);
});
await page.goto(`${url}/`);
await page.waitForFunction(() => window.__perf.firstText > 0 && (window.__metrics?.drawn ?? 0) >= 12, null, { timeout: 120_000 });

// Frame times through the heaviest animation: training at full speed while the induction moment
// draws itself in (from 200 steps before it until two seconds after).
await page.waitForFunction(() => (window.__metrics?.completed ?? 0) >= 1250, null, { timeout: 180_000 });
await page.evaluate(() => {
  window.__mframes = [];
  let last = performance.now();
  const tick = (t) => {
    window.__mframes.push(t - last);
    last = t;
    if (!window.__perf.moment && /learned to copy at step/.test(document.getElementById('moment')?.textContent ?? '')) window.__perf.moment = t;
    if (!window.__perf.moment || t - window.__perf.moment < 2000) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
});
await page.waitForFunction(() => window.__perf.moment > 0 && performance.now() - window.__perf.moment > 2100, null, { timeout: 180_000 });
const mframes = (await page.evaluate(() => window.__mframes)).slice(2).sort((a, b) => a - b);
const perf = await page.evaluate(() => window.__perf);

// How far the live text trails the run: every 250 ms for 8 s, the step being trained against
// the step whose text is on screen.
const lagSamples = [];
for (let i = 0; i < 32; i++) {
  lagSamples.push(
    await page.evaluate(() => ({
      t: performance.now(),
      trained: window.__metrics?.completed ?? 0,
      shown: Number(/step ([\d,]+)/.exec(document.getElementById('where').textContent)?.[1].replace(/,/g, '') ?? 0),
    })),
  );
  await page.waitForTimeout(250);
}
const rate = (lagSamples.at(-1).trained - lagSamples[0].trained) / ((lagSamples.at(-1).t - lagSamples[0].t) / 1000);
const lags = lagSamples.map((l) => l.trained - l.shown).sort((a, b) => a - b);
const lagAt = (p) => lags[Math.min(lags.length - 1, Math.floor(p * lags.length))];

// Frame times while dragging the cursor back and forth across the trace for five seconds.
const box = await page.locator('#scope').boundingBox();
await page.evaluate(() => {
  window.__frames = [];
  let last = performance.now();
  const tick = (t) => {
    window.__frames.push(t - last);
    last = t;
    if (window.__frames.length < 100000) window.__raf = requestAnimationFrame(tick);
  };
  window.__raf = requestAnimationFrame(tick);
});
await page.mouse.move(box.x + 40, box.y + box.height / 2);
await page.mouse.down();
const t0 = Date.now();
let i = 0;
while (Date.now() - t0 < 5000) {
  const f = (Math.sin(i / 12) + 1) / 2;
  await page.mouse.move(box.x + 40 + f * (box.width * 0.4), box.y + box.height / 2);
  i++;
}
await page.mouse.up();
const frames = await page.evaluate(() => {
  cancelAnimationFrame(window.__raf);
  return window.__frames.slice(2);
});
const sorted = [...frames].sort((a, b) => a - b);
const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
const mq = (p) => mframes[Math.min(mframes.length - 1, Math.floor(p * mframes.length))];
const metrics = await page.evaluate(() => window.__metrics);
// The end of the run, and the dissection: knockouts, patching, the autoencoder, its analysis.
await page.waitForFunction(() => window.__metrics?.state === 'finished', null, { timeout: 300_000 });
const finished = Date.now();
await page.click('#mode-features');
await page.waitForFunction(() => document.querySelectorAll('#feature-list .feature-row').length > 100 && /in [\d.]+ s on the GPU/.test(document.getElementById('sae-status').textContent) && !document.querySelector('.switch--live[aria-pressed="false"]'), null, { timeout: 300_000 });
const dissected = (Date.now() - finished) / 1000;
const saeFacts = await page.textContent('#sae-status');
await browser.close();
await server.close();

let js = 0;
for (const f of loaded.values()) if (fs.existsSync(f)) js += gz(f);
const all = fs.readdirSync(path.join(dist, 'assets')).filter((f) => f.endsWith('.js'));
const allJs = all.reduce((a, f) => a + gz(path.join(dist, 'assets', f)), 0);
const css = fs.readdirSync(path.join(dist, 'assets')).filter((f) => f.endsWith('.css')).reduce((a, f) => a + gz(path.join(dist, 'assets', f)), 0);

const rec = (f) => fs.statSync(path.join(dist, 'recorded', f)).size;
console.log(`first contentful paint     ${perf.fcp.toFixed(0)} ms`);
console.log(`recorded run on the trace  ${perf.ref.toFixed(0)} ms after navigation (run.json ${kb(rec('run.json'))}, ${kb(gz(path.join(dist, 'recorded/run.json')))} gzipped)`);
console.log(`induction moment fired     ${(perf.moment / 1000).toFixed(1)} s after navigation`);
console.log(`frames through the moment  ${mframes.length}: median ${mq(0.5).toFixed(1)} ms, p95 ${mq(0.95).toFixed(1)} ms, max ${mframes[mframes.length - 1].toFixed(1)} ms, ${mframes.filter((f) => f > 50).length} over 50 ms`);
console.log(`dissection after the run   ${dissected.toFixed(1)} s (${saeFacts.split('·')[1]?.trim() ?? ''})`);
console.log(`recorded downloads         features.json ${kb(rec('features.json'))} (${kb(gz(path.join(dist, 'recorded/features.json')))} gzipped) at the first visit to Features; weights.bin ${kb(rec('weights.bin'))}, only on request`);
console.log(`largest contentful paint   ${perf.lcp.toFixed(0)} ms`);
console.log(`model's text on screen     ${perf.firstText.toFixed(0)} ms after navigation`);
console.log(`JavaScript loaded at start ${kb(js)} gzipped (${loaded.size} files: page, workers, the default text)`);
console.log(`JavaScript in the build    ${kb(allJs)} gzipped, all texts included; CSS ${kb(css)}`);
console.log(`live text behind the run   median ${lagAt(0.5)} steps (${((lagAt(0.5) / rate) * 1000).toFixed(0)} ms), p95 ${lagAt(0.95)} steps, at ${rate.toFixed(0)} steps/s; checkpoints every 50–100 steps here`);
console.log(`frames while scrubbing     ${frames.length} in 5 s: median ${q(0.5).toFixed(1)} ms, p95 ${q(0.95).toFixed(1)} ms, max ${sorted[sorted.length - 1].toFixed(1)} ms, ${frames.filter((f) => f > 50).length} over 50 ms`);
console.log(`training meanwhile         step ${metrics.completed}, ${Math.round(metrics.tokensPerSecond).toLocaleString('en-GB')} tokens/s`);
