// Screenshots of every state of the page, at 1440×900 and 390×844, into shots/v3/ (or --out=dir).
//   node scripts/shoot.mjs [desktop] [mobile] [states] [--out=shots/v3-plex]
// One browser at a time, muted (see browser.mjs).
import fs from 'node:fs';
import path from 'node:path';
import { launchWithWebGPU, root, startServer } from './browser.mjs';

const out = path.join(root, process.argv.find((a) => a.startsWith('--out='))?.slice(6) ?? 'shots/v3');
fs.mkdirSync(out, { recursive: true });
const wanted = new Set(process.argv.slice(2).filter((a) => !a.startsWith('--')));
const want = (n) => wanted.size === 0 || wanted.has(n);
const { server, url } = await startServer({ dist: true });
const problems = [];

const SIZES = { desktop: { width: 1440, height: 900 }, mobile: { width: 390, height: 844 } };

async function open(size, query = '', opts = {}) {
  const { slow, before, expectFailure, ...launch } = opts;
  const launched = await launchWithWebGPU(url, { viewport: SIZES[size], deviceScaleFactor: 2, ...launch });
  const { page } = launched;
  page.on('console', (m) => {
    if (m.type() !== 'error' && m.type() !== 'warning') return;
    // The failed-download state is made by refusing the request: its network error is expected.
    if (expectFailure && /Failed to load resource|ERR_FAILED/.test(m.text())) return;
    problems.push(`${size}${query}: [${m.type()}] ${m.text()}`);
  });
  page.on('pageerror', (e) => problems.push(`${size}${query}: [pageerror] ${e.message}`));
  if (slow) {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('Network.enable');
    await cdp.send('Network.emulateNetworkConditions', { offline: false, latency: 400, downloadThroughput: 60 * 1024, uploadThroughput: 30 * 1024 });
  }
  if (before) await before(page);
  await page.goto(`${url}/${query}`, { waitUntil: 'commit' });
  return launched;
}

const shot = (page, size, name, full = false) => page.screenshot({ path: path.join(out, `${size}-${name}.png`), fullPage: full });
const until = (page, fn, arg, timeout = 240_000) =>
  page.waitForFunction(fn, arg, { timeout }).catch(async (e) => {
    const state = await page.evaluate(() => ({ key: document.getElementById('runstop')?.dataset.state, metrics: window.__metrics })).catch((x) => String(x));
    console.error('waiting failed; page state:', JSON.stringify(state));
    throw e;
  });
const headCell = (page, name) => page.locator('.head-cell', { has: page.locator('.head-name', { hasText: name }) }).first();

for (const size of ['desktop', 'mobile']) {
  if (!want(size)) continue;

  // Loading on a slow network: the shell before the code and the text arrive.
  {
    const { browser, page } = await open(size, '', { slow: true });
    await page.waitForTimeout(1200);
    await shot(page, size, '01-loading-slow-network');
    await browser.close();
  }

  // Reduced motion: nothing starts on its own; the recorded run is already on the trace.
  {
    const { browser, page } = await open(size, '', { reducedMotion: 'reduce' });
    await until(page, () => document.getElementById('runstop').dataset.state === 'armed' && document.getElementById('ref-legend').dataset.state === 'ready');
    await page.waitForTimeout(400);
    await shot(page, size, '03-armed-reduced-motion');
    await browser.close();
  }

  // The run: the opener, training, the moment, replay, inspection, the end, the microscope.
  {
    const { browser, page } = await open(size);
    await page.waitForTimeout(1500);
    await shot(page, size, '02-opener-1500ms');
    await until(page, () => (window.__metrics?.completed ?? 0) > 600, null, 60_000);
    await shot(page, size, '04-training');
    await until(page, () => /learned to copy at step/.test(document.getElementById('moment').textContent), null, 120_000);
    await page.waitForTimeout(1600);
    await shot(page, size, '05-the-moment');
    // Replay the first checkpoint.
    await page.focus('#scope');
    await page.keyboard.press('Home');
    await page.waitForTimeout(800);
    await shot(page, size, '06-replay-step-0');
    await page.keyboard.press('End');
    await until(page, () => window.__metrics?.state === 'finished');
    await page.waitForTimeout(1200);
    // Inspect a character: what the model looked at.
    const at = await page.evaluate(() => {
      const gs = [...document.querySelectorAll('#text-body .g[data-i]:not(.cut)')];
      let pick = gs[Math.floor(gs.length / 2)];
      for (let i = Math.floor(gs.length * 0.6); i < gs.length - 1; i++) {
        if (/\p{L}/u.test(gs[i].textContent) && /\p{L}/u.test(gs[i - 1].textContent)) {
          pick = gs[i];
          break;
        }
      }
      const r = pick.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    });
    await page.mouse.move(at.x, at.y);
    await page.waitForTimeout(900);
    await shot(page, size, '07-inspect');
    await page.mouse.move(2, 2);
    await page.waitForTimeout(300);
    await shot(page, size, '08-done');
    // Heads, once the knockouts and patching are in: the copy heads knocked out while it writes.
    await page.click('#mode-heads');
    await until(page, () => document.querySelector('table.ko') !== null, null, 120_000);
    const copyHeads = await page.evaluate(() => [...document.querySelectorAll('.head-cell--copies .head-name')].map((e) => e.textContent));
    for (const name of copyHeads) await headCell(page, name).click({ modifiers: ['Shift'] });
    if (copyHeads[0]) await headCell(page, copyHeads[0]).click();
    await until(page, () => (document.getElementById('copy-off').textContent ?? '').length > 10, null, 60_000);
    await page.waitForTimeout(600);
    await page.evaluate(() => window.scrollTo(0, 0));
    await shot(page, size, '09-heads', size === 'mobile');
    // Features, once the autoencoder is trained: a feature pinned at three times its strongest.
    await page.click('#mode-features');
    await until(page, () => document.querySelectorAll('#feature-list .feature-row').length > 20, null, 120_000);
    await page.getByRole('button', { name: 'Triple' }).click();
    await until(page, () => (document.getElementById('fd-pinned')?.textContent ?? '').length > 50, null, 60_000);
    await page.waitForTimeout(500);
    await page.evaluate(() => window.scrollTo(0, 0));
    await shot(page, size, '10-features', size === 'mobile');
    await page.fill('#feature-filter', 'after “');
    await page.waitForTimeout(400);
    await shot(page, size, '11-features-filtered');
    await page.click('#mode-train');
    await page.waitForTimeout(400);
    await shot(page, size, '12-full-page', true);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    if (overflow > 0) problems.push(`${size}: page is ${overflow}px wider than the viewport`);
    // A fault, as the page draws it (the error is injected; a real one comes from the GPU).
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.evaluate(() => window.__app.fault('Training stopped.', 'The GPU device was lost (a simulated fault, for this screenshot).', true));
    await page.waitForTimeout(300);
    await shot(page, size, '13-fault-simulated');
    await browser.close();
  }

  if (want('states')) {
    // No WebGPU: the CPU fallback trains live; the microscope shows the recorded run.
    {
      const { browser, page } = await open(size, '?cpu');
      await until(page, () => (window.__metrics?.drawn ?? 0) >= 4, null, 60_000);
      await page.waitForTimeout(800);
      await shot(page, size, '14-cpu-fallback');
      await page.click('#mode-features');
      await until(page, () => document.querySelectorAll('#feature-list .feature-row').length > 20, null, 60_000);
      await page.waitForTimeout(400);
      await shot(page, size, '15-recorded-weights-offered', size === 'mobile');
      await page.getByRole('button', { name: /Load it/ }).click();
      await until(page, () => (document.getElementById('fd-pinned')?.textContent ?? '').length > 50, null, 60_000);
      await page.waitForTimeout(400);
      await shot(page, size, '16-recorded-weights-loaded', size === 'mobile');
      await browser.close();
    }
    // The download fails: the page says so and stays useful.
    {
      const { browser, page } = await open(size, '?cpu', { expectFailure: true, before: (p) => p.route('**/recorded/weights.bin', (r) => r.abort('failed')) });
      await page.waitForTimeout(1500);
      await page.click('#mode-features');
      await until(page, () => document.querySelectorAll('#feature-list .feature-row').length > 20, null, 60_000);
      await page.getByRole('button', { name: /Load it/ }).click();
      await until(page, () => document.querySelector('.gate--failed') !== null, null, 30_000);
      await page.waitForTimeout(300);
      await shot(page, size, '17-download-failed', size === 'mobile');
      await browser.close();
    }
    // Your own text, empty.
    {
      const { browser, page } = await open(size, '?hold');
      await until(page, () => document.getElementById('runstop').dataset.state === 'armed');
      await page.locator('[data-id="own"]').click();
      await page.waitForTimeout(300);
      await shot(page, size, '18-paste-empty');
      await browser.close();
    }
  }
}

await server.close();
console.log(problems.length ? `problems:\n${problems.join('\n')}` : 'no console errors or warnings, no horizontal overflow');
process.exit(problems.length ? 1 : 0);
