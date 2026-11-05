// Cross-browser check of the production build in Chromium, WebKit and Firefox: the page loads
// with no console errors or warnings, trains (on the GPU where the browser offers WebGPU, on the
// CPU otherwise), and the scrubber, attention view and keyboard controls all work.
//   node scripts/browsers.mjs [chromium] [webkit] [firefox]
import fs from 'node:fs';
import path from 'node:path';
import { loadPlaywright, root, startServer } from './browser.mjs';

const pw = loadPlaywright();
const wanted = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const engines = wanted.length ? wanted : ['chromium', 'webkit', 'firefox'];
const { server, url } = await startServer({ dist: true });
const shots = path.join(root, 'shots');
fs.mkdirSync(shots, { recursive: true });
const results = [];

for (const name of engines) {
  const problems = [];
  const notes = [];
  const args = name === 'chromium' ? ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=metal', '--mute-audio'] : [];
  // Muted everywhere: the page plays nothing, and no automated browser on this machine may.
  const browser = await pw[name].launch({
    headless: true,
    ...(name === 'chromium' ? { channel: 'chromium' } : {}),
    ...(name === 'firefox' ? { firefoxUserPrefs: { 'media.volume_scale': '0.0' } } : {}),
    args,
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.on('console', (m) => {
      if (m.type() !== 'error' && m.type() !== 'warning') return;
      // Firefox logs its own notice when its WebGPU blocklist turns the API off. That message
      // comes from the browser, not this page: it is reported, not counted as a problem.
      if (name === 'firefox' && /WebGPU is disabled by blocklist/i.test(m.text())) notes.push(`browser notice: ${m.text().trim()}`);
      else problems.push(`[${m.type()}] ${m.text()}`);
    });
    page.on('pageerror', (e) => problems.push(`[pageerror] ${e.message}`));
    const t0 = Date.now();
    await page.goto(`${url}/`);
    const gpu = await page.evaluate(async () => {
      if (!('gpu' in navigator) || !navigator.gpu) return 'no navigator.gpu';
      const a = await navigator.gpu.requestAdapter();
      return a ? `adapter ${[a.info?.vendor, a.info?.architecture].filter(Boolean).join(' ') || 'present'}` : 'navigator.gpu but no adapter';
    });
    notes.push(`WebGPU: ${gpu}`);
    await page.waitForFunction(() => (window.__metrics?.completed ?? 0) > 5 && (window.__metrics?.drawn ?? 0) >= 2, null, { timeout: 90_000 });
    const m = await page.evaluate(() => window.__metrics);
    notes.push(
      `trains on the ${m.kind.toUpperCase()}: step ${m.completed} after ${((Date.now() - t0) / 1000).toFixed(1)} s, ${Math.round(m.tokensPerSecond).toLocaleString('en-GB')} tokens/s`,
    );
    await page.waitForFunction(() => document.getElementById('ref-legend').dataset.state === 'ready', null, { timeout: 20_000 });
    notes.push('recorded run on the trace');
    await page.click('#mode-heads');
    await page.waitForFunction(() => document.querySelectorAll('.head-cell').length >= 4, null, { timeout: 20_000 });
    notes.push(`heads pane: ${await page.locator('.head-cell').count()} heads`);
    await page.click('#mode-features');
    await page.waitForFunction(() => document.querySelectorAll('#feature-list .feature-row').length > 50, null, { timeout: 30_000 });
    notes.push(`features pane: ${await page.locator('#feature-list .feature-row').count()} features listed`);
    await page.click('#mode-train');
    // Scrub back to the first checkpoint with the keyboard.
    await page.focus('#scope');
    await page.keyboard.press('Home');
    await page.waitForFunction(() => /REPLAY step 0\b/.test(document.getElementById('where').textContent), null, { timeout: 15_000 });
    notes.push('scrubber: Home replays step 0');
    await page.keyboard.press('End');
    // Point at a character: what the model looked at when it wrote it.
    await page.locator('#text-body .g').nth(60).hover();
    await page.waitForFunction(() => /drew /.test(document.getElementById('text-note').textContent), null, { timeout: 20_000 });
    notes.push(`inspect: ${(await page.textContent('#text-note')).slice(0, 48)}…`);
    // Space holds and runs.
    await page.locator('body').click({ position: { x: 5, y: 5 } });
    await page.keyboard.press('Space');
    await page.waitForFunction(() => document.getElementById('runstop').dataset.state === 'held', null, { timeout: 10_000 });
    await page.keyboard.press('Space');
    await page.waitForFunction(() => document.getElementById('runstop').dataset.state === 'running', null, { timeout: 10_000 });
    notes.push('space: holds and runs');
    await page.screenshot({ path: path.join(shots, `browser-${name}.png`) });
  } catch (e) {
    problems.push(String(e).split('\n')[0]);
  } finally {
    await browser.close();
  }
  results.push({ name, notes, problems });
  console.log(`${problems.length ? 'FAIL' : 'pass'}  ${name}\n      ${notes.join('\n      ')}${problems.length ? `\n      ${problems.join('\n      ')}` : ''}`);
}
await server.close();
process.exit(results.some((r) => r.problems.length) ? 1 : 0);
