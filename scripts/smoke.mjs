// Browser test of the page's lifecycle races, against the production build in Chromium with WebGPU.
import { launchWithWebGPU, startServer } from './browser.mjs';

const { server, url } = await startServer({ dist: true });
const { browser, page } = await launchWithWebGPU(url, { reducedMotion: 'reduce' });
const problems = [];
page.on('pageerror', (e) => problems.push(`page error: ${e.message}`));
page.on('console', (m) => {
  if (m.type() === 'error' || m.type() === 'warning') problems.push(`console ${m.type()}: ${m.text()}`);
});
const metrics = () => page.evaluate(() => window.__metrics ?? null);
const key = () => page.evaluate(() => document.getElementById('runstop').dataset.state);
const where = () => page.evaluate(() => document.getElementById('where').textContent);
const text = () => page.evaluate(() => document.getElementById('text-body').textContent);
const check = (ok, what) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'}  ${what}`);
  if (!ok) problems.push(what);
};
const until = (fn, arg, timeout = 180_000) => page.waitForFunction(fn, arg, { timeout });

try {
  // With reduced motion nothing starts on its own; a double press must start exactly one run.
  await page.goto(`${url}/`);
  await until(() => document.getElementById('runstop').dataset.state === 'armed');
  check((await metrics()) === null, 'with reduced motion, nothing trains until RUN is pressed');
  await page.dblclick('#runstop');
  await until(() => (window.__metrics?.completed ?? 0) > 60 && (window.__metrics?.drawn ?? 0) >= 6);
  check((await metrics()).sessions === 1, 'a double press of RUN starts exactly one run');

  // Replaying a checkpoint gives the same text every time (the same random numbers).
  await page.focus('#scope');
  await page.keyboard.press('Home');
  await until(() => /REPLAY step 0\b/.test(document.getElementById('where').textContent));
  await until(() => document.querySelectorAll('#text-body .g').length > 100);
  const first = await text();
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(400);
  await page.keyboard.press('Home');
  await until(() => /REPLAY step 0\b/.test(document.getElementById('where').textContent));
  await page.waitForTimeout(300);
  check((await text()) === first, 'step 0 replays the identical text');
  await page.keyboard.press('End');

  // Pointing at a character holds the text still while training carries on; letting go catches up.
  await until(() => /LIVE/.test(document.getElementById('where').textContent));
  const glyph = await page.evaluate(() => {
    const r = document.querySelector('#text-body .g[data-i="40"]').getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  });
  await page.mouse.move(glyph.x, glyph.y);
  await until(() => /text held/.test(document.getElementById('where').textContent), null, 20_000);
  const heldText = await text();
  const heldWhere = await where();
  const trainedAt = (await metrics()).completed;
  await page.waitForTimeout(1500);
  check((await text()) === heldText && (await where()) === heldWhere && (await metrics()).completed > trainedAt, 'pointing at a character holds the text while training carries on');
  await page.mouse.move(5, 5);
  await until((w) => !/held/.test(document.getElementById('where').textContent) && document.getElementById('where').textContent !== w, heldWhere, 30_000);
  check(true, 'letting go of the character, the live text catches up');

  // A new prompt re-draws every checkpoint; live stays on the newest one, never jumping back.
  const before = await metrics();
  await page.fill('#prompt', 'Shall I');
  await page.waitForTimeout(1200);
  const w = await where();
  const shown = Number(/step ([\d,]+)/.exec(w)?.[1].replace(/,/g, '') ?? 0);
  check(/LIVE/.test(w) && shown >= before.completed - 400, `after a new prompt, live stays near the run (${w.trim()} at step ${before.completed})`);
  await page.fill('#prompt', '');

  // Time the kernels; hold during the timing; the run must stay held afterwards, then carry on.
  await page.locator('#system').scrollIntoViewIfNeeded();
  await until(() => document.querySelector('#system-body button') !== null);
  const steps = (await metrics()).completed;
  await page.locator('#system-body button').first().click();
  await page.locator('#runstop').click();
  await until(() => document.querySelector('.kernel-table') !== null, null, 60_000);
  await page.waitForTimeout(800);
  const held = await metrics();
  await page.waitForTimeout(800);
  const after = { key: await key(), steps: (await metrics()).completed };
  check(after.key === 'held' && after.steps === held.completed, `holding during kernel timing keeps the run held (key ${after.key}, steps ${held.completed} then ${after.steps})`);
  check(held.completed >= steps, 'the step count never goes backwards across timing');
  await page.locator('#runstop').click();
  await until((n) => (window.__metrics?.completed ?? 0) > n + 30, held.completed);
  const resumed = await key();
  check(resumed === 'running', `the run carries on after timing (key ${resumed})`);

  // Run to the end; then 25% more stays at the final learning rate.
  await until(() => window.__metrics?.state === 'finished', null, 300_000);
  check((await key()) === 'done', 'the key reads DONE when the run ends');
  check(/learned to copy at step/.test(await page.textContent('#moment')), `an induction head formed and was named (${(await page.textContent('#moment')).slice(0, 40)}…)`);

  // The microscope: knocking out the copying heads breaks copying; pinning a feature changes the writing.
  await page.click('#mode-heads');
  await until(() => document.querySelector('table.ko') !== null, null, 120_000);
  const copiers = await page.evaluate(() => [...document.querySelectorAll('.head-cell--copies .head-name')].map((e) => e.textContent));
  check(copiers.length > 0, `the heads pane marks the copying heads (${copiers.join(', ')})`);
  for (const name of copiers) await page.locator('.head-cell', { has: page.locator('.head-name', { hasText: name }) }).first().click({ modifiers: ['Shift'] });
  await until(() => /\d+\/\d+/.test(document.getElementById('copy-off').textContent) && /\d+\/\d+/.test(document.getElementById('copy-base').textContent), null, 60_000);
  const score = (id) => page.evaluate((i) => Number(/(\d+)\/(\d+)/.exec(document.getElementById(i).textContent)?.[1] ?? -1), id);
  const [on, off] = [await score('copy-base'), await score('copy-off')];
  check(on > off + 10, `with the copying heads knocked out it stops copying (${on} right as trained, ${off} without them)`);
  await page.click('#mode-features');
  await until(() => document.querySelectorAll('#feature-list .feature-row').length > 100, null, 120_000);
  await page.getByRole('button', { name: 'Triple' }).click();
  await until(() => (document.getElementById('fd-pinned')?.textContent ?? '').length > 50 && (document.getElementById('fd-base')?.textContent ?? '').length > 50, null, 60_000);
  const [base, pinned] = await page.evaluate(() => [document.getElementById('fd-base').textContent, document.getElementById('fd-pinned').textContent]);
  check(base !== pinned, 'pinning a feature changes what the model writes');
  await page.click('#mode-train');
  const done = await metrics();
  await page.locator('#runstop').click();
  await until((n) => (window.__metrics?.completed ?? 0) > n + 40, done.completed);
  const more = await metrics();
  check(Math.abs(more.lr - 3e-4) < 1e-9, `training more runs at the final learning rate (${more.lr})`);
  await page.locator('#runstop').click();
  await until(() => document.getElementById('runstop').dataset.state === 'held');

  // Pick another text while the kernels are being timed: a new run starts and trains normally.
  await page.locator('#system-body button').first().click().catch(() => undefined);
  await page.locator('[data-id="alice"]').click();
  await until(() => window.__metrics?.sessions === 2 && (window.__metrics?.completed ?? 0) > 80, null, 120_000);
  const second = await metrics();
  check(second.state === 'training' && Number.isFinite(second.loss), 'a new text after a run starts a fresh run that trains');
  check((await page.locator('.fault').count()) === 0, 'no fault on the page');
} catch (e) {
  problems.push(String(e).split('\n')[0]);
} finally {
  await browser.close();
  await server.close();
}
console.log(problems.length ? `\n${problems.length} problem(s):\n${problems.join('\n')}` : '\nsmoke test passed');
process.exit(problems.length ? 1 : 0);
