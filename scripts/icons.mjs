// Renders the PNG icons from public/favicon.svg, and the 1200×630 social card (public/og.png)
// from the real instrument mid-run. Needs WebGPU in Chromium, like the other browser scripts.
//   PLAYWRIGHT_FROM=/path/to/project-with-playwright node scripts/icons.mjs
import fs from 'node:fs';
import path from 'node:path';
import { launchPlain, launchWithWebGPU, root, startServer } from './browser.mjs';

const svg = fs.readFileSync(path.join(root, 'public/favicon.svg'), 'utf8');
{
  const { browser, page } = await launchPlain('', { viewport: { width: 200, height: 200 }, deviceScaleFactor: 1 });
  for (const [name, size] of [
    ['favicon-32.png', 32],
    ['apple-touch-icon.png', 180],
  ]) {
    await page.setContent(`<html><body style="margin:0;background:#121314">${svg.replace('<svg ', `<svg width="${size}" height="${size}" `)}</body></html>`);
    await page.locator('svg').screenshot({ path: path.join(root, 'public', name), omitBackground: false });
    console.log(`public/${name}`);
  }
  await browser.close();
}

{
  const { server, url } = await startServer({ dist: true });
  const { browser, page } = await launchWithWebGPU(url, { viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
  await page.goto(`${url}/`);
  // The card shows the moment: an induction head named on the trace, its lanes lit.
  await page.waitForFunction(() => /learned to copy at step/.test(document.getElementById('moment')?.textContent ?? ''), null, { timeout: 180_000 });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: path.join(root, 'public/og.png') });
  console.log('public/og.png');
  await browser.close();
  await server.close();
}
