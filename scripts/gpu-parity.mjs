// Runs tests/gpu/parity.ts in Chromium with WebGPU and reports every kernel's agreement with the
// CPU reference. Exits non-zero on any failure, or if no hardware GPU adapter is reachable.
import { launchWithWebGPU, loadPlaywright, startServer } from './browser.mjs';

// --browser webkit|firefox runs the suite in another engine's WebGPU implementation.
// Firefox blocklists WebGPU headless on macOS; --ignore-blocklist overrides that, for testing only.
const browserArg = process.argv.find((a) => a.startsWith('--browser='))?.split('=')[1] ?? 'chromium';

const { server, url } = await startServer();
let browser;
let code = 1;
try {
  const launched =
    browserArg === 'chromium'
      ? await launchWithWebGPU(url, { headedOnly: process.argv.includes('--headed') })
      : await (async () => {
          const pw = loadPlaywright();
          const firefoxUserPrefs = process.argv.includes('--ignore-blocklist') ? { 'dom.webgpu.enabled': true, 'gfx.webgpu.ignore-blocklist': true } : {};
          const browser = await pw[browserArg].launch({ headless: true, ...(browserArg === 'firefox' ? { firefoxUserPrefs } : {}) });
          const page = await browser.newPage();
          return { browser, page, mode: `${browserArg} headless`, adapter: 'see below', notes: [] };
        })();
  browser = launched.browser;
  const { page, mode, adapter, notes } = launched;
  for (const n of notes) console.log(`note: ${n}`);
  console.log(`${mode.startsWith(browserArg) ? mode : `Chromium ${mode}`}, adapter: ${adapter}\n`);
  page.on('console', (m) => {
    if (m.type() === 'error') console.log(`[page] ${m.text()}`);
  });
  await page.goto(`${url}/tests/gpu/index.html`);
  await page.waitForFunction(() => window.__parity?.done === true, null, { timeout: 180_000 });
  const report = await page.evaluate(() => window.__parity);
  if (report.fatal) {
    console.log(`fatal: ${report.fatal}`);
  } else {
    const w = Math.max(...report.results.map((r) => r.name.length));
    for (const r of report.results) {
      const status = r.pass ? 'pass' : 'FAIL';
      console.log(`${status}  ${r.name.padEnd(w)}  max rel err ${r.error.toExponential(2)}  tol ${r.tol.toExponential(0)}${r.note ? `  — ${r.note}` : ''}`);
    }
    const failed = report.results.filter((r) => !r.pass).length;
    console.log(`\n${report.results.length - failed}/${report.results.length} parity tests passed (${mode}).`);
    code = failed === 0 && report.results.length > 0 ? 0 : 1;
  }
} catch (e) {
  console.error(String(e));
} finally {
  await browser?.close();
  await server.close();
}
process.exit(code);
