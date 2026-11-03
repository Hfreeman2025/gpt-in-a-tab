// Dev helper: open tests/gpu/bench.html in Chromium with WebGPU and print its report.
import { launchWithWebGPU, startServer } from './browser.mjs';

const query = process.argv[2] ?? '';
const { server, url } = await startServer();
const { browser, page, mode, adapter } = await launchWithWebGPU(url);
console.log(`Chromium ${mode}, adapter: ${adapter}`);
await page.goto(`${url}/tests/gpu/bench.html${query}`);
await page.waitForFunction(() => window.__bench?.done === true, null, { timeout: 120_000 });
console.log(await page.locator('#out').textContent());
await browser.close();
await server.close();
