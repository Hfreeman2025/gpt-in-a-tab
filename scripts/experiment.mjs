// Dev helper: open tests/gpu/experiment.html in Chromium with WebGPU and print its report.
import { launchWithWebGPU, startServer } from './browser.mjs';

const query = process.argv[2] ?? '';
const { server, url } = await startServer();
const { browser, page, mode, adapter } = await launchWithWebGPU(url);
console.log(`Chromium ${mode}, adapter: ${adapter}`);
await page.goto(`${url}/tests/gpu/experiment.html${query}`);
await page.waitForFunction(() => window.__exp?.done === true, null, { timeout: 400_000 });
console.log(await page.locator('#out').textContent());
await browser.close();
await server.close();
