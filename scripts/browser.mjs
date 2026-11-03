// Shared helpers for the browser-driven scripts (GPU parity, measurement, screenshots).
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/**
 * Playwright is not a dependency of this project. It is resolved from this project if you have
 * installed it, or from the directory in PLAYWRIGHT_FROM (any project with playwright installed).
 */
export function loadPlaywright() {
  const dirs = [root, process.env.PLAYWRIGHT_FROM].filter(Boolean);
  for (const dir of dirs) {
    try {
      return createRequire(path.join(dir, 'package.json'))('playwright');
    } catch {
      // try the next location
    }
  }
  throw new Error('Playwright not found. Run `npm i -D playwright && npx playwright install chromium`, or set PLAYWRIGHT_FROM to a project that has it.');
}

/**
 * Serves the page: the Vite dev server by default, or the production build (built first) when
 * `--dist` is passed on the command line.
 */
export async function startServer({ dist = process.argv.includes('--dist') } = {}) {
  const vite = await import('vite');
  if (dist) {
    await vite.build({ root, logLevel: 'error' });
    const server = await vite.preview({ root, logLevel: 'error', preview: { port: 0, host: '127.0.0.1' } });
    const address = server.httpServer.address();
    return { server: { close: () => server.close() }, url: `http://127.0.0.1:${address.port}` };
  }
  const server = await vite.createServer({ root, logLevel: 'error', server: { port: 0, host: '127.0.0.1' } });
  await server.listen();
  const address = server.httpServer.address();
  return { server, url: `http://127.0.0.1:${address.port}` };
}

// --mute-audio: the page plays no sound, but no automated browser on this shared machine may.
const WEBGPU_ARGS = ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=metal', '--mute-audio'];

async function adapterInfo(page) {
  return page.evaluate(async () => {
    if (!navigator.gpu) return { ok: false, why: 'navigator.gpu missing' };
    const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!a) return { ok: false, why: 'no adapter' };
    return { ok: true, why: [a.info.vendor, a.info.architecture, a.info.description].filter(Boolean).join(' · '), fallback: a.info.isFallbackAdapter ?? false };
  });
}

/**
 * Launch Chromium with a real GPU. Tries headless first (new headless mode), and falls back to a
 * headed window when headless cannot reach the GPU. Returns which mode worked.
 */
export async function launchWithWebGPU(url, { headedOnly = false, viewport = { width: 1440, height: 900 }, deviceScaleFactor = 2, colorScheme = 'light', reducedMotion = 'no-preference', hasTouch = false } = {}) {
  const { chromium } = loadPlaywright();
  const attempts = headedOnly ? [{ headless: false }] : [{ headless: true, channel: 'chromium' }, { headless: false }];
  const notes = [];
  for (const opts of attempts) {
    const browser = await chromium.launch({ ...opts, args: WEBGPU_ARGS });
    const context = await browser.newContext({ viewport, deviceScaleFactor, colorScheme, reducedMotion, hasTouch });
    const page = await context.newPage();
    await page.goto(url + '/tests/gpu/blank.html').catch(() => page.goto(url));
    const info = await adapterInfo(page);
    const mode = opts.headless ? 'headless' : 'headed';
    if (info.ok && !info.fallback) return { browser, context, page, mode, adapter: info.why, notes };
    notes.push(`${mode}: ${info.why}${info.fallback ? ' (software fallback adapter)' : ''}`);
    await browser.close();
  }
  throw new Error(`No hardware WebGPU adapter in Chromium. ${notes.join('; ')}`);
}

export async function launchPlain(url, { viewport = { width: 1440, height: 900 }, deviceScaleFactor = 2, colorScheme = 'light', reducedMotion = 'no-preference', isMobile = false, hasTouch = false } = {}) {
  const { chromium } = loadPlaywright();
  const browser = await chromium.launch({ headless: true, channel: 'chromium', args: WEBGPU_ARGS });
  const context = await browser.newContext({ viewport, deviceScaleFactor, colorScheme, reducedMotion, isMobile, hasTouch });
  const page = await context.newPage();
  void url;
  return { browser, context, page };
}
