import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: './',
  // The largest chunk is the five tragedies as text (744 kB), fetched only when that sample is
  // picked; the code itself is about 110 kB. The limit is set above the text, not the code.
  build: { target: 'es2022', chunkSizeWarningLimit: 800 },
  worker: { format: 'es' },
  test: { include: ['tests/**/*.test.ts'], environment: 'node', testTimeout: 30_000 },
});
