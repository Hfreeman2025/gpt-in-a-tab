import { defineConfig } from 'vitest/config';

export default defineConfig({ test: { include: ['tests/experiments/**/*.exp.ts'], environment: 'node', testTimeout: 3_600_000 } });
