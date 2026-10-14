import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    testTimeout: 60000,
    hookTimeout: 60000,
    projects: [
      {
        extends: true,
        test: {
          name: 'node',
          environment: 'node',
          setupFiles: ['./tests/vitest.setup.ts', './tests/env.node.ts'],
          include: ['tests/**/*.test.ts'],
          exclude: [
            'tests/**/*.browser.test.ts',
            'tests-legacy/**/*.ts'
          ]
        }
      },
      {
        extends: true,
        test: {
          name: 'browser',
          setupFiles: ['./tests/vitest.setup.ts'],
          include: ['tests/**/*.test.ts'],
          exclude: [
            'tests/**/*.node.test.ts',
            'tests-legacy/**/*.ts'
          ],
          browser: {
            enabled: true,
            provider: 'playwright',
            instances: [
              { browser: 'chromium', headless: true }
            ]
          }
        }
      }
    ]
  }
});
