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
          setupFiles: ['./test/vitest.setup.ts', './test/env.node.ts'],
          include: ['test/tests/**/*.test.ts'],
          exclude: [
            'test/tests/**/*.browser.test.ts',
            'test/tests-legacy/**/*.ts'
          ]
        }
      },
      {
        extends: true,
        test: {
          name: 'browser',
          setupFiles: ['./test/vitest.setup.ts'],
          include: ['test/tests/**/*.test.ts'],
          exclude: [
            'test/tests/**/*.node.test.ts',
            'test/tests-legacy/**/*.ts'
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
