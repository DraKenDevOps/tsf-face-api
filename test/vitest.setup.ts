import { vi } from 'vitest';

const g = globalThis as Record<string, any>;
if (!g['jasmine']) {
  g['jasmine'] = {
    DEFAULT_TIMEOUT_INTERVAL: 60000,
    createSpy: () => vi.fn()
  };
}
if (!g['fail']) {
  g['fail'] = (msg?: string) => {
    throw new Error(msg || 'Test failed via fail()');
  };
}
