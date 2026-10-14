# Phase 5: Rollup 4 Build Pipeline, Vitest Test Suite & Final Validation

## Objective

Replace the legacy Rollup 2 + Uglify bundler (`[rollup.config.js](../rollup.config.js)`) and Karma + Jasmine test runner (`[karma.conf.js](../karma.conf.js)`, `[jasmine-node.js](../jasmine-node.js)`) in `../ts-face-api` with **Rollup 4** and **Vitest** (Node + Playwright Browser mode), and validate all build artifacts and tests.

---

## Step 5.1: Migrate `rollup.config.js` to `rollup.config.mjs` (Rollup 4)

### Why the Legacy Rollup Config Breaks
In `[rollup.config.js](../rollup.config.js)`:
- `rollup-plugin-uglify` only parses ES5 syntax and crashes on ES2020 output emitted by TypeScript 5.8 and `@tensorflow/tfjs-core@^4.22.0`.
- `rollup-plugin-commonjs`, `rollup-plugin-node-resolve`, and `rollup-plugin-typescript2` are superseded by official `@rollup/plugin-*` packages.

### Create `../ts-face-api/rollup.config.mjs`

```javascript
import commonjs from '@rollup/plugin-commonjs';
import resolve from '@rollup/plugin-node-resolve';
import terser from '@rollup/plugin-terser';
import typescript from '@rollup/plugin-typescript';

const { minify } = process.env;

export default {
  input: 'src/index.ts',
  plugins: [
    typescript({
      tsconfig: './tsconfig.json',
      compilerOptions: {
        module: 'ESNext',
        target: 'ES2020',
        declaration: false,
        declarationMap: false,
        outDir: 'dist'
      }
    }),
    resolve({
      browser: true,
      preferBuiltins: false
    }),
    commonjs({
      include: 'node_modules/**'
    }),
    ...(minify ? [terser()] : [])
  ],
  output: {
    extend: true,
    file: `dist/ts-face-api${minify ? '.min' : ''}.js`,
    format: 'umd',
    name: 'faceapi',
    globals: {
      crypto: 'crypto',
      fs: 'fs',
      path: 'path'
    },
    sourcemap: !minify
  },
  external: ['crypto', 'fs', 'path'],
  onwarn: (warning) => {
    const ignoreWarnings = [
      'CIRCULAR_DEPENDENCY',
      'CIRCULAR',
      'THIS_IS_UNDEFINED'
    ];
    if (ignoreWarnings.includes(warning.code)) {
      return;
    }
    if (warning.missing === 'alea') {
      return;
    }
    console.warn(warning.message);
  }
};
```

---

## Step 5.2: Update WebGL Backend Test Helpers in `test/utils.ts`

### Root Cause
In `[test/utils.ts](../test/utils.ts)` (lines 184-218):
```typescript
const gpgpu = tf.backend()['gpgpu']
...
  if (!(gpgpu instanceof tf.webgl.GPGPUContext)) {
    describe(description, specDefinitions)
    return
  }

  const defaultBackendName = tf.getBackend()
  const newBackendName = 'testBackend'
  const backend = new tf.webgl.MathBackendWebGL(gpgpu)
```
In TensorFlow.js 4.x:
1. `tf.webgl` is no longer a namespace on `@tensorflow/tfjs-core`; `GPGPUContext` and `MathBackendWebGL` live in `@tensorflow/tfjs-backend-webgl`.
2. `tf.backend()` may be initialized asynchronously on first tensor operation (`await tf.ready()`).

### Fix in `../ts-face-api/test/utils.ts`
Import `GPGPUContext` and `MathBackendWebGL` from `@tensorflow/tfjs-backend-webgl` and guard `tf.backend()` safely:

```typescript
import * as tf from '@tensorflow/tfjs-core';
import { GPGPUContext, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
```

And update lines 184-218 in `test/utils.ts`:
```typescript
const currentBackend = tf.backend() as any
const gpgpu = currentBackend ? currentBackend['gpgpu'] : undefined

if (gpgpu) {
  console.log('running tests on WebGL backend')
} else {
  console.log('running tests on CPU/Node backend')
}

export function describeWithBackend(description: string, specDefinitions: () => void) {
  if (!(gpgpu instanceof GPGPUContext)) {
    describe(description, specDefinitions)
    return
  }

  const defaultBackendName = tf.getBackend()
  const newBackendName = 'testBackend'
  const backend = new MathBackendWebGL(gpgpu)

  describe(description, () => {
    beforeAll(async () => {
      tf.registerBackend(newBackendName, () => backend)
      await tf.setBackend(newBackendName)
    })

    afterAll(async () => {
      await tf.setBackend(defaultBackendName)
      tf.removeBackend(newBackendName)
      backend.dispose()
    })

    specDefinitions()
  })
}
```

---

## Step 5.3: Migrate Jasmine & Karma to Vitest

### 1. Update `test/env.node.ts`
In `[test/env.node.ts](../test/env.node.ts)`, replace CommonJS `require` and `__dirname` with ESM-compatible imports for Node.js + Vitest:

```typescript
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import '@tensorflow/tfjs-node';
import * as canvas from 'canvas';

import { env, NeuralNetwork } from '../src';
import { TestEnv } from './Environment';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const { Canvas, Image, ImageData } = canvas as any;
env.monkeyPatch({ Canvas, Image, ImageData });

async function loadImageNode(uri: string): Promise<HTMLImageElement> {
  return (canvas as any).loadImage(path.resolve(__dirname, '../', uri));
}

async function loadJsonNode<T>(uri: string): Promise<T> {
  return JSON.parse(fs.readFileSync(path.resolve(__dirname, '../', uri)).toString());
}

export async function initNetNode<TNet extends NeuralNetwork<any>>(
  net: TNet,
  uncompressedFilename?: string | boolean,
  isUnusedModel = false
) {
  if (uncompressedFilename) {
    const filePath = path.resolve(
      __dirname,
      '../weights_uncompressed',
      typeof uncompressedFilename === 'string' ? uncompressedFilename : ''
    );
    const buf = fs.readFileSync(filePath);
    const weights = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    net.extractWeights(weights);
    return;
  }
  const weightsDir = isUnusedModel ? '../weights_unused' : '../weights';
  await net.loadFromDisk(path.resolve(__dirname, weightsDir));
}

const nodeTestEnv: TestEnv = {
  loadImage: loadImageNode,
  loadJson: loadJsonNode,
  initNet: initNetNode
};

(globalThis as Record<string, any>)['nodeTestEnv'] = nodeTestEnv;
```

### 2. Update `test/env.ts`
In `[test/env.ts](../test/env.ts)`, remove the Jasmine-specific `jasmine.DEFAULT_TIMEOUT_INTERVAL = 60000` (timeouts are configured in `vitest.config.ts`) and replace `global['nodeTestEnv']` with `(globalThis as Record<string, any>)['nodeTestEnv']`:

```typescript
import * as tf from '@tensorflow/tfjs-core';

import { fetchImage, fetchJson, fetchNetWeights, NeuralNetwork } from '../src';
import { TestEnv } from './Environment';

async function loadImageBrowser(uri: string): Promise<HTMLImageElement> {
  return fetchImage(`/${uri.startsWith('/') ? uri.slice(1) : uri}`);
}

async function loadJsonBrowser<T>(uri: string): Promise<T> {
  return fetchJson<T>(`/${uri.startsWith('/') ? uri.slice(1) : uri}`);
}

async function initNetBrowser<TNet extends NeuralNetwork<any>>(
  net: TNet,
  uncompressedFilename?: string | boolean,
  isUnusedModel = false
) {
  const url = uncompressedFilename
    ? await fetchNetWeights(`/weights_uncompressed/${uncompressedFilename}`)
    : (isUnusedModel ? '/weights_unused' : '/weights');
  await net.load(url);
}

const browserTestEnv: TestEnv = {
  loadImage: loadImageBrowser,
  loadJson: loadJsonBrowser,
  initNet: initNetBrowser
};

export function getTestEnv(): TestEnv {
  return (globalThis as Record<string, any>)['nodeTestEnv'] || browserTestEnv;
}
```

### 3. Jasmine Global Compatibility in Tests
The existing test files in `test/tests/**/*.test.ts` use standard BDD globals (`describe`, `it`, `expect`, `beforeAll`, `afterAll`, `beforeEach`, `afterEach`) which Vitest provides natively when `globals: true` is enabled.
Two minor Jasmine APIs used in a few tests (`jasmine.createSpy` or `fail()`) can be aliased in a tiny Vitest setup shim (`test/vitest.setup.ts`):

```typescript
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
```

### 4. Create `../ts-face-api/vitest.config.ts`

```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    testTimeout: 60000,
    hookTimeout: 60000,
    workspace: [
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
            name: 'chromium',
            headless: true
          }
        }
      }
    ]
  }
});
```

### 5. Remove Legacy Test Configs from `../ts-face-api`
Delete `../ts-face-api/karma.conf.js` and `../ts-face-api/jasmine-node.js` once `vitest.config.ts` is in place.

---

## Step 5.4: End-to-End Execution & Validation Commands

Run the following verification sequence inside `C:\Users\Anakin\Other\ts-face-api`:

```powershell
# 1. Verify dependencies are locked with pnpm
pnpm install --frozen-lockfile

# 2. Run strict TypeScript 5.8 typechecking across src/
pnpm run typecheck

# 3. Build UMD (dist/ts-face-api.js, dist/ts-face-api.min.js), CommonJS (build/commonjs), and ESM (build/es6)
pnpm run build

# 4. Run Node.js test suite (@tensorflow/tfjs-node + canvas)
pnpm run test:node

# 5. Install Playwright Chromium (first time only) and run Browser test suite (@tensorflow/tfjs-backend-webgl)
pnpm exec playwright install chromium
pnpm run test:browser

# 6. Verify original face-api.js project is completely untouched
git -C ..\face-api.js status
```

---

## Phase 5 Verification Checklist

- [ ] `pnpm run build` generates:
  - `dist/ts-face-api.js` and `dist/ts-face-api.min.js`
  - `build/commonjs/index.js` and `build/commonjs/index.d.ts`
  - `build/es6/index.js` and `build/es6/index.d.ts`
- [ ] `pnpm run test:node` passes all unit and model inference tests in Node.js without tensor memory leaks (`expectAllTensorsReleased`).
- [ ] `pnpm run test:browser` passes all DOM, fetch, and WebGL inference tests in headless Chromium.
- [ ] `C:\Users\Anakin\Other\face-api.js` remains untouched except for the newly added `migrations/` directory.
