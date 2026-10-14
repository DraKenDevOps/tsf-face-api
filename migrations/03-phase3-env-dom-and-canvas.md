# Phase 3: Environment, DOM & `canvas` 3.x Upgrade

## Objective

Modernize the Node.js and Browser environment layer (`src/env/`) and DOM/Canvas utilities (`src/dom/`) in `../ts-face-api` to support:
1. **`canvas@^3.1.0`** (`node-canvas` v3) in Node.js environments.
2. **Native Node.js 18+ `fetch`** without requiring external polyfills.
3. **`@tensorflow/tfjs-core@^4.22.0`** `tf.browser.fromPixels` and `tf.browser.toPixels` strict type contracts.

---

## Step 3.1: `node-canvas` 3.x Constructor & `monkeyPatch` Behavior

### Root Cause Analysis
In `[src/env/index.ts](../src/env/index.ts)` (line 44) and `[src/env/createNodejsEnv.ts](../src/env/createNodejsEnv.ts)` (line 11):
```typescript
environment.createCanvasElement = env.createCanvasElement || (() => new Canvas())
```
In `canvas@2.6.1`, `new Canvas()` with no arguments defaulted to a `0x0` canvas, whereas in `canvas@^3.1.0` (and certain Cairo versions), instantiating `new Canvas(width, height)` without numeric dimensions or drawing to a `0x0` canvas before setting `width`/`height` can throw `InvalidStateError` or dimension errors.

### Fix in `src/env/index.ts` and `src/env/createNodejsEnv.ts`
Pass default `(0, 0)` or fallback safely when constructing a Node `Canvas` instance:

In `[src/env/createNodejsEnv.ts](../src/env/createNodejsEnv.ts)`:
```typescript
const createCanvasElement = function(): HTMLCanvasElement {
  if (Canvas) {
    try {
      return new Canvas(0, 0)
    } catch {
      return new Canvas()
    }
  }
  throw new Error('createCanvasElement - missing Canvas implementation for nodejs environment')
}
```

In `[src/env/index.ts](../src/env/index.ts)` (lines 41-46):
```typescript
const { Canvas = environment.Canvas, Image = environment.Image } = env
environment.Canvas = Canvas
environment.Image = Image
environment.createCanvasElement = env.createCanvasElement || (() => {
  try {
    return new (Canvas as any)(0, 0)
  } catch {
    return new Canvas()
  }
})
environment.createImageElement = env.createImageElement || (() => new Image())
```

---

## Step 3.2: Native Node.js `fetch` & `FileSystem` Support

### 1. Native `globalThis.fetch` in `src/env/createNodejsEnv.ts`
Node.js 18, 20, and 22+ provide built-in `globalThis.fetch` (Undici). Ensure `createNodejsEnv.ts` binds `globalThis.fetch` when available:

```typescript
const fetch = typeof globalThis.fetch === 'function'
  ? globalThis.fetch.bind(globalThis)
  : function() {
      throw new Error('fetch - missing fetch implementation for nodejs environment')
    }
```

### 2. ESM/CJS Safe `fs` Resolution in `src/env/createFileSystem.ts`
In `[src/env/createFileSystem.ts](../src/env/createFileSystem.ts)` (lines 7-13):
```typescript
  if (!fs) {
    try {
      fs = require('fs')
    } catch (err) {
      requireFsError = err.toString()
    }
  }
```
When running in native ESM (such as Vitest or Node `"type": "module"`), `require` may not be defined globally unless created via `module.createRequire` or dynamic import. Update `src/env/createFileSystem.ts` so it works seamlessly in both CJS and ESM Node environments while remaining safe for browser bundlers:

```typescript
import { FileSystem } from './types';

export function createFileSystem(fs?: any): FileSystem {
  let requireFsError = ''

  if (!fs) {
    try {
      if (typeof require === 'function') {
        fs = require('fs')
      }
    } catch (err: any) {
      requireFsError = err ? err.toString() : 'unknown error'
    }
  }

  const readFile = fs
    ? function(filePath: string) {
        return new Promise<Buffer>((res, rej) => {
          fs.readFile(filePath, function(err: any, buffer: Buffer) {
            return err ? rej(err) : res(buffer)
          })
        })
      }
    : async function(filePath: string) {
        try {
          const nodeFs = await import('fs')
          return await nodeFs.promises.readFile(filePath)
        } catch (err: any) {
          throw new Error(`readFile - failed to load fs in nodejs environment: ${requireFsError || err}`)
        }
      }

  return {
    readFile
  }
}
```

### 3. Update `isNodejs()` in `src/env/isNodejs.ts`
In `[src/env/isNodejs.ts](../src/env/isNodejs.ts)`:
```typescript
export function isNodejs(): boolean {
  return typeof global === 'object'
    && typeof require === 'function'
    && typeof module !== 'undefined'
    && typeof process !== 'undefined' && !!process.version
}
```
In Node.js ESM (`"type": "module"`), `typeof require === 'function'` and `typeof module !== 'undefined'` are `false`! This causes `isNodejs()` to return `false` in ESM Node.js environments (including Vitest), breaking `env.initialize()`.

Fix `src/env/isNodejs.ts` to check `process.versions.node`:
```typescript
export function isNodejs(): boolean {
  return typeof process !== 'undefined'
    && !!process.versions
    && !!process.versions.node
}
```
*(Note: `src/env/index.ts` checks `if (isBrowser())` before `if (isNodejs())`, so Electron renderer processes continue to resolve as Browser first).*

---

## Step 3.3: `tf.browser.fromPixels` and `tf.browser.toPixels` Compatibility

### 1. `src/dom/NetInput.ts`
In `[src/dom/NetInput.ts](../src/dom/NetInput.ts)` (line 142):
```typescript
if (input instanceof env.getEnv().Canvas) {
  return tf.browser.fromPixels(imageToSquare(input, inputSize, isCenterInputs))
}
```
In Node.js with `canvas@^3.1.0`, `tf.browser.fromPixels` checks whether the input is `HTMLCanvasElement`, `ImageData`, or an object with `{ width, height, data }`. However, in Node.js (where `HTMLCanvasElement` is not a native DOM class unless polyfilled on `globalThis`), `tf.browser.fromPixels(canvas)` in TFJS 4.x checks `input instanceof HTMLCanvasElement`.
If `globalThis.HTMLCanvasElement` is not set to `canvas.Canvas`, `tf.browser.fromPixels(nodeCanvas)` in TFJS 4.x may fail with:
`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData...`

### Fix in `src/env/index.ts` (`monkeyPatch`)
When `env.monkeyPatch({ Canvas, Image, ImageData })` is called, also ensure `globalThis.HTMLCanvasElement`, `globalThis.HTMLImageElement`, and `globalThis.ImageData` are populated if missing (so `tf.browser.fromPixels` in `@tensorflow/tfjs-core@4.x` recognizes `node-canvas` instances via `instanceof`):

```typescript
  const { Canvas = environment.Canvas, Image = environment.Image } = env
  environment.Canvas = Canvas
  environment.Image = Image
  environment.createCanvasElement = env.createCanvasElement || (() => new (Canvas as any)(0, 0))
  environment.createImageElement = env.createImageElement || (() => new Image())

  environment.ImageData = env.ImageData || environment.ImageData
  environment.Video = env.Video || environment.Video
  environment.fetch = env.fetch || environment.fetch
  environment.readFile = env.readFile || environment.readFile

  if (typeof globalThis !== 'undefined') {
    const g = globalThis as Record<string, any>
    if (env.Canvas && !g['HTMLCanvasElement']) g['HTMLCanvasElement'] = env.Canvas
    if (env.Image && !g['HTMLImageElement']) g['HTMLImageElement'] = env.Image
    if (env.ImageData && !g['ImageData']) g['ImageData'] = env.ImageData
  }
```

### 2. `src/dom/imageTensorToCanvas.ts`
In `[src/dom/imageTensorToCanvas.ts](../src/dom/imageTensorToCanvas.ts)` (lines 14-15):
```typescript
const imgTensor3D = tf.tidy(() => imgTensor.as3D(height, width, numChannels).toInt())
await tf.browser.toPixels(imgTensor3D, targetCanvas)
```
In `@tensorflow/tfjs-core@^4.22.0`, `tf.browser.toPixels` signature is:
`toPixels(img: Tensor2D | Tensor3D | TensorLike, canvas?: HTMLCanvasElement): Promise<Uint8ClampedArray>`
Verify `imgTensor3D` is typed as `tf.Tensor3D` (`as tf.Tensor3D` after `.toInt()`) so TypeScript 5.8 overload resolution succeeds.

---

## Phase 3 Verification Checklist

- [ ] `isNodejs()` in `src/env/isNodejs.ts` detects Node ESM (`process.versions.node`) without relying on CommonJS `require`.
- [ ] `createFileSystem()` works in both CommonJS and ESM Node runtimes.
- [ ] `env.monkeyPatch({ Canvas, Image, ImageData })` registers globals needed by `tf.browser.fromPixels` in TFJS 4.x.
- [ ] `createNodejsEnv()` uses native `globalThis.fetch` automatically on Node 18+.
