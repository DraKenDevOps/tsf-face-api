# Phase 1: Project Scaffold, `pnpm` Setup & TypeScript 5.8 Configuration

## Objective

Create the standalone `ts-face-api` project in `../ts-face-api` (leaving `face-api.js` intact), initialize it with `pnpm`, install the latest stable versions of `@tensorflow/*`, `canvas`, and `typescript`, and update the TypeScript configuration and source annotations for TypeScript 5.8 compatibility.

---

## Step 1.1: Create Sibling Directory & Copy Source Assets

From `C:\Users\Anakin\Other\face-api.js`, create the sibling directory `C:\Users\Anakin\Other\ts-face-api` and copy the source, test, and model weight directories without modifying the original repo:

```powershell
# PowerShell commands run from C:\Users\Anakin\Other\face-api.js
New-Item -ItemType Directory -Force -Path ..\ts-face-api

Copy-Item -Recurse .\src ..\ts-face-api\src
Copy-Item -Recurse .\test ..\ts-face-api\test
Copy-Item -Recurse .\weights ..\ts-face-api\weights
Copy-Item -Recurse .\weights_uncompressed ..\ts-face-api\weights_uncompressed
Copy-Item -Recurse .\weights_unused ..\ts-face-api\weights_unused
Copy-Item .\.gitignore ..\ts-face-api\.gitignore
Copy-Item .\LICENSE ..\ts-face-api\LICENSE -ErrorAction SilentlyContinue
```

---

## Step 1.2: Initialize `pnpm` & Configure `package.json`

Create `../ts-face-api/.npmrc` to ensure native addons (`@tensorflow/tfjs-node`, `canvas`) can run their build/install scripts under `pnpm`:

```ini
enable-pre-post-scripts=true
strict-peer-dependencies=false
```

Create `../ts-face-api/package.json` with `name: "ts-face-api"`, modern `exports` map, and latest stable dependencies:

```json
{
  "name": "ts-face-api",
  "version": "1.0.0",
  "description": "TypeScript API for face detection and face recognition in browser and Node.js with TensorFlow.js 4.x",
  "type": "module",
  "main": "./build/commonjs/index.js",
  "module": "./build/es6/index.js",
  "types": "./build/commonjs/index.d.ts",
  "exports": {
    ".": {
      "types": "./build/es6/index.d.ts",
      "import": "./build/es6/index.js",
      "require": {
        "types": "./build/commonjs/index.d.ts",
        "default": "./build/commonjs/index.js"
      }
    }
  },
  "files": [
    "build",
    "dist",
    "src",
    "weights"
  ],
  "scripts": {
    "clean": "node -e \"fs.rmSync('./build', { recursive: true, force: true }); fs.rmSync('./dist', { recursive: true, force: true });\"",
    "typecheck": "tsc --noEmit",
    "rollup": "rollup -c rollup.config.mjs",
    "rollup:min": "rollup -c rollup.config.mjs --environment minify:true",
    "tsc:cjs": "tsc -p tsconfig.json",
    "tsc:esm": "tsc -p tsconfig.es6.json",
    "build": "pnpm run clean && pnpm run rollup && pnpm run rollup:min && pnpm run tsc:cjs && pnpm run tsc:esm",
    "test": "pnpm run test:node && pnpm run test:browser",
    "test:node": "vitest run --project node",
    "test:browser": "vitest run --project browser",
    "docs": "typedoc --options ./typedoc.config.mjs ./src/index.ts"
  },
  "keywords": [
    "face",
    "detection",
    "recognition",
    "tensorflow",
    "tfjs",
    "typescript",
    "ts-face-api"
  ],
  "license": "MIT",
  "dependencies": {
    "@tensorflow/tfjs-backend-cpu": "^4.22.0",
    "@tensorflow/tfjs-backend-webgl": "^4.22.0",
    "@tensorflow/tfjs-core": "^4.22.0",
    "tslib": "^2.8.1"
  },
  "devDependencies": {
    "@rollup/plugin-commonjs": "^28.0.3",
    "@rollup/plugin-node-resolve": "^16.0.1",
    "@rollup/plugin-terser": "^0.4.4",
    "@rollup/plugin-typescript": "^12.1.2",
    "@tensorflow/tfjs-node": "^4.22.0",
    "@types/node": "^22.13.0",
    "@vitest/browser": "^3.0.0",
    "canvas": "^3.1.0",
    "playwright": "^1.50.0",
    "rollup": "^4.34.0",
    "typedoc": "^0.27.0",
    "typescript": "^5.8.2",
    "vitest": "^3.0.0"
  },
  "peerDependencies": {
    "@tensorflow/tfjs-node": ">=4.0.0",
    "canvas": ">=2.11.0 || >=3.0.0"
  },
  "peerDependenciesMeta": {
    "@tensorflow/tfjs-node": {
      "optional": true
    },
    "canvas": {
      "optional": true
    }
  },
  "pnpm": {
    "onlyBuiltDependencies": [
      "@tensorflow/tfjs-node",
      "canvas"
    ]
  }
}
```

### Install Dependencies with `pnpm`

```powershell
cd ..\ts-face-api
pnpm install
```

> **Note on `canvas` & `@tensorflow/tfjs-node` native builds on Windows**:
> If `canvas@^3.1.0` or `@tensorflow/tfjs-node@^4.22.0` prompts for build approval under `pnpm` v10+, `pnpm.onlyBuiltDependencies` in `package.json` pre-approves both packages. If an older Node ABI environment lacks prebuilt binaries for `canvas@3.x`, `pnpm add -D canvas@^2.11.2` can be used as a drop-in fallback.

---

## Step 1.3: Upgrade `tsconfig.json` and `tsconfig.es6.json` for TypeScript 5.8

In `[tsconfig.json](../tsconfig.json)`, the legacy config uses `"target": "es5"`, `"suppressImplicitAnyIndexErrors": true`, and `"formatCodeOptions"`, which cause fatal errors in TypeScript 5.0+:
- `error TS5102: Option 'suppressImplicitAnyIndexErrors' has been removed.`
- `error TS5023: Unknown compiler option 'formatCodeOptions'.`

### Target `../ts-face-api/tsconfig.json` (CommonJS Output)

```json
{
  "compilerOptions": {
    "removeComments": false,
    "preserveConstEnums": true,
    "emitDecoratorMetadata": true,
    "experimentalDecorators": true,
    "sourceMap": true,
    "declaration": true,
    "noImplicitAny": true,
    "noImplicitReturns": true,
    "noImplicitThis": true,
    "noFallthroughCasesInSwitch": true,
    "strictNullChecks": true,
    "useUnknownInCatchVariables": false,
    "importHelpers": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "outDir": "build/commonjs",
    "module": "commonjs",
    "target": "ES2020",
    "moduleResolution": "node",
    "lib": ["ES2020", "DOM"]
  },
  "exclude": [
    "node_modules",
    "dist",
    "build",
    "test"
  ],
  "include": [
    "src"
  ]
}
```

### Target `../ts-face-api/tsconfig.es6.json` (ES2020 Module Output)

Replace `[tsconfig.es6.json](../tsconfig.es6.json)` so it emits modern ES2020 modules with declarations:

```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "outDir": "build/es6",
    "module": "ES2020",
    "target": "ES2020",
    "declaration": true
  },
  "include": [
    "src"
  ]
}
```

---

## Step 1.4: Fix Source Locations Reliant on `suppressImplicitAnyIndexErrors`

Because `suppressImplicitAnyIndexErrors` was removed in TypeScript 5.0, any bracket indexing on objects without an explicit index signature (`Record<string, any>`) will fail with `TS7053: Element implicitly has an 'any' type`.

### 1. `src/env/createNodejsEnv.ts`
In `[src/env/createNodejsEnv.ts](../src/env/createNodejsEnv.ts)` (lines 6-35), `global['Canvas']`, `global['Image']`, etc. index `typeof globalThis` directly.
Update to use `(globalThis as Record<string, any>)`:

```typescript
import { createFileSystem } from './createFileSystem';
import { Environment } from './types';

export function createNodejsEnv(): Environment {
  const g = globalThis as Record<string, any>
  const Canvas = g['Canvas'] || g['HTMLCanvasElement']
  const Image = g['Image'] || g['HTMLImageElement']

  const createCanvasElement = function() {
    if (Canvas) {
      return new Canvas()
    }
    throw new Error('createCanvasElement - missing Canvas implementation for nodejs environment')
  }

  const createImageElement = function() {
    if (Image) {
      return new Image()
    }
    throw new Error('createImageElement - missing Image implementation for nodejs environment')
  }

  const fetch = g['fetch'] || function() {
    throw new Error('fetch - missing fetch implementation for nodejs environment')
  }

  const fileSystem = createFileSystem()

  return {
    Canvas: Canvas || class {},
    CanvasRenderingContext2D: g['CanvasRenderingContext2D'] || class {},
    Image: Image || class {},
    ImageData: g['ImageData'] || class {},
    Video: g['HTMLVideoElement'] || class {},
    createCanvasElement,
    createImageElement,
    fetch,
    ...fileSystem
  }
}
```

### 2. `src/env/createBrowserEnv.ts`
In `[src/env/createBrowserEnv.ts](../src/env/createBrowserEnv.ts)` (line 5), `window['fetch']` triggers TS2774/TS7053 or strict callable checks when `window.fetch` is always defined on `Window`.
Update line 5:

```typescript
const fetch = window.fetch ? window.fetch.bind(window) : function() {
  throw new Error('fetch - missing fetch implementation for browser environment')
}
```

### 3. `src/xception/types.ts` and `src/xception/TinyXception.ts`
In `[src/xception/types.ts](../src/xception/types.ts)` (line 21), `middle_flow` is typed as `any`, or when indexed in `src/xception/extractParams.ts` / `src/xception/extractParamsFromWeigthMap.ts` via `middle_flow[`main_block_${idx}`]`, tighten `TinyXceptionParams`:

```typescript
export type TinyXceptionParams = {
  entry_flow: {
    conv_in: ConvParams
    reduction_block_0: ReductionBlockParams
    reduction_block_1: ReductionBlockParams
  }
  middle_flow: Record<string, MainBlockParams>
  exit_flow: {
    reduction_block: ReductionBlockParams
    separable_conv: SeparableConvParams
  }
}
```

### 4. `src/faceExpressionNet/FaceExpressions.ts`
Check bracket indexing on `FaceExpressions` instance in `src/faceExpressionNet/FaceExpressions.ts` where `this[expression]` is assigned dynamically. Cast `(this as Record<string, any>)[expression]` or add an index signature if flagged by TS7053.

---

## Phase 1 Verification Checklist

- [ ] `../ts-face-api` directory exists alongside `./face-api.js`.
- [ ] `git status` in `./face-api.js` shows zero changes to `src/`, `test/`, or `package.json`.
- [ ] `pnpm install` completes in `../ts-face-api` and generates `pnpm-lock.yaml`.
- [ ] `tsconfig.json` and `tsconfig.es6.json` contain no deprecated TS 5.0+ options.
