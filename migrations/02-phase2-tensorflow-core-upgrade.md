# Phase 2: `@tensorflow/*` 4.x Core & Operations Upgrade

## Objective

Migrate all TensorFlow.js usages across `../ts-face-api/src/` from `@tensorflow/tfjs-core@1.7.0` to `@tensorflow/tfjs-core@^4.22.0` (plus modular backends `@tensorflow/tfjs-backend-cpu@^4.22.0`, `@tensorflow/tfjs-backend-webgl@^4.22.0`, and `@tensorflow/tfjs-node@^4.22.0`).

---

## Step 2.1: Modular Backend Registration in `src/index.ts`

### Background
In `@tensorflow/tfjs-core@1.7.0`, the CPU and WebGL backends were bundled directly inside `@tensorflow/tfjs-core`. Starting in TensorFlow.js 2.0+ through 4.x, backends were split into separate packages (`@tensorflow/tfjs-backend-cpu` and `@tensorflow/tfjs-backend-webgl`) and must be imported to register their kernels.

### Change in `src/index.ts`
Update `[src/index.ts](../src/index.ts)` to import the CPU and WebGL backends so browser consumers and default environments have registered kernels out of the box:

```typescript
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import * as draw from './draw';
import * as utils from './utils';

export {
  draw,
  utils,
  tf
}

export * from './ageGenderNet/index';
export * from './classes/index';
export * from './dom/index';
export * from './env/index';
export * from './faceExpressionNet/index';
export * from './faceLandmarkNet/index';
export * from './faceRecognitionNet/index';
export * from './factories/index';
export * from './globalApi/index';
export * from './mtcnn/index';
export * from './ops/index';
export * from './ssdMobilenetv1/index';
export * from './tinyFaceDetector/index';
export * from './tinyYolov2/index';

export * from './euclideanDistance';
export * from './NeuralNetwork';
export * from './resizeResults';
```

---

## Step 2.2: Fix Pooled `Buffer.buffer` Slicing in `src/NeuralNetwork.ts`

### Critical Root Cause
In `[src/NeuralNetwork.ts](../src/NeuralNetwork.ts)` (lines 103-107):
```typescript
const fetchWeightsFromDisk = (filePaths: string[]) => Promise.all(
  filePaths.map(filePath => readFile(filePath).then(buf => buf.buffer))
)
```
In modern Node.js (`@types/node@^22.x`), small `Buffer` instances created by `fs.readFile` may share a pooled `ArrayBuffer` (`Buffer.poolSize = 8192`) with a non-zero `byteOffset`, and `buf.buffer` is typed as `ArrayBufferLike` (`ArrayBuffer | SharedArrayBuffer`). Passing `buf.buffer` directly without slicing reads from byte 0 of the pool instead of `buf.byteOffset`, corrupting model weights in `tf.io.weightsLoaderFactory` and failing TypeScript 5.8's `Promise<ArrayBuffer[]>` return check.

### Fix in `src/NeuralNetwork.ts` (lines 103-108)
Slice the exact byte range into a standalone `ArrayBuffer`:

```typescript
const fetchWeightsFromDisk = (filePaths: string[]): Promise<ArrayBuffer[]> => Promise.all(
  filePaths.map(filePath =>
    readFile(filePath).then(buf =>
      buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
    )
  )
)
const loadWeights = tf.io.weightsLoaderFactory(fetchWeightsFromDisk)
```

---

## Step 2.3: Update Tensor Creation & `tf.Variable` Freezing in `src/NeuralNetwork.ts`

In `[src/NeuralNetwork.ts](../src/NeuralNetwork.ts)` (lines 51-57):
```typescript
public freeze() {
  this.getTrainableParams().forEach(({ path, tensor: variable }) => {
    const tensor = tf.tensor(variable.dataSync())
    variable.dispose()
    this.reassignParamFromPath(path, tensor)
  })
}
```
In TFJS 4.x, `tf.tensor(variable.dataSync())` flattens the tensor to 1D unless `variable.shape` and `variable.dtype` are preserved:

```typescript
public freeze() {
  this.getTrainableParams().forEach(({ path, tensor: variable }) => {
    const tensor = tf.tensor(variable.dataSync(), variable.shape, variable.dtype)
    variable.dispose()
    this.reassignParamFromPath(path, tensor)
  })
}
```

---

## Step 2.4: Update Tensor Op Signatures & Return Types Across `src/`

In `@tensorflow/tfjs-core@^4.22.0`, several tensor methods tightened their TypeScript overloads and return types (`arraySync()`, `array()`, `tf.avgPool`, `tf.maxPool`, `tf.pad`, `tf.fill`).

### 1. `arraySync()` and `array()` Typing
In TFJS 1.7, `tensor.arraySync()` returned `any`. In TFJS 4.x:
- `Tensor2D.arraySync()` returns `number[][]`
- `Tensor4D.array()` returns `Promise<number[][][][]>`
- Generic `Tensor.arraySync()` returns `number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][]`

Verify and annotate in:
- `[src/ssdMobilenetv1/nonMaxSuppression.ts](../src/ssdMobilenetv1/nonMaxSuppression.ts)` (line 52):
  ```typescript
  const boxesData = boxes.arraySync() as number[][]
  ```
- `[src/ssdMobilenetv1/SsdMobilenetv1.ts](../src/ssdMobilenetv1/SsdMobilenetv1.ts)` (line 90):
  ```typescript
  const boxesData = boxes.arraySync() as number[][]
  ```
- `[src/tinyYolov2/TinyYolov2Base.ts](../src/tinyYolov2/TinyYolov2Base.ts)` (lines 204-205 and 246):
  ```typescript
  const scoresData = (await scoresTensor.array()) as number[][][][]
  const boxesData = (await boxesTensor.array()) as number[][][][]
  ```
  and in `extractPredictedClass`:
  ```typescript
  const classesData = (await classesTensor.array()) as number[][][][]
  ```

### 2. `tf.pad` Padding Tuple Types
In TFJS 4.x, `tf.pad` expects `Array<[number, number]>`.
Check:
- `[src/tinyYolov2/depthwiseSeparableConv.ts](../src/tinyYolov2/depthwiseSeparableConv.ts)` (line 8)
- `[src/tinyYolov2/convWithBatchNorm.ts](../src/tinyYolov2/convWithBatchNorm.ts)` (line 8)
Both pass `[[0, 0], [1, 1], [1, 1], [0, 0]]` which satisfies `Array<[number, number]>`.

### 3. `tf.slice3d` and `as3D` / `as4D` Integer Coordinates
In `[src/dom/extractFaceTensors.ts](../src/dom/extractFaceTensors.ts)` (lines 40-42):
```typescript
const faceTensors = boxes.map(({ x, y, width, height }) =>
  tf.slice3d(
    imageTensor.as3D(imgHeight, imgWidth, numChannels),
    [Math.round(y), Math.round(x), 0],
    [Math.round(height), Math.round(width), numChannels]
  )
)
```
In TFJS 4.x, `tf.slice3d` validates integer start/size indices more strictly on WebGL and Node backends. Ensure `x, y, width, height` are integer-rounded or floored if bounding boxes have floating-point coordinates.

### 4. `tf.batchNorm` in `src/ssdMobilenetv1/mobileNetV1.ts`
In `[src/ssdMobilenetv1/mobileNetV1.ts](../src/ssdMobilenetv1/mobileNetV1.ts)` (lines 16-22):
```typescript
out = tf.batchNorm<tf.Rank.R4>(
  out,
  params.batch_norm_mean,
  params.batch_norm_variance,
  params.batch_norm_offset,
  params.batch_norm_scale,
  epsilon
)
```
Verify `tf.batchNorm<tf.Rank.R4>` compiles cleanly against `@tensorflow/tfjs-core@^4.22.0` (signature `batchNorm<R extends Rank>(x, mean, variance, offset?, scale?, varianceEpsilon?)` is unchanged).

### 5. `tf.image.resizeBilinear` Method Relocation
In TFJS 1.7, `tf.image.resizeBilinear` was available on `tf.image`. In TFJS 4.x, `tf.image.resizeBilinear` remains exported on the `tf.image` namespace in `@tensorflow/tfjs-core`.
Files using `tf.image.resizeBilinear`:
- `[src/dom/NetInput.ts](../src/dom/NetInput.ts)` (line 135)
- `[src/mtcnn/stage1.ts](../src/mtcnn/stage1.ts)` (line 16)

### 6. Method Chaining vs Functional Ops (`tensor.toFloat()`, `tensor.toInt()`, `tensor.div()`, `tensor.mul()`)
In TFJS 4.x, `Tensor` prototype methods `.toFloat()`, `.toInt()`, `.div()`, `.mul()`, `.sub()`, `.expandDims()`, `.as3D()`, `.as4D()`, `.reshape()`, `.slice()` remain present on `tf.Tensor` when `@tensorflow/tfjs-core` is imported. However, if any chained call loses its `Tensor4D` rank type (e.g., `.div(tf.scalar(255))` returning `Tensor<Rank>`), keep or add `as tf.Tensor4D`:
- `[src/faceFeatureExtractor/FaceFeatureExtractor.ts](../src/faceFeatureExtractor/FaceFeatureExtractor.ts)` (line 28)
- `[src/faceFeatureExtractor/TinyFaceFeatureExtractor.ts](../src/faceFeatureExtractor/TinyFaceFeatureExtractor.ts)` (line 28)
- `[src/xception/TinyXception.ts](../src/xception/TinyXception.ts)` (line 53)
- `[src/tinyYolov2/TinyYolov2Base.ts](../src/tinyYolov2/TinyYolov2Base.ts)` (line 106)

---

## Phase 2 Verification Checklist

- [ ] `@tensorflow/tfjs-backend-cpu` and `@tensorflow/tfjs-backend-webgl` are imported in `src/index.ts`.
- [ ] `NeuralNetwork.loadFromDisk` slices `buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)` to prevent pooled buffer corruption.
- [ ] `NeuralNetwork.freeze` preserves `variable.shape` and `variable.dtype`.
- [ ] All `arraySync()` and `array()` calls have explicit array rank types (`number[][]` / `number[][][][]`).
