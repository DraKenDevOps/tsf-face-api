When this line runs in `example/nodejs/recognize.ts`:

```41:41:example/nodejs/recognize.ts
  const single = await faceapi.detectSingleFace(angryImg, faceDetectionOptions).withFaceLandmarks().withFaceDescriptor();
```

Execution happens in two distinct phases: **Synchronous Task Chain Construction** (left-to-right method chaining) followed by **Asynchronous Pipeline Execution** (triggered by `await` via `ComposableTask.then`, which resolves from the outermost task inward).

---

### Phase 1: Synchronous Task Chain Construction (Left-to-Right)

#### 1. `faceapi.detectSingleFace(angryImg, faceDetectionOptions)`
* **`detectSingleFace`** (`src/globalApi/detectFaces.ts:6`)
  * Instantiates and returns **`new DetectSingleFaceTask(input, options)`** (`src/globalApi/DetectFacesTasks.ts:68`), which calls `super()` on `DetectFacesTaskBase` $\rightarrow$ `ComposableTask`.

#### 2. `.withFaceLandmarks()`
* **`DetectSingleFaceTask.withFaceLandmarks(useTinyLandmarkNet = false)`** (`src/globalApi/DetectFacesTasks.ts:87`)
  * Immediately calls **`this.runAndExtendWithFaceDetection()`** (`src/globalApi/DetectFacesTasks.ts:80`), which creates a `new Promise` and **starts executing `DetectSingleFaceTask.run()`** up to its first `await`.
  * Instantiates and returns **`new DetectSingleFaceLandmarksTask(promise, input, false)`** (`src/globalApi/DetectFaceLandmarksTasks.ts:57`), which calls `super()` on `DetectFaceLandmarksTaskBase` $\rightarrow$ `ComposableTask`.

#### 3. `.withFaceDescriptor()`
* **`DetectSingleFaceLandmarksTask.withFaceDescriptor()`** (`src/globalApi/DetectFaceLandmarksTasks.ts:86`)
  * Instantiates and returns **`new ComputeSingleFaceDescriptorTask(this, input)`** (`src/globalApi/ComputeFaceDescriptorsTasks.ts:47`), passing the `DetectSingleFaceLandmarksTask` instance as `parentTask`.

---

### Phase 2: Asynchronous Execution Sequence (Triggered by `await`)

Because `ComputeSingleFaceDescriptorTask` inherits from **`ComposableTask`** (`src/globalApi/ComposableTask.ts`), it implements a `.then()` method (making it a "Thenable"). `await` invokes `.then()`, which calls `.run()` on the outermost task, which in turn `await`s its `parentTask` chain:

```1:4:src/globalApi/ComposableTask.ts
export class ComposableTask<T> {
  public async then(onfulfilled: (value: T) => T | PromiseLike<T>): Promise<T> {
    return onfulfilled(await this.run());
  }
```

#### Step A: Face Detection (`SsdMobilenetv1`)
1. **`DetectSingleFaceTask.runAndExtendWithFaceDetection()`** (`src/globalApi/DetectFacesTasks.ts:80`)
   * Calls **`DetectSingleFaceTask.run()`** (`src/globalApi/DetectFacesTasks.ts:69`).
   * Creates and `await`s **`new DetectAllFacesTask(input, options)`** (which triggers `ComposableTask.then` $\rightarrow$ **`DetectAllFacesTask.run()`** at line 25).
2. **`DetectAllFacesTask.run()`** (`src/globalApi/DetectFacesTasks.ts:25`)
   * Checks `options instanceof SsdMobilenetv1Options` and calls **`nets.ssdMobilenetv1.locateFaces(input, options)`** (`src/ssdMobilenetv1/SsdMobilenetv1.ts:46`).
3. **`SsdMobilenetv1.locateFaces()`** (`src/ssdMobilenetv1/SsdMobilenetv1.ts:46`)
   * Calls **`toNetInput(input)`** (`src/dom/index.ts:183`) $\rightarrow$ validates media, awaits `awaitMediaLoaded`, and returns **`new NetInput(inputArray)`** (`src/dom/NetInput.ts:16`), which converts `angryImg` to a canvas via `createCanvasFromMedia`.
   * Calls **`SsdMobilenetv1.forwardInput(netInput)`** (`src/ssdMobilenetv1/SsdMobilenetv1.ts:23`):
     * `netInput.toBatchTensor(512, false)` (`src/dom/NetInput.ts:111`) $\rightarrow$ `imageToSquare` $\rightarrow$ `tf.browser.fromPixels`
     * `mobileNetV1(x, params.mobilenetv1)` (`src/ssdMobilenetv1/mobileNetV1.ts`)
     * `predictionLayer(...)` (`src/ssdMobilenetv1/predictionLayer.ts`)
     * `outputLayer(...)` (`src/ssdMobilenetv1/outputLayer.ts`)
   * Runs **`nonMaxSuppression(...)`** (`src/ssdMobilenetv1/nonMaxSuppression.ts`) on the output boxes and scores.
   * Maps surviving indices to **`new FaceDetection(score, new Rectangle(...), dims)`** (`src/classes/FaceDetection.ts`).
4. **Back in `DetectSingleFaceTask.run()` & `runAndExtendWithFaceDetection()`** (`src/globalApi/DetectFacesTasks.ts:71-84`)
   * Picks the `FaceDetection` with the highest score.
   * Wraps it using **`extendWithFaceDetection({}, detection)`** (`src/factories/WithFaceDetection.ts:11`), returning `{ detection: FaceDetection }`.

---

#### Step B: Face Landmark Detection (`FaceLandmark68Net`)
5. **`DetectSingleFaceLandmarksTask.run()`** (`src/globalApi/DetectFaceLandmarksTasks.ts:61`)
   * `await this.parentTask` resolves with `{ detection }` from Step A.
   * Calls **`extractFaces(this.input, [detection])`** (`src/dom/index.ts:63`), which crops the detected face bounding box from the image into a new `Canvas` element.
   * Accesses `this.landmarkNet` (`src/globalApi/DetectFaceLandmarksTasks.ts:24`), which returns **`nets.faceLandmark68Net`** (`FaceLandmark68Net`).
   * Calls **`FaceLandmark68NetBase.detectLandmarks(faces[0])`** (`src/faceLandmarkNet/index.ts:63`):
     * **`toNetInput(faces[0])`** (`src/dom/index.ts:183`)
     * **`FaceLandmark68NetBase.forwardInput(netInput)`** (`src/faceLandmarkNet/index.ts:48`):
       * **`FaceProcessor.runNet(input)`** (`src/faceProcessor/index.ts:27`) $\rightarrow$ **`FaceFeatureExtractor.forwardInput(input)`** (resizes to `112x112`, runs dense blocks) $\rightarrow$ **`fullyConnectedLayer(...)`** (outputs 136 values).
       * **`FaceLandmark68NetBase.postProcess(...)`** (`src/faceLandmarkNet/index.ts:16`) normalizes the 68 `(x, y)` coordinates.
     * Instantiates and returns **`new FaceLandmarks68(points, dims)`** (`src/classes/FaceLandmarks68.ts`).
   * Calls **`extendWithFaceLandmarks(parentResult, landmarks)`** (`src/factories/WithFaceLandmarks.ts:21`):
     * Calls `unshiftedLandmarks.shiftBy(shift.x, shift.y)` (`src/classes/FaceLandmarks.ts:50`) to shift landmarks to original image coordinates.
     * Calls `landmarks.align()` (`src/classes/FaceLandmarks.ts:69`) $\rightarrow$ `alignMinBbox()` to compute `alignedRect`.
     * Returns `{ detection, landmarks, unshiftedLandmarks, alignedRect }`.

---

#### Step C: Face Descriptor / Embedding Computation (`FaceRecognitionNet`)
6. **`ComputeSingleFaceDescriptorTask.run()`** (`src/globalApi/ComputeFaceDescriptorsTasks.ts:51`)
   * `await this.parentTask` triggers `DetectSingleFaceLandmarksTask.then()` $\rightarrow$ resolves with `{ detection, landmarks, unshiftedLandmarks, alignedRect }` from Step B.
   * Calls **`extractSingleFaceAndComputeResult(...)`** (`src/globalApi/extractFacesAndComputeResults.ts:28`) $\rightarrow$ **`extractAllFacesAndComputeResults(...)`** (`src/globalApi/extractFacesAndComputeResults.ts:10`):
     * Calls the alignment callback: **`parentResult.landmarks.align(null, { useDlibAlignment: true })`** (`src/classes/FaceLandmarks.ts:69`):
       * Calls **`FaceLandmarks.alignDlib()`** (`src/classes/FaceLandmarks.ts:85`) $\rightarrow$ **`FaceLandmarks68.getRefPointsForAlignment()`** (`src/classes/FaceLandmarks68.ts:34`), computing the aligned bounding box from the left eye, right eye, and mouth centers.
     * Calls **`extractFaces(input, faceBoxes)`** (`src/dom/index.ts:63`) to crop the Dlib-aligned face region into a `Canvas`.
     * Calls **`nets.faceRecognitionNet.computeFaceDescriptor(face)`** (`src/faceRecognitionNet/index.ts:62`):
       * **`toNetInput(face)`** (`src/dom/index.ts:183`)
       * **`FaceRecognitionNet.forwardInput(netInput)`** (`src/faceRecognitionNet/index.ts:17`):
         * `netInput.toBatchTensor(150, true)` (`150x150` centered square tensor)
         * `opsNormalize(batchTensor, meanRgb)`
         * ResNet-34-like layers: `convDown`, `maxPool`, `residual` / `residualDown` blocks (`conv32_*`, `conv64_*`, `conv128_*`, `conv256_*`), `tf.mean` (global average pooling), and `tf.matMul(..., params.fc)`.
       * Extracts the `128`-d `Float32Array` embedding via `t.data()`.
   * Calls **`extendWithFaceDescriptor(parentResult, descriptor)`** (`src/utils/index.ts:177`), returning the final object:
     ```ts
     {
       detection: FaceDetection,
       landmarks: FaceLandmarks68,
       unshiftedLandmarks: FaceLandmarks68,
       alignedRect: FaceDetection,
       descriptor: Float32Array(128)
     }
     ```