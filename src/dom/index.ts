import * as tf from "@tensorflow/tfjs-core";
import { env } from "../env";
import { getModelUris, isTensor3D, isTensor4D } from "../utils";
import { TNetInput } from "./types";
import { FaceDetection } from "../classes/FaceDetection";
import { Rectangle } from "../classes/Rectangle";
import { getContext2dOrThrow, isMediaElement, isMediaLoaded, NetInput, resolveInput, createCanvas } from "./NetInput";

export function awaitMediaLoaded(media: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement) {
  return new Promise<Event | void>((resolve, reject) => {
    if (media instanceof env.getEnv().Canvas || isMediaLoaded(media)) {
      return resolve();
    }

    function onLoad(e: Event) {
      if (!e.currentTarget) return;
      e.currentTarget.removeEventListener("load", onLoad);
      e.currentTarget.removeEventListener("error", onError);
      resolve(e);
    }

    function onError(e: Event) {
      if (!e.currentTarget) return;
      e.currentTarget.removeEventListener("load", onLoad);
      e.currentTarget.removeEventListener("error", onError);
      reject(e);
    }

    media.addEventListener("load", onLoad);
    media.addEventListener("error", onError);
  });
}

export function blobToImage(blb: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    if (!(blb instanceof Blob)) {
      return reject("blobToImage - expected blob to be of type: Blob");
    }

    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result !== "string") {
        return reject("blobToImage - expected reader.result to be a string, in onload");
      }

      const img = env.getEnv().createImageElement();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(blb);
  });
}

/**
 * Extracts the image regions containing the detected faces.
 *
 * @param input The image that face detection has been performed on.
 * @param detections The face detection results or face bounding boxes for that image.
 * @returns The Canvases of the corresponding image region for each detected face.
 */
export async function extractFaces(input: TNetInput, detections: Array<FaceDetection | Rectangle>): Promise<HTMLCanvasElement[]> {
  const { Canvas } = env.getEnv();

  let canvas = input as HTMLCanvasElement;

  if (!(input instanceof Canvas)) {
    const netInput = await toNetInput(input);

    if (netInput.batchSize > 1) {
      throw new Error("extractFaces - batchSize > 1 not supported");
    }

    const tensorOrCanvas = netInput.getInput(0);
    canvas = tensorOrCanvas instanceof Canvas ? tensorOrCanvas : await imageTensorToCanvas(tensorOrCanvas);
  }

  const ctx = getContext2dOrThrow(canvas);
  const boxes = detections
    .map((det) => (det instanceof FaceDetection ? det.forSize(canvas.width, canvas.height).box.floor() : det))
    .map((box) => box.clipAtImageBorders(canvas.width, canvas.height));

  return boxes.map(({ x, y, width, height }) => {
    const faceImg = createCanvas({ width, height });
    getContext2dOrThrow(faceImg).putImageData(ctx.getImageData(x, y, width, height), 0, 0);
    return faceImg;
  });
}

/**
 * Extracts the tensors of the image regions containing the detected faces.
 * Useful if you want to compute the face descriptors for the face images.
 * Using this method is faster then extracting a canvas for each face and
 * converting them to tensors individually.
 *
 * @param imageTensor The image tensor that face detection has been performed on.
 * @param detections The face detection results or face bounding boxes for that image.
 * @returns Tensors of the corresponding image region for each detected face.
 */
export async function extractFaceTensors(imageTensor: tf.Tensor3D | tf.Tensor4D, detections: Array<FaceDetection | Rectangle>): Promise<tf.Tensor3D[]> {
  if (!isTensor3D(imageTensor) && !isTensor4D(imageTensor)) {
    throw new Error("extractFaceTensors - expected image tensor to be 3D or 4D");
  }

  if (isTensor4D(imageTensor) && imageTensor.shape[0] > 1) {
    throw new Error("extractFaceTensors - batchSize > 1 not supported");
  }

  return tf.tidy(() => {
    const [imgHeight, imgWidth, numChannels] = imageTensor.shape.slice(isTensor4D(imageTensor) ? 1 : 0);

    const boxes = detections
      .map((det) => (det instanceof FaceDetection ? det.forSize(imgWidth, imgHeight).box : det))
      .map((box) => box.clipAtImageBorders(imgWidth, imgHeight));

    const faceTensors = boxes.map(({ x, y, width, height }) =>
      tf.slice3d(
        tf.reshape(imageTensor, [imgHeight, imgWidth, numChannels]) as tf.Tensor3D,
        [Math.round(y), Math.round(x), 0],
        [Math.round(height), Math.round(width), numChannels]
      )
    );

    return faceTensors;
  });
}

export async function fetchOrThrow(url: string, init?: RequestInit): Promise<Response> {
  const fetch = env.getEnv().fetch;
  const res = await fetch(url, init);
  if (!(res.status < 400)) {
    throw new Error(`failed to fetch: (${res.status}) ${res.statusText}, from url: ${res.url}`);
  }
  return res;
}

export async function fetchImage(uri: string): Promise<HTMLImageElement> {
  const res = await fetchOrThrow(uri);
  const blob = await res.blob();

  if (!blob.type.startsWith("image/")) {
    throw new Error(`fetchImage - expected blob type to be of type image/*, instead have: ${blob.type}, for url: ${res.url}`);
  }
  return blobToImage(blob);
}

export async function fetchJson<T>(uri: string): Promise<T> {
  return (await fetchOrThrow(uri)).json();
}

export async function fetchNetWeights(uri: string): Promise<Float32Array> {
  return new Float32Array(await (await fetchOrThrow(uri)).arrayBuffer());
}

export async function imageTensorToCanvas(imgTensor: tf.Tensor, canvas?: HTMLCanvasElement): Promise<HTMLCanvasElement> {
  const targetCanvas = canvas || env.getEnv().createCanvasElement();

  const [height, width, numChannels] = imgTensor.shape.slice(isTensor4D(imgTensor) ? 1 : 0);
  const imgTensor3D = tf.tidy(() => tf.cast(tf.reshape(imgTensor, [height, width, numChannels]), "int32") as tf.Tensor3D);
  await tf.browser.toPixels(imgTensor3D, targetCanvas);

  imgTensor3D.dispose();

  return targetCanvas;
}

export async function loadWeightMap(uri: string | undefined, defaultModelName: string): Promise<tf.NamedTensorMap> {
  const { manifestUri, modelBaseUri } = getModelUris(uri, defaultModelName);

  const manifest = await fetchJson<tf.io.WeightsManifestConfig>(manifestUri);

  return tf.io.loadWeights(manifest, modelBaseUri);
}

/**
 * Validates the input to make sure, they are valid net inputs and awaits all media elements
 * to be finished loading.
 *
 * @param input The input, which can be a media element or an array of different media elements.
 * @returns A NetInput instance, which can be passed into one of the neural networks.
 */
export async function toNetInput(inputs: TNetInput): Promise<NetInput> {
  if (inputs instanceof NetInput) {
    return inputs;
  }

  let inputArgArray = Array.isArray(inputs) ? inputs : [inputs];

  if (!inputArgArray.length) {
    throw new Error("toNetInput - empty array passed as input");
  }

  const getIdxHint = (idx: number) => (Array.isArray(inputs) ? ` at input index ${idx}:` : "");

  const inputArray = inputArgArray.map(resolveInput);

  inputArray.forEach((input, i) => {
    if (!isMediaElement(input) && !isTensor3D(input) && !isTensor4D(input)) {
      if (typeof inputArgArray[i] === "string") {
        throw new Error(`toNetInput -${getIdxHint(i)} string passed, but could not resolve HTMLElement for element id ${inputArgArray[i]}`);
      }

      throw new Error(
        `toNetInput -${getIdxHint(i)} expected media to be of type HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | tf.Tensor3D, or to be an element id`
      );
    }

    if (isTensor4D(input)) {
      // if tf.Tensor4D is passed in the input array, the batch size has to be 1
      const batchSize = input.shape[0];
      if (batchSize !== 1) {
        throw new Error(`toNetInput -${getIdxHint(i)} tf.Tensor4D with batchSize ${batchSize} passed, but not supported in input array`);
      }
    }
  });

  // wait for all media elements being loaded
  await Promise.all(inputArray.map((input) => isMediaElement(input) && awaitMediaLoaded(input)));

  return new NetInput(inputArray, Array.isArray(inputs));
}
