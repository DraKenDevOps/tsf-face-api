import * as tf from "@tensorflow/tfjs-core";

import { Dimensions } from "../classes/Dimensions";
import { Point } from "../classes/Point";
import { ScaleLayerParams } from "../faceRecognitionNet/types";

export function isTensor(tensor: any, dim: number) {
  return tensor instanceof tf.Tensor && tensor.shape.length === dim;
}

export function isTensor1D(tensor: any): tensor is tf.Tensor1D {
  return isTensor(tensor, 1);
}

export function isTensor2D(tensor: any): tensor is tf.Tensor2D {
  return isTensor(tensor, 2);
}

export function isTensor3D(tensor: any): tensor is tf.Tensor3D {
  return isTensor(tensor, 3);
}

export function isTensor4D(tensor: any): tensor is tf.Tensor4D {
  return isTensor(tensor, 4);
}

export function isFloat(num: number) {
  return num % 1 !== 0;
}

export function isEven(num: number) {
  return num % 2 === 0;
}

export function round(num: number, prec: number = 2) {
  const f = Math.pow(10, prec);
  return Math.floor(num * f) / f;
}

export function isDimensions(obj: any): boolean {
  return obj && obj.width && obj.height;
}

export function computeReshapedDimensions({ width, height }: IDimensions, inputSize: number) {
  const scale = inputSize / Math.max(height, width);
  return new Dimensions(Math.round(width * scale), Math.round(height * scale));
}

export function getCenterPoint(pts: Point[]): Point {
  return pts.reduce((sum, pt) => sum.add(pt), new Point(0, 0)).div(new Point(pts.length, pts.length));
}

export function range(num: number, start: number, step: number): number[] {
  return Array(num)
    .fill(0)
    .map((_, i) => start + i * step);
}

export function isValidNumber(num: any) {
  return (!!num && num !== Infinity && num !== -Infinity && !isNaN(num)) || num === 0;
}

export function isValidProbablitiy(num: any) {
  return isValidNumber(num) && 0 <= num && num <= 1.0;
}

export function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

export function inverseSigmoid(x: number) {
  return Math.log(x / (1 - x));
}

export function euclideanDistance(arr1: number[] | Float32Array, arr2: number[] | Float32Array) {
  if (arr1.length !== arr2.length) throw new Error("euclideanDistance: arr1.length !== arr2.length");
  const desc1 = Array.from(arr1);
  const desc2 = Array.from(arr2);
  return Math.sqrt(desc1.map((val, i) => val - desc2[i]).reduce((res, diff) => res + Math.pow(diff, 2), 0));
}

export function extractWeightsFactory(weights: Float32Array) {
  let remainingWeights = weights;

  function extractWeights(numWeights: number): Float32Array {
    const ret = remainingWeights.slice(0, numWeights);
    remainingWeights = remainingWeights.slice(numWeights);
    return ret;
  }

  function getRemainingWeights(): Float32Array {
    return remainingWeights;
  }

  return {
    extractWeights,
    getRemainingWeights
  };
}

export function getModelUris(uri: string | undefined, defaultModelName: string) {
  const defaultManifestFilename = `${defaultModelName}-weights_manifest.json`;

  if (!uri) {
    return {
      modelBaseUri: "",
      manifestUri: defaultManifestFilename
    };
  }

  if (uri === "/") {
    return {
      modelBaseUri: "/",
      manifestUri: `/${defaultManifestFilename}`
    };
  }
  const protocol = uri.startsWith("http://") ? "http://" : uri.startsWith("https://") ? "https://" : "";
  uri = uri.replace(protocol, "");

  const parts = uri.split("/").filter((s) => s);

  const manifestFile = uri.endsWith(".json") ? parts[parts.length - 1] : defaultManifestFilename;

  let modelBaseUri = protocol + (uri.endsWith(".json") ? parts.slice(0, parts.length - 1) : parts).join("/");
  modelBaseUri = uri.startsWith("/") ? `/${modelBaseUri}` : modelBaseUri;

  return {
    modelBaseUri,
    manifestUri: modelBaseUri === "/" ? `/${manifestFile}` : `${modelBaseUri}/${manifestFile}`
  };
}

export function getSizesForScale(scale: number, [height, width]: number[]) {
  return {
    height: Math.floor(height * scale),
    width: Math.floor(width * scale)
  };
}

export function shuffleArray(inputArray: any[]) {
  const array = inputArray.slice();
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const x = array[i];
    array[i] = array[j];
    array[j] = x;
  }
  return array;
}

export function isBrowser(): boolean {
  return (
    typeof window === "object" &&
    typeof document !== "undefined" &&
    typeof HTMLImageElement !== "undefined" &&
    typeof HTMLCanvasElement !== "undefined" &&
    typeof HTMLVideoElement !== "undefined" &&
    typeof ImageData !== "undefined" &&
    typeof CanvasRenderingContext2D !== "undefined"
  );
}

export function isNodejs(): boolean {
  return typeof process !== "undefined" && !!process.versions && !!process.versions.node;
}

export function isWithAge(obj: any): obj is WithAge<{}> {
  return typeof obj["age"] === "number";
}

export function extendWithAge<TSource>(sourceObj: TSource, age: number): WithAge<TSource> {
  const extension = { age };
  return Object.assign({}, sourceObj, extension);
}

export function extendWithFaceDescriptor<TSource>(sourceObj: TSource, descriptor: Float32Array): WithFaceDescriptor<TSource> {
  const extension = { descriptor };
  return Object.assign({}, sourceObj, extension);
}

export function bgrToRgbTensor(tensor: tf.Tensor4D): tf.Tensor4D {
  return tf.tidy(() => tf.stack(tf.unstack(tensor, 3).reverse(), 3)) as tf.Tensor4D;
}

export function normalize(x: tf.Tensor4D): tf.Tensor4D {
  return tf.tidy(() => tf.mul(tf.sub(x, tf.scalar(127.5)), tf.scalar(0.0078125)));
}

export function prelu<T extends tf.Tensor>(x: T, alpha: tf.Tensor1D): T {
  return tf.tidy(() => tf.add(tf.relu(x), tf.mul(alpha, tf.neg(tf.relu(tf.neg(x))))));
}

/**
 * Pads the smaller dimension of an image tensor with zeros, such that width === height.
 *
 * @param imgTensor The image tensor.
 * @param isCenterImage (optional, default: false) If true, add an equal amount of padding on
 * both sides of the minor dimension oof the image.
 * @returns The padded tensor with width === height.
 */
export function padToSquare(imgTensor: tf.Tensor4D, isCenterImage = false): tf.Tensor4D {
  return tf.tidy(() => {
    const [height, width] = imgTensor.shape.slice(1);
    if (height === width) {
      return imgTensor;
    }

    const dimDiff = Math.abs(height - width);
    const paddingAmount = Math.round(dimDiff * (isCenterImage ? 0.5 : 1));
    const paddingAxis = height > width ? 2 : 1;

    const createPaddingTensor = (paddingAmount: number): tf.Tensor => {
      const paddingTensorShape = imgTensor.shape.slice();
      paddingTensorShape[paddingAxis] = paddingAmount;
      return tf.fill(paddingTensorShape, 0);
    };

    const paddingTensorAppend = createPaddingTensor(paddingAmount);
    const remainingPaddingAmount = dimDiff - (paddingTensorAppend.shape[paddingAxis] as number);

    const paddingTensorPrepend = isCenterImage && remainingPaddingAmount ? createPaddingTensor(remainingPaddingAmount) : null;

    const tensorsToStack = [paddingTensorPrepend, imgTensor, paddingTensorAppend]
      .filter((t): t is tf.Tensor => !!t)
      .map((t: tf.Tensor) => tf.cast(t, "float32")) as tf.Tensor4D[];
    return tf.concat(tensorsToStack, paddingAxis);
  });
}

export function opsNormalize(x: tf.Tensor4D, meanRgb: number[]): tf.Tensor4D {
  return tf.tidy(() => {
    const [r, g, b] = meanRgb;
    const avg_r = tf.fill([...x.shape.slice(0, 3), 1], r);
    const avg_g = tf.fill([...x.shape.slice(0, 3), 1], g);
    const avg_b = tf.fill([...x.shape.slice(0, 3), 1], b);
    const avg_rgb = tf.concat([avg_r, avg_g, avg_b], 3);

    return tf.sub(x, avg_rgb);
  });
}

export function leaky(x: tf.Tensor4D): tf.Tensor4D {
  return tf.tidy(() => {
    const min = tf.mul(x, tf.scalar(0.10000000149011612));
    return tf.add(tf.relu(tf.sub(x, min)), min);
    //return tf.maximum(x, min)
  });
}

export function scale(x: tf.Tensor4D, params: ScaleLayerParams): tf.Tensor4D {
  return tf.add(tf.mul(x, params.weights), params.biases);
}

export function seperateWeightMaps(weightMap: tf.NamedTensorMap) {
  const featureExtractorMap: tf.NamedTensorMap = {};
  const classifierMap: tf.NamedTensorMap = {};

  Object.keys(weightMap).forEach((key) => {
    const map = key.startsWith("fc") ? classifierMap : featureExtractorMap;
    map[key] = weightMap[key];
  });

  return { featureExtractorMap, classifierMap };
}