import * as tf from "@tensorflow/tfjs-core";
import { env } from "../env";
import { computeReshapedDimensions, isTensor3D, isTensor4D, padToSquare, range } from "../utils";
import { Dimensions } from "../classes/Dimensions";
import { TResolvedNetInput } from "./types";

export class NetInput {
  private _imageTensors: Array<tf.Tensor3D | tf.Tensor4D> = [];
  private _canvases: HTMLCanvasElement[] = [];
  private _batchSize: number;
  private _treatAsBatchInput = false;

  private _inputDimensions: number[][] = [];
  private _inputSize!: number;

  constructor(inputs: Array<TResolvedNetInput>, treatAsBatchInput = false) {
    if (!Array.isArray(inputs)) {
      throw new Error(`NetInput.constructor - expected inputs to be an Array of TResolvedNetInput or to be instanceof tf.Tensor4D, instead have ${inputs}`);
    }

    this._treatAsBatchInput = treatAsBatchInput;
    this._batchSize = inputs.length;

    inputs.forEach((input, idx) => {
      if (isTensor3D(input)) {
        this._imageTensors[idx] = input;
        this._inputDimensions[idx] = input.shape;
        return;
      }

      if (isTensor4D(input)) {
        const batchSize = input.shape[0];
        if (batchSize !== 1) {
          throw new Error(`NetInput - tf.Tensor4D with batchSize ${batchSize} passed, but not supported in input array`);
        }

        this._imageTensors[idx] = input;
        this._inputDimensions[idx] = input.shape.slice(1);
        return;
      }

      const canvas = input instanceof env.getEnv().Canvas ? input : createCanvasFromMedia(input);
      this._canvases[idx] = canvas;
      this._inputDimensions[idx] = [canvas.height, canvas.width, 3];
    });
  }

  public get imageTensors(): Array<tf.Tensor3D | tf.Tensor4D> {
    return this._imageTensors;
  }

  public get canvases(): HTMLCanvasElement[] {
    return this._canvases;
  }

  public get isBatchInput(): boolean {
    return this.batchSize > 1 || this._treatAsBatchInput;
  }

  public get batchSize(): number {
    return this._batchSize;
  }

  public get inputDimensions(): number[][] {
    return this._inputDimensions;
  }

  public get inputSize(): number | undefined {
    return this._inputSize;
  }

  public get reshapedInputDimensions(): Dimensions[] {
    return range(this.batchSize, 0, 1).map((_, batchIdx) => this.getReshapedInputDimensions(batchIdx));
  }

  public getInput(batchIdx: number): tf.Tensor3D | tf.Tensor4D | HTMLCanvasElement {
    return this.canvases[batchIdx] || this.imageTensors[batchIdx];
  }

  public getInputDimensions(batchIdx: number): number[] {
    return this._inputDimensions[batchIdx];
  }

  public getInputHeight(batchIdx: number): number {
    return this._inputDimensions[batchIdx][0];
  }

  public getInputWidth(batchIdx: number): number {
    return this._inputDimensions[batchIdx][1];
  }

  public getReshapedInputDimensions(batchIdx: number): Dimensions {
    if (typeof this.inputSize !== "number") {
      throw new Error("getReshapedInputDimensions - inputSize not set, toBatchTensor has not been called yet");
    }

    const width = this.getInputWidth(batchIdx);
    const height = this.getInputHeight(batchIdx);
    return computeReshapedDimensions({ width, height }, this.inputSize);
  }

  /**
   * Create a batch tensor from all input canvases and tensors
   * with size [batchSize, inputSize, inputSize, 3].
   *
   * @param inputSize Height and width of the tensor.
   * @param isCenterImage (optional, default: false) If true, add an equal amount of padding on
   * both sides of the minor dimension oof the image.
   * @returns The batch tensor.
   */
  public toBatchTensor(inputSize: number, isCenterInputs = true): tf.Tensor4D {
    this._inputSize = inputSize;

    return tf.tidy(() => {
      const inputTensors = range(this.batchSize, 0, 1).map((batchIdx) => {
        const input = this.getInput(batchIdx);

        if (input instanceof tf.Tensor) {
          let imgTensor = isTensor4D(input) ? input : tf.expandDims<tf.Tensor4D>(input);
          imgTensor = padToSquare(imgTensor, isCenterInputs);

          if (imgTensor.shape[1] !== inputSize || imgTensor.shape[2] !== inputSize) {
            imgTensor = tf.image.resizeBilinear(imgTensor, [inputSize, inputSize]);
          }

          return tf.reshape(imgTensor, [inputSize, inputSize, 3]) as tf.Tensor3D;
        }

        if (input instanceof env.getEnv().Canvas) {
          return tf.browser.fromPixels(imageToSquare(input, inputSize, isCenterInputs));
        }

        throw new Error(
          `toBatchTensor - at batchIdx ${batchIdx}, expected input to be instanceof tf.Tensor or instanceof HTMLCanvasElement, instead have ${input}`
        );
      });

      const batchTensor = tf.reshape(tf.stack(inputTensors.map((t) => tf.cast(t, "float32"))), [this.batchSize, inputSize, inputSize, 3]) as tf.Tensor4D;

      return batchTensor;
    });
  }
}

export function createCanvas({ width, height }: IDimensions): HTMLCanvasElement {
  const { createCanvasElement } = env.getEnv();
  const canvas = createCanvasElement();
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

export function isMediaElement(input: any) {
  const { Image, Canvas, Video } = env.getEnv();
  return input instanceof Image || input instanceof Canvas || input instanceof Video;
}

export function isMediaLoaded(media: HTMLImageElement | HTMLVideoElement): boolean {
  const { Image, Video } = env.getEnv();
  return (media instanceof Image && media.complete) || (media instanceof Video && media.readyState >= 3);
}

export function createCanvasFromMedia(media: HTMLImageElement | HTMLVideoElement | ImageData, dims?: IDimensions): HTMLCanvasElement {
  const { ImageData } = env.getEnv();

  if (!(media instanceof ImageData) && !isMediaLoaded(media)) {
    throw new Error("createCanvasFromMedia - media has not finished loading yet");
  }

  const { width, height } = dims || getMediaDimensions(media);
  const canvas = createCanvas({ width, height });

  if (media instanceof ImageData) {
    getContext2dOrThrow(canvas).putImageData(media, 0, 0);
  } else {
    getContext2dOrThrow(canvas).drawImage(media, 0, 0, width, height);
  }
  return canvas;
}

export function getContext2dOrThrow(canvasArg: string | HTMLCanvasElement | CanvasRenderingContext2D): CanvasRenderingContext2D {
  const { Canvas, CanvasRenderingContext2D } = env.getEnv();

  if (canvasArg instanceof CanvasRenderingContext2D) {
    return canvasArg;
  }

  const canvas = resolveInput(canvasArg);

  if (!(canvas instanceof Canvas)) {
    throw new Error("resolveContext2d - expected canvas to be of instance of Canvas");
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("resolveContext2d - canvas 2d context is null");
  }

  return ctx;
}

export function resolveInput(arg: string | any) {
  if (!env.isNodejs() && typeof arg === "string") {
    return document.getElementById(arg);
  }
  return arg;
}

export function imageToSquare(input: HTMLImageElement | HTMLCanvasElement, inputSize: number, centerImage = false) {
  const { Image, Canvas } = env.getEnv();

  if (!(input instanceof Image || input instanceof Canvas)) {
    throw new Error("imageToSquare - expected arg0 to be HTMLImageElement | HTMLCanvasElement");
  }

  const dims = getMediaDimensions(input);
  const scale = inputSize / Math.max(dims.height, dims.width);
  const width = scale * dims.width;
  const height = scale * dims.height;

  const targetCanvas = createCanvas({ width: inputSize, height: inputSize });
  const inputCanvas = input instanceof Canvas ? input : createCanvasFromMedia(input);

  const offset = Math.abs(width - height) / 2;
  const dx = centerImage && width < height ? offset : 0;
  const dy = centerImage && height < width ? offset : 0;
  getContext2dOrThrow(targetCanvas).drawImage(inputCanvas, dx, dy, width, height);

  return targetCanvas;
}

export function matchDimensions(input: IDimensions, reference: IDimensions, useMediaDimensions = false) {
  const { width, height } = useMediaDimensions ? getMediaDimensions(reference) : reference;
  input.width = width;
  input.height = height;
  return { width, height };
}

export function getMediaDimensions(input: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | IDimensions): Dimensions {
  const { Image, Video } = env.getEnv();

  if (input instanceof Image) {
    return new Dimensions(input.naturalWidth, input.naturalHeight);
  }
  if (input instanceof Video) {
    return new Dimensions(input.videoWidth, input.videoHeight);
  }
  return new Dimensions(input.width, input.height);
}