import * as tf from "@tensorflow/tfjs-core";

import { toNetInput } from "../dom";
import { NetInput } from "../dom/NetInput";
import { TNetInput } from "../dom/types";
import { NeuralNetwork } from "../NeuralNetwork";
import { ConvLayerParams, NetParams, ResidualLayerParams, ScaleLayerParams } from "./types";
import { extractWeightsFactory, isFloat, isTensor2D, opsNormalize, scale } from "../utils";
import { ConvParams } from "../common/types";
import { disposeUnusedWeightTensors, extractWeightEntryFactory } from "../common";

export class FaceRecognitionNet extends NeuralNetwork<NetParams> {
  constructor() {
    super("FaceRecognitionNet");
  }

  public forwardInput(input: NetInput): tf.Tensor2D {
    const { params } = this;

    if (!params) {
      throw new Error("FaceRecognitionNet - load model before inference");
    }

    return tf.tidy(() => {
      const batchTensor = tf.cast(input.toBatchTensor(150, true), "float32") as tf.Tensor4D;

      const meanRgb = [122.782, 117.001, 104.298];
      const normalized = tf.div(opsNormalize(batchTensor, meanRgb), tf.scalar(256)) as tf.Tensor4D;

      let out = convDown(normalized, params.conv32_down);
      out = tf.maxPool(out, 3, 2, "valid");

      out = residual(out, params.conv32_1);
      out = residual(out, params.conv32_2);
      out = residual(out, params.conv32_3);

      out = residualDown(out, params.conv64_down);
      out = residual(out, params.conv64_1);
      out = residual(out, params.conv64_2);
      out = residual(out, params.conv64_3);

      out = residualDown(out, params.conv128_down);
      out = residual(out, params.conv128_1);
      out = residual(out, params.conv128_2);

      out = residualDown(out, params.conv256_down);
      out = residual(out, params.conv256_1);
      out = residual(out, params.conv256_2);
      out = residualDown(out, params.conv256_down_out);

      const globalAvg = tf.mean(out, [1, 2]) as tf.Tensor2D;
      const fullyConnected = tf.matMul(globalAvg, params.fc);

      return fullyConnected as tf.Tensor2D;
    });
  }

  public async forward(input: TNetInput): Promise<tf.Tensor2D> {
    return this.forwardInput(await toNetInput(input));
  }

  public async computeFaceDescriptor(input: TNetInput): Promise<Float32Array | Float32Array[]> {
    const netInput = await toNetInput(input);

    const faceDescriptorTensors = tf.tidy(() => tf.unstack(this.forwardInput(netInput)));

    const faceDescriptorsForBatch = (await Promise.all(faceDescriptorTensors.map((t) => t.data()))) as Float32Array[];

    faceDescriptorTensors.forEach((t) => t.dispose());

    return netInput.isBatchInput ? faceDescriptorsForBatch : faceDescriptorsForBatch[0];
  }

  protected getDefaultModelName(): string {
    return "face_recognition_model";
  }

  protected extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeigthMap(weightMap);
  }

  protected extractParams(weights: Float32Array) {
    return extractParams(weights);
  }
}

export function createFaceRecognitionNet(weights: Float32Array) {
  const net = new FaceRecognitionNet();
  net.extractWeights(weights);
  return net;
}

function convLayer(x: tf.Tensor4D, params: ConvLayerParams, strides: [number, number], withRelu: boolean, padding: "valid" | "same" = "same"): tf.Tensor4D {
  const { filters, bias } = params.conv;

  let out = tf.conv2d(x, filters, strides, padding);
  out = tf.add(out, bias);
  out = scale(out, params.scale);
  return withRelu ? tf.relu(out) : out;
}

export function conv(x: tf.Tensor4D, params: ConvLayerParams) {
  return convLayer(x, params, [1, 1], true);
}

export function convNoRelu(x: tf.Tensor4D, params: ConvLayerParams) {
  return convLayer(x, params, [1, 1], false);
}

export function convDown(x: tf.Tensor4D, params: ConvLayerParams) {
  return convLayer(x, params, [2, 2], true, "valid");
}

function extractorsFactory(extractWeights: ExtractWeightsFunction, paramMappings: ParamMapping[]) {
  function extractFilterValues(numFilterValues: number, numFilters: number, filterSize: number): tf.Tensor4D {
    const weights = extractWeights(numFilterValues);
    const depth = weights.length / (numFilters * filterSize * filterSize);

    if (isFloat(depth)) {
      throw new Error(`depth has to be an integer: ${depth}, weights.length: ${weights.length}, numFilters: ${numFilters}, filterSize: ${filterSize}`);
    }

    return tf.tidy(() => tf.transpose(tf.tensor4d(weights, [numFilters, depth, filterSize, filterSize]), [2, 3, 1, 0]));
  }

  function extractConvParams(numFilterValues: number, numFilters: number, filterSize: number, mappedPrefix: string): ConvParams {
    const filters = extractFilterValues(numFilterValues, numFilters, filterSize);
    const bias = tf.tensor1d(extractWeights(numFilters));

    paramMappings.push({ paramPath: `${mappedPrefix}/filters` }, { paramPath: `${mappedPrefix}/bias` });

    return { filters, bias };
  }

  function extractScaleLayerParams(numWeights: number, mappedPrefix: string): ScaleLayerParams {
    const weights = tf.tensor1d(extractWeights(numWeights));
    const biases = tf.tensor1d(extractWeights(numWeights));

    paramMappings.push({ paramPath: `${mappedPrefix}/weights` }, { paramPath: `${mappedPrefix}/biases` });

    return {
      weights,
      biases
    };
  }

  function extractConvLayerParams(numFilterValues: number, numFilters: number, filterSize: number, mappedPrefix: string): ConvLayerParams {
    const conv = extractConvParams(numFilterValues, numFilters, filterSize, `${mappedPrefix}/conv`);
    const scale = extractScaleLayerParams(numFilters, `${mappedPrefix}/scale`);

    return { conv, scale };
  }

  function extractResidualLayerParams(
    numFilterValues: number,
    numFilters: number,
    filterSize: number,
    mappedPrefix: string,
    isDown = false
  ): ResidualLayerParams {
    const conv1 = extractConvLayerParams((isDown ? 0.5 : 1) * numFilterValues, numFilters, filterSize, `${mappedPrefix}/conv1`);
    const conv2 = extractConvLayerParams(numFilterValues, numFilters, filterSize, `${mappedPrefix}/conv2`);

    return { conv1, conv2 };
  }

  return {
    extractConvLayerParams,
    extractResidualLayerParams
  };
}

export function extractParams(weights: Float32Array): { params: NetParams; paramMappings: ParamMapping[] } {
  const { extractWeights, getRemainingWeights } = extractWeightsFactory(weights);

  const paramMappings: ParamMapping[] = [];

  const { extractConvLayerParams, extractResidualLayerParams } = extractorsFactory(extractWeights, paramMappings);

  const conv32_down = extractConvLayerParams(4704, 32, 7, "conv32_down");
  const conv32_1 = extractResidualLayerParams(9216, 32, 3, "conv32_1");
  const conv32_2 = extractResidualLayerParams(9216, 32, 3, "conv32_2");
  const conv32_3 = extractResidualLayerParams(9216, 32, 3, "conv32_3");

  const conv64_down = extractResidualLayerParams(36864, 64, 3, "conv64_down", true);
  const conv64_1 = extractResidualLayerParams(36864, 64, 3, "conv64_1");
  const conv64_2 = extractResidualLayerParams(36864, 64, 3, "conv64_2");
  const conv64_3 = extractResidualLayerParams(36864, 64, 3, "conv64_3");

  const conv128_down = extractResidualLayerParams(147456, 128, 3, "conv128_down", true);
  const conv128_1 = extractResidualLayerParams(147456, 128, 3, "conv128_1");
  const conv128_2 = extractResidualLayerParams(147456, 128, 3, "conv128_2");

  const conv256_down = extractResidualLayerParams(589824, 256, 3, "conv256_down", true);
  const conv256_1 = extractResidualLayerParams(589824, 256, 3, "conv256_1");
  const conv256_2 = extractResidualLayerParams(589824, 256, 3, "conv256_2");
  const conv256_down_out = extractResidualLayerParams(589824, 256, 3, "conv256_down_out");

  const fc = tf.tidy(() => tf.transpose(tf.tensor2d(extractWeights(256 * 128), [128, 256]), [1, 0]));
  paramMappings.push({ paramPath: `fc` });

  if (getRemainingWeights().length !== 0) {
    throw new Error(`weights remaing after extract: ${getRemainingWeights().length}`);
  }

  const params = {
    conv32_down,
    conv32_1,
    conv32_2,
    conv32_3,
    conv64_down,
    conv64_1,
    conv64_2,
    conv64_3,
    conv128_down,
    conv128_1,
    conv128_2,
    conv256_down,
    conv256_1,
    conv256_2,
    conv256_down_out,
    fc
  };

  return { params, paramMappings };
}

function extractorsFactoryWeightMap(weightMap: any, paramMappings: ParamMapping[]) {
  const extractWeightEntry = extractWeightEntryFactory(weightMap, paramMappings);

  function extractScaleLayerParams(prefix: string): ScaleLayerParams {
    const weights = extractWeightEntry<tf.Tensor1D>(`${prefix}/scale/weights`, 1);
    const biases = extractWeightEntry<tf.Tensor1D>(`${prefix}/scale/biases`, 1);

    return { weights, biases };
  }

  function extractConvLayerParams(prefix: string): ConvLayerParams {
    const filters = extractWeightEntry<tf.Tensor4D>(`${prefix}/conv/filters`, 4);
    const bias = extractWeightEntry<tf.Tensor1D>(`${prefix}/conv/bias`, 1);
    const scale = extractScaleLayerParams(prefix);

    return { conv: { filters, bias }, scale };
  }

  function extractResidualLayerParams(prefix: string): ResidualLayerParams {
    return {
      conv1: extractConvLayerParams(`${prefix}/conv1`),
      conv2: extractConvLayerParams(`${prefix}/conv2`)
    };
  }

  return {
    extractConvLayerParams,
    extractResidualLayerParams
  };
}

export function extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap): { params: NetParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];

  const { extractConvLayerParams, extractResidualLayerParams } = extractorsFactoryWeightMap(weightMap, paramMappings);

  const conv32_down = extractConvLayerParams("conv32_down");
  const conv32_1 = extractResidualLayerParams("conv32_1");
  const conv32_2 = extractResidualLayerParams("conv32_2");
  const conv32_3 = extractResidualLayerParams("conv32_3");

  const conv64_down = extractResidualLayerParams("conv64_down");
  const conv64_1 = extractResidualLayerParams("conv64_1");
  const conv64_2 = extractResidualLayerParams("conv64_2");
  const conv64_3 = extractResidualLayerParams("conv64_3");

  const conv128_down = extractResidualLayerParams("conv128_down");
  const conv128_1 = extractResidualLayerParams("conv128_1");
  const conv128_2 = extractResidualLayerParams("conv128_2");

  const conv256_down = extractResidualLayerParams("conv256_down");
  const conv256_1 = extractResidualLayerParams("conv256_1");
  const conv256_2 = extractResidualLayerParams("conv256_2");
  const conv256_down_out = extractResidualLayerParams("conv256_down_out");

  const fc = weightMap["fc"];
  paramMappings.push({ originalPath: "fc", paramPath: "fc" });

  if (!isTensor2D(fc)) {
    throw new Error(`expected weightMap[fc] to be a Tensor2D, instead have ${fc}`);
  }

  const params = {
    conv32_down,
    conv32_1,
    conv32_2,
    conv32_3,
    conv64_down,
    conv64_1,
    conv64_2,
    conv64_3,
    conv128_down,
    conv128_1,
    conv128_2,
    conv256_down,
    conv256_1,
    conv256_2,
    conv256_down_out,
    fc
  };

  disposeUnusedWeightTensors(weightMap, paramMappings);

  return { params, paramMappings };
}

export function residual(x: tf.Tensor4D, params: ResidualLayerParams): tf.Tensor4D {
  let out = conv(x, params.conv1);
  out = convNoRelu(out, params.conv2);
  out = tf.add(out, x);
  out = tf.relu(out);
  return out;
}

export function residualDown(x: tf.Tensor4D, params: ResidualLayerParams): tf.Tensor4D {
  let out = convDown(x, params.conv1);
  out = convNoRelu(out, params.conv2);

  let pooled = tf.avgPool(x, 2, 2, "valid") as tf.Tensor4D;
  const zeros = tf.zeros<tf.Rank.R4>(pooled.shape);
  const isPad = pooled.shape[3] !== out.shape[3];
  const isAdjustShape = pooled.shape[1] !== out.shape[1] || pooled.shape[2] !== out.shape[2];

  if (isAdjustShape) {
    const padShapeX = [...out.shape] as [number, number, number, number];
    padShapeX[1] = 1;
    const zerosW = tf.zeros<tf.Rank.R4>(padShapeX);
    out = tf.concat([out, zerosW], 1);

    const padShapeY = [...out.shape] as [number, number, number, number];
    padShapeY[2] = 1;
    const zerosH = tf.zeros<tf.Rank.R4>(padShapeY);
    out = tf.concat([out, zerosH], 2);
  }

  pooled = isPad ? tf.concat([pooled, zeros], 3) : pooled;
  out = tf.add(pooled, out) as tf.Tensor4D;

  out = tf.relu(out);
  return out;
}
