import * as tf from "@tensorflow/tfjs-core";

import { ConvParams, SeparableConvParams } from "../common/types";
import {
  depthwiseSeparableConv,
  disposeUnusedWeightTensors,
  extractConvParamsFactory,
  extractSeparableConvParamsFactory,
  extractWeightEntryFactory,
  loadSeparableConvParamsFactory,
  loadConvParamsFactory
} from "../common";
import { DenseBlock3Params, DenseBlock4Params } from "./types";
import { extractWeightsFactory, opsNormalize } from "../utils";
import { FaceFeatureExtractorParams, TinyFaceFeatureExtractorParams, IFaceFeatureExtractor } from "./types";
import { toNetInput } from "../dom";
import { NetInput } from "../dom/NetInput";
import { NeuralNetwork } from "../NeuralNetwork";
import { TNetInput } from "../dom/types";

export function denseBlock3(x: tf.Tensor4D, denseBlockParams: DenseBlock3Params, isFirstLayer = false) {
  return tf.tidy(() => {
    const out1 = tf.relu(
      isFirstLayer
        ? tf.add(tf.conv2d(x, (denseBlockParams.conv0 as ConvParams).filters, [2, 2], "same"), denseBlockParams.conv0.bias)
        : depthwiseSeparableConv(x, denseBlockParams.conv0 as SeparableConvParams, [2, 2])
    ) as tf.Tensor4D;
    const out2 = depthwiseSeparableConv(out1, denseBlockParams.conv1, [1, 1]) as tf.Tensor4D;

    const in3 = tf.relu(tf.add(out1, out2)) as tf.Tensor4D;
    const out3 = depthwiseSeparableConv(in3, denseBlockParams.conv2, [1, 1]);

    return tf.relu(tf.add(out1, tf.add(out2, out3))) as tf.Tensor4D;
  });
}

export function denseBlock4(x: tf.Tensor4D, denseBlockParams: DenseBlock4Params, isFirstLayer = false, isScaleDown = true) {
  return tf.tidy(() => {
    const out1 = tf.relu(
      isFirstLayer
        ? tf.add(tf.conv2d(x, (denseBlockParams.conv0 as ConvParams).filters, isScaleDown ? [2, 2] : [1, 1], "same"), denseBlockParams.conv0.bias)
        : depthwiseSeparableConv(x, denseBlockParams.conv0 as SeparableConvParams, isScaleDown ? [2, 2] : [1, 1])
    ) as tf.Tensor4D;
    const out2 = depthwiseSeparableConv(out1, denseBlockParams.conv1, [1, 1]);

    const in3 = tf.relu(tf.add(out1, out2)) as tf.Tensor4D;
    const out3 = depthwiseSeparableConv(in3, denseBlockParams.conv2, [1, 1]);

    const in4 = tf.relu(tf.add(out1, tf.add(out2, out3))) as tf.Tensor4D;
    const out4 = depthwiseSeparableConv(in4, denseBlockParams.conv3, [1, 1]);

    return tf.relu(tf.add(out1, tf.add(out2, tf.add(out3, out4)))) as tf.Tensor4D;
  });
}

export function extractorsFactory(extractWeights: ExtractWeightsFunction, paramMappings: ParamMapping[]) {
  const extractConvParams = extractConvParamsFactory(extractWeights, paramMappings);
  const extractSeparableConvParams = extractSeparableConvParamsFactory(extractWeights, paramMappings);

  function extractDenseBlock3Params(channelsIn: number, channelsOut: number, mappedPrefix: string, isFirstLayer = false): DenseBlock3Params {
    const conv0 = isFirstLayer
      ? extractConvParams(channelsIn, channelsOut, 3, `${mappedPrefix}/conv0`)
      : extractSeparableConvParams(channelsIn, channelsOut, `${mappedPrefix}/conv0`);
    const conv1 = extractSeparableConvParams(channelsOut, channelsOut, `${mappedPrefix}/conv1`);
    const conv2 = extractSeparableConvParams(channelsOut, channelsOut, `${mappedPrefix}/conv2`);

    return { conv0, conv1, conv2 };
  }

  function extractDenseBlock4Params(channelsIn: number, channelsOut: number, mappedPrefix: string, isFirstLayer = false): DenseBlock4Params {
    const { conv0, conv1, conv2 } = extractDenseBlock3Params(channelsIn, channelsOut, mappedPrefix, isFirstLayer);
    const conv3 = extractSeparableConvParams(channelsOut, channelsOut, `${mappedPrefix}/conv3`);

    return { conv0, conv1, conv2, conv3 };
  }

  return {
    extractDenseBlock3Params,
    extractDenseBlock4Params
  };
}

export function extractParams(weights: Float32Array): { params: FaceFeatureExtractorParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];

  const { extractWeights, getRemainingWeights } = extractWeightsFactory(weights);
  const { extractDenseBlock4Params } = extractorsFactory(extractWeights, paramMappings);

  const dense0 = extractDenseBlock4Params(3, 32, "dense0", true);
  const dense1 = extractDenseBlock4Params(32, 64, "dense1");
  const dense2 = extractDenseBlock4Params(64, 128, "dense2");
  const dense3 = extractDenseBlock4Params(128, 256, "dense3");

  if (getRemainingWeights().length !== 0) {
    throw new Error(`weights remaing after extract: ${getRemainingWeights().length}`);
  }

  return {
    paramMappings,
    params: { dense0, dense1, dense2, dense3 }
  };
}

export function extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap): { params: FaceFeatureExtractorParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];

  const { extractDenseBlock4Params } = loadParamsFactory(weightMap, paramMappings);

  const params = {
    dense0: extractDenseBlock4Params("dense0", true),
    dense1: extractDenseBlock4Params("dense1"),
    dense2: extractDenseBlock4Params("dense2"),
    dense3: extractDenseBlock4Params("dense3")
  };

  disposeUnusedWeightTensors(weightMap, paramMappings);

  return { params, paramMappings };
}

export function extractParamsFromWeigthMapTiny(weightMap: tf.NamedTensorMap): { params: TinyFaceFeatureExtractorParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];

  const { extractDenseBlock3Params } = loadParamsFactory(weightMap, paramMappings);

  const params = {
    dense0: extractDenseBlock3Params("dense0", true),
    dense1: extractDenseBlock3Params("dense1"),
    dense2: extractDenseBlock3Params("dense2")
  };

  disposeUnusedWeightTensors(weightMap, paramMappings);

  return { params, paramMappings };
}

export function extractParamsTiny(weights: Float32Array): { params: TinyFaceFeatureExtractorParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];

  const { extractWeights, getRemainingWeights } = extractWeightsFactory(weights);

  const { extractDenseBlock3Params } = extractorsFactory(extractWeights, paramMappings);

  const dense0 = extractDenseBlock3Params(3, 32, "dense0", true);
  const dense1 = extractDenseBlock3Params(32, 64, "dense1");
  const dense2 = extractDenseBlock3Params(64, 128, "dense2");

  if (getRemainingWeights().length !== 0) {
    throw new Error(`weights remaing after extract: ${getRemainingWeights().length}`);
  }

  return {
    paramMappings,
    params: { dense0, dense1, dense2 }
  };
}

export class FaceFeatureExtractor extends NeuralNetwork<FaceFeatureExtractorParams> implements IFaceFeatureExtractor<FaceFeatureExtractorParams> {
  constructor() {
    super("FaceFeatureExtractor");
  }

  public forwardInput(input: NetInput): tf.Tensor4D {
    const { params } = this;

    if (!params) {
      throw new Error("FaceFeatureExtractor - load model before inference");
    }

    return tf.tidy(() => {
      const batchTensor = input.toBatchTensor(112, true);
      const meanRgb = [122.782, 117.001, 104.298];
      const normalized = tf.div(opsNormalize(batchTensor, meanRgb), tf.scalar(255)) as tf.Tensor4D;

      let out = denseBlock4(normalized, params.dense0, true);
      out = denseBlock4(out, params.dense1);
      out = denseBlock4(out, params.dense2);
      out = denseBlock4(out, params.dense3);
      out = tf.avgPool(out, [7, 7], [2, 2], "valid");

      return out;
    });
  }

  public async forward(input: TNetInput): Promise<tf.Tensor4D> {
    return this.forwardInput(await toNetInput(input));
  }

  protected getDefaultModelName(): string {
    return "face_feature_extractor_model";
  }

  protected extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeigthMap(weightMap);
  }

  protected extractParams(weights: Float32Array) {
    return extractParams(weights);
  }
}

export function loadParamsFactory(weightMap: any, paramMappings: ParamMapping[]) {
  const extractWeightEntry = extractWeightEntryFactory(weightMap, paramMappings);

  const extractConvParams = loadConvParamsFactory(extractWeightEntry);
  const extractSeparableConvParams = loadSeparableConvParamsFactory(extractWeightEntry);

  function extractDenseBlock3Params(prefix: string, isFirstLayer = false): DenseBlock3Params {
    const conv0 = isFirstLayer ? extractConvParams(`${prefix}/conv0`) : extractSeparableConvParams(`${prefix}/conv0`);
    const conv1 = extractSeparableConvParams(`${prefix}/conv1`);
    const conv2 = extractSeparableConvParams(`${prefix}/conv2`);

    return { conv0, conv1, conv2 };
  }

  function extractDenseBlock4Params(prefix: string, isFirstLayer = false): DenseBlock4Params {
    const conv0 = isFirstLayer ? extractConvParams(`${prefix}/conv0`) : extractSeparableConvParams(`${prefix}/conv0`);
    const conv1 = extractSeparableConvParams(`${prefix}/conv1`);
    const conv2 = extractSeparableConvParams(`${prefix}/conv2`);
    const conv3 = extractSeparableConvParams(`${prefix}/conv3`);

    return { conv0, conv1, conv2, conv3 };
  }

  return {
    extractDenseBlock3Params,
    extractDenseBlock4Params
  };
}

export class TinyFaceFeatureExtractor extends NeuralNetwork<TinyFaceFeatureExtractorParams> implements IFaceFeatureExtractor<TinyFaceFeatureExtractorParams> {
  constructor() {
    super("TinyFaceFeatureExtractor");
  }

  public forwardInput(input: NetInput): tf.Tensor4D {
    const { params } = this;

    if (!params) throw new Error("TinyFaceFeatureExtractor - load model before inference");

    return tf.tidy(() => {
      const batchTensor = input.toBatchTensor(112, true);
      const meanRgb = [122.782, 117.001, 104.298];
      const normalized = tf.div(opsNormalize(batchTensor, meanRgb), tf.scalar(255)) as tf.Tensor4D;

      let out = denseBlock3(normalized, params.dense0, true);
      out = denseBlock3(out, params.dense1);
      out = denseBlock3(out, params.dense2);
      out = tf.avgPool(out, [14, 14], [2, 2], "valid");

      return out;
    });
  }

  public async forward(input: TNetInput): Promise<tf.Tensor4D> {
    return this.forwardInput(await toNetInput(input));
  }

  protected getDefaultModelName(): string {
    return "face_feature_extractor_tiny_model";
  }

  protected extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeigthMapTiny(weightMap);
  }

  protected extractParams(weights: Float32Array) {
    return extractParamsTiny(weights);
  }
}
