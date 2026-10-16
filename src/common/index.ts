import * as tf from "@tensorflow/tfjs-core";

import { ConvParams, FCParams, SeparableConvParams } from "./types";
import { isTensor } from "../utils";

export function convLayer(x: tf.Tensor4D, params: ConvParams, padding: "valid" | "same" = "same", withRelu = false): tf.Tensor4D {
  return tf.tidy(() => {
    const out = tf.add<tf.Tensor4D>(tf.conv2d(x, params.filters, [1, 1], padding), params.bias);
    return withRelu ? tf.relu(out) : out;
  });
}

export function depthwiseSeparableConv(x: tf.Tensor4D, params: SeparableConvParams, stride: [number, number]): tf.Tensor4D {
  return tf.tidy(() => {
    let out = tf.separableConv2d(x, params.depthwise_filter, params.pointwise_filter, stride, "same");
    out = tf.add(out, params.bias);
    return out;
  });
}

export function disposeUnusedWeightTensors(weightMap: tf.NamedTensorMap, paramMappings: ParamMapping[]) {
  Object.keys(weightMap).forEach((path) => {
    if (!paramMappings.some((pm) => pm.originalPath === path)) {
      weightMap[path].dispose();
    }
  });
}

export function extractConvParamsFactory(extractWeights: ExtractWeightsFunction, paramMappings: ParamMapping[]) {
  return function (channelsIn: number, channelsOut: number, filterSize: number, mappedPrefix: string): ConvParams {
    const filters = tf.tensor4d(extractWeights(channelsIn * channelsOut * filterSize * filterSize), [filterSize, filterSize, channelsIn, channelsOut]);
    const bias = tf.tensor1d(extractWeights(channelsOut));

    paramMappings.push({ paramPath: `${mappedPrefix}/filters` }, { paramPath: `${mappedPrefix}/bias` });

    return { filters, bias };
  };
}

export function extractFCParamsFactory(extractWeights: ExtractWeightsFunction, paramMappings: ParamMapping[]) {
  return function (channelsIn: number, channelsOut: number, mappedPrefix: string): FCParams {
    const fc_weights = tf.tensor2d(extractWeights(channelsIn * channelsOut), [channelsIn, channelsOut]);
    const fc_bias = tf.tensor1d(extractWeights(channelsOut));

    paramMappings.push({ paramPath: `${mappedPrefix}/weights` }, { paramPath: `${mappedPrefix}/bias` });

    return {
      weights: fc_weights,
      bias: fc_bias
    };
  };
}

export function extractSeparableConvParamsFactory(extractWeights: ExtractWeightsFunction, paramMappings: ParamMapping[]) {
  return function (channelsIn: number, channelsOut: number, mappedPrefix: string): SeparableConvParams {
    const depthwise_filter = tf.tensor4d(extractWeights(3 * 3 * channelsIn), [3, 3, channelsIn, 1]);
    const pointwise_filter = tf.tensor4d(extractWeights(channelsIn * channelsOut), [1, 1, channelsIn, channelsOut]);
    const bias = tf.tensor1d(extractWeights(channelsOut));

    paramMappings.push(
      { paramPath: `${mappedPrefix}/depthwise_filter` },
      { paramPath: `${mappedPrefix}/pointwise_filter` },
      { paramPath: `${mappedPrefix}/bias` }
    );

    return new SeparableConvParams(depthwise_filter, pointwise_filter, bias);
  };
}

export function loadSeparableConvParamsFactory(extractWeightEntry: <T>(originalPath: string, paramRank: number) => T) {
  return function (prefix: string): SeparableConvParams {
    const depthwise_filter = extractWeightEntry<tf.Tensor4D>(`${prefix}/depthwise_filter`, 4);
    const pointwise_filter = extractWeightEntry<tf.Tensor4D>(`${prefix}/pointwise_filter`, 4);
    const bias = extractWeightEntry<tf.Tensor1D>(`${prefix}/bias`, 1);

    return new SeparableConvParams(depthwise_filter, pointwise_filter, bias);
  };
}

export function extractWeightEntryFactory(weightMap: any, paramMappings: ParamMapping[]) {
  return function <T>(originalPath: string, paramRank: number, mappedPath?: string): T {
    const tensor = weightMap[originalPath];

    if (!isTensor(tensor, paramRank)) {
      throw new Error(`expected weightMap[${originalPath}] to be a Tensor${paramRank}D, instead have ${tensor}`);
    }

    paramMappings.push({ originalPath, paramPath: mappedPath || originalPath });

    return tensor;
  };
}

export function loadConvParamsFactory(extractWeightEntry: <T>(originalPath: string, paramRank: number) => T) {
  return function (prefix: string): ConvParams {
    const filters = extractWeightEntry<tf.Tensor4D>(`${prefix}/filters`, 4);
    const bias = extractWeightEntry<tf.Tensor1D>(`${prefix}/bias`, 1);

    return { filters, bias };
  };
}

export function fullyConnectedLayer(x: tf.Tensor2D, params: FCParams): tf.Tensor2D {
  return tf.tidy(() => tf.add(tf.matMul(x, params.weights), params.bias));
}
