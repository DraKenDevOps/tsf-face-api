import * as tf from "@tensorflow/tfjs-core";

import {
  depthwiseSeparableConv,
  disposeUnusedWeightTensors,
  loadSeparableConvParamsFactory,
  extractWeightEntryFactory,
  loadConvParamsFactory,
  extractConvParamsFactory,
  extractSeparableConvParamsFactory
} from "../common";
import { ConvParams } from "../common/types";
import { toNetInput } from "../dom";
import { NetInput } from "../dom/NetInput";
import { TNetInput } from "../dom/types";
import { NeuralNetwork } from "../NeuralNetwork";
import { opsNormalize, range, extractWeightsFactory } from "../utils";
import { MainBlockParams, ReductionBlockParams, TinyXceptionParams } from "./types";

function conv(x: tf.Tensor4D, params: ConvParams, stride: [number, number]): tf.Tensor4D {
  return tf.add(tf.conv2d(x, params.filters, stride, "same"), params.bias);
}

function reductionBlock(x: tf.Tensor4D, params: ReductionBlockParams, isActivateInput = true): tf.Tensor4D {
  let out = isActivateInput ? tf.relu(x) : x;
  out = depthwiseSeparableConv(out, params.separable_conv0, [1, 1]);
  out = depthwiseSeparableConv(tf.relu(out), params.separable_conv1, [1, 1]);
  out = tf.maxPool(out, [3, 3], [2, 2], "same");
  out = tf.add(out, conv(x, params.expansion_conv, [2, 2]));
  return out;
}

function mainBlock(x: tf.Tensor4D, params: MainBlockParams): tf.Tensor4D {
  let out = depthwiseSeparableConv(tf.relu(x), params.separable_conv0, [1, 1]);
  out = depthwiseSeparableConv(tf.relu(out), params.separable_conv1, [1, 1]);
  out = depthwiseSeparableConv(tf.relu(out), params.separable_conv2, [1, 1]);
  out = tf.add(out, x);
  return out;
}

function extractorsFactory(extractWeights: ExtractWeightsFunction, paramMappings: ParamMapping[]) {
  const extractConvParams = extractConvParamsFactory(extractWeights, paramMappings);
  const extractSeparableConvParams = extractSeparableConvParamsFactory(extractWeights, paramMappings);

  function extractReductionBlockParams(channelsIn: number, channelsOut: number, mappedPrefix: string): ReductionBlockParams {
    const separable_conv0 = extractSeparableConvParams(channelsIn, channelsOut, `${mappedPrefix}/separable_conv0`);
    const separable_conv1 = extractSeparableConvParams(channelsOut, channelsOut, `${mappedPrefix}/separable_conv1`);
    const expansion_conv = extractConvParams(channelsIn, channelsOut, 1, `${mappedPrefix}/expansion_conv`);

    return { separable_conv0, separable_conv1, expansion_conv };
  }

  function extractMainBlockParams(channels: number, mappedPrefix: string): MainBlockParams {
    const separable_conv0 = extractSeparableConvParams(channels, channels, `${mappedPrefix}/separable_conv0`);
    const separable_conv1 = extractSeparableConvParams(channels, channels, `${mappedPrefix}/separable_conv1`);
    const separable_conv2 = extractSeparableConvParams(channels, channels, `${mappedPrefix}/separable_conv2`);

    return { separable_conv0, separable_conv1, separable_conv2 };
  }

  return {
    extractConvParams,
    extractSeparableConvParams,
    extractReductionBlockParams,
    extractMainBlockParams
  };
}

function loadParamsFactoryWeightMap(weightMap: any, paramMappings: ParamMapping[]) {
  const extractWeightEntry = extractWeightEntryFactory(weightMap, paramMappings);

  const extractConvParams = loadConvParamsFactory(extractWeightEntry);
  const extractSeparableConvParams = loadSeparableConvParamsFactory(extractWeightEntry);

  function extractReductionBlockParams(mappedPrefix: string): ReductionBlockParams {
    const separable_conv0 = extractSeparableConvParams(`${mappedPrefix}/separable_conv0`);
    const separable_conv1 = extractSeparableConvParams(`${mappedPrefix}/separable_conv1`);
    const expansion_conv = extractConvParams(`${mappedPrefix}/expansion_conv`);

    return { separable_conv0, separable_conv1, expansion_conv };
  }

  function extractMainBlockParams(mappedPrefix: string): MainBlockParams {
    const separable_conv0 = extractSeparableConvParams(`${mappedPrefix}/separable_conv0`);
    const separable_conv1 = extractSeparableConvParams(`${mappedPrefix}/separable_conv1`);
    const separable_conv2 = extractSeparableConvParams(`${mappedPrefix}/separable_conv2`);

    return { separable_conv0, separable_conv1, separable_conv2 };
  }

  return {
    extractConvParams,
    extractSeparableConvParams,
    extractReductionBlockParams,
    extractMainBlockParams
  };
}

export function extractParams(weights: Float32Array, numMainBlocks: number): { params: TinyXceptionParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];

  const { extractWeights, getRemainingWeights } = extractWeightsFactory(weights);

  const { extractConvParams, extractSeparableConvParams, extractReductionBlockParams, extractMainBlockParams } = extractorsFactory(
    extractWeights,
    paramMappings
  );

  const entry_flow_conv_in = extractConvParams(3, 32, 3, "entry_flow/conv_in");
  const entry_flow_reduction_block_0 = extractReductionBlockParams(32, 64, "entry_flow/reduction_block_0");
  const entry_flow_reduction_block_1 = extractReductionBlockParams(64, 128, "entry_flow/reduction_block_1");

  const entry_flow = {
    conv_in: entry_flow_conv_in,
    reduction_block_0: entry_flow_reduction_block_0,
    reduction_block_1: entry_flow_reduction_block_1
  };

  const middle_flow: Record<string, MainBlockParams> = {};
  range(numMainBlocks, 0, 1).forEach((idx) => {
    middle_flow[`main_block_${idx}`] = extractMainBlockParams(128, `middle_flow/main_block_${idx}`);
  });

  const exit_flow_reduction_block = extractReductionBlockParams(128, 256, "exit_flow/reduction_block");
  const exit_flow_separable_conv = extractSeparableConvParams(256, 512, "exit_flow/separable_conv");

  const exit_flow = {
    reduction_block: exit_flow_reduction_block,
    separable_conv: exit_flow_separable_conv
  };

  if (getRemainingWeights().length !== 0) {
    throw new Error(`weights remaing after extract: ${getRemainingWeights().length}`);
  }

  return {
    paramMappings,
    params: { entry_flow, middle_flow, exit_flow }
  };
}

export function extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap, numMainBlocks: number): { params: TinyXceptionParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];

  const { extractConvParams, extractSeparableConvParams, extractReductionBlockParams, extractMainBlockParams } = loadParamsFactoryWeightMap(
    weightMap,
    paramMappings
  );

  const entry_flow_conv_in = extractConvParams("entry_flow/conv_in");
  const entry_flow_reduction_block_0 = extractReductionBlockParams("entry_flow/reduction_block_0");
  const entry_flow_reduction_block_1 = extractReductionBlockParams("entry_flow/reduction_block_1");

  const entry_flow = {
    conv_in: entry_flow_conv_in,
    reduction_block_0: entry_flow_reduction_block_0,
    reduction_block_1: entry_flow_reduction_block_1
  };

  const middle_flow: Record<string, MainBlockParams> = {};
  range(numMainBlocks, 0, 1).forEach((idx) => {
    middle_flow[`main_block_${idx}`] = extractMainBlockParams(`middle_flow/main_block_${idx}`);
  });

  const exit_flow_reduction_block = extractReductionBlockParams("exit_flow/reduction_block");
  const exit_flow_separable_conv = extractSeparableConvParams("exit_flow/separable_conv");

  const exit_flow = {
    reduction_block: exit_flow_reduction_block,
    separable_conv: exit_flow_separable_conv
  };

  disposeUnusedWeightTensors(weightMap, paramMappings);

  return { params: { entry_flow, middle_flow, exit_flow }, paramMappings };
}

export class TinyXception extends NeuralNetwork<TinyXceptionParams> {
  private _numMainBlocks: number;

  constructor(numMainBlocks: number) {
    super("TinyXception");
    this._numMainBlocks = numMainBlocks;
  }

  public forwardInput(input: NetInput): tf.Tensor4D {
    const { params } = this;

    if (!params) {
      throw new Error("TinyXception - load model before inference");
    }

    return tf.tidy(() => {
      const batchTensor = input.toBatchTensor(112, true);
      const meanRgb = [122.782, 117.001, 104.298];
      const normalized = tf.div(opsNormalize(batchTensor, meanRgb), tf.scalar(256)) as tf.Tensor4D;

      let out = tf.relu(conv(normalized, params.entry_flow.conv_in, [2, 2]));
      out = reductionBlock(out, params.entry_flow.reduction_block_0, false);
      out = reductionBlock(out, params.entry_flow.reduction_block_1);

      range(this._numMainBlocks, 0, 1).forEach((idx) => {
        out = mainBlock(out, params.middle_flow[`main_block_${idx}`]);
      });

      out = reductionBlock(out, params.exit_flow.reduction_block);
      out = tf.relu(depthwiseSeparableConv(out, params.exit_flow.separable_conv, [1, 1]));
      return out;
    });
  }

  public async forward(input: TNetInput): Promise<tf.Tensor4D> {
    return this.forwardInput(await toNetInput(input));
  }

  protected getDefaultModelName(): string {
    return "tiny_xception_model";
  }

  protected extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap) {
    return extractParamsFromWeigthMap(weightMap, this._numMainBlocks);
  }

  protected extractParams(weights: Float32Array) {
    return extractParams(weights, this._numMainBlocks);
  }
}
