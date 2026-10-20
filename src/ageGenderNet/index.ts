import * as tf from "@tensorflow/tfjs-core";

import { fullyConnectedLayer } from "../common";
import { seperateWeightMaps } from "../utils";
import { TinyXception } from "../xception";
import { NetOutput, NetParams } from "./types";
import { NeuralNetwork } from "../NeuralNetwork";
import { toNetInput } from "../dom";
import { NetInput } from "../dom/NetInput";
import { TNetInput } from "../dom/types";
import { Gender } from "../utils/config";
import { disposeUnusedWeightTensors, extractFCParamsFactory, extractWeightEntryFactory } from "../common";
import { FCParams } from "../common/types";
import { extractWeightsFactory } from "../utils";

export class AgeGenderNet extends NeuralNetwork<NetParams> {
  private _faceFeatureExtractor: TinyXception;

  constructor(faceFeatureExtractor: TinyXception = new TinyXception(2)) {
    super("AgeGenderNet");
    this._faceFeatureExtractor = faceFeatureExtractor;
  }

  public get faceFeatureExtractor(): TinyXception {
    return this._faceFeatureExtractor;
  }

  public runNet(input: NetInput | tf.Tensor4D): NetOutput {
    const { params } = this;

    if (!params) {
      throw new Error(`${this._name} - load model before inference`);
    }

    return tf.tidy(() => {
      const bottleneckFeatures = input instanceof NetInput ? this.faceFeatureExtractor.forwardInput(input) : input;

      const pooled = tf.reshape(tf.avgPool(bottleneckFeatures, [7, 7], [2, 2], "valid"), [bottleneckFeatures.shape[0], -1]) as tf.Tensor2D;
      const age = tf.reshape(fullyConnectedLayer(pooled, params.fc.age), [-1]) as tf.Tensor1D;
      const gender = fullyConnectedLayer(pooled, params.fc.gender);
      return { age, gender };
    });
  }

  public forwardInput(input: NetInput | tf.Tensor4D): NetOutput {
    return tf.tidy(() => {
      const { age, gender } = this.runNet(input);
      return { age, gender: tf.softmax(gender) };
    });
  }

  public async forward(input: TNetInput): Promise<NetOutput> {
    return this.forwardInput(await toNetInput(input));
  }

  public async predictAgeAndGender(input: TNetInput): Promise<AgeAndGenderPrediction | AgeAndGenderPrediction[]> {
    const netInput = await toNetInput(input);
    const out = this.forwardInput(netInput);

    const ages = tf.unstack(out.age);
    const genders = tf.unstack(out.gender);
    const ageAndGenderTensors = ages.map((ageTensor, i) => ({
      ageTensor,
      genderTensor: genders[i]
    }));

    const predictionsByBatch = await Promise.all(
      ageAndGenderTensors.map(async ({ ageTensor, genderTensor }) => {
        const age = (await ageTensor.data())[0];
        const probMale = (await genderTensor.data())[0];
        const isMale = probMale > 0.5;
        const gender = isMale ? Gender.MALE : Gender.FEMALE;
        const genderProbability = isMale ? probMale : 1 - probMale;

        ageTensor.dispose();
        genderTensor.dispose();
        return { age, gender, genderProbability };
      })
    );
    out.age.dispose();
    out.gender.dispose();

    return netInput.isBatchInput ? predictionsByBatch : predictionsByBatch[0];
  }

  protected getDefaultModelName(): string {
    return "age_gender_model";
  }

  public dispose(throwOnRedispose = true) {
    this.faceFeatureExtractor.dispose(throwOnRedispose);
    super.dispose(throwOnRedispose);
  }

  public loadClassifierParams(weights: Float32Array) {
    const { params, paramMappings } = this.extractClassifierParams(weights);
    this._params = params;
    this._paramMappings = paramMappings;
  }

  public extractClassifierParams(weights: Float32Array) {
    return extractParams(weights);
  }

  protected extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap) {
    const { featureExtractorMap, classifierMap } = seperateWeightMaps(weightMap);

    this.faceFeatureExtractor.loadFromWeightMap(featureExtractorMap);

    return extractParamsFromWeigthMap(classifierMap);
  }

  protected extractParams(weights: Float32Array) {
    const classifierWeightSize = 512 * 1 + 1 + (512 * 2 + 2);

    const featureExtractorWeights = weights.slice(0, weights.length - classifierWeightSize);
    const classifierWeights = weights.slice(weights.length - classifierWeightSize);

    this.faceFeatureExtractor.extractWeights(featureExtractorWeights);
    return this.extractClassifierParams(classifierWeights);
  }
}

export function extractParams(weights: Float32Array): { params: NetParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];
  const { extractWeights, getRemainingWeights } = extractWeightsFactory(weights);
  const extractFCParams = extractFCParamsFactory(extractWeights, paramMappings);

  const age = extractFCParams(512, 1, "fc/age");
  const gender = extractFCParams(512, 2, "fc/gender");

  if (getRemainingWeights().length !== 0) {
    throw new Error(`weights remaing after extract: ${getRemainingWeights().length}`);
  }
  return {
    paramMappings,
    params: { fc: { age, gender } }
  };
}

export function extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap): { params: NetParams; paramMappings: ParamMapping[] } {
  const paramMappings: ParamMapping[] = [];
  const extractWeightEntry = extractWeightEntryFactory(weightMap, paramMappings);
  function extractFcParams(prefix: string): FCParams {
    const weights = extractWeightEntry<tf.Tensor2D>(`${prefix}/weights`, 2);
    const bias = extractWeightEntry<tf.Tensor1D>(`${prefix}/bias`, 1);
    return { weights, bias };
  }
  const params = {
    fc: {
      age: extractFcParams("fc/age"),
      gender: extractFcParams("fc/gender")
    }
  };
  disposeUnusedWeightTensors(weightMap, paramMappings);
  return { params, paramMappings };
}
