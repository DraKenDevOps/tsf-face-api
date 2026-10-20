import * as tf from "@tensorflow/tfjs-core";

import { convLayer, fullyConnectedLayer } from "../common";
import { prelu } from "../utils";
import { sharedLayer } from "./functions";
import { ONetParams, PNetParams, RNetParams } from "./types";

export function ONet(x: tf.Tensor4D, params: ONetParams): { scores: tf.Tensor1D; regions: tf.Tensor2D; points: tf.Tensor2D } {
  return tf.tidy(() => {
    let out = sharedLayer(x, params);
    out = tf.maxPool(out, [2, 2], [2, 2], "same");
    out = convLayer(out, params.conv4, "valid");
    out = prelu<tf.Tensor4D>(out, params.prelu4_alpha);

    const vectorized = tf.reshape(out, [out.shape[0], params.fc1.weights.shape[0]]) as tf.Tensor2D;
    const fc1 = fullyConnectedLayer(vectorized, params.fc1);
    const prelu5 = prelu<tf.Tensor2D>(fc1, params.prelu5_alpha);
    const fc2_1 = fullyConnectedLayer(prelu5, params.fc2_1);
    const max = tf.expandDims(tf.max(fc2_1, 1), 1);

    const prob = tf.softmax(tf.sub(fc2_1, max), 1) as tf.Tensor2D;
    const regions = fullyConnectedLayer(prelu5, params.fc2_2);
    const points = fullyConnectedLayer(prelu5, params.fc2_3);

    const scores = tf.unstack(prob, 1)[1] as tf.Tensor1D;
    return { scores, regions, points };
  });
}

export function PNet(x: tf.Tensor4D, params: PNetParams): { prob: tf.Tensor4D; regions: tf.Tensor4D } {
  return tf.tidy(() => {
    let out = sharedLayer(x, params, true);
    const conv = convLayer(out, params.conv4_1, "valid");
    const max = tf.expandDims(tf.max(conv, 3), 3);
    const prob = tf.softmax(tf.sub(conv, max), 3) as tf.Tensor4D;
    const regions = convLayer(out, params.conv4_2, "valid");

    return { prob, regions };
  });
}

export function RNet(x: tf.Tensor4D, params: RNetParams): { scores: tf.Tensor1D; regions: tf.Tensor2D } {
  return tf.tidy(() => {
    const convOut = sharedLayer(x, params);
    const vectorized = tf.reshape(convOut, [convOut.shape[0], params.fc1.weights.shape[0]]) as tf.Tensor2D;
    const fc1 = fullyConnectedLayer(vectorized, params.fc1);
    const prelu4 = prelu<tf.Tensor2D>(fc1, params.prelu4_alpha);
    const fc2_1 = fullyConnectedLayer(prelu4, params.fc2_1);
    const max = tf.expandDims(tf.max(fc2_1, 1), 1);
    const prob = tf.softmax(tf.sub(fc2_1, max), 1) as tf.Tensor2D;
    const regions = fullyConnectedLayer(prelu4, params.fc2_2);

    const scores = tf.unstack(prob, 1)[1] as tf.Tensor1D;
    return { scores, regions };
  });
}