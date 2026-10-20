
import * as tf from "@tensorflow/tfjs-core";

import { convLayer } from "../common";
import { prelu } from "../utils";
import { SharedParams } from "./types";
import { CELL_SIZE } from "../utils/config";

export function pyramidDown(minFaceSize: number, scaleFactor: number, dims: number[]): number[] {
  const [height, width] = dims;
  const m = CELL_SIZE / minFaceSize;

  const scales = [];

  let minLayer = Math.min(height, width) * m;
  let exp = 0;
  while (minLayer >= 12) {
    scales.push(m * Math.pow(scaleFactor, exp));
    minLayer = minLayer * scaleFactor;
    exp += 1;
  }
  return scales;
}

export function sharedLayer(x: tf.Tensor4D, params: SharedParams, isPnet = false) {
  return tf.tidy(() => {
    let out = convLayer(x, params.conv1, "valid");
    out = prelu<tf.Tensor4D>(out, params.prelu1_alpha);
    out = tf.maxPool(out, isPnet ? [2, 2] : [3, 3], [2, 2], "same");
    out = convLayer(out, params.conv2, "valid");
    out = prelu<tf.Tensor4D>(out, params.prelu2_alpha);
    out = isPnet ? out : tf.maxPool(out, [3, 3], [2, 2], "valid");
    out = convLayer(out, params.conv3, "valid");
    out = prelu<tf.Tensor4D>(out, params.prelu3_alpha);

    return out;
  });
}
