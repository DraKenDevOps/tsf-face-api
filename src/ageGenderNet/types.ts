import * as tf from "@tensorflow/tfjs-core";
import { FCParams } from "../common/types";

export type NetOutput = { age: tf.Tensor1D; gender: tf.Tensor2D };
export type NetParams = {
  fc: {
    age: FCParams;
    gender: FCParams;
  };
};
