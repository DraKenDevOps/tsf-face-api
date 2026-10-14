import * as tf from "@tensorflow/tfjs-core";

import { ParamMapping } from "./types";

export function disposeUnusedWeightTensors(weightMap: tf.NamedTensorMap, paramMappings: ParamMapping[]) {
  Object.keys(weightMap).forEach((path) => {
    if (!paramMappings.some((pm) => pm.originalPath === path)) {
      weightMap[path].dispose();
    }
  });
}
