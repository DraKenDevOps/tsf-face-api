/// <reference path="./utils/types.d.ts" />
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

import * as utils from "./utils";
import * as draw from "./draw";

export { utils, draw, tf };

export * from "./ageGenderNet";
export * from "./ageGenderNet/types";
export * from "./dom";
export * from "./dom/types";
export * from "./env";

export * from "./classes/BoundingBox";
export * from "./classes/Box";
export * from "./classes/Dimensions";
export * from "./classes/FaceDetection";
export * from "./classes/FaceLandmarks";
export * from "./classes/FaceLandmarks5";
export * from "./classes/FaceLandmarks68";
export * from "./classes/FaceMatch";
export * from "./classes/LabeledBox";
export * from "./classes/LabeledFaceDescriptors";
export * from "./classes/ObjectDetection";
export * from "./classes/Point";
export * from "./classes/PredictedBox";
export * from "./classes/Rectangle";

export * from "./faceExpressionNet/FaceExpressionNet";
export * from "./faceExpressionNet/FaceExpressions";

export * from "./faceLandmarkNet";
export * from "./factories/WithFaceDetection";
export * from "./factories/WithFaceExpressions";
export * from "./factories/WithFaceLandmarks";
export * from "./factories/WithGender";

export * from "./globalApi/allFaces";
export * from "./globalApi/ComposableTask";
export * from "./globalApi/ComputeFaceDescriptorsTasks";
export * from "./globalApi/detectFaces";
export * from "./globalApi/DetectFacesTasks";
export * from "./globalApi/DetectFaceLandmarksTasks";
export * from "./globalApi/FaceMatcher";
export * from "./globalApi/nets";
export * from "./globalApi/types";

export * from "./mtcnn/Mtcnn";
export * from "./mtcnn/MtcnnOptions";

export * from "./ops";
export * from "./ssdMobilenetv1/SsdMobilenetv1";
export * from "./ssdMobilenetv1/SsdMobilenetv1Options";

export * from "./tinyFaceDetector";

export * from "./tinyYolov2/config";
export * from "./tinyYolov2/TinyYolov2";
export * from "./tinyYolov2/TinyYolov2Options";
export * from "./tinyYolov2/types";

export * from "./utils";

export * from "./NeuralNetwork";
export * from "./resizeResults";
