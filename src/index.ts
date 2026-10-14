import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

import * as utils from "./utils";
import { DrawBox } from "./draw/DrawBox";
import { drawContour } from "./draw/drawContour";
import { drawDetections } from "./draw/drawDetections";
import { drawFaceExpressions } from "./draw/drawFaceExpressions";
import { DrawFaceLandmarks, drawFaceLandmarks } from "./draw/DrawFaceLandmarks";
import { DrawTextField } from "./draw/DrawTextField";

export const draw = {
  drawContour,
  drawDetections,
  drawFaceExpressions,
  drawFaceLandmarks,
  DrawBox,
  DrawFaceLandmarks,
  DrawTextField
};

export { utils, tf };

export * from "./ageGenderNet/AgeGenderNet";
export * from "./ageGenderNet/types";

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
export * from "./classes/Rect";

export * from "./dom/awaitMediaLoaded";
export * from "./dom/blobToImage";
export * from "./dom/createCanvas";
export * from "./dom/extractFaces";
export * from "./dom/extractFaceTensors";
export * from "./dom/fetchImage";
export * from "./dom/fetchJson";
export * from "./dom/fetchNetWeights";
export * from "./dom/fetchOrThrow";
export * from "./dom/getContext2dOrThrow";
export * from "./dom/getMediaDimensions";
export * from "./dom/imageTensorToCanvas";
export * from "./dom/imageToSquare";
export * from "./dom/isMediaElement";
export * from "./dom/isMediaLoaded";
export * from "./dom/loadWeightMap";
export * from "./dom/matchDimensions";
export * from "./dom/NetInput";
export * from "./dom/resolveInput";
export * from "./dom/toNetInput";
export * from "./dom/types";

export * from "./draw/drawContour";
export * from "./draw/drawDetections";
export * from "./draw/drawFaceExpressions";
export * from "./draw/DrawBox";
export * from "./draw/DrawFaceLandmarks";
export * from "./draw/DrawTextField";

export * from "./env";

export * from "./faceExpressionNet/FaceExpressionNet";
export * from "./faceExpressionNet/FaceExpressions";

export * from "./faceLandmarkNet/FaceLandmark68Net";
export * from "./faceLandmarkNet/FaceLandmark68TinyNet";

export * from "./faceRecognitionNet/FaceRecognitionNet";

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

export * from "./ops/iou";
export * from "./ops/minBbox";
export * from "./ops/nonMaxSuppression";

export * from "./ssdMobilenetv1/SsdMobilenetv1";
export * from "./ssdMobilenetv1/SsdMobilenetv1Options";

export * from "./tinyFaceDetector/TinyFaceDetector";
export * from "./tinyFaceDetector/TinyFaceDetectorOptions";

export * from "./tinyYolov2/config";
export * from "./tinyYolov2/TinyYolov2";
export * from "./tinyYolov2/TinyYolov2Options";
export * from "./tinyYolov2/types";

export * from "./utils";

export * from "./NeuralNetwork";
export * from "./resizeResults";
