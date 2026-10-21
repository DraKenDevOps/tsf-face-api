import * as faceapi from "../../src";
// import * as tf from "@tensorflow/tfjs-node"
import canvas from "canvas";
// import fs from "fs";
import path from "path";

faceapi.env.monkeyPatch({
  Canvas: canvas.Canvas,
  Image: canvas.Image,
  ImageData: canvas.ImageData,
  createCanvasElement: () => canvas.createCanvas(0, 0),
  createImageElement: () => new canvas.Image()
});
// const faceDetectionNet = faceapi.nets.ssdMobilenetv1;
const faceDetectionNet = faceapi.nets.tinyFaceDetector;

// SsdMobilenetv1Options
const minConfidence = 0.5;
// TinyFaceDetectorOptions
const inputSize = 416;
const scoreThreshold = 0.5;

function getFaceDetectorOptions(net: faceapi.NeuralNetwork<any>) {
  return net === faceapi.nets.ssdMobilenetv1
    ? new faceapi.SsdMobilenetv1Options({ minConfidence })
    : new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold });
}

const angry = path.resolve(process.cwd() + "/tests/images/angry.jpg");
const faces = path.resolve(process.cwd() + "/tests/images/faces.jpg");

const modelPath = path.resolve(process.cwd() + "/weights");
(async () => {
  try {
    await import("@tensorflow/tfjs-node-gpu");
  } catch {
    console.warn("No GPU (T_T)");
    try {
      console.warn("Use CPU (^_^)");
      await import("@tensorflow/tfjs-node");
    } catch {
      // console.error(err);
      // Fallback to CPU backend
      console.log("Fallback to CPU backend");
    }
  }

  await faceDetectionNet.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);

  // const imgBuffer = fs.readFileSync(angry);
  // const angryTensor = tf.node.decodeImage(imgBuffer, 3) as tf.Tensor3D;

  const angryImg = await canvas.loadImage(angry);
  // const facesImg = await canvas.loadImage(faces);
  const faceDetectionOptions = getFaceDetectorOptions(faceDetectionNet);

  console.time("recognize");
  const single = await faceapi.detectSingleFace(angryImg, faceDetectionOptions).withFaceLandmarks(true).withFaceDescriptor();
  // const all = await faceapi.detectAllFaces(facesImg, faceDetectionOptions).withFaceLandmarks().withFaceDescriptors();
  console.log(single?.detection.score);
  console.log("single", JSON.stringify(Array.from(single?.descriptor ?? [])));
  // console.log("all", all.map((f) => Array.from(f.descriptor)));
  console.timeEnd("recognize");
  // angryTensor.dispose();
})();

import descriptors from "./descriptors.json";
const distance = faceapi.euclideanDistance(descriptors[0], descriptors[1]);
const cosim = faceapi.cosineSimilarity(descriptors[0], descriptors[1]);

console.log("distance", distance);
console.log("same face", distance < 0.6);
console.log("cosim", cosim);
console.log("same face", cosim > 0.6);
