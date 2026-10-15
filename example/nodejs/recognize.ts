import * as faceapi from "../../src";
import canvas from "canvas";
import path from "path";

faceapi.env.monkeyPatch({
  Canvas: canvas.Canvas,
  Image: canvas.Image,
  ImageData: canvas.ImageData,
  createCanvasElement: () => canvas.createCanvas(0, 0),
  createImageElement: () => new canvas.Image()
});

const faceDetectionNet = faceapi.nets.ssdMobilenetv1;

// SsdMobilenetv1Options
const minConfidence = 0.5;
// TinyFaceDetectorOptions
const inputSize = 408;
const scoreThreshold = 0.5;

function getFaceDetectorOptions(net: faceapi.NeuralNetwork<any>) {
  return net === faceapi.nets.ssdMobilenetv1
    ? new faceapi.SsdMobilenetv1Options({ minConfidence })
    : new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold });
}

const imagePath = path.resolve(process.cwd() + "/tests/images/face1.png");
const modelPath = path.resolve(process.cwd() + "/weights");
(async () => {
  await faceDetectionNet.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);

  const referenceImage = await canvas.loadImage(imagePath);
  const faceDetectionOptions = getFaceDetectorOptions(faceDetectionNet);

  console.time("recognize");
  const resultsRef = await faceapi.detectSingleFace(referenceImage, faceDetectionOptions).withFaceLandmarks().withFaceDescriptor();
  console.log("resultsRef", Array.from(resultsRef?.descriptor || []));
  console.timeEnd("recognize");
})();
