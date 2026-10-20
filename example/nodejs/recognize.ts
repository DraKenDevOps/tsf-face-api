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

const angry = path.resolve(process.cwd() + "/tests/images/angry.jpg");
const faces = path.resolve(process.cwd() + "/tests/images/faces.jpg");

const modelPath = path.resolve(process.cwd() + "/weights");
(async () => {
  await faceDetectionNet.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);

  const angryImg = await canvas.loadImage(angry);
  const facesImg = await canvas.loadImage(faces);
  const faceDetectionOptions = getFaceDetectorOptions(faceDetectionNet);

  console.time("recognize");
  const single = await faceapi.detectSingleFace(angryImg, faceDetectionOptions).withFaceLandmarks().withFaceDescriptor();
  // const all = await faceapi.detectAllFaces(facesImg, faceDetectionOptions).withFaceLandmarks().withFaceDescriptors();
  console.log(single?.detection.score);
  // console.log("single", Array.from(single?.descriptor || []));
  // console.log("all", all.map((f) => Array.from(f.descriptor)));
  console.timeEnd("recognize");
})();
