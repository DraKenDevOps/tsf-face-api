import { FaceDetection } from "../classes/FaceDetection";
import { TNetInput } from "../dom/types";
import { MtcnnOptions } from "../mtcnn/MtcnnOptions";
import { SsdMobilenetv1Options } from "../ssdMobilenetv1/SsdMobilenetv1Options";
import { TinyFaceDetectorOptions } from "../tinyFaceDetector";
import { TinyYolov2Options } from "../tinyYolov2/TinyYolov2Options";

export type FaceDetectionOptions = TinyFaceDetectorOptions | SsdMobilenetv1Options | MtcnnOptions | TinyYolov2Options;

export type FaceDetectionFunction = (input: TNetInput) => Promise<FaceDetection[]>;
