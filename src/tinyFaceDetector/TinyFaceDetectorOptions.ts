import { TinyYolov2Options } from "../tinyYolov2/TinyYolov2Options";

export interface ITinyFaceDetectorOptions extends ITinyYolov2Options {}

export class TinyFaceDetectorOptions extends TinyYolov2Options {
  protected _name: string = "TinyFaceDetectorOptions";
}
