interface IBoundingBox {
  left: number;
  top: number;
  right: number;
  bottom: number;
}
interface IDimensions {
  width: number;
  height: number;
}
interface IFaceMatch {
  label: string;
  distance: number;
}
interface IPoint {
  x: number;
  y: number;
}
interface IRect {
  x: number;
  y: number;
  width: number;
  height: number;
}
interface IDrawBoxOptions {
  boxColor?: string;
  lineWidth?: number;
  drawLabelOptions?: IDrawTextFieldOptions;
  label?: string;
}
interface IDrawFaceLandmarksOptions {
  drawLines?: boolean;
  drawPoints?: boolean;
  lineWidth?: number;
  pointSize?: number;
  lineColor?: string;
  pointColor?: string;
}
interface IDrawTextFieldOptions {
  anchorPosition?: AnchorPosition;
  backgroundColor?: string;
  fontColor?: string;
  fontSize?: number;
  fontStyle?: string;
  padding?: number;
}

interface IMtcnnOptions {
  minFaceSize?: number;
  scaleFactor?: number;
  maxNumScales?: number;
  scoreThresholds?: number[];
  scaleSteps?: number[];
}
interface ISsdMobilenetv1Options {
  minConfidence?: number;
  maxResults?: number;
}

declare enum TinyYolov2SizeType {
  XS = 224,
  SM = 320,
  MD = 416,
  LG = 608
}

interface ITinyYolov2Options {
  inputSize?: number;
  scoreThreshold?: number;
}

type AgeAndGenderPrediction = {
  age: number;
  gender: Gender;
  genderProbability: number;
}

declare enum Gender {
  FEMALE = "female",
  MALE = "male"
}

type ParamMapping = {
  originalPath?: string;
  paramPath: string;
}

type ExtractWeightsFunction = (numWeights: number) => Float32Array;
interface IFileSystem extends FileSystem {
  readFile: (filePath: string) => Promise<Buffer>;
}

type Environment = IFileSystem & {
  Canvas: typeof HTMLCanvasElement;
  CanvasRenderingContext2D: typeof CanvasRenderingContext2D;
  Image: typeof HTMLImageElement;
  ImageData: typeof ImageData;
  Video: typeof HTMLVideoElement;
  createCanvasElement: () => HTMLCanvasElement;
  createImageElement: () => HTMLImageElement;
  fetch: (url: string, init?: RequestInit) => Promise<Response>;
}

type WithAge<TSource> = TSource & {
  age: number;
}

type WithFaceDescriptor<TSource> = TSource & {
  descriptor: Float32Array;
}

declare enum AnchorPosition {
  TOP_LEFT = "TOP_LEFT",
  TOP_RIGHT = "TOP_RIGHT",
  BOTTOM_LEFT = "BOTTOM_LEFT",
  BOTTOM_RIGHT = "BOTTOM_RIGHT"
}
