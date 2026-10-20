import { Point } from "../classes/Point";
import { Box } from "../classes/Box";
import { FaceDetection } from "../classes/FaceDetection";
import { isWithFaceDetection, WithFaceDetection } from "../factories/WithFaceDetection";
import { FaceExpressions } from "../faceExpressionNet/FaceExpressions";
import { isWithFaceExpressions, WithFaceExpressions } from "../factories/WithFaceExpressions";
import { round } from "../utils";
import { FaceLandmarks } from "../classes/FaceLandmarks";
import { FaceLandmarks68 } from "../classes/FaceLandmarks68";
import { resolveInput, getContext2dOrThrow } from "../dom/NetInput";
import { isWithFaceLandmarks, WithFaceLandmarks } from "../factories/WithFaceLandmarks";
import { AnchorPosition } from "../utils/config";

export type TDrawDetectionsInput = IRect | IBoundingBox | FaceDetection | WithFaceDetection<{}>;

export class DrawBoxOptions {
  public boxColor: string;
  public lineWidth: number;
  public drawLabelOptions: DrawTextFieldOptions;
  public label?: string;

  constructor(options: IDrawBoxOptions = {}) {
    const { boxColor, lineWidth, label, drawLabelOptions } = options;
    this.boxColor = boxColor || "rgba(255, 0, 0, 1)";
    this.lineWidth = lineWidth || 2;
    this.label = label;

    const defaultOptions = {
      anchorPosition: AnchorPosition.BOTTOM_LEFT,
      backgroundColor: this.boxColor
    };
    this.drawLabelOptions = new DrawTextFieldOptions(Object.assign({}, defaultOptions, drawLabelOptions));
  }
}

export class DrawBox {
  public box: Box;
  public options: DrawBoxOptions;

  constructor(box: IBoundingBox | IRect, options: IDrawBoxOptions = {}) {
    this.box = new Box(box);
    this.options = new DrawBoxOptions(options);
  }

  draw(canvasArg: string | HTMLCanvasElement | CanvasRenderingContext2D) {
    const ctx = getContext2dOrThrow(canvasArg);
    const { boxColor, lineWidth } = this.options;
    const { x, y, width, height } = this.box;
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x, y, width, height);
    const { label } = this.options;
    if (label) new DrawTextField([label], { x: x - lineWidth / 2, y }, this.options.drawLabelOptions).draw(canvasArg);
  }
}

export function drawContour(ctx: CanvasRenderingContext2D, points: Point[], isClosed = false) {
  ctx.beginPath();
  points.slice(1).forEach(({ x, y }, prevIdx) => {
    const from = points[prevIdx];
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(x, y);
  });
  if (isClosed) {
    const from = points[points.length - 1];
    const to = points[0];
    if (!from || !to) return;
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
  }
  ctx.stroke();
}

export function drawDetections(canvasArg: string | HTMLCanvasElement, detections: TDrawDetectionsInput | Array<TDrawDetectionsInput>) {
  const detectionsArray = Array.isArray(detections) ? detections : [detections];
  detectionsArray.forEach((det) => {
    const score = det instanceof FaceDetection ? det.score : isWithFaceDetection(det) ? det.detection.score : undefined;
    const box = det instanceof FaceDetection ? det.box : isWithFaceDetection(det) ? det.detection.box : new Box(det);
    const label = score ? `${round(score)}` : undefined;
    new DrawBox(box, { label }).draw(canvasArg);
  });
}

export type DrawFaceExpressionsInput = FaceExpressions | WithFaceExpressions<{}>;

export function drawFaceExpressions(
  canvasArg: string | HTMLCanvasElement,
  faceExpressions: DrawFaceExpressionsInput | Array<DrawFaceExpressionsInput>,
  minConfidence = 0.1,
  textFieldAnchor?: IPoint
) {
  const faceExpressionsArray = Array.isArray(faceExpressions) ? faceExpressions : [faceExpressions];

  faceExpressionsArray.forEach((e) => {
    const expr = e instanceof FaceExpressions ? e : isWithFaceExpressions(e) ? e.expressions : undefined;
    if (!expr) {
      throw new Error("drawFaceExpressions - expected faceExpressions to be FaceExpressions | WithFaceExpressions<{}> or array thereof");
    }

    const sorted = expr.asSortedArray();
    const resultsToDisplay = sorted.filter((expr) => expr.probability > minConfidence);

    const anchor = isWithFaceDetection(e) ? e.detection.box.bottomLeft : textFieldAnchor || new Point(0, 0);

    const drawTextField = new DrawTextField(
      resultsToDisplay.map((expr) => `${expr.expression} (${round(expr.probability)})`),
      anchor
    );
    drawTextField.draw(canvasArg);
  });
}

export class DrawFaceLandmarksOptions {
  public drawLines: boolean;
  public drawPoints: boolean;
  public lineWidth: number;
  public pointSize: number;
  public lineColor: string;
  public pointColor: string;

  constructor(options: IDrawFaceLandmarksOptions = {}) {
    const { drawLines = true, drawPoints = true, lineWidth, lineColor, pointSize, pointColor } = options;
    this.drawLines = drawLines;
    this.drawPoints = drawPoints;
    this.lineWidth = lineWidth || 1;
    this.pointSize = pointSize || 2;
    this.lineColor = lineColor || "rgba(0, 255, 255, 1)";
    this.pointColor = pointColor || "rgba(255, 0, 255, 1)";
  }
}

export class DrawFaceLandmarks {
  public faceLandmarks: FaceLandmarks;
  public options: DrawFaceLandmarksOptions;

  constructor(faceLandmarks: FaceLandmarks, options: IDrawFaceLandmarksOptions = {}) {
    this.faceLandmarks = faceLandmarks;
    this.options = new DrawFaceLandmarksOptions(options);
  }

  draw(canvasArg: string | HTMLCanvasElement | CanvasRenderingContext2D) {
    const ctx = getContext2dOrThrow(canvasArg);

    const { drawLines, drawPoints, lineWidth, lineColor, pointSize, pointColor } = this.options;

    if (drawLines && this.faceLandmarks instanceof FaceLandmarks68) {
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = lineWidth;
      drawContour(ctx, this.faceLandmarks.getJawOutline());
      drawContour(ctx, this.faceLandmarks.getLeftEyeBrow());
      drawContour(ctx, this.faceLandmarks.getRightEyeBrow());
      drawContour(ctx, this.faceLandmarks.getNose());
      drawContour(ctx, this.faceLandmarks.getLeftEye(), true);
      drawContour(ctx, this.faceLandmarks.getRightEye(), true);
      drawContour(ctx, this.faceLandmarks.getMouth(), true);
    }

    if (drawPoints) {
      ctx.strokeStyle = pointColor;
      ctx.fillStyle = pointColor;

      const drawPoint = (pt: IPoint) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, pointSize, 0, 2 * Math.PI);
        ctx.fill();
      };
      this.faceLandmarks.positions.forEach(drawPoint);
    }
  }
}

export type DrawFaceLandmarksInput = FaceLandmarks | WithFaceLandmarks<WithFaceDetection<{}>>;

export function drawFaceLandmarks(canvasArg: string | HTMLCanvasElement, faceLandmarks: DrawFaceLandmarksInput | Array<DrawFaceLandmarksInput>) {
  const faceLandmarksArray = Array.isArray(faceLandmarks) ? faceLandmarks : [faceLandmarks];
  faceLandmarksArray.forEach((f) => {
    const landmarks = f instanceof FaceLandmarks ? f : isWithFaceLandmarks(f) ? f.landmarks : undefined;
    if (!landmarks) {
      throw new Error("drawFaceLandmarks - expected faceExpressions to be FaceLandmarks | WithFaceLandmarks<WithFaceDetection<{}>> or array thereof");
    }

    new DrawFaceLandmarks(landmarks).draw(canvasArg);
  });
}

export class DrawTextFieldOptions implements IDrawTextFieldOptions {
  public anchorPosition: AnchorPosition;
  public backgroundColor: string;
  public fontColor: string;
  public fontSize: number;
  public fontStyle: string;
  public padding: number;

  constructor(options: IDrawTextFieldOptions = {}) {
    const { anchorPosition, backgroundColor, fontColor, fontSize, fontStyle, padding } = options;
    this.anchorPosition = anchorPosition || AnchorPosition.TOP_LEFT;
    this.backgroundColor = backgroundColor || "rgba(0, 0, 0, 0.5)";
    this.fontColor = fontColor || "rgba(255, 255, 255, 1)";
    this.fontSize = fontSize || 14;
    this.fontStyle = fontStyle || "Georgia";
    this.padding = padding || 4;
  }
}

export class DrawTextField {
  public text: string[];
  public anchor: IPoint;
  public options: DrawTextFieldOptions;

  constructor(text: string | string[] | DrawTextField, anchor: IPoint, options: IDrawTextFieldOptions = {}) {
    this.text = typeof text === "string" ? [text] : text instanceof DrawTextField ? text.text : text;
    this.anchor = anchor;
    this.options = new DrawTextFieldOptions(options);
  }

  measureWidth(ctx: CanvasRenderingContext2D): number {
    const { padding } = this.options;
    return this.text.map((l) => ctx.measureText(l).width).reduce((w0, w1) => (w0 < w1 ? w1 : w0), 0) + 2 * padding;
  }

  measureHeight(): number {
    const { fontSize, padding } = this.options;
    return this.text.length * fontSize + 2 * padding;
  }

  getUpperLeft(ctx: CanvasRenderingContext2D, canvasDims?: IDimensions): IPoint {
    const { anchorPosition } = this.options;
    const isShiftLeft = anchorPosition === AnchorPosition.BOTTOM_RIGHT || anchorPosition === AnchorPosition.TOP_RIGHT;
    const isShiftTop = anchorPosition === AnchorPosition.BOTTOM_LEFT || anchorPosition === AnchorPosition.BOTTOM_RIGHT;

    const textFieldWidth = this.measureWidth(ctx);
    const textFieldHeight = this.measureHeight();
    const x = isShiftLeft ? this.anchor.x - textFieldWidth : this.anchor.x;
    const y = isShiftTop ? this.anchor.y - textFieldHeight : this.anchor.y;

    // adjust anchor if text box exceeds canvas borders
    if (canvasDims) {
      const { width, height } = canvasDims;
      const newX = Math.max(Math.min(x, width - textFieldWidth), 0);
      const newY = Math.max(Math.min(y, height - textFieldHeight), 0);
      return { x: newX, y: newY };
    }
    return { x, y };
  }

  draw(canvasArg: string | HTMLCanvasElement | CanvasRenderingContext2D) {
    const canvas = resolveInput(canvasArg);
    const ctx = getContext2dOrThrow(canvas);

    const { backgroundColor, fontColor, fontSize, fontStyle, padding } = this.options;

    ctx.font = `${fontSize}px ${fontStyle}`;
    const maxTextWidth = this.measureWidth(ctx);
    const textHeight = this.measureHeight();

    ctx.fillStyle = backgroundColor;
    const upperLeft = this.getUpperLeft(ctx, canvas);
    ctx.fillRect(upperLeft.x, upperLeft.y, maxTextWidth, textHeight);

    ctx.fillStyle = fontColor;
    this.text.forEach((textLine, i) => {
      const x = padding + upperLeft.x;
      const y = padding + upperLeft.y + (i + 1) * fontSize;
      ctx.fillText(textLine, x, y);
    });
  }
}
