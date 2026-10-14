import { Box } from "../classes/Box";
import { getContext2dOrThrow } from "../dom/getContext2dOrThrow";
import { DrawTextField, DrawTextFieldOptions } from "./DrawTextField";

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

    const defaultDrawLabelOptions = {
      anchorPosition: AnchorPosition.BOTTOM_LEFT,
      backgroundColor: this.boxColor
    };
    this.drawLabelOptions = new DrawTextFieldOptions(Object.assign({}, defaultDrawLabelOptions, drawLabelOptions));
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
    if (label) {
      new DrawTextField([label], { x: x - lineWidth / 2, y }, this.options.drawLabelOptions).draw(canvasArg);
    }
  }
}
