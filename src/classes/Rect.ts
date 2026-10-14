import { Box } from "./Box";

export class Rect extends Box<Rect> implements IRect {
  constructor(x: number, y: number, width: number, height: number, allowNegativeDimensions = false) {
    super({ x, y, width, height }, allowNegativeDimensions);
  }
}
