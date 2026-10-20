import { Box } from "./Box";

export class Rectangle extends Box<Rectangle> implements IRect {
  constructor(x: number, y: number, width: number, height: number, allowNegativeDimensions = false) {
    super({ x, y, width, height }, allowNegativeDimensions);
  }
}
