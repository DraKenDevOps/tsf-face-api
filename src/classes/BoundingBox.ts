import { Box } from "./Box";

export class BoundingBox extends Box<BoundingBox> implements IBoundingBox {
  constructor(left: number, top: number, right: number, bottom: number, allowNegativeDimensions = false) {
    super({ left, top, right, bottom }, allowNegativeDimensions);
  }
}
