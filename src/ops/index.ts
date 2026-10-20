import { Box } from "../classes/Box";
import { BoundingBox } from "../classes/BoundingBox";

export function iou(box1: Box, box2: Box, isIOU = true) {
  const width = Math.max(0.0, Math.min(box1.right, box2.right) - Math.max(box1.left, box2.left));
  const height = Math.max(0.0, Math.min(box1.bottom, box2.bottom) - Math.max(box1.top, box2.top));
  const interSection = width * height;
  return isIOU ? interSection / (box1.area + box2.area - interSection) : interSection / Math.min(box1.area, box2.area);
}

export function minBbox(pts: IPoint[]): BoundingBox {
  const xs = pts.map((pt) => pt.x);
  const ys = pts.map((pt) => pt.y);
  const minX = xs.reduce((min, x) => (x < min ? x : min), Infinity);
  const minY = ys.reduce((min, y) => (y < min ? y : min), Infinity);
  const maxX = xs.reduce((max, x) => (max < x ? x : max), 0);
  const maxY = ys.reduce((max, y) => (max < y ? y : max), 0);

  return new BoundingBox(minX, minY, maxX, maxY);
}

export function nonMaxSuppression(boxes: Box[], scores: number[], iouThreshold: number, isIOU = true): number[] {
  let indicesSortedByScore = scores
    .map((score, boxIndex) => ({ score, boxIndex }))
    .sort((c1, c2) => c1.score - c2.score)
    .map((c) => c.boxIndex);

  const pick: number[] = [];

  while (indicesSortedByScore.length > 0) {
    const curr = indicesSortedByScore.pop() as number;
    pick.push(curr);
    const indices = indicesSortedByScore;
    const outputs: number[] = [];
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      const currBox = boxes[curr];
      const idxBox = boxes[idx];
      outputs.push(iou(currBox, idxBox, isIOU));
    }
    indicesSortedByScore = indicesSortedByScore.filter((_, j) => outputs[j] <= iouThreshold);
  }

  return pick;
}
