import { createFileSystem } from "./createFileSystem";
import { Environment } from "./types";

export function createNodejsEnv(): Environment {
  const g = globalThis as Record<string, any>;
  const Canvas = g["Canvas"] || g["HTMLCanvasElement"];
  const Image = g["Image"] || g["HTMLImageElement"];

  const createCanvasElement = function (): HTMLCanvasElement {
    if (Canvas) {
      try {
        return new Canvas(0, 0);
      } catch {
        return new Canvas();
      }
    }
    throw new Error("createCanvasElement - missing Canvas implementation for nodejs environment");
  };

  const createImageElement = function (): HTMLImageElement {
    if (Image) {
      return new Image();
    }
    throw new Error("createImageElement - missing Image implementation for nodejs environment");
  };

  const fetch =
    typeof globalThis.fetch === "function"
      ? globalThis.fetch.bind(globalThis)
      : g["fetch"] ||
        function () {
          throw new Error("fetch - missing fetch implementation for nodejs environment");
        };

  const fileSystem = createFileSystem();

  return {
    Canvas: Canvas || class {},
    CanvasRenderingContext2D: g["CanvasRenderingContext2D"] || class {},
    Image: Image || class {},
    ImageData: g["ImageData"] || class {},
    Video: g["HTMLVideoElement"] || class {},
    createCanvasElement,
    createImageElement,
    fetch,
    ...fileSystem
  };
}
