export function createBrowserEnv(): Environment {
  const fetch = window.fetch
    ? window.fetch.bind(window)
    : function () {
        throw new Error("fetch - missing fetch implementation for browser environment");
      };

  const readFile = function () {
    throw new Error("readFile - filesystem not available for browser environment");
  };

  return {
    Canvas: HTMLCanvasElement,
    CanvasRenderingContext2D: CanvasRenderingContext2D,
    Image: HTMLImageElement,
    ImageData: ImageData,
    Video: HTMLVideoElement,
    createCanvasElement: () => document.createElement("canvas"),
    createImageElement: () => document.createElement("img"),
    fetch,
    readFile,
    name: "browser",
    root: {} as FileSystemDirectoryEntry
  };
}
