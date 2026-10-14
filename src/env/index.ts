import { isBrowser, isNodejs } from "../utils";
import { createBrowserEnv } from "./createBrowserEnv";
import { createFileSystem } from "./createFileSystem";
import { createNodejsEnv } from "./createNodejsEnv";

let environment: Environment | null;

function getEnv(): Environment {
  if (!environment) {
    throw new Error("getEnv - environment is not defined, check isNodejs() and isBrowser()");
  }
  return environment;
}

function setEnv(env: Environment) {
  environment = env;
}

function initialize() {
  // check for isBrowser() first to prevent electron renderer process
  // to be initialized with wrong environment due to isNodejs() returning true
  if (isBrowser()) {
    return setEnv(createBrowserEnv());
  }
  if (isNodejs()) {
    return setEnv(createNodejsEnv());
  }
}

function monkeyPatch(env: Partial<Environment>) {
  if (!environment) {
    initialize();
  }

  if (!environment) {
    throw new Error("monkeyPatch - environment is not defined, check isNodejs() and isBrowser()");
  }

  const { Canvas = environment.Canvas, Image = environment.Image } = env;
  environment.Canvas = Canvas;
  environment.Image = Image;
  environment.createCanvasElement =
    env.createCanvasElement ||
    (() => {
      try {
        return new (Canvas as any)(0, 0);
      } catch {
        return new Canvas();
      }
    });
  environment.createImageElement = env.createImageElement || (() => new Image());

  environment.ImageData = env.ImageData || environment.ImageData;
  environment.Video = env.Video || environment.Video;
  environment.fetch = env.fetch || environment.fetch;
  environment.readFile = env.readFile || environment.readFile;

  if (typeof globalThis !== "undefined") {
    const g = globalThis as Record<string, any>;
    if (environment.Canvas && !g["HTMLCanvasElement"]) g["HTMLCanvasElement"] = environment.Canvas;
    if (environment.Image && !g["HTMLImageElement"]) g["HTMLImageElement"] = environment.Image;
    if (environment.ImageData && !g["ImageData"]) g["ImageData"] = environment.ImageData;
  }
}

export const env = {
  getEnv,
  setEnv,
  initialize,
  createBrowserEnv,
  createFileSystem,
  createNodejsEnv,
  monkeyPatch,
  isBrowser,
  isNodejs
};

initialize();
