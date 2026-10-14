import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

import { env, NeuralNetwork } from '../src';
import { TestEnv } from './Environment';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function setupNodeEnv() {
  try {
    await import('@tensorflow/tfjs-node');
  } catch {
    // Fallback to CPU backend registered via src/index.ts if native tfjs-node addon is not built
  }

  try {
    const canvas = await import('canvas');
    const { Canvas, Image, ImageData, loadImage } = (canvas.default || canvas) as any;
    if (Canvas && Image && ImageData) {
      env.monkeyPatch({ Canvas, Image, ImageData });
    }

    async function loadImageNode(uri: string): Promise<HTMLImageElement> {
      return loadImage(path.resolve(__dirname, '../', uri));
    }

    async function loadJsonNode<T>(uri: string): Promise<T> {
      return JSON.parse(fs.readFileSync(path.resolve(__dirname, '../', uri)).toString());
    }

    async function initNetNode<TNet extends NeuralNetwork<any>>(
      net: TNet,
      uncompressedFilename?: string | boolean,
      isUnusedModel: boolean = false
    ) {
      if (uncompressedFilename) {
        const filePath = path.resolve(
          __dirname,
          '../weights_uncompressed',
          typeof uncompressedFilename === 'string' ? uncompressedFilename : ''
        );
        if (fs.existsSync(filePath)) {
          const buf = fs.readFileSync(filePath);
          const weights = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
          net.extractWeights(weights);
          return;
        }
      }
      const weightsDir = isUnusedModel && fs.existsSync(path.resolve(__dirname, '../weights_unused'))
        ? '../weights_unused'
        : '../weights';
      await net.loadFromDisk(path.resolve(__dirname, weightsDir));
    }

    const nodeTestEnv: TestEnv = {
      loadImage: loadImageNode,
      loadJson: loadJsonNode,
      initNet: initNetNode
    };

    (globalThis as Record<string, any>)['nodeTestEnv'] = nodeTestEnv;
  } catch (err) {
    console.warn('test/env.node.ts: canvas setup warning:', err);
  }
}

await setupNodeEnv();
