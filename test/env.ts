import * as tf from '@tensorflow/tfjs-core';

import { fetchImage, fetchJson, fetchNetWeights, NeuralNetwork } from '../src';
import { TestEnv } from './Environment';

const g = globalThis as Record<string, any>;
if (g['jasmine']) {
  g['jasmine'].DEFAULT_TIMEOUT_INTERVAL = 60000;
}

if (typeof window !== 'undefined' && (window as any)['__karma__'] && ((window as any)['__karma__'].config.jasmine.args as string[]).some(arg => arg === 'backend_cpu')) {
  tf.setBackend('cpu');
}

async function loadImageBrowser(uri: string): Promise<HTMLImageElement> {
  return fetchImage(`/${uri.startsWith('/') ? uri.slice(1) : uri}`);
}

async function loadJsonBrowser<T>(uri: string): Promise<T> {
  return fetchJson<T>(`/${uri.startsWith('/') ? uri.slice(1) : uri}`);
}

async function initNetBrowser<TNet extends NeuralNetwork<any>>(
  net: TNet,
  uncompressedFilename?: string | boolean,
  isUnusedModel: boolean = false
) {
  const url = uncompressedFilename
    ? await fetchNetWeights(`/weights_uncompressed/${uncompressedFilename}`)
    : (isUnusedModel ? '/weights_unused' : '/weights');
  await net.load(url);
}

const browserTestEnv: TestEnv = {
  loadImage: loadImageBrowser,
  loadJson: loadJsonBrowser,
  initNet: initNetBrowser
};

export function getTestEnv(): TestEnv {
  return (globalThis as Record<string, any>)['nodeTestEnv'] || browserTestEnv;
}
