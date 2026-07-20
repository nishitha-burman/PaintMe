/**
 * PaintMe — Inference module
 * Manages ONNX Runtime Web sessions, model loading, IndexedDB caching,
 * and WebNN/WASM execution provider selection.
 */

import * as ort from 'onnxruntime-web';

// IndexedDB config for model caching
const DB_NAME = 'paintme-models';
const DB_VERSION = 1;
const STORE_NAME = 'onnx-models';

// Track which execution provider is in use
let activeProvider = 'wasm';

// Base URL for ONNX Model Zoo style transfer models (Git LFS via media.githubusercontent.com)
const MODEL_ZOO_BASE = 'https://media.githubusercontent.com/media/onnx/models/main/validated/vision/style_transfer/fast_neural_style/model';

/**
 * Available style models with their metadata.
 * Models are fetched from the ONNX Model Zoo on GitHub and cached in IndexedDB.
 * These models accept dynamic input sizes (we use 256x256 or 512x512).
 */
export const STYLES = [
  { name: 'candy', label: 'Candy', thumbnail: '/thumbnails/candy.jpg', file: 'candy-9.onnx' },
  { name: 'mosaic', label: 'Mosaic', thumbnail: '/thumbnails/mosaic.jpg', file: 'mosaic-9.onnx' },
  { name: 'rain_princess', label: 'Rain Princess', thumbnail: '/thumbnails/rain_princess.jpg', file: 'rain-princess-9.onnx' },
  { name: 'pointilism', label: 'Pointilism', thumbnail: '/thumbnails/pointilism.jpg', file: 'pointilism-9.onnx' },
  { name: 'udnie', label: 'Udnie', thumbnail: '/thumbnails/udnie.jpg', file: 'udnie-9.onnx' },
];

/**
 * Open (or create) the IndexedDB database for model caching
 */
function openModelDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Get a cached model from IndexedDB
 */
async function getCachedModel(key) {
  const db = await openModelDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const request = store.get(key);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Store a model in IndexedDB for future offline use
 */
async function cacheModel(key, data) {
  const db = await openModelDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const request = store.put(data, key);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

/**
 * Download a model file with progress tracking
 * @param {string} url - Model file URL
 * @param {function} onProgress - Progress callback (0-100)
 * @returns {Promise<ArrayBuffer>} Model data
 */
async function downloadModel(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.statusText}`);
  }

  const contentLength = response.headers.get('content-length');
  if (!contentLength) {
    // No content-length header; download without progress
    const buffer = await response.arrayBuffer();
    onProgress(100);
    return buffer;
  }

  const total = parseInt(contentLength, 10);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(Math.round((received / total) * 100));
  }

  // Combine chunks into a single ArrayBuffer
  const buffer = new ArrayBuffer(received);
  const view = new Uint8Array(buffer);
  let offset = 0;
  for (const chunk of chunks) {
    view.set(chunk, offset);
    offset += chunk.length;
  }

  return buffer;
}

/**
 * Determine which execution providers are available.
 * Prefers WebNN (NPU/GPU) and falls back to WASM.
 */
function getExecutionProviders() {
  // WebNN is available in Chromium-based browsers with the flag enabled
  const providers = [];

  // Try WebNN first
  if ('ml' in navigator) {
    providers.push('webnn');
  }

  // Always include WASM as fallback
  providers.push('wasm');

  return providers;
}

/**
 * Create an ONNX Runtime inference session for the given style
 * @param {string} styleName - Style model name (e.g., 'starry_night')
 * @param {number} resolution - Input resolution (256 or 512)
 * @param {function} onProgress - Progress callback
 * @returns {Promise<ort.InferenceSession>}
 */
export async function createInferenceSession(styleName, resolution, onProgress) {
  const style = STYLES.find(s => s.name === styleName);
  if (!style) throw new Error(`Unknown style: ${styleName}`);

  const modelKey = `${styleName}_v9`;
  // Fetch from ONNX Model Zoo on GitHub (cached in IndexedDB after first load)
  const modelUrl = `${MODEL_ZOO_BASE}/${style.file}`;

  // Try to load from IndexedDB cache first
  let modelData = await getCachedModel(modelKey);

  if (!modelData) {
    // Download the model and cache it
    modelData = await downloadModel(modelUrl, (percent) => onProgress(percent, 'download'));
    await cacheModel(modelKey, modelData);
  } else {
    onProgress(100, 'cached');
  }

  // Signal that we're now creating the session (the slow part after download)
  onProgress(90, 'creating');

  // Configure session options with execution provider fallback
  const providers = getExecutionProviders();
  const sessionOptions = {
    executionProviders: providers.map(p => {
      if (p === 'webnn') {
        return { name: 'webnn', deviceType: 'gpu' };  // Prefer GPU, falls back to NPU
      }
      return p;
    }),
    graphOptimizationLevel: 'all',
  };

  // Create the session (this parses the model and builds the execution graph)
  const session = await ort.InferenceSession.create(modelData, sessionOptions);

  // Detect which provider was actually used
  activeProvider = detectActiveProvider(session);

  return session;
}

/**
 * Detect which execution provider the session is actually using
 */
function detectActiveProvider(session) {
  // ONNX Runtime Web doesn't expose this directly in all versions,
  // so we infer from available context
  if ('ml' in navigator) {
    return 'WebNN (GPU/NPU)';
  }
  return 'WASM';
}

/**
 * Run inference on an input tensor
 * @param {ort.InferenceSession} session - Active ONNX session
 * @param {ort.Tensor} inputTensor - Preprocessed input tensor [1, 3, H, W]
 * @returns {Promise<Float32Array>} Output tensor data
 */
export async function runInference(session, inputTensor) {
  const feeds = {};
  const inputName = session.inputNames[0];
  feeds[inputName] = inputTensor;

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  return results[outputName].data;
}

/**
 * Get the active execution provider name
 */
export function getProvider() {
  return activeProvider;
}
