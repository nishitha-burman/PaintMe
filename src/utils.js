/**
 * PaintMe — Utility functions
 * Tensor preprocessing (image → NCHW float tensor) and
 * postprocessing (output tensor → ImageData for canvas).
 */

import * as ort from 'onnxruntime-web';

/**
 * Preprocess a video/image source into an ONNX tensor.
 * Resizes to target resolution, converts to NCHW format, normalizes to [0, 255].
 * 
 * Johnson et al. style transfer models expect input in range [0, 255] with shape [1, 3, H, W].
 * 
 * @param {HTMLVideoElement|HTMLImageElement} source - Source element
 * @param {number} resolution - Target resolution (256 or 512)
 * @returns {ort.Tensor} Input tensor ready for inference
 */
export function preprocessFrame(source, resolution) {
  // Draw source to an offscreen canvas at target resolution
  const offscreen = new OffscreenCanvas(resolution, resolution);
  const ctx = offscreen.getContext('2d');

  // Draw with aspect-fill (center crop)
  const srcWidth = source.videoWidth || source.naturalWidth || source.width;
  const srcHeight = source.videoHeight || source.naturalHeight || source.height;
  const scale = Math.max(resolution / srcWidth, resolution / srcHeight);
  const scaledW = srcWidth * scale;
  const scaledH = srcHeight * scale;
  const offsetX = (resolution - scaledW) / 2;
  const offsetY = (resolution - scaledH) / 2;

  ctx.drawImage(source, offsetX, offsetY, scaledW, scaledH);

  // Get pixel data
  const imageData = ctx.getImageData(0, 0, resolution, resolution);
  const { data } = imageData;

  // Convert RGBA interleaved to NCHW planar format
  // Shape: [1, 3, resolution, resolution]
  const numPixels = resolution * resolution;
  const float32Data = new Float32Array(3 * numPixels);

  for (let i = 0; i < numPixels; i++) {
    const rgbaIdx = i * 4;
    // Red channel
    float32Data[i] = data[rgbaIdx];
    // Green channel
    float32Data[numPixels + i] = data[rgbaIdx + 1];
    // Blue channel
    float32Data[2 * numPixels + i] = data[rgbaIdx + 2];
  }

  // Create ONNX tensor with shape [1, 3, H, W]
  return new ort.Tensor('float32', float32Data, [1, 3, resolution, resolution]);
}

/**
 * Postprocess model output tensor into an ImageData object.
 * Converts from NCHW [1, 3, H, W] float (range ~0-255) back to RGBA uint8.
 * 
 * @param {Float32Array} outputData - Raw output tensor data
 * @param {number} resolution - Output resolution
 * @returns {ImageData} Styled frame as ImageData
 */
export function postprocessOutput(outputData, resolution) {
  const numPixels = resolution * resolution;
  const rgba = new Uint8ClampedArray(numPixels * 4);

  for (let i = 0; i < numPixels; i++) {
    // Output is in NCHW format — read from each channel plane
    const r = outputData[i];
    const g = outputData[numPixels + i];
    const b = outputData[2 * numPixels + i];

    const rgbaIdx = i * 4;
    rgba[rgbaIdx] = clamp(r, 0, 255);
    rgba[rgbaIdx + 1] = clamp(g, 0, 255);
    rgba[rgbaIdx + 2] = clamp(b, 0, 255);
    rgba[rgbaIdx + 3] = 255; // Full alpha
  }

  return new ImageData(rgba, resolution, resolution);
}

/**
 * Blend two ImageData frames together
 * @param {ImageData} original - Original frame
 * @param {ImageData} styled - Styled frame
 * @param {number} alpha - Blend factor (0 = original, 1 = styled)
 * @returns {ImageData} Blended result
 */
export function blendFrames(original, styled, alpha) {
  const length = original.data.length;
  const result = new Uint8ClampedArray(length);

  for (let i = 0; i < length; i += 4) {
    result[i] = original.data[i] * (1 - alpha) + styled.data[i] * alpha;
    result[i + 1] = original.data[i + 1] * (1 - alpha) + styled.data[i + 1] * alpha;
    result[i + 2] = original.data[i + 2] * (1 - alpha) + styled.data[i + 2] * alpha;
    result[i + 3] = 255;
  }

  return new ImageData(result, original.width, original.height);
}

/**
 * Clamp a value between min and max
 */
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
