/**
 * PaintMe — WebGPU Pipeline
 * 
 * Strategy: Use WebGPU for postprocessing (NCHW→RGBA + blend + mirror + render to canvas).
 * Preprocessing stays in JS because ONNX Runtime needs CPU-accessible tensor data anyway.
 * This avoids the expensive JS pixel loop in postprocessing and renders directly to a
 * WebGPU canvas without createImageBitmap/putImageData overhead.
 * 
 * When MLTensor + exportToGPU becomes available (behind WebNN flag in Edge Canary),
 * the full zero-copy path (GPU preprocess → WebNN inference → GPU postprocess) can be enabled.
 */

import { preprocessFrame } from './utils.js';

// ============================================================
// State
// ============================================================

let gpuDevice = null;
let gpuContext = null;
let canvasEl = null;

// Pipelines
let postprocessPipeline = null;
let renderPipeline = null;

// Buffers and textures
let outputTensorBuffer = null;  // GPUBuffer for model output (NCHW float32)
let outputRgbaTexture = null;   // GPU texture after postprocessing (RGBA)
let sampler = null;

// Uniform buffer for postprocess params
let postprocessUniformBuffer = null;

// Config
let resolution = 224;

/**
 * Check if WebGPU is available
 */
export function isWebGPUAvailable() {
  return 'gpu' in navigator;
}

/**
 * Check if WebNN MLTensor with GPU interop is available
 */
export async function isMLTensorAvailable() {
  if (!('ml' in navigator)) return false;
  try {
    const ctx = await navigator.ml.createContext({ deviceType: 'gpu' });
    return typeof ctx.createTensor === 'function';
  } catch {
    return false;
  }
}

/**
 * Initialize the WebGPU pipeline
 * @param {HTMLCanvasElement} canvas - Target canvas for rendering
 * @param {number} res - Model input resolution (224)
 * @returns {Promise<boolean>} Whether initialization succeeded
 */
export async function initWebGPUPipeline(canvas, res = 224) {
  resolution = res;
  canvasEl = canvas;

  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');

    gpuDevice = await adapter.requestDevice();

    // Configure canvas context for WebGPU rendering
    gpuContext = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    gpuContext.configure({
      device: gpuDevice,
      format,
      alphaMode: 'opaque',
    });

    // Create shader pipelines
    createPostprocessPipeline();
    createRenderPipeline(format);

    // Allocate persistent buffers
    allocateBuffers();

    // Create sampler for rendering
    sampler = gpuDevice.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });

    console.log('[PaintMe WebGPU] Pipeline initialized — GPU postprocessing active');
    return true;
  } catch (err) {
    console.error('[PaintMe WebGPU] Initialization failed:', err);
    return false;
  }
}

/**
 * Allocate GPU buffers for the pipeline
 */
function allocateBuffers() {
  const tensorSize = 3 * resolution * resolution * 4; // float32 bytes for NCHW

  // Buffer for model output (NCHW float32) — written from CPU after inference
  outputTensorBuffer = gpuDevice.createBuffer({
    size: tensorSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Texture for postprocessed RGBA output
  outputRgbaTexture = gpuDevice.createTexture({
    size: [resolution, resolution],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  // Uniform buffer for postprocess params (resolution, blend, mirror)
  postprocessUniformBuffer = gpuDevice.createBuffer({
    size: 16, // resolution(u32) + blend(f32) + mirror(u32) + padding
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

// ============================================================
// Postprocess Compute Shader
// Takes NCHW float32 buffer → outputs RGBA texture
// Applies blend with original feed and optional mirror
// ============================================================

function createPostprocessPipeline() {
  const shaderCode = /* wgsl */`
    @group(0) @binding(0) var<storage, read> styledBuffer: array<f32>;
    @group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
    @group(0) @binding(2) var<storage, read> originalBuffer: array<f32>;
    @group(0) @binding(3) var<uniform> params: PostprocessParams;

    struct PostprocessParams {
      resolution: u32,
      blend: f32,
      mirror: u32,
      _pad: u32,
    }

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let res = params.resolution;
      let x = gid.x;
      let y = gid.y;

      if (x >= res || y >= res) {
        return;
      }

      let pixelIdx = y * res + x;
      let planeSize = res * res;

      // Read styled pixel from NCHW output — clamp to [0,255] then normalize to [0,1]
      let sr = clamp(styledBuffer[pixelIdx], 0.0, 255.0) / 255.0;
      let sg = clamp(styledBuffer[planeSize + pixelIdx], 0.0, 255.0) / 255.0;
      let sb = clamp(styledBuffer[2u * planeSize + pixelIdx], 0.0, 255.0) / 255.0;

      // Read original pixel from NCHW input (same format, [0,255])
      let or_ = clamp(originalBuffer[pixelIdx], 0.0, 255.0) / 255.0;
      let og = clamp(originalBuffer[planeSize + pixelIdx], 0.0, 255.0) / 255.0;
      let ob = clamp(originalBuffer[2u * planeSize + pixelIdx], 0.0, 255.0) / 255.0;

      // Blend styled with original
      let blend = params.blend;
      let r = mix(or_, sr, blend);
      let g = mix(og, sg, blend);
      let b = mix(ob, sb, blend);

      // Apply mirror (horizontal flip) for output position
      var outX = x;
      if (params.mirror == 1u) {
        outX = res - 1u - x;
      }

      textureStore(outputTexture, vec2<u32>(outX, y), vec4<f32>(r, g, b, 1.0));
    }
  `;

  const shaderModule = gpuDevice.createShaderModule({ code: shaderCode });

  postprocessPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  });
}

// ============================================================
// Render Pipeline — draws postprocessed texture to canvas
// ============================================================

function createRenderPipeline(format) {
  const shaderCode = /* wgsl */`
    @group(0) @binding(0) var texSampler: sampler;
    @group(0) @binding(1) var texInput: texture_2d<f32>;

    struct VertexOutput {
      @builtin(position) position: vec4<f32>,
      @location(0) uv: vec2<f32>,
    }

    @vertex
    fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
      var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
      );
      var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
      );

      var out: VertexOutput;
      out.position = vec4<f32>(positions[idx], 0.0, 1.0);
      out.uv = uvs[idx];
      return out;
    }

    @fragment
    fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
      return textureSample(texInput, texSampler, input.uv);
    }
  `;

  const shaderModule = gpuDevice.createShaderModule({ code: shaderCode });

  renderPipeline = gpuDevice.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [{ format }],
    },
  });
}

// ============================================================
// Frame Processing
// ============================================================

// Persistent buffer for original frame (avoids re-creating per frame)
let originalTensorBuffer = null;

/**
 * Process a single frame: JS preprocess → ONNX inference → GPU postprocess → GPU render
 * 
 * @param {HTMLVideoElement} videoSource - Camera video element
 * @param {object} session - ONNX Runtime inference session
 * @param {object} options - { blend: 0-1, mirror: boolean }
 * @param {object} preprocessing - { preprocessFn, resolution } from the JS pipeline
 */
export async function processFrameWebGPU(videoSource, session, options = {}) {
  if (!gpuDevice || !gpuContext) return null;

  const blend = options.blend ?? 1.0;
  const mirror = options.mirror ?? true;

  // --- Step 1: Preprocess in JS (same as JS pipeline for identical input) ---
  const inputTensor = preprocessFrame(videoSource, resolution);
  const inputData = inputTensor.data; // Float32Array in NCHW [0,255]

  // --- Step 2: Run inference (timed) ---
  const inferStart = performance.now();
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const feeds = { [inputName]: inputTensor };
  const results = await session.run(feeds);
  const outputData = results[outputName].data; // Float32Array in NCHW
  const inferEnd = performance.now();

  // --- Step 3: GPU postprocess (timed) ---
  const postStart = performance.now();

  // Ensure original buffer exists
  const tensorByteSize = 3 * resolution * resolution * 4;
  if (!originalTensorBuffer) {
    originalTensorBuffer = gpuDevice.createBuffer({
      size: tensorByteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  // Upload styled output and original input to GPU
  gpuDevice.queue.writeBuffer(outputTensorBuffer, 0, outputData);
  gpuDevice.queue.writeBuffer(originalTensorBuffer, 0, inputData);

  // Update postprocess uniforms
  const paramsBuffer = new ArrayBuffer(16);
  new Uint32Array(paramsBuffer, 0, 1)[0] = resolution;
  new Float32Array(paramsBuffer, 4, 1)[0] = blend;
  new Uint32Array(paramsBuffer, 8, 1)[0] = mirror ? 1 : 0;
  new Uint32Array(paramsBuffer, 12, 1)[0] = 0;
  gpuDevice.queue.writeBuffer(postprocessUniformBuffer, 0, paramsBuffer);

  // Bind group for postprocess
  const postprocessBG = gpuDevice.createBindGroup({
    layout: postprocessPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: outputTensorBuffer } },
      { binding: 1, resource: outputRgbaTexture.createView() },
      { binding: 2, resource: { buffer: originalTensorBuffer } },
      { binding: 3, resource: { buffer: postprocessUniformBuffer } },
    ],
  });

  const encoder = gpuDevice.createCommandEncoder();

  // Postprocess compute pass
  const computePass = encoder.beginComputePass();
  computePass.setPipeline(postprocessPipeline);
  computePass.setBindGroup(0, postprocessBG);
  computePass.dispatchWorkgroups(
    Math.ceil(resolution / 16),
    Math.ceil(resolution / 16)
  );
  computePass.end();

  // --- Step 4: Render texture to canvas ---
  const canvasTexture = gpuContext.getCurrentTexture();
  const renderBG = gpuDevice.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: outputRgbaTexture.createView() },
    ],
  });

  const renderPass = encoder.beginRenderPass({
    colorAttachments: [{
      view: canvasTexture.createView(),
      loadOp: 'clear',
      storeOp: 'store',
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    }],
  });
  renderPass.setPipeline(renderPipeline);
  renderPass.setBindGroup(0, renderBG);
  renderPass.draw(6);
  renderPass.end();

  gpuDevice.queue.submit([encoder.finish()]);
  const postEnd = performance.now();

  return { infer: inferEnd - inferStart, post: postEnd - postStart };
}

/**
 * Update pipeline settings
 */
export function updateWebGPUSettings(opts) {
  if (opts.resolution !== undefined) resolution = opts.resolution;
}

/**
 * Clean up WebGPU resources (does NOT destroy the device — allows re-init)
 */
export function destroyWebGPUPipeline() {
  outputTensorBuffer?.destroy();
  outputRgbaTexture?.destroy();
  originalTensorBuffer?.destroy();
  postprocessUniformBuffer?.destroy();

  outputTensorBuffer = null;
  outputRgbaTexture = null;
  originalTensorBuffer = null;
  postprocessUniformBuffer = null;

  if (gpuContext) {
    gpuContext.unconfigure();
    gpuContext = null;
  }

  gpuDevice?.destroy();
  gpuDevice = null;
  canvasEl = null;
}
