/**
 * PaintMe — Main entry point
 * Sets up camera, render loop, and coordinates modules.
 */

import { createInferenceSession, runInference, getProvider } from './inference.js';
import { initUI, showProgress, hideProgress, updateProviderBadge, updateFPS } from './ui.js';
import { preprocessFrame, postprocessOutput, blendFrames } from './utils.js';
import { isWebGPUAvailable, initWebGPUPipeline, processFrameWebGPU } from './webgpu-pipeline.js';

// App state
const state = {
  stream: null,
  session: null,
  currentStyle: null,
  isRunning: false,
  mirror: true,
  blend: 1.0,
  resolution: 224,
  useStaticImage: false,
  staticImage: null,
  useGPUPipeline: false,       // Toggle: false = JS pipeline, true = WebGPU pipeline
  gpuPipelineReady: false,     // Whether WebGPU pipeline initialized successfully
};

// DOM elements
const landing = document.getElementById('landing');
const app = document.getElementById('app');
const startBtn = document.getElementById('start-btn');
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

/**
 * Initialize the application
 */
function init() {
  // Set up UI controls and wire event handlers
  initUI({
    onStyleSelect: handleStyleSelect,
    onBlendChange: (value) => { state.blend = value / 100; },
    onMirrorToggle: (checked) => { state.mirror = checked; },
    onResolutionToggle: (_checked) => {
      // Disabled: ONNX Model Zoo style transfer models have fixed 224×224 input
    },
    onGPUPipelineToggle: handleGPUPipelineToggle,
    onCameraToggle: handleCameraToggle,
    onSnap: handleSnap,
    onRecord: handleRecord,
  });

  // Camera start button
  startBtn.addEventListener('click', startCamera);

  // Drag-and-drop fallback for camera denial
  setupDropZone();
}

/**
 * Start the webcam stream
 */
async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });
    webcam.srcObject = state.stream;
    await webcam.play();

    // Switch to app view
    landing.classList.add('hidden');
    app.classList.remove('hidden');

    // Set canvas size
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;

    // Start render loop (initially shows raw camera feed)
    state.isRunning = true;
    requestAnimationFrame(renderLoop);
  } catch (err) {
    console.warn('Camera access denied:', err.message);
    // Show drop zone more prominently
    dropZone.style.borderColor = 'var(--accent)';
    dropZone.querySelector('span').textContent = 'Camera unavailable — drop an image instead';
  }
}

/**
 * Handle camera toggle — stops/restarts the camera stream
 */
function handleCameraToggle(enabled) {
  if (!enabled) {
    // Stop camera
    if (state.stream) {
      state.stream.getTracks().forEach(track => track.stop());
    }
    state.isRunning = false;
    webcam.srcObject = null;
    // Clear the canvas to black
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  } else {
    // Restart camera
    navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    }).then((stream) => {
      state.stream = stream;
      webcam.srcObject = stream;
      return webcam.play();
    }).then(() => {
      state.isRunning = true;
      requestAnimationFrame(renderLoop);
    }).catch((err) => {
      console.error('Failed to restart camera:', err);
      document.getElementById('camera-toggle').checked = false;
    });
  }
}

/**
 * Set up drag-and-drop image upload as camera fallback
 */
function setupDropZone() {
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => handleImageFile(e.target.files[0]));

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleImageFile(file);
    }
  });
}

/**
 * Handle image file upload (fallback for no camera)
 */
function handleImageFile(file) {
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    state.useStaticImage = true;
    state.staticImage = img;

    // Switch to app view
    landing.classList.add('hidden');
    app.classList.remove('hidden');

    // Set canvas to image dimensions (capped at 640px)
    const scale = Math.min(1, 640 / img.width);
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;

    // Draw the initial image
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    state.isRunning = true;
    requestAnimationFrame(renderLoop);
  };
  img.src = URL.createObjectURL(file);
}

/**
 * Handle style selection — loads the model and starts inference
 */
async function handleStyleSelect(styleName) {
  // Pause current inference while loading new model
  state.session = null;
  state.currentStyle = styleName;

  showProgress(`Downloading ${styleName} model...`, 0);

  try {
    state.session = await createInferenceSession(styleName, state.resolution, (progress, phase) => {
      if (phase === 'download') {
        showProgress(`Downloading ${styleName} model...`, progress);
      } else if (phase === 'cached') {
        showProgress(`Loading ${styleName} from cache...`, 50);
      } else if (phase === 'creating') {
        showProgress(`Preparing ${styleName} model...`, 90);
      }
    });

    // Show success briefly so user sees it finished
    showProgress(`${styleName} ready!`, 100);
    updateProviderBadge(getProviderLabel());

    // Keep "ready" message visible for a moment before hiding
    setTimeout(() => hideProgress(), 1200);
  } catch (err) {
    console.error('Failed to load model:', err);
    showProgress(`Failed to load ${styleName}. Check console.`, 0);
    setTimeout(() => hideProgress(), 3000);
    state.session = null;
    state.currentStyle = null;
  }
}

/**
 * Build the provider badge label showing inference + postprocessing pipeline
 */
function getProviderLabel() {
  const infer = getProvider(); // 'WebNN' or 'WASM'
  const post = state.useGPUPipeline ? 'GPU Post' : 'JS Post';
  return `${infer} · ${post}`;
}

/**
 * Handle GPU Pipeline toggle.
 * Initializes WebGPU on first enable. Never destroys — just swaps which canvas is visible.
 * This avoids all race conditions from tearing down GPU resources mid-frame.
 */
async function handleGPUPipelineToggle(enabled) {
  if (enabled) {
    // If already initialized, just flip the flag — no re-init needed
    if (state.gpuPipelineReady) {
      state.useGPUPipeline = true;
      updateProviderBadge(getProviderLabel());
      return;
    }

    if (!isWebGPUAvailable()) {
      showProgress('WebGPU not available in this browser', 0);
      setTimeout(() => hideProgress(), 3000);
      document.getElementById('gpu-pipeline-toggle').checked = false;
      return;
    }

    showProgress('Initializing WebGPU pipeline...', 50);

    // Create a separate canvas for WebGPU (WebGPU needs its own context)
    let gpuCanvas = document.getElementById('gpu-canvas');
    if (!gpuCanvas) {
      gpuCanvas = document.createElement('canvas');
      gpuCanvas.id = 'gpu-canvas';
      gpuCanvas.width = canvas.width;
      gpuCanvas.height = canvas.height;
      gpuCanvas.style.display = 'none';
      gpuCanvas.style.maxWidth = '100%';
      gpuCanvas.style.maxHeight = '100%';
      gpuCanvas.style.borderRadius = 'var(--radius-sm)';
      canvas.parentNode.insertBefore(gpuCanvas, canvas.nextSibling);
    }

    const success = await initWebGPUPipeline(gpuCanvas, state.resolution);
    if (success) {
      state.useGPUPipeline = true;
      state.gpuPipelineReady = true;
      showProgress('GPU Pipeline ready!', 100);
      updateProviderBadge(getProviderLabel());
      setTimeout(() => hideProgress(), 1200);
    } else {
      showProgress('WebGPU pipeline init failed — using JS pipeline', 0);
      setTimeout(() => hideProgress(), 3000);
      document.getElementById('gpu-pipeline-toggle').checked = false;
    }
  } else {
    // Just flip the flag — no teardown, no resource destruction
    state.useGPUPipeline = false;

    // Swap canvases
    canvas.style.display = 'block';
    const gpuCanvas = document.getElementById('gpu-canvas');
    if (gpuCanvas) gpuCanvas.style.display = 'none';

    updateProviderBadge(getProviderLabel());
  }
}

/**
 * Main render loop — runs inference on each frame.
 * Uses a flag to prevent overlapping inference calls (since inference is async).
 */
let lastFrameTime = 0;
let frameCount = 0;
let fpsInterval = 0;
let inferenceInProgress = false;
let droppedFrames = 0;

// Performance metrics state — rolling averages updated once per second
const perfMetrics = {
  frameTime: 0,
  inferTime: 0,
  postTime: 0,
  droppedPerSec: 0,
  // Accumulators for averaging
  _frameSamples: [],
  _inferSamples: [],
  _postSamples: [],
  _droppedCount: 0,
  _lastUpdate: 0,
};

const perfOverlay = document.getElementById('perf-overlay');
const perfFrameEl = document.getElementById('perf-frame');
const perfInferEl = document.getElementById('perf-infer');
const perfPostEl = document.getElementById('perf-post');
const perfDroppedEl = document.getElementById('perf-dropped');

function recordPerfSample(timing) {
  if (!timing) return;
  const now = performance.now();
  perfMetrics._frameSamples.push(timing.frame);
  perfMetrics._inferSamples.push(timing.infer);
  perfMetrics._postSamples.push(timing.post);

  // Update display once per second
  if (now - perfMetrics._lastUpdate > 1000) {
    const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    perfFrameEl.textContent = avg(perfMetrics._frameSamples).toFixed(1);
    perfInferEl.textContent = avg(perfMetrics._inferSamples).toFixed(1);
    perfPostEl.textContent = avg(perfMetrics._postSamples).toFixed(1);
    perfDroppedEl.textContent = perfMetrics._droppedCount + '/s';
    perfMetrics._frameSamples = [];
    perfMetrics._inferSamples = [];
    perfMetrics._postSamples = [];
    perfMetrics._droppedCount = 0;
    perfMetrics._lastUpdate = now;
  }
}

function showPerfOverlay() {
  perfOverlay.classList.remove('hidden');
}

function hidePerfOverlay() {
  perfOverlay.classList.add('hidden');
  perfMetrics._droppedCount = 0;
}

function renderLoop(timestamp) {
  if (!state.isRunning) return;

  // FPS calculation
  frameCount++;
  fpsInterval += timestamp - lastFrameTime;
  lastFrameTime = timestamp;
  if (fpsInterval >= 1000) {
    updateFPS(frameCount);
    frameCount = 0;
    fpsInterval = 0;
  }

  // Get source frame (webcam or static image)
  const source = state.useStaticImage ? state.staticImage : webcam;

  if (state.session && state.currentStyle && !inferenceInProgress) {
    inferenceInProgress = true;
    showPerfOverlay();
    const frameStart = performance.now();

    // Capture which pipeline to use at the START of this frame
    // (so toggling mid-frame doesn't cause issues)
    const useGPU = state.useGPUPipeline && state.gpuPipelineReady && !state.useStaticImage;

    if (useGPU) {
      // Show GPU canvas, hide 2D canvas for styled WebGPU rendering
      const gpuCanvas = document.getElementById('gpu-canvas');
      if (gpuCanvas && gpuCanvas.style.display === 'none') {
        gpuCanvas.style.display = 'block';
        canvas.style.display = 'none';
      }
      // WebGPU pipeline path — postprocessing on GPU
      processFrameWebGPU(source, state.session, {
        blend: state.blend,
        mirror: state.mirror,
      }).then((timing) => {
        const frameTime = performance.now() - frameStart;
        if (timing) {
          recordPerfSample({ frame: frameTime, infer: timing.infer, post: timing.post });
        }
      }).catch((err) => {
        console.warn('[PaintMe] GPU frame error:', err.message);
      }).finally(() => {
        inferenceInProgress = false;
      });
    } else {
      // Show 2D canvas, hide GPU canvas for JS rendering
      const gpuCanvas = document.getElementById('gpu-canvas');
      if (gpuCanvas && gpuCanvas.style.display !== 'none') {
        gpuCanvas.style.display = 'none';
        canvas.style.display = 'block';
      }
      // JS pipeline path (original)
      processFrame(source).then((timing) => {
        const frameTime = performance.now() - frameStart;
        if (timing) {
          recordPerfSample({ frame: frameTime, infer: timing.infer, post: timing.post });
        }
      }).finally(() => {
        inferenceInProgress = false;
      });
    }
  } else if (state.session && state.currentStyle && inferenceInProgress) {
    // Frame dropped — inference still in progress
    perfMetrics._droppedCount++;
  } else if (!state.session || !state.currentStyle) {
    // No style selected — show raw feed on 2D canvas
    hidePerfOverlay();
    if (state.useGPUPipeline) {
      const gpuCanvas = document.getElementById('gpu-canvas');
      if (gpuCanvas && gpuCanvas.style.display !== 'none') {
        gpuCanvas.style.display = 'none';
        canvas.style.display = 'block';
      }
    }
    drawRawFrame(source);
  }

  requestAnimationFrame(renderLoop);
}

/**
 * Process a single frame through the style transfer model (JS pipeline)
 * Returns timing breakdown: { infer, post }
 */
async function processFrame(source) {
  try {
    // Preprocess: resize to model input and convert to tensor
    const inputTensor = preprocessFrame(source, state.resolution);

    // Run inference (timed)
    const inferStart = performance.now();
    const outputTensor = await runInference(state.session, inputTensor);
    const inferEnd = performance.now();

    // Postprocess: convert output tensor back to ImageData (timed)
    const postStart = performance.now();
    const styledImageData = postprocessOutput(outputTensor, state.resolution);

    // Draw to main canvas with blending
    ctx.save();

    if (state.mirror) {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }

    // Draw original frame first if blending
    if (state.blend < 1.0) {
      ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = state.blend;
    }

    // Draw styled frame — use createImageBitmap for reliable canvas drawing
    const bitmap = await createImageBitmap(styledImageData);
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();

    ctx.restore();
    const postEnd = performance.now();

    return { infer: inferEnd - inferStart, post: postEnd - postStart };
  } catch (err) {
    console.error('Inference error:', err);
    showProgress(`Error: ${err.message}`, 0);
    drawRawFrame(source);
    return null;
  }
}

/**
 * Draw the raw camera frame (no style applied)
 */
function drawRawFrame(source) {
  ctx.save();
  if (state.mirror) {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  ctx.restore();
}

/**
 * Snap the current frame as a PNG download
 */
function handleSnap() {
  const link = document.createElement('a');
  link.download = `paintme-${Date.now()}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
}

/**
 * Record a few seconds of the canvas as WebM
 */
let mediaRecorder = null;

function handleRecord() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    // Stop recording
    mediaRecorder.stop();
    return;
  }

  const stream = canvas.captureStream(30);
  const chunks = [];

  mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm; codecs=vp9' });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    const blob = new Blob(chunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = `paintme-${Date.now()}.webm`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);

    // Update button state
    document.getElementById('record-btn').classList.remove('recording');
    document.getElementById('record-btn').textContent = '🎥 Record';
  };

  // Start recording for 5 seconds
  mediaRecorder.start();
  document.getElementById('record-btn').classList.add('recording');
  document.getElementById('record-btn').textContent = '⏹ Stop';

  // Auto-stop after 5 seconds
  setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }
  }, 5000);
}

// Boot the app
init();
