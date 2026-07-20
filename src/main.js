/**
 * PaintMe — Main entry point
 * Sets up camera, render loop, and coordinates modules.
 */

import { createInferenceSession, runInference, getProvider } from './inference.js';
import { initUI, showProgress, hideProgress, updateProviderBadge, updateFPS } from './ui.js';
import { preprocessFrame, postprocessOutput, blendFrames } from './utils.js';

// App state
const state = {
  stream: null,
  session: null,
  currentStyle: null,
  isRunning: false,
  mirror: true,
  blend: 1.0,
  resolution: 256,
  useStaticImage: false,
  staticImage: null,
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
    onResolutionToggle: (checked) => {
      state.resolution = checked ? 512 : 256;
      // Reload model at new resolution if one is active
      if (state.currentStyle) {
        handleStyleSelect(state.currentStyle);
      }
    },
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
    updateProviderBadge(getProvider());

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
 * Main render loop — runs inference on each frame.
 * Uses a flag to prevent overlapping inference calls (since inference is async).
 */
let lastFrameTime = 0;
let frameCount = 0;
let fpsInterval = 0;
let inferenceInProgress = false;

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
    // Run style transfer inference (async, but we gate on the flag)
    inferenceInProgress = true;
    processFrame(source).finally(() => {
      inferenceInProgress = false;
    });
  } else if (!state.session || !state.currentStyle) {
    // No style selected — show raw feed
    drawRawFrame(source);
  }
  // If inference is in progress, we skip this frame (canvas keeps showing previous result)

  requestAnimationFrame(renderLoop);
}

/**
 * Process a single frame through the style transfer model
 */
async function processFrame(source) {
  try {
    // Preprocess: resize to model input and convert to tensor
    const inputTensor = preprocessFrame(source, state.resolution);

    // Run inference
    const outputTensor = await runInference(state.session, inputTensor);

    // Postprocess: convert output tensor back to ImageData
    const styledImageData = postprocessOutput(outputTensor, state.resolution);

    // Create offscreen canvas for the styled frame
    const offscreen = new OffscreenCanvas(state.resolution, state.resolution);
    const offCtx = offscreen.getContext('2d');
    offCtx.putImageData(styledImageData, 0, 0);

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

    // Draw styled frame
    ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
    ctx.restore();
  } catch (err) {
    // If inference fails, just show the raw frame
    drawRawFrame(source);
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
