/**
 * PaintMe — UI module
 * Handles all DOM interactions: style picker, controls, progress, badges.
 */

import { STYLES } from './inference.js';

// DOM references (populated on init)
let stylePicker;
let blendSlider;
let blendValue;
let mirrorToggle;
let resolutionToggle;
let snapBtn;
let recordBtn;
let providerBadge;
let fpsBadge;
let progressContainer;
let progressBar;
let progressText;

/**
 * Initialize UI elements and wire up event handlers
 * @param {object} handlers - Callback handlers for UI events
 */
export function initUI(handlers) {
  // Cache DOM references
  stylePicker = document.getElementById('style-picker');
  blendSlider = document.getElementById('blend-slider');
  blendValue = document.getElementById('blend-value');
  mirrorToggle = document.getElementById('mirror-toggle');
  resolutionToggle = document.getElementById('resolution-toggle');
  snapBtn = document.getElementById('snap-btn');
  recordBtn = document.getElementById('record-btn');
  providerBadge = document.getElementById('provider-badge');
  fpsBadge = document.getElementById('fps-badge');
  progressContainer = document.getElementById('progress-container');
  progressBar = document.getElementById('progress-bar');
  progressText = document.getElementById('progress-text');

  // Build style picker cards
  buildStylePicker(handlers.onStyleSelect);

  // Blend slider
  blendSlider.addEventListener('input', (e) => {
    const value = parseInt(e.target.value);
    blendValue.textContent = `${value}%`;
    handlers.onBlendChange(value);
  });

  // Mirror toggle
  mirrorToggle.addEventListener('change', (e) => {
    handlers.onMirrorToggle(e.target.checked);
  });

  // Resolution toggle
  resolutionToggle.addEventListener('change', (e) => {
    handlers.onResolutionToggle(e.target.checked);
  });

  // GPU Pipeline toggle
  const gpuPipelineToggle = document.getElementById('gpu-pipeline-toggle');
  gpuPipelineToggle.addEventListener('change', (e) => {
    handlers.onGPUPipelineToggle(e.target.checked);
  });

  // Camera toggle
  const cameraToggle = document.getElementById('camera-toggle');
  cameraToggle.addEventListener('change', (e) => {
    handlers.onCameraToggle(e.target.checked);
  });

  // Snap button
  snapBtn.addEventListener('click', handlers.onSnap);

  // Record button
  recordBtn.addEventListener('click', handlers.onRecord);
}

/**
 * Build the horizontal style picker strip with thumbnail cards
 */
function buildStylePicker(onSelect) {
  STYLES.forEach((style) => {
    const card = document.createElement('div');
    card.className = 'style-card';
    card.dataset.style = style.name;

    card.innerHTML = `
      <img src="${style.thumbnail}" alt="${style.label}" 
           onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 70%22><rect fill=%22%231a1a2e%22 width=%22100%22 height=%2270%22/><text x=%2250%22 y=%2240%22 text-anchor=%22middle%22 fill=%22%23666%22 font-size=%2212%22>${style.label}</text></svg>'" />
      <div class="style-name">${style.label}</div>
    `;

    card.addEventListener('click', () => {
      // Update active state
      document.querySelectorAll('.style-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      onSelect(style.name);
    });

    stylePicker.appendChild(card);
  });
}

/**
 * Show the progress bar with a message
 * @param {string} message - Status message
 * @param {number} percent - Progress percentage (0-100)
 */
export function showProgress(message, percent = 0) {
  progressContainer.classList.remove('hidden');
  progressContainer.classList.toggle('success', percent === 100 && message.includes('ready'));
  progressBar.style.setProperty('--progress', `${percent}%`);
  progressText.textContent = message;
}

/**
 * Hide the progress bar
 */
export function hideProgress() {
  progressContainer.classList.add('hidden');
}

/**
 * Update the execution provider badge
 * @param {string} provider - Provider name
 */
export function updateProviderBadge(provider) {
  providerBadge.textContent = provider;
}

/**
 * Update the FPS counter badge
 * @param {number} fps - Current frames per second
 */
export function updateFPS(fps) {
  fpsBadge.textContent = `${fps} Hz`;
}
