# Architecture

## Overview

PaintMe is a client-side web application that performs real-time neural style transfer on camera frames using ONNX Runtime Web. The entire pipeline runs in the browser — no server-side computation required.

## Module Responsibilities

### `src/main.js` — Orchestrator
- Manages application state (stream, session, settings)
- Handles camera initialization via `getUserMedia`
- Owns the `requestAnimationFrame` render loop
- Coordinates between inference, UI, and utility modules
- Implements fallback to drag-and-drop image upload

### `src/inference.js` — Model Management
- Manages ONNX Runtime Web session lifecycle
- Handles execution provider selection (WebNN → WASM fallback)
- Implements model downloading with progress tracking
- Manages IndexedDB caching for offline model access
- Exposes the list of available styles with metadata

### `src/ui.js` — Interface Layer
- Builds the style picker card strip from style metadata
- Wires DOM event handlers to application callbacks
- Manages progress bar visibility and state
- Updates status badges (provider, FPS)

### `src/utils.js` — Tensor Operations
- Preprocesses video frames: resize, center-crop, RGBA → NCHW conversion
- Postprocesses inference output: NCHW → RGBA, clamping to [0, 255]
- Provides frame blending utility for the blend slider

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Render Loop (rAF)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐          │
│  │  Camera  │───▶│ Preprocess   │───▶│   Inference   │          │
│  │  Frame   │    │ (utils.js)   │    │(inference.js) │          │
│  └──────────┘    │              │    │               │          │
│                  │ - Resize to  │    │ - Feed tensor │          │
│                  │   256/512    │    │ - Run session │          │
│                  │ - Center crop│    │ - WebNN/WASM  │          │
│                  │ - RGBA→NCHW  │    │               │          │
│                  └──────────────┘    └───────┬───────┘          │
│                                              │                   │
│  ┌──────────┐    ┌──────────────┐            │                   │
│  │  Canvas  │◀───│ Postprocess  │◀───────────┘                   │
│  │ (output) │    │ (utils.js)   │                                │
│  └──────────┘    │              │                                │
│       ▲          │ - NCHW→RGBA  │                                │
│       │          │ - Clamp 0-255│                                │
│       │          │ - Blend      │                                │
│       │          └──────────────┘                                │
│       │                                                          │
│  ┌────┴─────┐                                                    │
│  │  Mirror  │                                                    │
│  │  + Draw  │                                                    │
│  └──────────┘                                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Execution Provider Fallback Strategy

```
1. Check: Is navigator.ml available?
   ├─ YES → Try creating session with 'webnn' provider (deviceType: 'gpu')
   │        ├─ Success → Use WebNN (best performance, NPU/GPU accelerated)
   │        └─ Fail → Fall through to WASM
   └─ NO → Use WASM directly

2. WASM provider is always available as universal fallback
   - Uses SharedArrayBuffer + WebWorker for threading
   - Requires COOP/COEP headers (configured in vite.config.js)
```

The execution provider list `['webnn', 'wasm']` is passed to `InferenceSession.create()`. ONNX Runtime Web internally handles the fallback — if the first provider fails, it tries the next.

## Caching Layer (IndexedDB)

```
Model Request
     │
     ▼
┌─────────────────┐     ┌───────────────────┐
│ Check IndexedDB │────▶│ Cache Hit?        │
│ (paintme-models)│     │                   │
└─────────────────┘     └───────┬───────────┘
                                │
                    ┌───────────┼───────────┐
                    │ YES                   │ NO
                    ▼                       ▼
          ┌─────────────┐       ┌─────────────────┐
          │ Use cached  │       │ Fetch from       │
          │ ArrayBuffer │       │ /models/*.onnx   │
          └─────────────┘       │ (with progress)  │
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │ Store in IDB    │
                                │ for next time   │
                                └─────────────────┘
```

- **Database**: `paintme-models` (version 1)
- **Object store**: `onnx-models`
- **Keys**: `{style_name}_{resolution}` (e.g., `starry_night_256`)
- **Values**: `ArrayBuffer` of the full `.onnx` file

This ensures models are downloaded only once and available offline on repeat visits.

## Render Loop Design

The render loop uses `requestAnimationFrame` for smooth animation tied to the display refresh rate:

1. **FPS tracking** — Counts frames per second, updates badge every 1000ms
2. **Source selection** — Uses webcam video element or static image (fallback)
3. **Inference path** — If a model session is active, runs the full preprocess → infer → postprocess pipeline
4. **Raw path** — If no model is selected, draws the raw camera feed directly
5. **Compositing** — Applies mirror transform and blend alpha before final draw

The loop is non-blocking: if inference takes longer than one frame, the next `requestAnimationFrame` callback simply waits for the previous inference to complete before starting a new one. This naturally throttles to the inference speed (typically 15-30 FPS depending on hardware).

## WebGPU Pipeline (v2.0+)

An optional GPU pipeline can be enabled via the "GPU Pipeline" toggle. When active:

```
┌───────────────────────────────────────────────────────────────────┐
│                  WebGPU Pipeline Path                              │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐           │
│  │  Camera  │───▶│ JS Preprocess│───▶│   Inference   │           │
│  │  Frame   │    │ (utils.js)   │    │ (same ONNX)   │           │
│  └──────────┘    └──────────────┘    └───────┬───────┘           │
│                                              │                    │
│                                              ▼                    │
│  ┌──────────────┐    ┌───────────────────────────────┐           │
│  │ WebGPU Canvas│◀───│ GPU Postprocess (compute shader)│          │
│  │  (render)    │    │ - Upload NCHW to GPU buffer    │          │
│  └──────────────┘    │ - NCHW→RGBA via compute shader │          │
│                      │ - Blend + Mirror on GPU        │          │
│                      │ - Render to WebGPU canvas      │          │
│                      └────────────────────────────────┘          │
└───────────────────────────────────────────────────────────────────┘
```

**What moves to GPU**: Postprocessing (NCHW→RGBA conversion, blending with original frame, mirror transform, and final canvas render) — all done via WGSL compute shaders instead of JavaScript pixel loops.

**What stays on CPU**: Preprocessing and ONNX Runtime inference. Without MLTensor/exportToGPU support (available behind WebNN flag in Edge Canary), data must round-trip through CPU for inference.

**Future: Full zero-copy path** — When `MLContext.exportToGPU()` is available, the full pipeline (preprocess → inference → postprocess) can stay on GPU via MLTensor → GPUBuffer interop, eliminating all CPU readback.

### `src/webgpu-pipeline.js` — GPU Module

- Initializes WebGPU device and canvas context
- Creates compute shader pipelines for postprocessing
- Creates a render pipeline (full-screen quad) for final output
- Manages GPU buffer lifecycle and bind groups
- Gracefully handles teardown when toggled off mid-frame

## Performance Metrics Overlay

When a style is active, a performance overlay displays rolling 1-second averages:

- **Frame** — Total end-to-end time per styled frame (ms)
- **Infer** — Time spent in ONNX Runtime `session.run()` (ms)
- **Post** — Time for postprocessing + canvas draw (ms) — key differentiator between pipelines
- **Dropped** — Frames where `requestAnimationFrame` fired but inference was still busy

This allows direct A/B comparison between JS and WebGPU pipelines.

## Resolution Modes

| Mode | Input Size | Speed | Quality |
|------|-----------|-------|---------|
| Fast (default) | 224×224 | 5-30 FPS | Good for real-time |
| Detailed | 448×448 | 2-15 FPS | Better detail, slower |

The resolution toggle reloads the model at the new size (models are resolution-specific for optimal performance).
