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

## Resolution Modes

| Mode | Input Size | Speed | Quality |
|------|-----------|-------|---------|
| Fast (default) | 256×256 | 25-30 FPS | Good for real-time |
| Detailed | 512×512 | 10-15 FPS | Better detail, slower |

The resolution toggle reloads the model at the new size (models are resolution-specific for optimal performance).
