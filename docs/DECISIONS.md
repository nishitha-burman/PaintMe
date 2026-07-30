# Technical Decisions

This document captures key architectural and technology choices made during PaintMe's development, along with the reasoning behind each.

---

## Why Vite (not Webpack, Parcel, or plain ES modules)?

**Decision**: Use Vite as the dev server and build tool.

**Rationale**:
- **Native ESM dev server** — No bundling during development, instant HMR. Critical for iteration speed when tweaking UI.
- **Simple config** — Vite works out of the box for vanilla JS projects. Our `vite.config.js` is ~10 lines.
- **COOP/COEP headers** — Easy to configure custom headers needed for `SharedArrayBuffer` (required by ONNX Runtime WASM threading).
- **Production build** — Rollup-based bundling with code splitting for production.
- **Static asset serving** — Serves `public/` directory as-is, perfect for ONNX model files.

**Alternatives considered**:
- **Webpack**: More config overhead, slower dev server for a vanilla JS project.
- **No bundler**: Would need a separate server for COOP/COEP headers and would lose HMR.
- **Parcel**: Would work, but Vite has better ecosystem support and documentation.

---

## Why ONNX Runtime Web (not raw WebNN API)?

**Decision**: Use `onnxruntime-web` as the inference runtime, not the WebNN API directly.

**Rationale**:
- **Abstraction over providers** — ONNX Runtime handles WebNN/WASM fallback internally. We pass a provider list and it picks the best available.
- **ONNX model ecosystem** — Massive library of pre-trained models in ONNX format. No need to manually convert to WebNN graph operations.
- **Battle-tested** — ONNX Runtime is used in production across Microsoft products. Handles edge cases (memory management, session lifecycle, graph optimization).
- **WASM fallback** — The raw WebNN API has no fallback — if WebNN isn't available, you're stuck. ONNX Runtime gives us WASM for free.
- **Simpler code** — Loading a model is `InferenceSession.create(buffer, options)`. With raw WebNN, you'd need to build the computation graph node-by-node.

**Alternatives considered**:
- **Raw WebNN API**: More control, but requires building graph manually (hundreds of lines for a style transfer model). No fallback. Very limited browser support currently.
- **TensorFlow.js**: Heavier dependency, less control over WebNN provider selection, not as optimized for ONNX models.
- **MediaPipe**: Focused on specific use cases (hands, face), not general style transfer.

---

## Why IndexedDB for Model Caching (not Cache API or localStorage)?

**Decision**: Cache downloaded ONNX model files in IndexedDB as `ArrayBuffer` values.

**Rationale**:
- **Large binary storage** — ONNX models are 1.7-6.5 MB. IndexedDB supports large binary blobs natively.
- **Persistent** — Survives page reloads, browser restarts, and is not subject to Cache API eviction policies.
- **Simple key-value access** — We just need `get(key)` and `put(key, buffer)`. No complex querying.
- **No service worker needed** — Cache API would require a service worker for programmatic access. IndexedDB works directly from the main thread.
- **Offline support** — Once cached, models load without any network request.

**Alternatives considered**:
- **Cache API**: Designed for HTTP responses, not raw binary data. Requires service worker for non-fetch use. Subject to browser eviction.
- **localStorage**: 5-10 MB limit per origin. Can only store strings (would need base64 encoding, bloating size 33%).
- **No caching**: Models are 6.5 MB — re-downloading on every visit would be a poor experience on slow connections.

---

## Why No TypeScript?

**Decision**: Use plain ES modules with JSDoc comments instead of TypeScript.

**Rationale**:
- **Learning accessibility** — PaintMe is intended as a demo/learning project. TypeScript adds a compilation step and type system that may be unfamiliar to the target audience (creative coders, ML enthusiasts exploring WebNN).
- **Fewer moving parts** — No build step in development (Vite serves ESM natively). No `tsconfig.json`, no type declaration files for ONNX Runtime.
- **JSDoc provides types** — We use JSDoc `@param` and `@returns` annotations for documentation. IDE support (VS Code) still provides autocomplete and type checking.
- **Small codebase** — With 4 source files totaling ~400 lines, the benefits of TypeScript's type safety are minimal.

**When we'd reconsider**:
- If the project grows beyond 10 source files
- If we add a plugin/extension system
- If multiple contributors need type contracts

---

## Why 224×224 Fixed Resolution?

**Decision**: Use 224×224 fixed input resolution. The Hi-Res toggle is hidden because the ONNX Model Zoo models have static input shapes.

**Rationale**:
- **Model constraint** — The ONNX Model Zoo style transfer models (mosaic, candy, rain_princess, etc.) are exported with fixed 224×224 input dimensions. Feeding 448×448 causes an `OrtRun` dimension mismatch error.
- **Real-time performance** — 224×224 achieves 15-30 FPS on mid-range hardware with WASM.
- **Memory budget** — Input tensor at 224×224×3 = 150 KB. Small tensors mean faster memory copies and better cache behavior.

**The tradeoff**: 224px output stretched to a 640px canvas can look soft. Multi-resolution support requires re-exporting models with dynamic input shapes or sourcing higher-resolution variants.

**When we'd re-enable the toggle**:
- If we source or re-export models with dynamic input dimensions
- If higher-resolution pre-trained models become available in the ONNX Model Zoo

---

## Why requestAnimationFrame (not setInterval or Web Workers)?

**Decision**: Drive the render loop with `requestAnimationFrame`, running inference on the main thread.

**Rationale**:
- **Natural throttling** — rAF fires at display refresh rate (60Hz) but our inference takes 30-60ms, so we naturally run at 15-30 FPS without overloading.
- **Browser optimization** — rAF pauses when the tab is hidden, saving resources. setInterval would keep running.
- **Canvas synchronization** — rAF guarantees our draw calls are synchronized with the browser's paint cycle, avoiding tearing.
- **Simplicity** — A single render loop is easier to reason about than a worker-based pipeline with message passing.

**Why not Web Workers for inference**:
- ONNX Runtime WASM already uses its own worker threads internally for parallel computation.
- Transferring large tensors between main thread and workers (via postMessage) adds latency that can exceed the inference time itself.
- Canvas API is main-thread only, so we'd need to transfer results back anyway.

**Future consideration**: If WebNN becomes widely available and inference drops to <5ms, we could decouple the render loop from inference and run at full 60 FPS with a separate inference scheduler.

---

## Why Selfie Mirror Enabled by Default?

**Decision**: The canvas is horizontally flipped (mirrored) by default.

**Rationale**:
- **User expectation** — Every selfie camera (phone, laptop, video call) mirrors by default. Not mirroring feels disorienting ("why is everything backwards?").
- **Interaction alignment** — When users move left, they expect their on-screen representation to move left.
- **Toggle available** — Users can disable mirror mode for non-selfie use cases (e.g., pointing camera at a scene).

---

## Why WebM for Recording (not MP4)?

**Decision**: Use `MediaRecorder` with `video/webm; codecs=vp9` for the record feature.

**Rationale**:
- **Browser-native** — `canvas.captureStream()` + `MediaRecorder` is the only way to record a canvas without external libraries.
- **No encoding libraries needed** — MP4/H.264 encoding in the browser requires heavy WASM-based encoders (ffmpeg.wasm, ~25MB).
- **VP9 quality** — Good compression with high visual quality at 30 FPS.
- **Cross-browser** — WebM recording works in Chrome, Edge, and Firefox.

**Limitation**: Safari doesn't support WebM recording. A future enhancement could detect Safari and offer frame-by-frame GIF export instead.

---

## Why WebGPU Postprocessing Only (not full GPU pipeline)?

**Decision**: The GPU Pipeline toggle moves only postprocessing to WebGPU compute shaders. Preprocessing and inference stay on CPU/ONNX Runtime.

**Rationale**:
- **MLTensor not available yet** — True zero-copy (WebNN output → GPUBuffer) requires `MLContext.exportToGPU()` which is behind experimental flags in Edge Canary and not yet stable.
- **Preprocess must match exactly** — Any difference in preprocessing (resize algorithm, normalization) produces visually different results. Keeping JS preprocessing ensures identical inputs regardless of pipeline toggle.
- **Postprocess is the bottleneck we can solve today** — The JS postprocessing path involves a pixel-by-pixel loop + `createImageBitmap` + `putImageData`. A compute shader does NCHW→RGBA, blend, and mirror in a single GPU dispatch.
- **Safe A/B comparison** — With identical preprocess + inference, users can toggle between pipelines and see the postprocessing performance difference in isolation.

**Future**: When `exportToGPU` and `MLContext.createTensor(GPUBuffer)` are stable, the full pipeline can stay on GPU end-to-end.

---

## Why Init-Once / Never-Destroy for GPU Pipeline Toggle?

**Decision**: WebGPU device, buffers, and shader pipelines are initialized once on first toggle-on and kept alive for the entire page lifetime. Toggling off just flips a boolean flag — no resource teardown.

**Rationale**:
- **Race conditions** — Earlier versions destroyed GPU resources on toggle-off, which crashed in-flight GPU commands and froze the page. `gpuDevice.destroy()` invalidated all GPU objects immediately. Even nulling references + `setTimeout` wasn't reliable because the async render loop could still reference destroyed buffers.
- **Toggle is for A/B comparison** — Users flip back and forth rapidly. Recreating the device, compiling shaders, and allocating buffers on every toggle added 100-500ms latency.
- **Negligible VRAM cost** — Two 224×224 buffers and a compute shader use ~2-3 MB. Not worth the complexity of lifecycle management.
- **Instant re-enable** — Second toggle-on skips initialization entirely since resources are already alive.

**Tradeoff**: GPU resources stay allocated even when using the JS pipeline. Acceptable given the small memory footprint — the browser reclaims everything on tab close.

---

## Why Rolling Averages for Performance Metrics?

**Decision**: Display performance numbers as rolling 1-second averages rather than per-frame values.

**Rationale**:
- **Readability** — Per-frame numbers change every 30-60ms, making them impossible to read. A 1-second rolling average gives stable, readable values.
- **Meaningful comparison** — When toggling between JS and GPU pipelines, users can see steady numbers for each mode instead of a blur of changing digits.
- **Low overhead** — Simply accumulate samples into an array, compute mean every 1000ms, and reset.
