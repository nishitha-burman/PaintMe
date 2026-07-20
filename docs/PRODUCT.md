# Product

## Vision

PaintMe turns your webcam into a magic art canvas. Point your camera at anything and watch it transform into a painting in real-time — no cloud, no signup, no wait. It's the fastest path from "I wonder what this would look like as a Van Gogh" to seeing it happen live.

## Target Audience

- **Creative explorers** — People who enjoy visual experiments and art
- **Web AI enthusiasts** — Developers curious about WebNN and on-device ML
- **Educators** — Teachers demonstrating neural networks and computer vision
- **Social sharers** — Users who want unique, artistically-styled photos/videos

## UX Principles

1. **Instant delight** — The first styled frame should appear within seconds of choosing a style
2. **Zero friction** — No accounts, no downloads, no configuration. Click and go.
3. **Transparent performance** — Show the user what's happening (FPS, provider, loading progress)
4. **Graceful degradation** — Every fallback (WASM, image upload, low-res) should feel intentional, not broken
5. **Playful exploration** — Encourage trying different styles, blends, and interactions

## Emotional Flow

```
Curiosity → "What does this do?"
     │
     ▼
Delight → "Whoa, I'm a painting!"
     │
     ▼
Play → "Let me try Mosaic... what about at 50% blend?"
     │
     ▼
Share → "I need to save this and show people"
```

## Feature List

### MVP (Current)

| Feature | Priority | Status |
|---------|----------|--------|
| Webcam capture | P0 | ✅ |
| Style picker (8 styles) | P0 | ✅ |
| Real-time inference | P0 | ✅ |
| WebNN with WASM fallback | P0 | ✅ |
| IndexedDB model caching | P0 | ✅ |
| Blend slider | P1 | ✅ |
| Mirror toggle | P1 | ✅ |
| Resolution toggle | P1 | ✅ |
| Snap to PNG | P1 | ✅ |
| Record to WebM | P1 | ✅ |
| FPS + provider badges | P1 | ✅ |
| Drag-and-drop fallback | P1 | ✅ |
| Progress bar for download | P1 | ✅ |
| Dark glassmorphism UI | P2 | ✅ |

### Future Enhancements

| Feature | Priority | Notes |
|---------|----------|-------|
| Custom style upload | P2 | Let users upload their own .onnx models |
| Style intensity slider | P2 | Control how strongly the style is applied (separate from blend) |
| Multiple face detection | P3 | Apply style only to faces / background |
| Gallery mode | P3 | Browse and share styled captures |
| PWA / installable | P2 | Full offline support with service worker |
| Comparison split view | P2 | Side-by-side or slider comparing original/styled |
| Animated style transitions | P3 | Smooth morph between styles |
| Rear camera support | P2 | Toggle front/back camera on mobile |
| Batch process photos | P3 | Upload multiple images, apply styles in bulk |
| Model quantization (int8) | P2 | Faster inference on more devices |

## Success Metrics

- **Time to first styled frame**: < 5 seconds (after model cached)
- **Inference FPS**: ≥ 15 FPS on mid-range hardware at 256px
- **Model load time**: < 3 seconds from cache, < 15 seconds first download
- **Works on**: Chrome, Edge (WebNN); Firefox, Safari (WASM fallback)
