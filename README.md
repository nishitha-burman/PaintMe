# 🎨 PaintMe

**Real-time neural style transfer camera app powered by WebNN and ONNX Runtime Web.**

🌐 **[Live Demo](https://nishitha-burman.github.io/PaintMe/)**

Transform your webcam feed into living art — Starry Night, The Scream, Mosaic, and more — running entirely in the browser with hardware-accelerated AI inference.

![PaintMe Demo](https://img.shields.io/badge/status-prototype-blueviolet)

## ✨ Features

- 🎥 **Live camera** — Real-time style transfer on your webcam feed
- 🖼️ **8 art styles** — Starry Night, The Scream, Mosaic, Candy, Udnie, Rain Princess, Pointilism, La Muse
- ⚡ **WebNN accelerated** — Uses NPU/GPU when available, falls back to WASM
- 📱 **Responsive** — Works on desktop and mobile browsers
- 💾 **Offline-ready** — Models cached in IndexedDB after first download
- 📸 **Snap & Record** — Save styled frames as PNG or record WebM video
- 🎛️ **Controls** — Blend slider, mirror mode, resolution toggle

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- ONNX model files (see below)

### Setup

```bash
# Install dependencies
npm install

# Add model files to public/models/
# See public/models/README.md for instructions

# Start dev server
npm run dev
```

Open `http://localhost:5173` in a Chromium-based browser (Chrome, Edge) for WebNN support.

### Adding Models

Download or export ONNX style transfer models and place them in `public/models/`:

```
public/models/starry_night_256.onnx
public/models/the_scream_256.onnx
public/models/mosaic_256.onnx
...
```

See [`public/models/README.md`](public/models/README.md) for detailed instructions on obtaining models.

## 🏗️ Architecture

```
Camera → Preprocessing → ONNX Inference → Postprocessing → Canvas
         (resize/NCHW)   (WebNN/WASM)     (NCHW→RGBA)     (blend/mirror)
```

### Project Structure

```
├── index.html              # Landing page and app shell
├── src/
│   ├── main.js            # Entry point, camera, render loop
│   ├── inference.js       # ONNX Runtime session management
│   ├── ui.js              # UI controls and style picker
│   └── utils.js           # Tensor preprocessing/postprocessing
├── styles/
│   └── main.css           # Dark glassmorphism theme
├── public/
│   ├── models/            # ONNX model files (.gitignored)
│   └── thumbnails/        # Style preview images
├── docs/                  # Architecture & design docs
├── package.json           # Vite + onnxruntime-web
└── vite.config.js         # Dev server config (COOP/COEP headers)
```

### Execution Provider Fallback

1. **WebNN (GPU/NPU)** — Hardware-accelerated, best performance
2. **WASM** — Universal fallback, runs everywhere

The app automatically detects WebNN availability and selects the best provider.

## 🛠️ Development

```bash
npm run dev      # Start Vite dev server with HMR
npm run build    # Production build to dist/
npm run preview  # Preview production build
```

### Browser Requirements

| Feature | Chrome/Edge | Firefox | Safari |
|---------|-------------|---------|--------|
| WebNN   | ✅ (flag)   | ❌      | ❌     |
| WASM    | ✅          | ✅      | ✅     |
| Camera  | ✅          | ✅      | ✅     |

Enable WebNN in Edge: `edge://flags/#web-machine-learning-neural-network` → set to **Enabled** → Relaunch

Enable WebNN in Chrome: `chrome://flags/#enable-web-machine-learning-neural-network-api`

## 📖 Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design and data flow
- [Product](docs/PRODUCT.md) — Vision, UX principles, feature priorities
- [Models](docs/MODELS.md) — How to source and add ONNX models
- [Decisions](docs/DECISIONS.md) — Technical decisions and rationale

## 📄 License

MIT
