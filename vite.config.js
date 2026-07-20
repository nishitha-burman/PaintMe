import { defineConfig } from 'vite';

export default defineConfig({
  // Serve ONNX Runtime WASM files from node_modules
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  server: {
    headers: {
      // Required for SharedArrayBuffer (used by ONNX Runtime WASM)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  }
});
