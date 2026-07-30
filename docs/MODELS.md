# Models

## Overview

PaintMe uses fast neural style transfer models based on the Johnson et al. architecture. These models take an image as input and produce a stylized version as output in a single forward pass (no iterative optimization needed), making them suitable for real-time video processing.

## Architecture: Johnson et al. (Perceptual Losses)

The model architecture consists of:

1. **Encoder** — 3 convolutional layers with stride 2 (downsampling)
2. **Residual blocks** — 5 residual blocks for feature transformation
3. **Decoder** — 3 transposed convolutional layers (upsampling)
4. **Instance normalization** — Used instead of batch normalization for style transfer

Reference: *"Perceptual Losses for Real-Time Style Transfer and Super-Resolution"* (Johnson et al., 2016)

## Expected Input/Output Shapes

| Property | Specification |
|----------|---------------|
| Input name | `input` (may vary) |
| Input shape | `[1, 3, H, W]` (batch=1, channels=RGB) |
| Input range | `[0, 255]` (uint8 pixel values as float32) |
| Output name | `output` (may vary) |
| Output shape | `[1, 3, H, W]` (same as input) |
| Output range | `~[0, 255]` (may slightly exceed, clamped in postprocessing) |
| Data type | `float32` |

**Important**: The model expects RGB channel order (not BGR). Pixel values are in [0, 255] range — NOT normalized to [0, 1].

## Supported Resolution

The current ONNX Model Zoo models have **fixed 224×224 input shapes**. The app hardcodes resolution to 224.

| Resolution | Status | Notes |
|------------|--------|-------|
| 224×224 | ✅ Active | Fixed input shape from ONNX Model Zoo models |
| 448×448+ | ⏸️ Future | Requires re-exporting models with dynamic shapes |

The Hi-Res toggle in the UI is hidden until multi-resolution models are available.

## How to Source Models

### Option 1: ONNX Model Zoo

The ONNX Model Zoo has pre-trained fast neural style transfer models:

```
https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style
```

These are already in the correct format. Download and rename to match the expected naming convention.

### Option 2: Export from PyTorch

If you have trained PyTorch models or want to train new ones:

```python
import torch
import torch.onnx

# Load your trained model
model = TransformerNet()  # Johnson et al. architecture
model.load_state_dict(torch.load('starry_night.pth', map_location='cpu'))
model.eval()

# Export to ONNX
resolution = 256
dummy_input = torch.randn(1, 3, resolution, resolution)

torch.onnx.export(
    model,
    dummy_input,
    f'starry_night_{resolution}.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=13,
    dynamic_axes=None  # Fixed size for better WebNN optimization
)
```

### Option 3: Convert from TensorFlow/other formats

Use the ONNX converter tools:

```bash
# From TensorFlow SavedModel
python -m tf2onnx.convert --saved-model ./saved_model --output model.onnx --opset 13

# From TFLite
python -m tf2onnx.convert --tflite model.tflite --output model.onnx --opset 13
```

## How to Add a New Style

1. **Get the model** — Train or download an ONNX model for your style
2. **Name it correctly** — `{style_name}_{resolution}.onnx` (e.g., `wave_256.onnx`)
3. **Place in public/models/** — The file will be served by Vite
4. **Add a thumbnail** — Create a 200×140px JPEG preview in `public/thumbnails/{style_name}.jpg`
5. **Register in code** — Add an entry to the `STYLES` array in `src/inference.js`:

```javascript
{ name: 'wave', label: 'Great Wave', thumbnail: '/thumbnails/wave.jpg' }
```

6. **Test** — Run `npm run dev` and verify the style loads and produces good output

## Quantization Guidance

For better performance on resource-constrained devices, models can be quantized:

### INT8 Quantization (recommended for WASM)

```python
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    'starry_night_256.onnx',
    'starry_night_256_int8.onnx',
    weight_type=QuantType.QInt8
)
```

### Float16 Quantization (recommended for WebNN/GPU)

```python
from onnxruntime.transformers import float16
import onnx

model = onnx.load('starry_night_256.onnx')
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, 'starry_night_256_fp16.onnx')
```

**Notes on quantization:**
- INT8 reduces model size by ~4x but may introduce visual artifacts
- FP16 reduces size by ~2x with minimal quality loss
- WebNN GPU provider handles FP16 natively
- Always visually compare quantized output vs. original before shipping

## Model Size Reference

| Style | FP32 (256) | FP32 (512) | INT8 (256) |
|-------|-----------|-----------|-----------|
| Typical | ~6.5 MB | ~6.5 MB | ~1.7 MB |

The model weights are the same regardless of resolution (the architecture is fully convolutional), but we export at fixed sizes for runtime optimization.

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| "Failed to fetch model" | Model file not in public/models/ | Check file path and naming |
| Garbled output | Wrong input range (using 0-1 instead of 0-255) | Ensure preprocessing outputs [0, 255] |
| BGR/RGB swap | Channel order mismatch | Verify model expects RGB |
| Session creation fails | Unsupported ONNX opset | Re-export with opset_version=13 |
| Very slow inference | Large model on WASM | Use 256px resolution or quantized model |
