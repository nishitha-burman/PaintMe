# ONNX Style Transfer Models

Place your `.onnx` model files here. They are git-ignored due to their size.

## Expected naming convention

```
{style_name}_{resolution}.onnx
```

Examples:
- `starry_night_256.onnx`
- `starry_night_512.onnx`
- `the_scream_256.onnx`
- `mosaic_256.onnx`

## How to get models

### Option 1: Export from PyTorch (recommended)

Use the Johnson et al. fast style transfer architecture. Train or use pre-trained checkpoints, then export:

```python
import torch

model = StyleTransferNet()
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model, dummy_input, 'starry_night_256.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {2: 'height', 3: 'width'},
                  'output': {2: 'height', 3: 'width'}}
)
```

### Option 2: ONNX Model Zoo

Download pre-trained style transfer models from:
- https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style

### Option 3: Use the provided script (coming soon)

```bash
python scripts/download_models.py
```

## Input/Output specification

- **Input**: `float32[1, 3, H, W]` — RGB image, pixel values in range [0, 255]
- **Output**: `float32[1, 3, H, W]` — Styled RGB image, pixel values in range [0, 255]

The models use the Johnson et al. architecture with instance normalization.
