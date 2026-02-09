#!/usr/bin/env python3
"""Quick test for foreground preview functionality."""

import numpy as np
from utils import preview_foreground, load_models, get_device, move_models_to_device

# Load models
print("Loading models...")
tokenizer, text_encoder, vae, unet, rmbg = load_models('stablediffusionapi/realistic-vision-v51')
device = get_device()
text_encoder, vae, unet, rmbg = move_models_to_device(device, text_encoder, vae, unet, rmbg)
print(f"Models loaded on {device}")

# Create a simple test image (512x512 RGB with a circle in the middle)
print("\nCreating test image...")
img = np.zeros((512, 512, 3), dtype=np.uint8)
# Draw a white circle
center = (256, 256)
radius = 100
for y in range(512):
    for x in range(512):
        if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
            img[y, x] = [255, 255, 255]

print("Running preview_foreground...")
original, foreground_rgba, alpha_viz = preview_foreground(img, rmbg, device)

# Verify outputs
print("\nVerifying outputs:")
print(f"  Original shape: {original.shape}, dtype: {original.dtype}")
print(f"  Foreground RGBA shape: {foreground_rgba.shape}, dtype: {foreground_rgba.dtype}")
print(f"  Alpha visualization shape: {alpha_viz.shape}, dtype: {alpha_viz.dtype}")

assert original.shape == (512, 512, 3), "Original should be RGB"
assert foreground_rgba.shape == (512, 512, 4), "Foreground should be RGBA"
assert alpha_viz.shape == (512, 512, 3), "Alpha viz should be RGB"
assert original.dtype == np.uint8, "All outputs should be uint8"
assert foreground_rgba.dtype == np.uint8, "All outputs should be uint8"
assert alpha_viz.dtype == np.uint8, "All outputs should be uint8"

print("\n✅ All tests passed! Preview functionality works correctly.")
