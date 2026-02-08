# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IC-Light ("Imposing Consistent Light") is a research project for manipulating image illumination using Stable Diffusion 1.5. It provides two relighting models exposed via Gradio web UIs:

- **Text-conditioned** (`gradio_demo.py`): Relights foreground images using text prompts and lighting direction preferences
- **Background-conditioned** (`gradio_demo_bg.py`): Relights foreground images using a background image for lighting context

## Running

```bash
# Setup
conda create -n iclight python=3.10
conda activate iclight

# For CUDA (NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For Apple Silicon (MPS):
pip install torch torchvision

pip install -r requirements.txt

# Text-conditioned relighting
python gradio_demo.py

# Background-conditioned relighting
python gradio_demo_bg.py
```

Models download automatically to `./models/` on first run. The Gradio UI launches on `0.0.0.0`.

**Device Support**: Code automatically detects and uses MPS (Apple Silicon), CUDA (NVIDIA GPU), or CPU. On MPS, VAE uses float16 instead of bfloat16 for compatibility.

## Architecture

### UNet Modification Pattern (critical to understand)

Both demos modify the SD1.5 UNet's input convolution to accept extra latent channels, then monkey-patch `unet.forward` via `hooked_unet_forward` to concatenate conditioning latents at inference time:

- **Text-conditioned** (`gradio_demo.py`): Expands `conv_in` from 4 to **8** channels (4 noise + 4 foreground latent). Uses model `iclight_sd15_fc.safetensors`.
- **Background-conditioned** (`gradio_demo_bg.py`): Expands `conv_in` from 4 to **12** channels (4 noise + 4 foreground + 4 background latent). Uses model `iclight_sd15_fbc.safetensors`.

Model weights are loaded as **offsets** added to the base SD1.5 weights (`sd_merged = sd_origin + sd_offset`), not replacements.

### Two-pass Generation Pipeline

Both demos use a two-pass approach:
1. **Low-res pass**: Generate at base resolution (default 512xN). Text-conditioned uses `i2i_pipe` with a gradient initial latent (or `t2i_pipe` if BGSource.NONE); background-conditioned always uses `t2i_pipe`.
2. **High-res pass**: Upscale result, re-encode, and run `i2i_pipe` with `highres_denoise` strength for refinement.

### Key Components

- `briarmbg.py`: BRIA RMBG 1.4 background removal model (U2-Net architecture). Used to extract foreground alpha mattes via `run_rmbg()`. Non-commercial license — replace with BiRefNet for commercial use.
- `db_examples.py`: Hardcoded example data (image paths, prompts, settings) for the Gradio UI example galleries.
- `imgs/`: Example input images and pre-computed outputs. `imgs/bgs/` contains background images for the background-conditioned demo.

### Tensor Conversion Conventions

- `numpy2pytorch`: Normalizes uint8 [0,255] to float [-1,1] using `x/127.0 - 1.0` (so 127 maps to exactly 0.0)
- `pytorch2numpy`: Inverse mapping with optional quantization to uint8
- Image tensors use NCHW format (PyTorch) ↔ NHWC/HWC (numpy)

### dtype Strategy

- UNet and text encoder: `float16`
- VAE: `bfloat16` (CUDA/CPU) or `float16` (MPS - for compatibility)
- RMBG: `float32`

## Dependencies

Pinned versions that matter: `diffusers==0.27.2`, `transformers==4.36.2`, `gradio==3.41.2`, `protobuf==3.20`. Base model is `stablediffusionapi/realistic-vision-v51` (SD1.5 fine-tune).