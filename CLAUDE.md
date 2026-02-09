# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IC-Light ("Imposing Consistent Light") is a research project for manipulating image illumination using Stable Diffusion 1.5. It provides two relighting models exposed via Gradio web UIs:

- **Text-conditioned** (`gradio_demo.py`): Relights foreground images using text prompts and lighting direction preferences
- **Background-conditioned** (`gradio_demo_bg.py`): Relights foreground images using a background image for lighting context

## Running

```bash
# Apple Silicon / macOS (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# CUDA / Windows / Linux
conda create -n iclight python=3.12
conda activate iclight
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# Text-conditioned relighting
python gradio_demo.py

# Background-conditioned relighting
python gradio_demo_bg.py

# CLI options (both demos)
python gradio_demo.py --host 0.0.0.0 --port 7860 --model-dir ./models --model stablediffusionapi/realistic-vision-v51 --output-dir ./outputs

# Tests and linting
pytest tests/
ruff check .
```

Models download automatically to `./models/` (or `--model-dir`) on first run. The Gradio UI launches on `0.0.0.0:7860` by default (configurable via `--host`/`--port`). Generated images are auto-saved to `./outputs/` (or `--output-dir`).

**Device Support**: Code automatically detects and uses MPS (Apple Silicon), CUDA (NVIDIA GPU), or CPU. On MPS, VAE uses float16 instead of bfloat16 for compatibility.

**Scheduler Note**: MPS defaults to DDIM scheduler (DPMSolverMultistepScheduler has off-by-one indexing errors on MPS). CUDA defaults to DPM++ 2M SDE Karras. Users can switch schedulers via the dropdown in Advanced options (DDIM, Euler a, DPM++ 2M SDE Karras).

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

- `utils.py`: Shared utility module containing device detection, model loading, schedulers, pipelines, image conversion, prompt encoding, background removal, and GPU memory management. Both demos import from here.
- `pyproject.toml`: Project metadata, dependencies, and tool configuration (ruff linter). Entry points: `python gradio_demo.py` (fc) and `python gradio_demo_bg.py` (fbc).
- `briarmbg.py`: BRIA RMBG 1.4 background removal model (U2-Net architecture). Used to extract foreground alpha mattes via `run_rmbg()`. Non-commercial license — replace with BiRefNet for commercial use.
- `db_examples.py`: Hardcoded example data (image paths, prompts, settings) for the Gradio UI example galleries.
- `imgs/`: Example input images and pre-computed outputs. `imgs/bgs/` contains background images for the background-conditioned demo.
- `TECHNICAL.md`: In-depth technical documentation covering pipeline flow, weight merging, normal map computation, precision handling, and improvement paths.
- `tests/test_utils.py`: Unit tests for utility functions (`pytest tests/`).

### Tensor Conversion Conventions

- `numpy2pytorch`: Normalizes uint8 [0,255] to float [-1,1] using `x/127.0 - 1.0` (so 127 maps to exactly 0.0)
- `pytorch2numpy`: Inverse mapping with optional quantization to uint8
- Image tensors use NCHW format (PyTorch) ↔ NHWC/HWC (numpy)

### dtype Strategy

- UNet and text encoder: `float16`
- VAE: `bfloat16` (CUDA/CPU) or `float16` (MPS - for compatibility)
- RMBG: `float32`

## Dependencies

Key versions: `diffusers>=0.36.0`, `transformers>=5.1.0`, `gradio>=6.5.0`, `peft>=0.18.0`, `protobuf==3.20`, `numpy<2.0`. Base model is `stablediffusionapi/realistic-vision-v51` (SD1.5 fine-tune).

**Python version**: 3.12 recommended. Python 3.13+ removed the `audioop` module which breaks some Gradio dependencies.

### Gradio 6 API Notes

- Use `sources=['upload']` (not `source='upload'`) for `gr.Image`
- Gallery selection uses `evt.index` to access original data lists
- Image data must be uint8 numpy arrays for proper display
- `gr.Examples` has a bug generating malformed file URLs — use `gr.Gallery` + `select` handler instead
- `block.launch()` requires `allowed_paths=[os.path.abspath('imgs/')]` to serve example images
- `gr.Progress(track_tqdm=True)` as last param in handler functions captures diffusers' internal tqdm progress bars in the Gradio UI. Do NOT use `callback_on_step_end` — it doesn't flush to the SSE frontend.
- Gradio 6 info text DOM: `[data-testid='block-info']` is the **label** span, not the info text. The info tip text is the next sibling `<div>` containing `.prose`. Use `span.nextElementSibling` in JS to target info tips.

### Error Handling

- `process_relight` and `process_normal` are wrapped with try/except that catches OOM errors and surfaces user-friendly `gr.Error` messages
- RuntimeError with "out of memory" triggers `clear_gpu_cache()` before raising `gr.Error`
- All unexpected exceptions are logged with `logger.exception()` before being re-raised as `gr.Error`

### GPU Memory Management

- `clear_gpu_cache(device)` in `utils.py` calls `torch.cuda.empty_cache()` or `torch.mps.empty_cache()` based on device type
- Called at the end of each `process()` function after generating results

### Image Output Saving

- `save_outputs(images, output_dir, prefix)` in `utils.py` saves generated images as timestamped PNGs
- Called automatically after each generation in `process_relight()` and `process_normal()`
- Output directory configurable via `--output-dir` CLI arg (default: `./outputs/`)
- Files named `{prefix}_{timestamp}_{index}.png` (e.g., `fc_relight_20250207_143022_00.png`)