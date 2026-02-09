# IC-Light Technical Documentation

## Table of Contents

1. [What IC-Light Is (And Isn't)](#1-what-ic-light-is-and-isnt)
2. [The Core Scientific Idea](#2-the-core-scientific-idea)
3. [The Two Models](#3-the-two-models)
4. [Exact Pipeline Flow](#4-exact-pipeline-flow)
5. [The Weight Merging Trick](#5-the-weight-merging-trick)
6. [The Normal Map Computation](#6-the-normal-map-computation)
7. [Precision and Device Handling](#7-precision-and-device-handling)
8. [How To Improve It](#8-how-to-improve-it)
9. [Key Limitations](#9-key-limitations)

---

## 1. What IC-Light Is (And Isn't)

**IC-Light does NOT learn.** It is purely an inference-time application. All neural networks are pre-trained and frozen. Every time you click "Relight," it runs forward passes through fixed models. No weights are updated, no training occurs, no data is saved to improve future runs.

The system is a **carefully orchestrated pipeline of 4 pre-trained neural networks** that work together:

| Model | Size | Purpose | Source |
|---|---|---|---|
| CLIP Text Encoder | ~500MB | Converts text prompts to embedding vectors | `stablediffusionapi/realistic-vision-v51` |
| UNet2D (modified) | ~1.7GB | The core diffusion denoiser | SD1.5 base + IC-Light offset weights |
| VAE (AutoencoderKL) | ~165MB | Encodes images to/from latent space | `stablediffusionapi/realistic-vision-v51` |
| BRIA RMBG 1.4 | ~176MB | Removes backgrounds from input images | `briaai/RMBG-1.4` |

Total model memory: ~3.2GB on disk, ~2-4GB in GPU VRAM depending on precision.

---

## 2. The Core Scientific Idea

IC-Light stands for **"Imposing Consistent Light."** The key insight is from physics:

> In HDR (High Dynamic Range) space, light transport is linear. This means: blending the *appearances* of two separate light sources is mathematically equivalent to the *appearance under both light sources combined*.

The authors trained the UNet to satisfy this consistency constraint using MLPs in latent space. The result: the model produces relightings so physically consistent that you can derive normal maps from 4 directional relightings (left, right, top, bottom) - despite never being trained on normal map data.

**Citation:**
```
@inproceedings{
    zhang2025scaling,
    title={Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization
           and Editing by Imposing Consistent Light Transport},
    author={Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=u1cQYxRI1H}
}
```

---

## 3. The Two Models

### Foreground-Conditioned (FC) - `gradio_demo.py`

- **UNet input: 8 channels** (4 noisy latent + 4 foreground latent)
- Weights: `iclight_sd15_fc.safetensors` (1.7GB)
- Uses text + foreground image to relight
- Optional directional lighting bias via synthetic gradient backgrounds

### Foreground+Background-Conditioned (FBC) - `gradio_demo_bg.py`

- **UNet input: 12 channels** (4 noisy latent + 4 foreground latent + 4 background latent)
- Weights: `iclight_sd15_fbc.safetensors` (1.7GB)
- Uses text + foreground + background image for relighting
- Can also compute normal maps (4x generation)

---

## 4. Exact Pipeline Flow

Here's what happens when you click "Relight" in `gradio_demo.py`:

```
INPUT: photo.jpg + "sunshine from window" + "Left Light"
```

### Step 1: Background Removal

**File:** `utils.py` function `run_rmbg` (line 322-334)

```
photo.jpg --> BRIA RMBG 1.4 --> alpha matte
result = 127 + (pixel - 127) * alpha    <-- composites onto neutral gray (127)
```

The formula `127 + (img - 127) * alpha` blends the foreground onto a flat gray background. The value 127 is chosen because it maps to exactly 0.0 in the normalized [-1, 1] tensor space (see `numpy2pytorch`: `x / 127.0 - 1.0`).

### Step 2: Generate Synthetic Background

**File:** `gradio_demo.py` (lines 78-93)

```
"Left Light" --> numpy gradient: 255 (left) --> 0 (right)
Tiled into a (H, W, 3) uint8 image
```

### Step 3: Encode Foreground into Latent Space

**File:** `gradio_demo.py` (lines 99-102)

```
foreground (512x640 uint8)
  --> numpy2pytorch: normalize to [-1, 1] float tensor
  --> VAE encoder: 512x640x3 --> 64x80x4 latent
  --> multiply by scaling_factor (0.18215)
  = concat_conds (foreground latent conditioning)
```

### Step 4: Encode Text Prompt

**File:** `utils.py` function `encode_prompt_pair` (lines 250-268)

```
"sunshine from window, best quality"
  --> CLIP Tokenizer: split into token IDs
  --> Handle long prompts: chunk into 75-token segments with BOS/EOS
  --> CLIP Text Encoder: token IDs --> hidden states (1, 77, 768)
  --> Same for negative prompt
  --> Pad/repeat to match lengths
  = conds, unconds (text embeddings)
```

### Step 5: Low-Resolution Diffusion

**File:** `gradio_demo.py` (lines 108-138)

Two paths depending on whether a background was selected:
- **No background (BGSource.NONE):** Text-to-Image pipeline generates from pure noise
- **With background (gradient):** Image-to-Image pipeline starts from the gradient's latent

The **magic happens in the hooked UNet forward pass** (`setup_unet` in `utils.py` lines 118-123):

```python
def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds']
    new_sample = torch.cat([sample, c_concat], dim=1)  # 4ch noise + 4ch fg = 8ch
    return unet_original_forward(new_sample, ...)
```

At every denoising step, the foreground latent is concatenated channel-wise with the noisy latent. This is how the UNet "sees" the foreground - not through cross-attention (like text), but through direct channel concatenation at the input.

For the FBC model, it's `4ch noise + 4ch foreground + 4ch background = 12ch`.

### Step 6: Decode, Upscale, Re-encode

**File:** `gradio_demo.py` (lines 140-156)

```
low-res latent (64x80x4)
  --> VAE decode --> pixel image (512x640)
  --> Upscale by highres_scale (1.5x) --> 768x960
  --> Re-encode through VAE --> larger latent (96x120x4)
  --> Re-encode foreground at new size
```

### Step 7: High-Resolution Refinement

**File:** `gradio_demo.py` (lines 158-175)

```
Upscaled latent + re-encoded foreground
  --> Image-to-Image pipeline at highres_denoise strength
  --> VAE decode --> final output (768x960)
```

The two-pass approach (low-res generation + high-res refinement) is critical for quality. The first pass establishes composition and lighting, the second adds detail.

---

## 5. The Weight Merging Trick

**File:** `utils.py` function `load_and_merge_weights` (lines 148-151)

The IC-Light weights are stored as **offsets** from the base SD1.5 model:

```python
sd_offset = sf.load_file(model_path)       # IC-Light delta weights
sd_origin = unet.state_dict()               # Base SD1.5 weights
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
```

This means: **IC-Light = SD1.5 + learned relighting delta.**

This is why you can swap the base model (`--model` flag) - using Realistic Vision v5.1 gives more photorealistic results than vanilla SD1.5, while the relighting capability comes from the offset.

---

## 6. The Normal Map Computation

**File:** `gradio_demo_bg.py` function `process_normal` (lines 204-254)

```
1. Generate 4 relightings: left, right, bottom, top
2. Compute ambient = average of all 4
3. For each direction: compute relative difference from ambient
4. u = (right - left) * 0.5         <-- horizontal component
5. v = (top - bottom) * 0.5         <-- vertical component
6. h = (1 - u^2 - v^2)^(sigma/2)   <-- height from unit normal constraint
7. normal = normalize([u, v, h])
8. Mask by alpha matte
```

This works because the relighting model is trained with consistent light transport. The difference between left-lit and right-lit images approximates the surface normal's X component.

---

## 7. Precision and Device Handling

**File:** `utils.py` function `move_models_to_device` (lines 65-77)

| Component | MPS (Apple Silicon) | CUDA (NVIDIA) |
|---|---|---|
| Text Encoder | float16 | float16 |
| VAE | **float16** | **bfloat16** |
| UNet | float16 | float16 |
| RMBG | float32 | float32 |
| Default Scheduler | DDIM | DPM++ 2M SDE Karras |

**Why these choices:**
- MPS can't use bfloat16 (not supported in Metal)
- RMBG stays float32 because background removal needs precision for clean mattes
- DDIM is forced on MPS because DPMSolverMultistepScheduler has an off-by-one indexing bug in its timestep handling on MPS

---

## 8. How To Improve It

### A. No retraining needed

| Improvement | How | Impact |
|---|---|---|
| **Better base model** | `--model` flag with a different SD1.5 checkpoint | Different art style, better faces |
| **Better background removal** | Replace BRIA RMBG 1.4 with BiRefNet or SAM2 | Cleaner foreground extraction |
| **ControlNet integration** | Add depth/canny/pose conditioning alongside IC-Light | Better structure preservation |
| **Batch processing** | Add CLI batch mode to process folders of images | Productivity |
| **LoRA support** | Load style LoRAs on top of the base model | Custom art styles |
| **Inpainting** | Mask specific regions for selective relighting | More control |
| **Multiple resolutions** | Increase max slider to 2048 | Higher output quality (needs more VRAM) |

### B. Requires retraining

| Improvement | What's needed | Difficulty |
|---|---|---|
| **SDXL IC-Light** | Retrain on SDXL architecture | High |
| **Video relighting** | Train temporal consistency across frames | Very High |
| **Higher consistency** | More training data, longer training | High |
| **Custom subject training** | Fine-tune with DreamBooth/LoRA for specific subjects | Medium |

### C. Code-level improvements

| Area | Current State | Improvement |
|---|---|---|
| **Memory** | All models loaded into VRAM at once | CPU offloading for idle models |
| **Speed** | Two full diffusion passes | Turbo/LCM distilled models for fewer steps |
| **Quantization** | float16 everywhere | INT8 quantization for faster inference |
| **Caching** | No prompt caching | Cache CLIP embeddings for repeated prompts |
| **API** | Gradio only | Add REST API endpoint for integration |

---

## 9. Key Limitations

1. **SD1.5 resolution ceiling** - The model was trained at 512px, so quality degrades significantly above ~768px even with the highres pass
2. **No temporal coherence** - Each frame is independent; cannot be used for consistent video relighting
3. **Foreground-dependent** - Background removal quality directly impacts output quality
4. **Limited prompt understanding** - CLIP's text encoder has a 77-token limit per chunk and doesn't understand complex spatial descriptions well
5. **Single light source bias** - The gradient-based lighting preference is a simple approximation; complex multi-light setups aren't well supported
6. **No explicit light placement** - Cannot drag a light source to a specific position; limited to text descriptions and directional gradients

---

## Architecture Diagram

```
                    +------------------+
                    |   Input Image    |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   BRIA RMBG 1.4  |  Background Removal
                    |   (float32)      |  Produces alpha matte
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v----------+     +------------v-----------+
    | Foreground (on gray)|    | Synthetic Background   |
    | 127 + (px-127)*alpha|    | (gradient or uploaded) |
    +--------+-----------+     +-----------+------------+
              |                             |
    +---------v----------+     +------------v-----------+
    |  VAE Encoder       |     |  VAE Encoder           |
    |  512x640 -> 64x80  |     |  512x640 -> 64x80      |
    +--------+-----------+     +-----------+------------+
              |                             |
              +--------> concat <-----------+
                           |
              +------------v-----------+
              |                        |     +------------------+
              |   Modified UNet        |<----| CLIP Text Encoder|
              |   (8ch FC / 12ch FBC)  |     | "prompt" -> embeds|
              |   Iterative Denoising  |     +------------------+
              |   (25 steps default)   |
              +------------+-----------+
                           |
              +------------v-----------+
              |    VAE Decoder         |  Low-res output
              |    64x80 -> 512x640    |
              +------------+-----------+
                           |
              +------------v-----------+
              |    Upscale (1.5x)      |  768x960
              |    + VAE Re-encode     |
              +------------+-----------+
                           |
              +------------v-----------+
              |   Modified UNet        |  High-res refinement
              |   (img2img, 0.5 str)   |
              +------------+-----------+
                           |
              +------------v-----------+
              |    VAE Decoder         |  Final output
              |    96x120 -> 768x960   |
              +------------+-----------+
                           |
                    +------v-------+
                    | Output Image |
                    +--------------+
```

---

## File Reference

| File | Purpose |
|---|---|
| `gradio_demo.py` | Text-conditioned demo (8-channel UNet, fc model) |
| `gradio_demo_bg.py` | Background-conditioned demo (12-channel UNet, fbc model) |
| `utils.py` | Shared utilities: device detection, model loading, schedulers, pipelines, image ops, prompt encoding, background removal, GPU cache management |
| `briarmbg.py` | BRIA RMBG 1.4 model architecture (U2-Net variant with RSU blocks) |
| `db_examples.py` | Example data for UI galleries |
| `pyproject.toml` | Project metadata, dependencies, ruff config |
| `tests/test_utils.py` | Unit tests for utility functions |
