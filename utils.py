import argparse
import json
import logging
import os
import math
from datetime import datetime
from typing import Optional, Any, Union
import numpy as np
import torch
import safetensors.torch as sf

from PIL import Image

logger = logging.getLogger(__name__)
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from torch.hub import download_url_to_file


# --- CLI Arguments ---

def parse_common_args(description: str = 'IC-Light Demo') -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=7860, help='Server port (default: 7860)')
    parser.add_argument('--model-dir', type=str, default='./models', help='Directory for model weights (default: ./models)')
    parser.add_argument('--model', type=str, default='stablediffusionapi/realistic-vision-v51', help='HuggingFace model name for SD1.5 base (default: stablediffusionapi/realistic-vision-v51)')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Directory for saved outputs (default: ./outputs)')
    return parser.parse_args()


# --- Logging ---

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )


# --- Device Detection ---

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device


def move_models_to_device(device: torch.device, text_encoder: CLIPTextModel, vae: AutoencoderKL, unet: UNet2DConditionModel, rmbg: BriaRMBG) -> tuple[CLIPTextModel, AutoencoderKL, UNet2DConditionModel, BriaRMBG]:
    # MPS doesn't support bfloat16 as well as CUDA, so use float16 for all on MPS
    if device.type == 'mps':
        text_encoder = text_encoder.to(device=device, dtype=torch.float16)
        vae = vae.to(device=device, dtype=torch.float16)
        unet = unet.to(device=device, dtype=torch.float16)
        rmbg = rmbg.to(device=device, dtype=torch.float32)
    else:
        text_encoder = text_encoder.to(device=device, dtype=torch.float16)
        vae = vae.to(device=device, dtype=torch.bfloat16)
        unet = unet.to(device=device, dtype=torch.float16)
        rmbg = rmbg.to(device=device, dtype=torch.float32)
    return text_encoder, vae, unet, rmbg


def clear_gpu_cache(device: torch.device) -> None:
    """Clear GPU memory cache to free up unused memory."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def save_outputs(images: list[np.ndarray], output_dir: str, prefix: str = 'relight', seed: Optional[int] = None) -> list[str]:
    """Save output images to the output directory with optional seed in filename."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_paths = []
    for i, img in enumerate(images):
        if seed is not None:
            filename = f'{prefix}_{timestamp}_seed{seed}_{i:02d}.png'
        else:
            filename = f'{prefix}_{timestamp}_{i:02d}.png'
        filepath = os.path.join(output_dir, filename)
        Image.fromarray(img).save(filepath)
        saved_paths.append(filepath)
        logger.info("Saved output to %s", filepath)
    return saved_paths


# --- UNet Setup ---

def setup_unet(unet: UNet2DConditionModel, in_channels: int) -> UNet2DConditionModel:
    """Modify UNet conv_in for extra channels and hook forward pass."""
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(
            in_channels, unet.conv_in.out_channels,
            unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    unet_original_forward = unet.forward

    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

    unet.forward = hooked_unet_forward
    return unet


# --- Model Loading ---

def load_models(sd15_name: str = 'stablediffusionapi/realistic-vision-v51') -> tuple[CLIPTokenizer, CLIPTextModel, AutoencoderKL, UNet2DConditionModel, BriaRMBG]:
    logger.info("Loading models from %s...", sd15_name)
    tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
    rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
    logger.info("Models loaded successfully")
    return tokenizer, text_encoder, vae, unet, rmbg


def load_and_merge_weights(unet: UNet2DConditionModel, model_path: str, model_url: str) -> None:
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    if not os.path.exists(model_path):
        logger.info("Downloading model weights to %s...", model_path)
        download_url_to_file(url=model_url, dst=model_path)

    sd_offset = sf.load_file(model_path)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged


# --- Schedulers ---

def create_schedulers() -> tuple[DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler]:
    ddim = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    euler_a = EulerAncestralDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        steps_offset=1
    )

    dpmpp_2m_sde_karras = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        steps_offset=1
    )

    return ddim, euler_a, dpmpp_2m_sde_karras


def get_default_scheduler(device: torch.device, ddim: DDIMScheduler, dpmpp_2m_sde_karras: DPMSolverMultistepScheduler) -> Union[DDIMScheduler, DPMSolverMultistepScheduler]:
    # Use DDIM scheduler for MPS compatibility (DPMSolver has indexing issues on MPS)
    return ddim if device.type == 'mps' else dpmpp_2m_sde_karras


# --- Pipelines ---

def create_pipelines(vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: Union[DDIMScheduler, DPMSolverMultistepScheduler]) -> tuple[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline]:
    t2i_pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None
    )

    i2i_pipe = StableDiffusionImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None
    )

    return t2i_pipe, i2i_pipe


def enable_sdp(unet: UNet2DConditionModel, vae: AutoencoderKL) -> None:
    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())


# --- Prompt Encoding ---

@torch.inference_mode()
def encode_prompt_inner(txt: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: torch.device) -> torch.Tensor:
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x: list[int], p: int, i: int) -> list[int]:
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt: str, negative_prompt: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    c = encode_prompt_inner(positive_prompt, tokenizer, text_encoder, device)
    uc = encode_prompt_inner(negative_prompt, tokenizer, text_encoder, device)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


# --- Image Conversion ---

@torch.inference_mode()
def pytorch2numpy(imgs: torch.Tensor, quant: bool = True) -> list[np.ndarray]:
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs: list[np.ndarray]) -> torch.Tensor:
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


# --- Image Resizing ---

def resize_and_center_crop(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


# --- Mask Refinement ---

def refine_mask(mask: np.ndarray, blur_radius: int = 0, expand: int = 0, threshold: float = 0.5) -> np.ndarray:
    """
    Refine alpha mask with blur, expand/contract, and threshold operations.

    Args:
        mask: Alpha mask (H, W) or (H, W, 1) with values in [0, 1]
        blur_radius: Gaussian blur radius in pixels (0 = no blur, 1-10 typical range)
        expand: Pixels to expand (positive) or contract (negative) the mask (-50 to +50 typical)
        threshold: Binary threshold value (0.0 to 1.0). Values above threshold become 1.0, below become 0.0.
                  Use 0.5 for no thresholding effect.

    Returns:
        Refined mask with same shape as input
    """
    import cv2

    # Ensure 2D mask
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    result = mask.copy()

    # 1. Expand/contract (morphological operations)
    if expand != 0:
        kernel_size = abs(expand) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if expand > 0:
            # Dilate to expand
            result = cv2.dilate(result, kernel, iterations=1)
        else:
            # Erode to contract
            result = cv2.erode(result, kernel, iterations=1)

    # 2. Blur
    if blur_radius > 0:
        kernel_size = blur_radius * 2 + 1
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)

    # 3. Threshold (only apply if significantly different from 0.5)
    if abs(threshold - 0.5) > 0.01:
        result = np.where(result > threshold, 1.0, 0.0).astype(np.float32)

    return result.clip(0, 1)


# --- Background Removal ---

@torch.inference_mode()
def run_rmbg(img: np.ndarray, rmbg_model: BriaRMBG, device: torch.device, sigma: float = 0.0, blend_strength: float = 1.0, mask_blur: int = 0, mask_expand: int = 0, mask_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove background using RMBG model.

    Args:
        img: Input image
        rmbg_model: Background removal model
        device: Torch device
        sigma: Brightness adjustment applied to foreground before blending (-255 to +255)
        blend_strength: Controls alpha mask strength (0.0 to 1.0). Lower values make background removal less aggressive.

    Returns:
        Tuple of (foreground image, alpha mask)
    """
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg_model(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)

    # Apply mask refinement (blur, expand/contract, threshold)
    if mask_blur > 0 or mask_expand != 0 or abs(mask_threshold - 0.5) > 0.01:
        alpha[:, :, 0] = refine_mask(alpha[:, :, 0], blur_radius=mask_blur, expand=mask_expand, threshold=mask_threshold)

    # Apply blend strength to alpha mask
    alpha = alpha * blend_strength
    alpha = alpha.clip(0, 1)

    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


# --- Favorite Seeds Management ---

def get_favorites_path(output_dir: str) -> str:
    """Get the path to the favorite_seeds.json file."""
    return os.path.join(output_dir, 'favorite_seeds.json')


def load_favorite_seeds(output_dir: str) -> list[dict[str, Any]]:
    """Load favorite seeds from JSON file."""
    favorites_path = get_favorites_path(output_dir)
    if not os.path.exists(favorites_path):
        return []

    try:
        with open(favorites_path, 'r') as f:
            favorites = json.load(f)
            logger.info("Loaded %d favorite seeds from %s", len(favorites), favorites_path)
            return favorites
    except Exception as e:
        logger.error("Failed to load favorite seeds: %s", e)
        return []


def save_favorite_seed(output_dir: str, seed: int, label: str = "", settings: Optional[dict[str, Any]] = None) -> bool:
    """Save a favorite seed to JSON file."""
    favorites = load_favorite_seeds(output_dir)

    # Create new favorite entry
    new_favorite = {
        'seed': int(seed),
        'label': label or f"Seed {seed}",
        'timestamp': datetime.now().isoformat(),
    }

    # Add settings if provided
    if settings:
        new_favorite['settings'] = settings

    # Check if seed already exists
    existing_idx = None
    for idx, fav in enumerate(favorites):
        if fav['seed'] == seed:
            existing_idx = idx
            break

    if existing_idx is not None:
        # Update existing entry
        favorites[existing_idx] = new_favorite
        logger.info("Updated existing favorite seed %d", seed)
    else:
        # Add new entry
        favorites.append(new_favorite)
        logger.info("Added new favorite seed %d", seed)

    # Save to file
    favorites_path = get_favorites_path(output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(favorites_path, 'w') as f:
            json.dump(favorites, f, indent=2)
        logger.info("Saved favorite seeds to %s", favorites_path)
        return True
    except Exception as e:
        logger.error("Failed to save favorite seeds: %s", e)
        return False


def get_favorite_seeds_choices(output_dir: str) -> list[tuple[str, int]]:
    """Get favorite seeds as choices for Gradio dropdown."""
    favorites = load_favorite_seeds(output_dir)
    if not favorites:
        return []

    # Return list of (label, seed) tuples
    choices = [(f"{fav['label']} (seed: {fav['seed']})", fav['seed']) for fav in favorites]
    return choices


# --- Preset Management ---

def get_presets_path(output_dir: str) -> str:
    """Get the path to the presets.json file."""
    return os.path.join(output_dir, 'presets.json')


def load_presets(output_dir: str) -> list[dict[str, Any]]:
    """Load presets from JSON file."""
    presets_path = get_presets_path(output_dir)
    if not os.path.exists(presets_path):
        return []

    try:
        with open(presets_path, 'r') as f:
            presets = json.load(f)
            logger.info("Loaded %d presets from %s", len(presets), presets_path)
            return presets
    except Exception as e:
        logger.error("Failed to load presets: %s", e)
        return []


def save_preset(output_dir: str, name: str, settings: dict[str, Any]) -> bool:
    """Save a complete preset with all settings."""
    presets = load_presets(output_dir)

    # Create new preset entry
    new_preset = {
        'name': name,
        'timestamp': datetime.now().isoformat(),
        'settings': settings
    }

    # Check if preset with same name already exists
    existing_idx = None
    for idx, preset in enumerate(presets):
        if preset['name'] == name:
            existing_idx = idx
            break

    if existing_idx is not None:
        # Update existing entry
        presets[existing_idx] = new_preset
        logger.info("Updated existing preset '%s'", name)
    else:
        # Add new entry
        presets.append(new_preset)
        logger.info("Added new preset '%s'", name)

    # Save to file
    presets_path = get_presets_path(output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(presets_path, 'w') as f:
            json.dump(presets, f, indent=2)
        logger.info("Saved presets to %s", presets_path)
        return True
    except Exception as e:
        logger.error("Failed to save presets: %s", e)
        return False


def delete_preset(output_dir: str, name: str) -> bool:
    """Delete a preset by name."""
    presets = load_presets(output_dir)

    # Find and remove preset
    presets = [p for p in presets if p['name'] != name]

    # Save to file
    presets_path = get_presets_path(output_dir)
    try:
        with open(presets_path, 'w') as f:
            json.dump(presets, f, indent=2)
        logger.info("Deleted preset '%s'", name)
        return True
    except Exception as e:
        logger.error("Failed to delete preset: %s", e)
        return False


def get_preset_choices(output_dir: str) -> list[str]:
    """Get preset names as choices for Gradio dropdown."""
    presets = load_presets(output_dir)
    if not presets:
        return []

    # Return list of preset names
    choices = [preset['name'] for preset in presets]
    return choices


def get_preset_by_name(output_dir: str, name: str) -> Optional[dict[str, Any]]:
    """Get a specific preset by name."""
    presets = load_presets(output_dir)
    for preset in presets:
        if preset['name'] == name:
            return preset
    return None
