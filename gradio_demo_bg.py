import logging
import os
import random
from enum import Enum

import gradio as gr
import numpy as np
import torch

import db_examples
from utils import (
    clear_gpu_cache,
    create_pipelines,
    create_schedulers,
    delete_preset,
    enable_sdp,
    encode_prompt_pair,
    get_default_scheduler,
    get_device,
    get_favorite_seeds_choices,
    get_preset_by_name,
    get_preset_choices,
    load_and_merge_weights,
    load_models,
    move_models_to_device,
    numpy2pytorch,
    parse_common_args,
    preprocess_foreground,
    pytorch2numpy,
    resize_and_center_crop,
    resize_without_crop,
    run_rmbg,
    save_favorite_seed,
    save_outputs,
    save_preset,
    setup_logging,
    setup_unet,
)

# CLI args and logging
args = parse_common_args(description='IC-Light: Background Conditioned Relighting')
setup_logging()
logger = logging.getLogger(__name__)

# Load models
sd15_name = args.model
tokenizer, text_encoder, vae, unet, rmbg = load_models(sd15_name)

# Change UNet (12-channel for foreground+background-conditioned model)
unet = setup_unet(unet, in_channels=12)

# Load weights
model_path = os.path.join(args.model_dir, 'iclight_sd15_fbc.safetensors')
load_and_merge_weights(
    unet,
    model_path=model_path,
    model_url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors',
)

# Device
device = get_device()
text_encoder, vae, unet, rmbg = move_models_to_device(device, text_encoder, vae, unet, rmbg)

# SDP
enable_sdp(unet, vae)

# Samplers
ddim_scheduler, euler_a_scheduler, dpmpp_2m_sde_karras_scheduler = create_schedulers()
default_scheduler = get_default_scheduler(device, ddim_scheduler, dpmpp_2m_sde_karras_scheduler)

# Pipelines
t2i_pipe, i2i_pipe = create_pipelines(vae, text_encoder, tokenizer, unet, default_scheduler)

# Scheduler selection
scheduler_map = {
    'DDIM': ddim_scheduler,
    'Euler a': euler_a_scheduler,
    'DPM++ 2M SDE Karras': dpmpp_2m_sde_karras_scheduler,
}
default_scheduler_name = 'DDIM' if device.type == 'mps' else 'DPM++ 2M SDE Karras'


@torch.inference_mode()
def process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, fg_brightness=0, fg_contrast=1.0, fg_saturation=1.0, scheduler_name=None, progress=None):
    if input_fg is None:
        raise gr.Error("Please upload a foreground image.")

    # Apply preprocessing to foreground
    input_fg = preprocess_foreground(input_fg, brightness=fg_brightness, contrast=fg_contrast, saturation=fg_saturation)

    image_width = int(image_width) // 64 * 64
    image_height = int(image_height) // 64 * 64

    scheduler = scheduler_map.get(scheduler_name, default_scheduler)
    t2i_pipe.scheduler = scheduler
    i2i_pipe.scheduler = scheduler

    bg_source = BGSource(bg_source)

    # Handle background source - synthetic backgrounds don't need input_bg
    if bg_source == BGSource.UPLOAD:
        if input_bg is None:
            raise gr.Error("Please upload a background image when using 'Use Background Image' mode")
    elif bg_source == BGSource.UPLOAD_FLIP:
        if input_bg is None:
            raise gr.Error("Please upload a background image when using 'Use Flipped Background Image' mode")
        input_bg = np.fliplr(input_bg)
    elif bg_source == BGSource.GREY:
        input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(224, 32, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(32, 224, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(224, 32, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(32, 224, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise ValueError('Wrong background source!')

    rng = torch.Generator(device=device).manual_seed(seed)

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    # For synthetic backgrounds, they're already the right size, so just use them directly
    if bg_source in [BGSource.GREY, BGSource.LEFT, BGSource.RIGHT, BGSource.TOP, BGSource.BOTTOM]:
        bg = input_bg
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    if progress:
        progress(0.12, desc="Encoding text prompt with CLIP...")
    conds, unconds = encode_prompt_pair(prompt + ', ' + a_prompt, n_prompt, tokenizer, text_encoder, device)

    if progress:
        progress(0.15, desc="Low-res generation: denoising with UNet...")
    latents = t2i_pipe(
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=steps,
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    if progress:
        progress(0.48, desc="Decoding low-res result...")
    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)

    if progress:
        progress(0.50, desc=f"Upscaling to {highres_scale}x and re-encoding...")
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    if progress:
        progress(0.55, desc="High-res refinement: denoising with UNet...")
    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    if progress:
        progress(0.95, desc="Decoding final image...")
    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels, quant=False)

    clear_gpu_cache(device)
    return pixels, [fg, bg]


@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, fg_brightness, fg_contrast, fg_saturation, fg_sigma, fg_blend_strength, mask_blur, mask_expand, mask_threshold, scheduler_name, progress=gr.Progress(track_tqdm=True)):
    try:
        progress(0, desc="Removing background with RMBG...")
        input_fg, matting = run_rmbg(input_fg, rmbg, device, sigma=fg_sigma, blend_strength=fg_blend_strength, mask_blur=mask_blur, mask_expand=mask_expand, mask_threshold=mask_threshold)
        progress(0.1, desc="Encoding foreground + background into latent space...")
        results, extra_images = process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, fg_brightness=fg_brightness, fg_contrast=fg_contrast, fg_saturation=fg_saturation, scheduler_name=scheduler_name, progress=progress)
        results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
        extra_images = [img if img.dtype == np.uint8 else (img * 255.0).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.clip(0, 255).astype(np.uint8) for img in extra_images]
        save_outputs(results, args.output_dir, prefix='fbc_relight', seed=seed)
        progress(1.0, desc="Done!")
        return results + extra_images
    except gr.Error:
        raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            clear_gpu_cache(device)
            raise gr.Error("Out of GPU memory. Try reducing image size or number of samples.")
        logger.exception("Runtime error during relighting")
        raise gr.Error(f"An error occurred: {e}")
    except Exception as e:
        logger.exception("Unexpected error during relighting")
        raise gr.Error(f"An error occurred: {e}")


@torch.inference_mode()
def process_normal(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, fg_brightness, fg_contrast, fg_saturation, fg_sigma, fg_blend_strength, mask_blur, mask_expand, mask_threshold, scheduler_name, progress=gr.Progress(track_tqdm=True)):
    try:
        progress(0, desc="Removing background...")
        input_fg, matting = run_rmbg(input_fg, rmbg, device, sigma=fg_sigma if fg_sigma != 0 else 16, blend_strength=fg_blend_strength, mask_blur=mask_blur, mask_expand=mask_expand, mask_threshold=mask_threshold)

        progress(0.05, desc="Computing normals: left...")
        left = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.LEFT.value, scheduler_name=scheduler_name)[0][0]

        progress(0.25, desc="Computing normals: right...")
        right = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.RIGHT.value, scheduler_name=scheduler_name)[0][0]

        progress(0.50, desc="Computing normals: bottom...")
        bottom = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.BOTTOM.value, scheduler_name=scheduler_name)[0][0]

        progress(0.75, desc="Computing normals: top...")
        top = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.TOP.value, scheduler_name=scheduler_name)[0][0]

        inner_results = [left * 2.0 - 1.0, right * 2.0 - 1.0, bottom * 2.0 - 1.0, top * 2.0 - 1.0]

        progress(0.95, desc="Computing normal map...")
        ambient = (left + right + bottom + top) / 4.0
        h, w, _ = ambient.shape
        matting = resize_and_center_crop((matting[..., 0] * 255.0).clip(0, 255).astype(np.uint8), w, h).astype(np.float32)[..., None] / 255.0

        def safa_divide(a, b):
            e = 1e-5
            return ((a + e) / (b + e)) - 1.0

        left = safa_divide(left, ambient)
        right = safa_divide(right, ambient)
        bottom = safa_divide(bottom, ambient)
        top = safa_divide(top, ambient)

        u = (right - left) * 0.5
        v = (top - bottom) * 0.5

        sigma = 10.0
        u = np.mean(u, axis=2)
        v = np.mean(v, axis=2)
        h = (1.0 - u ** 2.0 - v ** 2.0).clip(0, 1e5) ** (0.5 * sigma)
        z = np.zeros_like(h)

        normal = np.stack([u, v, h], axis=2)
        normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
        normal = normal * matting + np.stack([z, z, 1 - z], axis=2) * (1 - matting)

        results = [normal, left, right, bottom, top] + inner_results
        results = [(x * 127.5 + 127.5).clip(0, 255).astype(np.uint8) for x in results]
        save_outputs(results, args.output_dir, prefix='fbc_normal', seed=seed)
        progress(1.0, desc="Done!")
        return results
    except gr.Error:
        raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            clear_gpu_cache(device)
            raise gr.Error("Out of GPU memory. Try reducing image size or number of samples.")
        logger.exception("Runtime error during normal computation")
        raise gr.Error(f"An error occurred: {e}")
    except Exception as e:
        logger.exception("Unexpected error during normal computation")
        raise gr.Error(f"An error occurred: {e}")


quick_prompts = [
    'beautiful woman',
    'handsome man',
    'beautiful woman, cinematic lighting',
    'handsome man, cinematic lighting',
    'beautiful woman, natural lighting',
    'handsome man, natural lighting',
    'beautiful woman, neo punk lighting, cyberpunk',
    'handsome man, neo punk lighting, cyberpunk',
]
quick_prompts = [[x] for x in quick_prompts]


class BGSource(Enum):
    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## IC-Light (Relighting with Foreground and Background Condition)")
        show_tips = gr.Checkbox(label="Show Help Tips", value=False, scale=0)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Image(sources=['upload'], type="numpy", label="Foreground", height=480)
                input_bg = gr.Image(sources=['upload'], type="numpy", label="Background", height=480)
            prompt = gr.Textbox(label="Prompt", info="Simple prompts work best here. Example: 'beautiful woman, cinematic lighting'. The background image provides most of the lighting context.")
            bg_source = gr.Radio(choices=[e.value for e in BGSource],
                                 value=BGSource.UPLOAD.value,
                                 label="Background Source", type='value',
                                 info="Use Background Image: Your uploaded background sets the lighting. Flipped: Mirrors the background horizontally. Left/Right/Top/Bottom: Synthetic directional light gradient. Ambient: Flat neutral gray, even lighting.")

            example_prompts = gr.Dataset(samples=quick_prompts, label='Prompt Quick List', components=[prompt])
            bg_gallery = gr.Gallery(height=450, object_fit='contain', label='Background Quick List', value=db_examples.bg_samples, columns=5, allow_preview=False)

            preview_button = gr.Button(value="Preview Foreground", variant="secondary")
            preview_gallery = gr.Gallery(
                label="Foreground Preview",
                height=400,
                columns=3,
                object_fit='contain',
                visible=False
            )

            with gr.Row():
                relight_button = gr.Button(value="Relight", variant="primary", scale=3)
                cancel_button = gr.Button(value="Cancel", variant="stop", scale=1)

            with gr.Row():
                aspect_ratio = gr.Dropdown(
                    choices=["Portrait (512x640)", "Square (512x512)", "Landscape (640x512)", "Custom"],
                    value="Portrait (512x640)", label="Aspect Ratio",
                    info="Quick size presets. Choose 'Custom' to set width/height manually.", scale=1)
                quality_preset = gr.Dropdown(
                    choices=["Fast Draft", "Balanced", "High Quality", "Custom"],
                    value="Fast Draft", label="Quality Preset",
                    info="Fast Draft: 15 steps, no upscale. Balanced: 20 steps, 1.5x upscale. High Quality: 35 steps, 2x upscale.", scale=1)

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1, info="1: Single image, fastest. 2-4: Compare variations. 5+: Batch generation, multiplies processing time linearly.")
                    seed = gr.Number(label="Seed", value=12345, precision=0, minimum=0, maximum=2**31, info="Controls randomness (0 to 2,147,483,648). Same seed + same settings = identical output. Nearby seeds produce completely unrelated results — use Randomize to explore variations.")
                    randomize_seed = gr.Button("Randomize", scale=0, min_width=90)

                with gr.Row():
                    favorite_seeds_dropdown = gr.Dropdown(
                        label="Load Favorite Seed",
                        choices=get_favorite_seeds_choices(args.output_dir),
                        info="Select a previously saved favorite seed to load it.",
                        interactive=True
                    )
                    seed_label = gr.Textbox(label="Seed Label (optional)", placeholder="e.g., Golden sunset", scale=1)
                    save_seed_btn = gr.Button("💾 Save Seed", scale=0, min_width=110)

                with gr.Row():
                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64, info="256-384: Thumbnail/fast preview. 512: Default, best model quality. 768+: Larger but may degrade since model was trained at 512px.")
                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64, info="256-384: Thumbnail/fast preview. 512-640: Default portrait size. 768+: Larger but may degrade since model was trained at 512px.")

            with gr.Accordion("Advanced options", open=False):
                scheduler_dropdown = gr.Dropdown(
                    choices=list(scheduler_map.keys()), value=default_scheduler_name,
                    label="Scheduler",
                    info="DDIM: Stable, MPS-compatible, consistent results. Euler a: Fast, slightly random variations. DPM++ 2M SDE Karras: Highest quality, CUDA-only.")
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1,
                    info="10-15: Fast draft, visible noise. 20-30: Best quality/speed balance. 50+: Diminishing returns, much slower. Each step removes noise from the image.")
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=7.0, step=0.01,
                    info="1-2: Creative, loose interpretation of prompt. 3-5: Natural balance. 7-12: Strict prompt adherence. 15+: Over-saturated, artifacts likely. Controls how strongly the text prompt guides generation.")
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01,
                    info="1.0: No upscale (faster, less detail). 1.5: Default, good detail boost. 2.0+: Large output, sharp details, but uses significantly more memory and time.")
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=0.9, value=0.5, step=0.01,
                    info="0.1-0.3: Subtle sharpening, preserves low-res output closely. 0.4-0.5: Balanced refinement, adds detail. 0.6+: Major rework during upscale, may change composition.")

                gr.Markdown("### Foreground Controls")
                fg_brightness = gr.Slider(label="Brightness", minimum=-100, maximum=100, value=0, step=1,
                    info="-100 to +100. Adjust foreground brightness before processing. Negative values darken, positive brighten.")
                fg_contrast = gr.Slider(label="Contrast", minimum=0.5, maximum=2.0, value=1.0, step=0.01,
                    info="0.5 to 2.0. Adjust foreground contrast. <1.0 reduces contrast, >1.0 increases it.")
                fg_saturation = gr.Slider(label="Saturation", minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                    info="0.0 to 2.0. Adjust color intensity. 0.0 = grayscale, 1.0 = original, >1.0 = vivid.")
                fg_sigma = gr.Slider(label="RMBG Sigma", minimum=-255, maximum=255, value=0, step=1,
                    info="-255 to +255. Brightness shift applied to foreground during background removal. Adjust if edges are too bright/dark. For normal computation, defaults to 16 if set to 0.")
                fg_blend_strength = gr.Slider(label="Blend Strength", minimum=0.0, maximum=1.0, value=1.0, step=0.01,
                    info="0.0 to 1.0. Controls alpha mask intensity. Lower values make background removal less aggressive, keeping more of original image.")

                gr.Markdown("### Mask Refinement")
                mask_blur = gr.Slider(label="Mask Blur", minimum=0, maximum=20, value=0, step=1,
                    info="0 to 20 pixels. Softens mask edges. 0 = no blur, 3-5 = subtle, 10+ = very soft. Helps blend foreground naturally.")
                mask_expand = gr.Slider(label="Mask Expand/Contract", minimum=-50, maximum=50, value=0, step=1,
                    info="-50 to +50 pixels. Negative contracts (shrinks) mask, positive expands it. Use +5 to +15 to include more edges, -5 to -15 to remove edge artifacts.")
                mask_threshold = gr.Slider(label="Mask Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                    info="0.0 to 1.0. Creates sharp mask boundary. 0.5 = no effect (smooth gradient). <0.5 = more background, >0.5 = more foreground. Use 0.7-0.9 for clean cutout.")

                a_prompt = gr.Textbox(label="Added Prompt", value='best quality',
                    info="Automatically appended to your prompt. Common boosters: 'best quality', 'detailed face', 'sharp focus', '8k'.")
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='lowres, bad anatomy, bad hands, cropped, worst quality',
                                      info="Steers generation away from these concepts. Helps prevent deformed hands, low resolution, and common diffusion artifacts.")
                normal_button = gr.Button(value="Compute Normal (4x Slower)",
                    variant="secondary")
        with gr.Column():
            result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs')

            # Preset Management Panel
            with gr.Accordion("📸 Preset Library", open=False):
                gr.Markdown("Save complete workflows (prompt + settings + seed) for reproducible results.")

                with gr.Row():
                    preset_name_input = gr.Textbox(
                        label="Preset Name",
                        placeholder="e.g., Golden Hour Portrait",
                        scale=3
                    )
                    save_preset_btn = gr.Button("💾 Save Current as Preset", scale=1, variant="primary")

                preset_status = gr.Markdown("")

                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        label="Load Preset",
                        choices=get_preset_choices(args.output_dir),
                        info="Select a preset to load all its settings",
                        interactive=True,
                        scale=3
                    )
                    load_preset_btn = gr.Button("📂 Load", scale=1)
                    delete_preset_btn = gr.Button("🗑️ Delete", scale=1, variant="stop")

                with gr.Accordion("Preset Details", open=False):
                    preset_details = gr.JSON(label="Settings", value={})

            with gr.Accordion("Pipeline Stages (what the progress bar is doing)", open=False):
                gr.Markdown("""| Stage | Progress | What's Happening |
|---|---|---|
| Background Removal | 0-10% | BRIA RMBG extracts the foreground subject from the uploaded image |
| Encoding | 10-15% | VAE compresses foreground + background to latent space, CLIP encodes your text prompt |
| Low-Res Generation | 15-50% | UNet denoises a latent image step-by-step, guided by prompt, foreground, and background |
| Upscale + Re-encode | ~50% | Low-res result is upscaled by Highres Scale factor, then re-encoded to latent space |
| High-Res Refinement | 50-95% | UNet refines the upscaled image, adding detail at the higher resolution |
| Decode + Save | 95-100% | VAE decodes final latent to pixels, image saved to outputs/ directory |""")
    with gr.Row():
        example_gallery = gr.Gallery(
            height=200, object_fit='contain', label='Example Inputs',
            value=[ex[0] for ex in db_examples.background_conditioned_examples],
            columns=5, allow_preview=False
        )

    def example_selected(evt: gr.SelectData):
        ex = db_examples.background_conditioned_examples[evt.index]
        return ex[0], ex[1], ex[2], ex[3], ex[4], ex[5], ex[6]

    example_gallery.select(
        example_selected, inputs=None,
        outputs=[input_fg, input_bg, prompt, bg_source, image_width, image_height, seed]
    )

    randomize_seed.click(fn=lambda: random.randint(0, 2**31), inputs=[], outputs=[seed], show_progress=False, queue=False)

    def handle_save_seed(current_seed, label):
        """Save the current seed to favorites."""
        if current_seed is None:
            return gr.update(choices=get_favorite_seeds_choices(args.output_dir)), "⚠️ No seed to save"

        success = save_favorite_seed(args.output_dir, int(current_seed), label)
        if success:
            # Refresh dropdown choices
            return gr.update(choices=get_favorite_seeds_choices(args.output_dir)), f"✅ Saved seed {int(current_seed)}"
        else:
            return gr.update(choices=get_favorite_seeds_choices(args.output_dir)), "❌ Failed to save seed"

    def handle_load_favorite(choice):
        """Load a favorite seed from the dropdown."""
        if choice is None:
            return gr.update()
        # Extract seed from choice (format is "Label (seed: 12345)")
        try:
            seed_str = choice.split("seed: ")[1].rstrip(")")
            return int(seed_str)
        except (IndexError, ValueError):
            return gr.update()

    save_seed_btn.click(
        handle_save_seed,
        inputs=[seed, seed_label],
        outputs=[favorite_seeds_dropdown, seed_label],
        show_progress=False,
        queue=False
    )

    favorite_seeds_dropdown.change(
        handle_load_favorite,
        inputs=[favorite_seeds_dropdown],
        outputs=[seed],
        show_progress=False,
        queue=False
    )

    # Preset handlers
    def handle_save_preset(name, current_prompt, current_bg_source, current_seed, current_steps, current_cfg,
                          current_image_width, current_image_height, current_num_samples,
                          current_highres_scale, current_highres_denoise,
                          current_a_prompt, current_n_prompt, current_scheduler):
        """Save current settings as a preset."""
        if not name or not name.strip():
            return gr.update(), gr.update(), "⚠️ Please enter a preset name"

        # Collect all settings (note: bg-conditioned demo doesn't have lowres_denoise)
        settings = {
            'prompt': current_prompt,
            'bg_source': current_bg_source,
            'seed': int(current_seed) if current_seed is not None else 12345,
            'steps': int(current_steps),
            'cfg': float(current_cfg),
            'image_width': int(current_image_width),
            'image_height': int(current_image_height),
            'num_samples': int(current_num_samples),
            'highres_scale': float(current_highres_scale),
            'highres_denoise': float(current_highres_denoise),
            'a_prompt': current_a_prompt,
            'n_prompt': current_n_prompt,
            'scheduler': current_scheduler
        }

        success = save_preset(args.output_dir, name.strip(), settings)
        if success:
            return (
                gr.update(choices=get_preset_choices(args.output_dir)),
                gr.update(value=settings),
                f"✅ Saved preset '{name.strip()}'"
            )
        else:
            return gr.update(), gr.update(), "❌ Failed to save preset"

    def handle_load_preset(preset_name):
        """Load a preset and return all settings."""
        if not preset_name:
            return [gr.update()] * 13 + [{}] + [""]

        preset = get_preset_by_name(args.output_dir, preset_name)
        if not preset:
            return [gr.update()] * 13 + [{}] + ["❌ Preset not found"]

        s = preset['settings']
        return [
            s.get('prompt', ''),
            s.get('bg_source', 'Upload'),
            s.get('seed', 12345),
            s.get('steps', 20),
            s.get('cfg', 7.0),
            s.get('image_width', 512),
            s.get('image_height', 640),
            s.get('num_samples', 1),
            s.get('highres_scale', 1.5),
            s.get('highres_denoise', 0.5),
            s.get('a_prompt', 'best quality'),
            s.get('n_prompt', 'lowres, bad anatomy, bad hands, cropped, worst quality'),
            s.get('scheduler', 'DDIM'),
            s,  # preset_details
            f"✅ Loaded preset '{preset_name}'"
        ]

    def handle_delete_preset(preset_name):
        """Delete a preset."""
        if not preset_name:
            return gr.update(), gr.update(), "⚠️ No preset selected"

        success = delete_preset(args.output_dir, preset_name)
        if success:
            return (
                gr.update(choices=get_preset_choices(args.output_dir), value=None),
                gr.update(value={}),
                f"✅ Deleted preset '{preset_name}'"
            )
        else:
            return gr.update(), gr.update(), "❌ Failed to delete preset"

    def handle_preset_selection(preset_name):
        """Update preset details when selection changes."""
        if not preset_name:
            return {}

        preset = get_preset_by_name(args.output_dir, preset_name)
        if preset:
            return preset['settings']
        return {}

    save_preset_btn.click(
        handle_save_preset,
        inputs=[
            preset_name_input, prompt, bg_source, seed, steps, cfg,
            image_width, image_height, num_samples,
            highres_scale, highres_denoise,
            a_prompt, n_prompt, scheduler_dropdown
        ],
        outputs=[preset_dropdown, preset_details, preset_status],
        show_progress=False,
        queue=False
    )

    load_preset_btn.click(
        handle_load_preset,
        inputs=[preset_dropdown],
        outputs=[
            prompt, bg_source, seed, steps, cfg,
            image_width, image_height, num_samples,
            highres_scale, highres_denoise,
            a_prompt, n_prompt, scheduler_dropdown,
            preset_details, preset_status
        ],
        show_progress=False,
        queue=False
    )

    delete_preset_btn.click(
        handle_delete_preset,
        inputs=[preset_dropdown],
        outputs=[preset_dropdown, preset_details, preset_status],
        show_progress=False,
        queue=False
    )

    preset_dropdown.change(
        handle_preset_selection,
        inputs=[preset_dropdown],
        outputs=[preset_details],
        show_progress=False,
        queue=False
    )

    def set_aspect_ratio(choice):
        presets = {"Portrait (512x640)": (512, 640), "Square (512x512)": (512, 512), "Landscape (640x512)": (640, 512)}
        return presets.get(choice, (gr.update(), gr.update()))

    aspect_ratio.change(set_aspect_ratio, inputs=[aspect_ratio], outputs=[image_width, image_height], show_progress=False, queue=False)

    def set_quality_preset(choice):
        presets = {"Fast Draft": (15, 7.0, 1.0, 0.5), "Balanced": (20, 7.0, 1.5, 0.5), "High Quality": (35, 7.0, 2.0, 0.4)}
        if choice in presets:
            return presets[choice]
        return gr.update(), gr.update(), gr.update(), gr.update()

    quality_preset.change(set_quality_preset, inputs=[quality_preset], outputs=[steps, cfg, highres_scale, highres_denoise], show_progress=False, queue=False)

    show_tips.change(
        fn=None, inputs=[show_tips], outputs=[],
        js="""(v) => {
            document.querySelectorAll('[data-testid="block-info"]').forEach(span => {
                let sib = span.nextElementSibling;
                if (sib && sib.querySelector('.prose')) {
                    sib.style.display = v ? '' : 'none';
                }
            });
        }"""
    )

    ips = [input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, fg_brightness, fg_contrast, fg_saturation, fg_sigma, fg_blend_strength, mask_blur, mask_expand, mask_threshold, scheduler_dropdown]
    relight_event = relight_button.click(fn=process_relight, inputs=ips, outputs=[result_gallery])
    normal_event = normal_button.click(fn=process_normal, inputs=ips, outputs=[result_gallery])
    cancel_button.click(fn=None, inputs=None, outputs=None, cancels=[relight_event, normal_event])

    preview_button.click(
        fn=handle_preview_foreground,
        inputs=[input_fg],
        outputs=[preview_gallery],
        show_progress=True
    ).then(
        fn=lambda x: gr.update(visible=True),
        inputs=[preview_gallery],
        outputs=[preview_gallery]
    )

    example_prompts.click(lambda x: x[0], inputs=example_prompts, outputs=prompt, show_progress=False, queue=False)

    def bg_gallery_selected(evt: gr.SelectData):
        # Use the index to get the image path directly from the original list
        # This avoids Gradio 6's complex gallery data structures
        return db_examples.bg_samples[evt.index]

    bg_gallery.select(bg_gallery_selected, inputs=None, outputs=input_bg)


block.launch(server_name=args.host, server_port=args.port, allowed_paths=[os.path.abspath('imgs/')])
