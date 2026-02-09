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
args = parse_common_args(description='IC-Light: Foreground Conditioned Relighting')
setup_logging()
logger = logging.getLogger(__name__)

# Load models
sd15_name = args.model
tokenizer, text_encoder, vae, unet, rmbg = load_models(sd15_name)

# Change UNet (8-channel for foreground-conditioned model)
unet = setup_unet(unet, in_channels=8)

# Load weights
model_path = os.path.join(args.model_dir, 'iclight_sd15_fc.safetensors')
load_and_merge_weights(
    unet,
    model_path=model_path,
    model_url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors',
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
def process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, scheduler_name=None, progress=None):
    if input_fg is None:
        raise gr.Error("Please upload an input image.")
    image_width = int(image_width) // 64 * 64
    image_height = int(image_height) // 64 * 64

    scheduler = scheduler_map.get(scheduler_name, default_scheduler)
    t2i_pipe.scheduler = scheduler
    i2i_pipe.scheduler = scheduler

    bg_source = BGSource(bg_source)
    input_bg = None

    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise ValueError('Wrong initial latent!')

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    if progress:
        progress(0.12, desc="Encoding text prompt with CLIP...")
    conds, unconds = encode_prompt_pair(prompt + ', ' + a_prompt, n_prompt, tokenizer, text_encoder, device)

    if progress:
        progress(0.15, desc="Low-res generation: denoising with UNet...")
    if input_bg is None:
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
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = i2i_pipe(
            image=bg_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / lowres_denoise)),
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
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

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

    clear_gpu_cache(device)
    return pytorch2numpy(pixels)


@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, scheduler_name, progress=gr.Progress(track_tqdm=True)):
    try:
        progress(0, desc="Removing background with RMBG...")
        input_fg, matting = run_rmbg(input_fg, rmbg, device)
        progress(0.1, desc="Encoding foreground into latent space...")
        results = process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, scheduler_name=scheduler_name, progress=progress)
        save_outputs(results, args.output_dir, prefix='fc_relight', seed=seed)
        progress(1.0, desc="Done!")
        return input_fg, results
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


quick_prompts = [
    'sunshine from window',
    'neon light, city',
    'sunset over sea',
    'golden time',
    'sci-fi RGB glowing, cyberpunk',
    'natural lighting',
    'warm atmosphere, at home, bedroom',
    'magic lit',
    'evil, gothic, Yharnam',
    'light and shadow',
    'shadow from window',
    'soft studio lighting',
    'home atmosphere, cozy bedroom illumination',
    'neon, Wong Kar-wai, warm'
]
quick_prompts = [[x] for x in quick_prompts]


quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]
quick_subjects = [[x] for x in quick_subjects]


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## IC-Light (Relighting with Foreground Condition)")
        show_tips = gr.Checkbox(label="Show Help Tips", value=True, scale=0)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Image(sources=['upload'], type="numpy", label="Image", height=480)
                output_bg = gr.Image(type="numpy", label="Preprocessed Foreground", height=480)
            prompt = gr.Textbox(label="Prompt", info="Combine a subject + lighting style. Example: 'beautiful woman, detailed face, sunshine from window'. Use the quick lists below for inspiration.")
            bg_source = gr.Radio(choices=[e.value for e in BGSource],
                                 value=BGSource.NONE.value,
                                 label="Lighting Preference (Initial Latent)", type='value',
                                 info="None: Model decides lighting freely from prompt only. Left/Right/Top/Bottom: Creates a light-to-dark gradient as a starting point, biasing light toward that direction.")
            example_quick_subjects = gr.Dataset(samples=quick_subjects, label='Subject Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Lighting Quick List', samples_per_page=1000, components=[prompt])
            relight_button = gr.Button(value="Relight")

            with gr.Row():
                aspect_ratio = gr.Dropdown(
                    choices=["Portrait (512x640)", "Square (512x512)", "Landscape (640x512)", "Custom"],
                    value="Portrait (512x640)", label="Aspect Ratio",
                    info="Quick size presets. Choose 'Custom' to set width/height manually.", scale=1)
                quality_preset = gr.Dropdown(
                    choices=["Fast Draft", "Balanced", "High Quality", "Custom"],
                    value="Balanced", label="Quality Preset",
                    info="Fast Draft: 15 steps, no upscale. Balanced: 25 steps, 1.5x upscale. High Quality: 40 steps, 2x upscale.", scale=1)

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
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1,
                    info="10-15: Fast draft, visible noise. 20-30: Best quality/speed balance. 50+: Diminishing returns, much slower. Each step removes noise from the image.")
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=2, step=0.01,
                    info="1-2: Creative, loose interpretation of prompt. 3-5: Natural balance. 7-12: Strict prompt adherence. 15+: Over-saturated, artifacts likely. Controls how strongly the text prompt guides generation.")
                lowres_denoise = gr.Slider(label="Lowres Denoise (for initial latent)", minimum=0.1, maximum=1.0, value=0.9, step=0.01,
                    info="0.1-0.3: Keeps most of the lighting gradient intact. 0.5-0.7: Moderate reshaping. 0.8-1.0: Full creative freedom, ignores gradient. Only active when a Light Direction is selected.")
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01,
                    info="1.0: No upscale (faster, less detail). 1.5: Default, good detail boost. 2.0+: Large output, sharp details, but uses significantly more memory and time.")
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=1.0, value=0.5, step=0.01,
                    info="0.1-0.3: Subtle sharpening, preserves low-res output closely. 0.4-0.5: Balanced refinement, adds detail. 0.6+: Major rework during upscale, may change composition.")
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality',
                    info="Automatically appended to your prompt. Common boosters: 'best quality', 'detailed face', 'sharp focus', '8k'.")
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality',
                    info="Steers generation away from these concepts. Helps prevent deformed hands, low resolution, and common diffusion artifacts.")
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
| Encoding | 10-15% | VAE compresses foreground to latent space, CLIP encodes your text prompt |
| Low-Res Generation | 15-50% | UNet denoises a latent image step-by-step, guided by your prompt and foreground |
| Upscale + Re-encode | ~50% | Low-res result is upscaled by Highres Scale factor, then re-encoded to latent space |
| High-Res Refinement | 50-95% | UNet refines the upscaled image, adding detail at the higher resolution |
| Decode + Save | 95-100% | VAE decodes final latent to pixels, image saved to outputs/ directory |""")
    with gr.Row():
        example_gallery = gr.Gallery(
            height=200, object_fit='contain', label='Example Inputs',
            value=[ex[0] for ex in db_examples.foreground_conditioned_examples],
            columns=8, allow_preview=False
        )

    def example_selected(evt: gr.SelectData):
        ex = db_examples.foreground_conditioned_examples[evt.index]
        return ex[0], ex[1], ex[2], ex[3], ex[4], ex[5]

    example_gallery.select(
        example_selected, inputs=None,
        outputs=[input_fg, prompt, bg_source, image_width, image_height, seed]
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
                          current_lowres_denoise, current_highres_scale, current_highres_denoise,
                          current_a_prompt, current_n_prompt, current_scheduler):
        """Save current settings as a preset."""
        if not name or not name.strip():
            return gr.update(), gr.update(), "⚠️ Please enter a preset name"

        # Collect all settings
        settings = {
            'prompt': current_prompt,
            'bg_source': current_bg_source,
            'seed': int(current_seed) if current_seed is not None else 12345,
            'steps': int(current_steps),
            'cfg': float(current_cfg),
            'image_width': int(current_image_width),
            'image_height': int(current_image_height),
            'num_samples': int(current_num_samples),
            'lowres_denoise': float(current_lowres_denoise),
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
            return [gr.update()] * 14 + [{}] + [""]

        preset = get_preset_by_name(args.output_dir, preset_name)
        if not preset:
            return [gr.update()] * 14 + [{}] + ["❌ Preset not found"]

        s = preset['settings']
        return [
            s.get('prompt', ''),
            s.get('bg_source', 'None'),
            s.get('seed', 12345),
            s.get('steps', 25),
            s.get('cfg', 2.0),
            s.get('image_width', 512),
            s.get('image_height', 640),
            s.get('num_samples', 1),
            s.get('lowres_denoise', 0.9),
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
            lowres_denoise, highres_scale, highres_denoise,
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
            lowres_denoise, highres_scale, highres_denoise,
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
        presets = {"Fast Draft": (15, 2.0, 1.0, 0.5), "Balanced": (25, 2.0, 1.5, 0.5), "High Quality": (40, 2.0, 2.0, 0.4)}
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

    ips = [input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, scheduler_dropdown]
    relight_button.click(fn=process_relight, inputs=ips, outputs=[output_bg, result_gallery])
    example_quick_prompts.click(lambda x, y: ', '.join(y.split(', ')[:2] + [x[0]]), inputs=[example_quick_prompts, prompt], outputs=prompt, show_progress=False, queue=False)
    example_quick_subjects.click(lambda x: x[0], inputs=example_quick_subjects, outputs=prompt, show_progress=False, queue=False)


block.launch(server_name=args.host, server_port=args.port, allowed_paths=[os.path.abspath('imgs/')])
