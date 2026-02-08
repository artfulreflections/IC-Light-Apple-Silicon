import logging
import os
import gradio as gr
import numpy as np
import torch
import db_examples

from enum import Enum
from utils import (
    parse_common_args, setup_logging,
    get_device, load_models, setup_unet, load_and_merge_weights,
    move_models_to_device, enable_sdp, create_schedulers,
    get_default_scheduler, create_pipelines,
    encode_prompt_pair, pytorch2numpy, numpy2pytorch,
    resize_and_center_crop, resize_without_crop, run_rmbg,
    clear_gpu_cache,
)

# CLI args and logging
args = parse_common_args(description='IC-Light: Foreground Conditioned Relighting')
setup_logging()
logger = logging.getLogger(__name__)

# Load models
sd15_name = 'stablediffusionapi/realistic-vision-v51'
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


@torch.inference_mode()
def process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, progress=None):
    if input_fg is None:
        raise gr.Error("Please upload an input image.")
    image_width = int(image_width) // 64 * 64
    image_height = int(image_height) // 64 * 64

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
        progress(0.15, desc="Encoding prompts...")
    conds, unconds = encode_prompt_pair(prompt + ', ' + a_prompt, n_prompt, tokenizer, text_encoder, device)

    if progress:
        progress(0.25, desc="Generating (low-res)...")
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

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    if progress:
        progress(0.55, desc="Refining (high-res)...")
    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

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
        progress(0.9, desc="Decoding...")
    pixels = vae.decode(latents).sample

    clear_gpu_cache(device)
    return pytorch2numpy(pixels)


@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, progress=gr.Progress()):
    try:
        progress(0, desc="Removing background...")
        input_fg, matting = run_rmbg(input_fg, rmbg, device)
        progress(0.1, desc="Processing...")
        results = process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, progress=progress)
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
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Image(sources=['upload'], type="numpy", label="Image", height=480)
                output_bg = gr.Image(type="numpy", label="Preprocessed Foreground", height=480)
            prompt = gr.Textbox(label="Prompt")
            bg_source = gr.Radio(choices=[e.value for e in BGSource],
                                 value=BGSource.NONE.value,
                                 label="Lighting Preference (Initial Latent)", type='value')
            example_quick_subjects = gr.Dataset(samples=quick_subjects, label='Subject Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Lighting Quick List', samples_per_page=1000, components=[prompt])
            relight_button = gr.Button(value="Relight")

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Number(label="Seed", value=12345, precision=0)

                with gr.Row():
                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64)

            with gr.Accordion("Advanced options", open=False):
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=2, step=0.01)
                lowres_denoise = gr.Slider(label="Lowres Denoise (for initial latent)", minimum=0.1, maximum=1.0, value=0.9, step=0.01)
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=1.0, value=0.5, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
        with gr.Column():
            result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs')
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

    ips = [input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source]
    relight_button.click(fn=process_relight, inputs=ips, outputs=[output_bg, result_gallery])
    example_quick_prompts.click(lambda x, y: ', '.join(y.split(', ')[:2] + [x[0]]), inputs=[example_quick_prompts, prompt], outputs=prompt, show_progress=False, queue=False)
    example_quick_subjects.click(lambda x: x[0], inputs=example_quick_subjects, outputs=prompt, show_progress=False, queue=False)


block.launch(server_name=args.host, server_port=args.port, allowed_paths=[os.path.abspath('imgs/')])
