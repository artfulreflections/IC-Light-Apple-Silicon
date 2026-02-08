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
)

# CLI args and logging
args = parse_common_args(description='IC-Light: Background Conditioned Relighting')
setup_logging()
logger = logging.getLogger(__name__)

# Load models
sd15_name = 'stablediffusionapi/realistic-vision-v51'
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


@torch.inference_mode()
def process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    if input_fg is None:
        raise gr.Error("Please upload a foreground image.")
    image_width = int(image_width) // 64 * 64
    image_height = int(image_height) // 64 * 64

    bg_source = BGSource(bg_source)

    # Handle background source - synthetic backgrounds don't need input_bg
    if bg_source == BGSource.UPLOAD:
        if input_bg is None:
            raise ValueError("Please upload a background image when using 'Use Background Image' mode")
    elif bg_source == BGSource.UPLOAD_FLIP:
        if input_bg is None:
            raise ValueError("Please upload a background image when using 'Use Flipped Background Image' mode")
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

    conds, unconds = encode_prompt_pair(prompt + ', ' + a_prompt, n_prompt, tokenizer, text_encoder, device)

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

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
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

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels, quant=False)

    return pixels, [fg, bg]


@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    input_fg, matting = run_rmbg(input_fg, rmbg, device)
    results, extra_images = process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source)
    results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
    # Ensure extra_images (fg, bg) are also in uint8 format for Gradio
    extra_images = [img if img.dtype == np.uint8 else (img * 255.0).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.clip(0, 255).astype(np.uint8) for img in extra_images]
    final_output = results + extra_images
    return final_output


@torch.inference_mode()
def process_normal(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    input_fg, matting = run_rmbg(input_fg, rmbg, device, sigma=16)

    print('left ...')
    left = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.LEFT.value)[0][0]

    print('right ...')
    right = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.RIGHT.value)[0][0]

    print('bottom ...')
    bottom = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.BOTTOM.value)[0][0]

    print('top ...')
    top = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.TOP.value)[0][0]

    inner_results = [left * 2.0 - 1.0, right * 2.0 - 1.0, bottom * 2.0 - 1.0, top * 2.0 - 1.0]

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
    return results


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
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Image(sources=['upload'], type="numpy", label="Foreground", height=480)
                input_bg = gr.Image(sources=['upload'], type="numpy", label="Background", height=480)
            prompt = gr.Textbox(label="Prompt")
            bg_source = gr.Radio(choices=[e.value for e in BGSource],
                                 value=BGSource.UPLOAD.value,
                                 label="Background Source", type='value')

            example_prompts = gr.Dataset(samples=quick_prompts, label='Prompt Quick List', components=[prompt])
            bg_gallery = gr.Gallery(height=450, object_fit='contain', label='Background Quick List', value=db_examples.bg_samples, columns=5, allow_preview=False)
            relight_button = gr.Button(value="Relight")

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Number(label="Seed", value=12345, precision=0)
                with gr.Row():
                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64)

            with gr.Accordion("Advanced options", open=False):
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=7.0, step=0.01)
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=0.9, value=0.5, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='lowres, bad anatomy, bad hands, cropped, worst quality')
                normal_button = gr.Button(value="Compute Normal (4x Slower)")
        with gr.Column():
            result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs')
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

    ips = [input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source]
    relight_button.click(fn=process_relight, inputs=ips, outputs=[result_gallery])
    normal_button.click(fn=process_normal, inputs=ips, outputs=[result_gallery])
    example_prompts.click(lambda x: x[0], inputs=example_prompts, outputs=prompt, show_progress=False, queue=False)

    def bg_gallery_selected(evt: gr.SelectData):
        # Use the index to get the image path directly from the original list
        # This avoids Gradio 6's complex gallery data structures
        return db_examples.bg_samples[evt.index]

    bg_gallery.select(bg_gallery_selected, inputs=None, outputs=input_bg)


block.launch(server_name=args.host, server_port=args.port, allowed_paths=[os.path.abspath('imgs/')])
