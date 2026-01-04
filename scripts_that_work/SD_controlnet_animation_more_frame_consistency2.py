OUTPUT_DIR = 'outputs/SD_controlnet_animation_boob'
ANIMATION_DIR = "data/boob_animation"
STYLE_IMAGE_PATH = "data/content/sciency.webp"
PROMPT = "A scientific diagram"

ADAPTER_SCALE = 1.0 # strength of style influence on output image
TRANSFORM_STRENGTH = 0.75 # strength allowed deviation from original image
CONTROLNET_CONDITIONING_SCALE = 0.5 # strength of edges, style constraint
NUM_INFERENCE_STEPS = 15 # balanced steps
GUIDANCE_SCALE = 4.0 # moderate guidance (MPS can be unstable with high values)
PREV_FRAME_INFLUENCE = 0.7 # 0.0 = pure content, 1.0 = pure previous styled frame

import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch')
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'huggingface')

import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import cv2
import numpy as np
from PIL import Image
from glob import glob

def load_video_style_pipe():
    # 1. Load ControlNet (Canny is best for keeping animation lines)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32,
    )

    # 2. Load the main Pipeline
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to("mps")

    # 3. Use DPM++ scheduler (better quality) & stronger IP-Adapter for style
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(ADAPTER_SCALE)

    return pipe

def get_canny_image(image):
    # This creates the "edge map" from your content frame
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

def style_frame(pipe, content_img, prompt, style_embeds, prev_styled_frame=None):
    content_img = content_img.resize((512, 512))
    canny_img = get_canny_image(content_img)

    # Blend previous styled frame with current content for init image
    # This prevents style drift while maintaining frame consistency
    if prev_styled_frame is not None:
        prev_arr = np.array(prev_styled_frame).astype(np.float32)
        curr_arr = np.array(content_img).astype(np.float32)
        blended = (PREV_FRAME_INFLUENCE * prev_arr + (1 - PREV_FRAME_INFLUENCE) * curr_arr)
        init_image = Image.fromarray(blended.astype(np.uint8))
    else:
        init_image = content_img

    # The magic happens here: 
    # image = previous styled output (or content for first frame)
    # control_image = edge guide from current content frame
    # ip_adapter_image_embeds = style guide
    result = pipe(
        prompt=prompt,
        image=init_image,
        control_image=canny_img,
        ip_adapter_image_embeds=style_embeds,
        strength=TRANSFORM_STRENGTH,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    ).images[0]
    return result

def main():
    pipe = load_video_style_pipe()
    style_image = load_image(STYLE_IMAGE_PATH).resize((512, 512))
    
    # Pre-encode style image to avoid IP-Adapter + ControlNet tuple bug
    # encode_image returns (image_embeds, uncond_image_embeds)
    image_embeds, uncond_embeds = pipe.encode_image(style_image, device=pipe.device, num_images_per_prompt=1)
    # Pipeline expects negative + positive concatenated, then chunks them
    # Both need to be 3D tensors
    if image_embeds.dim() == 2:
        image_embeds = image_embeds.unsqueeze(0)
        uncond_embeds = uncond_embeds.unsqueeze(0)
    # Concatenate: [negative, positive] so chunk(2) works
    style_embeds = [torch.cat([uncond_embeds, image_embeds], dim=0)]

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    animation_frames = sorted(glob(f'{ANIMATION_DIR}/*.png'))
    
    prev_styled_frame = None
    for i, frame_path in enumerate(animation_frames):
        print(f"Styling frame {i} to {OUTPUT_DIR}...")
        content_image = load_image(frame_path)
        result = style_frame(pipe, content_image, PROMPT, style_embeds, prev_styled_frame)
        # Save at original resolution
        result.resize(content_image.size).save(f"{OUTPUT_DIR}/{i}.png")
        # Keep 512x512 result to influence next frame
        prev_styled_frame = result

if __name__ == "__main__":
    main()