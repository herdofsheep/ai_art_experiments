OUTPUT_DIR = 'outputs/SD_controlnet_animation'
ANIMATION_DIR = "data/animation_paler"
STYLE_IMAGE_PATH = "data/style2/sex_doll2.jpg"
PROMPT = "A sex doll, detailed, realistic"
ADAPTER_SCALE = 0.6

import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, LCMScheduler
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

    # 3. Add LCM for speed & IP-Adapter for style
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(0.8)

    return pipe

def get_canny_image(image):
    # This creates the "edge map" from your content frame
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

def style_frame(pipe, content_img, prompt, style_embeds):
    content_img = content_img.resize((512, 512))
    canny_img = get_canny_image(content_img)

    # The magic happens here: 
    # image = structural guide, control_image = edge guide, ip_adapter_image_embeds = style guide
    result = pipe(
        prompt=prompt,
        image=content_img,
        control_image=canny_img,
        ip_adapter_image_embeds=style_embeds,
        strength=0.6,            # How much to change the original
        controlnet_conditioning_scale=0.8, 
        num_inference_steps=4,   # LCM speed
        guidance_scale=1.5
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
    
    animation_frames = sorted(glob(f'{ANIMATION_DIR}/*.png'))
    for i, frame_path in enumerate(animation_frames):
        print(f"Styling frame {frame_path}...")
        content_image = load_image(frame_path)
        result = style_frame(pipe, content_image, PROMPT, style_embeds)
        result.save(f"{OUTPUT_DIR}/{i}.png")

if __name__ == "__main__":
    main()