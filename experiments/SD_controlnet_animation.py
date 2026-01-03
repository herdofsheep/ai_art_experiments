
OUTPUT_DIR = 'outputs/stable_diffusion_animation_more_blend'
ANIMATION_DIR = "data/animation_paler"
STYLE_IMAGE_PATH = "data/style2/sex_doll2.jpg"
PROMPT = "A sex doll, 8k, detailed, realistic"

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
    
    pipe.set_ip_adapter_scale(0.6)
    pipe.enable_attention_slicing()
    return pipe

def get_canny_image(image):
    # This creates the "edge map" from your content frame
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

def style_frame(pipe, content_path, style_image, prompt):
    print(f"Styling frame {content_path}...")
    content_img = load_image(content_path).resize((512, 512))
    canny_img = get_canny_image(content_img)

    # The magic happens here: 
    # image = structural guide, control_image = edge guide, ip_adapter = style guide
    result = pipe(
        prompt=prompt,
        image=content_img,
        control_image=canny_img,
        ip_adapter_image=style_image,
        strength=0.6,            # How much to change the original
        controlnet_conditioning_scale=0.8, 
        num_inference_steps=4,   # LCM speed
        guidance_scale=1.5
    ).images[0]
    return result

def main():
    pipe = load_video_style_pipe()
    style_image = load_image(STYLE_IMAGE_PATH)
    animation_frames = sorted(glob(f'{ANIMATION_DIR}/*.png'))
    for i, frame in enumerate(animation_frames):
        content_image = load_image(frame)
        frame = style_frame(pipe, content_image, style_image, PROMPT)
        frame.save(f"{OUTPUT_DIR}/{i}.png")

if __name__ == "__main__":
    main()