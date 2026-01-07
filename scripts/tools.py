import torch
from PIL import Image
import os
import re

def load_img_to_img_pipeline(adapter_scale=0.8):
    print("Loading image to image generation pipeline with IP-Adapter...")
    from diffusers import StableDiffusionImg2ImgPipeline
    
    # Use SD 1.5 - more stable on macOS
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=False,
        safety_checker=None,  # Disable NSFW filter
    )
    
    # Load IP-Adapter for image-based style conditioning
    print("Loading IP-Adapter...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin"
    )
    pipe.set_ip_adapter_scale(adapter_scale)  # Controls style influence (0-1)
    
    pipe = pipe.to("mps")
    print("Pipeline loaded!")
    return pipe


def generate_img_to_img(pipe, style_image_path, content_image_path, output_path=None):
    # Load and resize images
    style_image = Image.open(style_image_path).convert("RGB")
    style_image = style_image.resize((512, 512))
    
    content_image = Image.open(content_image_path).convert("RGB")
    content_image = content_image.resize((512, 512))

    with torch.inference_mode():
        image = pipe(
            prompt="",  # Empty prompt - style comes from IP-Adapter
            image=content_image,
            ip_adapter_image=style_image,  # Style reference image
            strength=0.7,
            num_inference_steps=30
        ).images[0]
    
    if output_path:
        image.save(output_path)
        print(f"Image saved to {output_path}")
    return image



def get_last_styled_frame(output_dir):
    existing = [f for f in os.listdir(output_dir) if re.match(r'^\d+\.png$', f)]
    if existing:
        nums = [int(re.match(r'^(\d+)\.png$', f).group(1)) for f in existing]
        last_frame_index = max(nums)
    else:
        last_frame_index = 0

    if os.path.exists(f"{output_dir}/{last_frame_index}.png"):
        image = Image.open(f"{output_dir}/{last_frame_index}.png")
    else:
        image = None

    return last_frame_index, image