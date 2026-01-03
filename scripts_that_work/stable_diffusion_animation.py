import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch')
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'huggingface')

from huggingface_hub import whoami
import torch
from PIL import Image
from dotenv import load_dotenv
from glob import glob

load_dotenv()

OUTPUT_DIR = 'outputs/stable_diffusion_animation_test'
ANIMATION_PATH = 'data/animation'
STYLE_IMAGE_PATH = 'data/style/sex_doll2.png'

def load_img_to_img_pipeline():
    print("Loading image to image generation pipeline with IP-Adapter...")
    from diffusers import StableDiffusionImg2ImgPipeline
    from transformers import CLIPTokenizer
    
    # Use SD 1.5 - more stable on macOS
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Load tokenizer with use_fast=False to avoid Rust threading issues
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
        use_fast=False
    )
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        use_safetensors=False,
    )
    
    # Load IP-Adapter for image-based style conditioning
    print("Loading IP-Adapter...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin"
    )
    pipe.set_ip_adapter_scale(0.8)  # Controls style influence (0-1)
    
    pipe = pipe.to("cpu")
    print("Pipeline loaded!")
    return pipe


def generate_img_to_img_image(pipe, style_image_path, content_image_path, frame_number):
    print("Generating styled image...")
    
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
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    image.save(os.path.join(OUTPUT_DIR, f"{frame_number}.png"))
    print(f"Image saved to {OUTPUT_DIR}/{frame_number}.png")
    return image


def main():
    # HF_TOKEN from .env is automatically used
    print("Logged in as:", whoami()['name'])
    img_to_img_pipe = load_img_to_img_pipeline()
    animation_frames = sorted(glob(f'{ANIMATION_PATH}/*.png'))

    for frame in animation_frames:
        frame_number = frame.split('/')[-1].split('.')[0]
        generate_img_to_img_image(
            img_to_img_pipe,
            style_image_path=STYLE_IMAGE_PATH,  # Style to apply
            content_image_path=frame  # Content to transform
            frame_number=frame_number
        )


if __name__ == "__main__":
    main()