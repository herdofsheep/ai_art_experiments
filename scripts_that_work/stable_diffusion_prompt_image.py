import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch')
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'huggingface')

from huggingface_hub import whoami
import torch
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from diffusers import AutoPipelineForImage2Image

DEVICE = "mps" 
DTYPE = torch.float16

OUTPUT_DIR = 'outputs/stable_diffusion_prompt_to_img_test'


def load_pipeline():
    print("Loading SDXL Image-to-Image pipeline...")
    # Using AutoPipelineForImage2Image automatically handles the i2i logic
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=DTYPE,
        variant="fp16", # Loads smaller files to save RAM
        use_safetensors=True
    )
    
    # Critical for Mac performance
    pipe.to(DEVICE)
    
    # Enable attention slicing to save memory if you run other apps
    pipe.enable_attention_slicing()
    
    return pipe


def process_frame(pipe, content_path, style_prompt):
    print(f"Processing: {content_path}")
    
    content_image = Image.open(content_path).convert("RGB")
    content_image = content_image.resize((1024, 1024))

    # Strength: 0.0 = original image, 1.0 = entirely new image.
    # For style transfer, 0.45 - 0.6 is the sweet spot.
    image = pipe(
        prompt=style_prompt,
        image=content_image,
        strength=0.5,
        guidance_scale=7.5,
        num_inference_steps=30
    ).images[0]
    
    save_path = os.path.join(OUTPUT_DIR, "output_styled.png")
    image.save(save_path)
    print(f"Saved to {save_path}")


def main():
    # HF_TOKEN from .env is automatically used
    print("Logged in as:", whoami()['name'])
    # Load pipeline once
    pipe = load_pipeline()
    
    # Define your style in the prompt
    style_desc = "A sex doll, 8k, detailed, realistic"
    
    process_frame(
        pipe, 
        "data/content/grid1.jpg",
        style_desc
    )


if __name__ == "__main__":
    main()