#
# Stable Diffusion Prompt+Image to Image Pipeline
# Interestingly, way slower than the image to image pipeline
# Also, results are not as good as the image to image pipeline.
#

OUTPUT_PATH = 'outputs/stable_diffusion_prompt_to_img_test.png'
CONTENT_IMAGE_PATH = 'data/content/grid1.jpg'
STYLE_PROMPT = "A sex doll, 8k, detailed, realistic"

import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch')
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'huggingface')

from huggingface_hub import whoami
import torch
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def load_pipeline():
    print("Loading SDXL Image-to-Image pipeline...")
    from diffusers import AutoPipelineForImage2Image

    model_id = "runwayml/stable-diffusion-v1-5"
    # Using AutoPipelineForImage2Image automatically handles the i2i logic
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float32,
        variant="fp16", # Loads smaller files to save RAM
        use_safetensors=True
    )
    
    # Controls style influence (0-1)
    # 0 = original image, 1 = entirely new image.
    pipe.set_ip_adapter_scale(0.8)

    # Try mps
    pipe.to("mps")
    # Enable attention slicing to save memory if you run other apps
    pipe.enable_attention_slicing()
    print("Pipeline loaded!")

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
    
    image.save(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")


def main():
    # HF_TOKEN from .env is automatically used
    print("Logged in as:", whoami()['name'])
    # Load pipeline once
    pipe = load_pipeline()
    
    process_frame(
        pipe,
        content_path=CONTENT_IMAGE_PATH,
        style_prompt=STYLE_PROMPT,
    )


if __name__ == "__main__":
    main()