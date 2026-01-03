import os

# Fix macOS threading issues - MUST be before other imports
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch')
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'huggingface')

from huggingface_hub import whoami
import torch
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = 'outputs/stable_diffusion_test'

def load_prompt_pipeline():
    print("Loading prompt image generation pipeline... This may take a few minutes...")
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", 
        torch_dtype=torch.float32,  # Use float32 instead of bfloat16
        low_cpu_mem_usage=True,
    )
    pipe.to("cpu")  # Stay on CPU entirely
    print("Pipeline loaded successfully!")
    return pipe


def load_img_to_img_pipeline():
    print("Loading image to image generation pipeline...")
    from diffusers import StableDiffusion3Img2ImgPipeline
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pipe.to("cpu")
    print("Pipeline loaded!")
    return pipe


def generate_prompt_image(pipe, prompt):
    print("Generating image from prompt...")
    image = pipe(prompt).images[0]
    image.save(os.path.join(OUTPUT_DIR, "output.png"))
    print(f"Image saved to {OUTPUT_DIR}/output.png")
    return image

def generate_img_to_img_image(pipe, style_image_path, content_image_path):
    print("Generating image from content image...")
    style_image = Image.open(style_image_path).convert("RGB")
    style_image = style_image.resize((1024, 1024))

    content_image = Image.open(content_image_path).convert("RGB")
    content_image = content_image.resize((1024, 1024))

    # SD3 Img2Img uses 'image' and 'prompt', not style_image/content_image
    image = pipe(prompt="", image=content_image, strength=0.7).images[0]
    image.save(os.path.join(OUTPUT_DIR, "output_img2img.png"))
    print(f"Image saved to {OUTPUT_DIR}/output_img2img.png")
    return image

def main():
    # HF_TOKEN from .env is automatically used
    print("Logged in as:", whoami()['name'])

    prompt_pipe = load_prompt_pipeline()

    generate_prompt_image(
        prompt_pipe, 
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    img_to_img_pipe = load_img_to_img_pipeline()
    generate_img_to_img_image(
        img_to_img_pipe, 
        "../data/style/sex_doll2.jpg", 
        "../data/content/grid1.jpg"
    )


if __name__ == "__main__":
    main()