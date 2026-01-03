import os

# Must be set BEFORE any imports - fixes mutex issues on macOS
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch')
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'huggingface')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# macOS-specific: disable fork safety checks that cause mutex issues
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['no_proxy'] = '*'  # Disable proxy to avoid urllib threading issues

from huggingface_hub import whoami
import torch
from PIL import Image
from dotenv import load_dotenv

# Limit PyTorch internal threading
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

load_dotenv()

OUTPUT_DIR = 'outputs/stable_diffusion_test'


def load_img_to_img_pipeline():
    print("Loading image to image generation pipeline...")
    from diffusers import StableDiffusionImg2ImgPipeline
    from transformers import CLIPTokenizer
    
    # Use SD 1.5 instead of SD3 - much simpler and more stable on macOS
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
    pipe = pipe.to("cpu")
    print("Pipeline loaded!")
    return pipe


def generate_img_to_img_image(pipe, content_image_path):
    print("Generating image from content image...")
    content_image = Image.open(content_image_path).convert("RGB")
    content_image = content_image.resize((512, 512))  # SD 1.5 uses 512x512

    with torch.inference_mode():
        image = pipe(
            prompt="artistic style transfer",
            image=content_image,
            strength=0.7,
            num_inference_steps=30
        ).images[0]
    image.save(os.path.join(OUTPUT_DIR, "output_img2img.png"))
    print(f"Image saved to {OUTPUT_DIR}/output_img2img.png")
    return image


def main():
    # HF_TOKEN from .env is automatically used
    print("Logged in as:", whoami()['name'])
    img_to_img_pipe = load_img_to_img_pipeline()

    generate_img_to_img_image(
        img_to_img_pipe, 
        "data/content/grid1.jpg"
    )


if __name__ == "__main__":
    main()