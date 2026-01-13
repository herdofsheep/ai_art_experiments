import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'torch')
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'huggingface')

import torch
from diffusers import FluxControlNetImg2ImgPipeline, FluxControlNetModel
from diffusers.utils import load_image
import cv2
import numpy as np
from PIL import Image
from glob import glob
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

from scripts.tools import get_last_styled_frame

# ============================================================================
# FLUX.1-schnell Animation Pipeline
# 
# This script uses Black Forest Labs' FLUX.1-schnell model for animation
# style transfer. FLUX is a 12B parameter rectified flow transformer that
# generates high-quality images in just 1-4 steps.
# ============================================================================

OUTPUT_DIR = 'outputs/animation_paler'
ANIMATION_DIR = "data/flux_animation_grid2"
STYLE_PROMPT = "Close up bare flesh, dripping with sweat, wet, shiny, realistic. A tangle of bodies, lit by a pale, pink, romatic light."

# STYLE_PROMPT = "close up wet shiny flesh, realistic"
# next? a tangle of naked bodies, lit by a pale, pink, romatic light. bare flesh, sweaty, realistic. 

# FLUX.1-schnell settings
NUM_INFERENCE_STEPS = 4  # schnell is optimized for 1-4 steps
GUIDANCE_SCALE = 0.0  # schnell uses 0.0 guidance (distilled model)

# Style intensity settings
CONTROLNET_CONDITIONING_SCALE = 0.7  # strength of edge preservation (0.5-0.9 works well)
TRANSFORM_STRENGTH = 0.77  # img2img strength (how much to deviate from input) higher increases style influence

# Anti-flicker: blend content frames (smooths input, no style accumulation)
CONTENT_FRAME_BLEND = 0.5  # Blend with previous content frame (0.0-0.5)


# Resolution
RESOLUTION = 512

# Device settings
DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16

# Model IDs
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
CONTROLNET_MODEL_ID = "InstantX/FLUX.1-dev-Controlnet-Canny"


def load_flux_pipeline():
    """Load FLUX.1-schnell with Canny ControlNet."""
    print("Loading FLUX.1-schnell ControlNet pipeline...")
    
    controlnet = FluxControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID,
        torch_dtype=TORCH_DTYPE,
    )
    
    pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
        FLUX_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=TORCH_DTYPE,
    )
    
    # Memory optimizations for limited VRAM
    pipe.enable_sequential_cpu_offload()
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing(slice_size="auto")
    
    print("FLUX pipeline loaded!")
    return pipe


def get_canny_image(image, low_threshold=100, high_threshold=200):
    """Create edge map from content frame using Canny edge detection."""
    image = np.array(image)
    edges = cv2.Canny(image, low_threshold, high_threshold)
    edges = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


def style_frame(pipe, content_img, prompt, prev_content_img=None):
    """Apply style to a single animation frame using FLUX + ControlNet."""
    content_img = content_img.resize((RESOLUTION, RESOLUTION))
    canny_img = get_canny_image(content_img)
    
    # Anti-flicker: blend with previous content frame
    if prev_content_img is not None and CONTENT_FRAME_BLEND > 0:
        prev_content_img = prev_content_img.resize((RESOLUTION, RESOLUTION))
        prev_arr = np.array(prev_content_img).astype(np.float32)
        curr_arr = np.array(content_img).astype(np.float32)
        blended = (CONTENT_FRAME_BLEND * prev_arr + (1 - CONTENT_FRAME_BLEND) * curr_arr)
        init_image = Image.fromarray(blended.astype(np.uint8))
    else:
        init_image = content_img
    
    result = pipe(
        prompt=prompt,
        image=init_image,
        control_image=canny_img,
        strength=TRANSFORM_STRENGTH,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        max_sequence_length=256,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]
    
    return result


def main():
    login(token=os.getenv("HF_TOKEN"))
    pipe = load_flux_pipeline()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    last_styled_frame_index, _ = get_last_styled_frame(OUTPUT_DIR)
    animation_frames = sorted(glob(f'{ANIMATION_DIR}/*.png'))
    
    print(f"Found {len(animation_frames)} animation frames")
    print(f"Resuming from frame {last_styled_frame_index + 1}")
    
    prev_content_img = None
    
    for i, frame_path in enumerate(animation_frames):
        content_image = load_image(frame_path)
        
        if i <= last_styled_frame_index:
            prev_content_img = content_image
            continue
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Styling frame {i}/{len(animation_frames)-1}...")
        result = style_frame(pipe, content_image, STYLE_PROMPT, prev_content_img)
        
        output_path = f"{OUTPUT_DIR}/{i}.png"
        result.resize(content_image.size).save(output_path)
        print(f"  Saved to {output_path}")
        
        prev_content_img = content_image
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
