#
# Stable Diffusion Animation Pipeline 1
# This is a basic stable diffusion animation pipeline
# It uses the image to image pipeline to generate an image from a style and a content image
# It then blends the image with the previous frame to reduce flicker
# Not great results, but it's a starting point.
#


OUTPUT_DIR = 'outputs/stable_diffusion_animation_more_blend'
ANIMATION_DIR = 'data/animation_paler'
STYLE_IMAGE_PATH = 'data/style2/sex_doll2.jpg'

# Temporal blending: 0 = no blending, 0.3 = mild smoothing, 0.5+ = strong (may ghost)
TEMPORAL_BLEND = 0.5
ADAPTER_SCALE = 0.6

import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', '..',  'models', 'torch')
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'huggingface')

from huggingface_hub import whoami
from dotenv import load_dotenv
from glob import glob
from PIL import Image
import numpy as np

from scripts.tools import load_img_to_img_pipeline, generate_img_to_img

load_dotenv()


def blend_images(current: Image.Image, previous: Image.Image, blend_factor: float) -> Image.Image:
    """Blend current frame with previous frame to reduce flicker."""
    current_arr = np.array(current, dtype=np.float32)
    previous_arr = np.array(previous, dtype=np.float32)
    blended = (1 - blend_factor) * current_arr + blend_factor * previous_arr
    return Image.fromarray(blended.astype(np.uint8))


def main():
    # HF_TOKEN from .env is automatically used
    print("Logged in as:", whoami()['name'])
    img_to_img_pipe = load_img_to_img_pipeline(adapter_scale=ADAPTER_SCALE)
    animation_frames = sorted(glob(f'{ANIMATION_DIR}/*.png'))

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    prev_image = None
    
    for i, frame in enumerate(animation_frames):
        frame_number = frame.split('/')[-1].split('.')[0]
        print(f"Generating styled image for frame {frame_number} ({i+1}/{len(animation_frames)})...")
        
        styled_image = generate_img_to_img(
            img_to_img_pipe,
            style_image_path=STYLE_IMAGE_PATH,
            content_image_path=frame,
            output_path=None  # Don't save yet - we'll blend first
        )
        
        # Blend with previous frame to reduce flicker
        if prev_image is not None and TEMPORAL_BLEND > 0:
            styled_image = blend_images(styled_image, prev_image, TEMPORAL_BLEND)
        
        prev_image = styled_image
        styled_image.save(f"{OUTPUT_DIR}/{frame_number}.png")
        print(f"Saved {OUTPUT_DIR}/{frame_number}.png")

if __name__ == "__main__":
    main()