import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch')
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'huggingface')

from huggingface_hub import whoami
from dotenv import load_dotenv
from glob import glob

from tools import load_img_to_img_pipeline, generate_img_to_img

load_dotenv()

OUTPUT_DIR = 'outputs/stable_diffusion_animation'
ANIMATION_DIR = 'data/animation_paler'
STYLE_IMAGE_PATH = 'data/style2/sex_doll2.jpg'

def main():
    # HF_TOKEN from .env is automatically used
    print("Logged in as:", whoami()['name'])
    img_to_img_pipe = load_img_to_img_pipeline(adapter_scale=0.4)
    animation_frames = sorted(glob(f'{ANIMATION_DIR}/*.png'))

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for frame in animation_frames:
        frame_number = frame.split('/')[-1].split('.')[0]
        print(f"Generating styled image for frame {frame_number}...")
        generate_img_to_img(
            img_to_img_pipe,
            style_image_path=STYLE_IMAGE_PATH,  # Style to apply
            content_image_path=frame,  # Content to transform
            output_path=f"{OUTPUT_DIR}/{frame_number}.png"
        )

if __name__ == "__main__":
    main()