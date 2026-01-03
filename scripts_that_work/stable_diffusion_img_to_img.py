import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch')
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'huggingface')

from huggingface_hub import whoami
from dotenv import load_dotenv

from tools import load_img_to_img_pipeline, generate_img_to_img

load_dotenv()

OUTPUT_PATH = 'outputs/stable_diffusion_test_NSFW.png'
STYLE_IMAGE_PATH = 'data/style2/sex_doll2.jpg'
CONTENT_IMAGE_PATH = 'data/content/images.png'


def main():
    # HF_TOKEN from .env is automatically used
    print("Logged in as:", whoami()['name'])
    img_to_img_pipe = load_img_to_img_pipeline()

    generate_img_to_img(
        img_to_img_pipe,
        style_image_path=STYLE_IMAGE_PATH,  # Style to apply
        content_image_path=CONTENT_IMAGE_PATH,  # Content to transform
        output_path=OUTPUT_PATH
    )


if __name__ == "__main__":
    main()