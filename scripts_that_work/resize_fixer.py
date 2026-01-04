import os
from glob import glob
from PIL import Image

SIZE_REF = "data/animation_paler/0001.png"
IMAGE_DIR_TO_RESIZE = "outputs/SD_controlnet_animation"
IMAGE_DIR_TO_SAVE = "outputs/SD_controlnet_animation_resized"

def main():
    size_ref = Image.open(SIZE_REF)
    size_ref_width, size_ref_height = size_ref.size
    for file in glob(f'{IMAGE_DIR_TO_RESIZE}/*.png'):
        image = Image.open(file)
        image = image.resize((size_ref_width, size_ref_height))
        image.save(f'{IMAGE_DIR_TO_SAVE}/{file.split("/")[-1]}')

if __name__ == "__main__":
    main()