from glob import glob
import os
from PIL import Image
import numpy as np

ANIMATION_DIR = "data/tits_mask"
OUTPUT_DIR = "outputs/tits_mask_fg"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, path in enumerate(sorted(glob(f'{ANIMATION_DIR}/*.png'))):
        print(f"Styling frame {path}...")
        px = np.array(Image.open(path).convert("RGBA"))
        mask = px[:, :, 3] != 0
        px[:] = 0
        px[:, :, 3] = np.where(mask, 255, 0)
        Image.fromarray(px, "RGBA").save(f"{OUTPUT_DIR}/{i}.png")

if __name__ == "__main__":
    main()