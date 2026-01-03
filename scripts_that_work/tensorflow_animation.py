#
# TensorFlow Animation Pipeline
# Really quick results.
# It uses the tensorflow hub model to generate an image 
# from a style and a content image.
# It then blends the image with the previous frame to reduce flicker
# Not great results, but it's a starting point.
#

OUTPUT_DIR = 'outputs/animation_script_test'
ANIMATION_DIR = 'data/animation'
STYLE_DIR = 'data/style'

import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch')

import tensorflow as tf
import numpy as np
import PIL
import glob
import kagglehub
import tensorflow_hub as hub

def imagepaths_list_from_folder(folder_path):
    return [f"{folder_path}/{f}" for f in os.listdir(folder_path) if not f.startswith('.')]


def model_download() -> tf.Module:
    print("Downloading model...")
    path = kagglehub.model_download("google/arbitrary-image-stylization-v1/tensorFlow1/256")
    print("Model downloaded!")
    return hub.load(path)


def tensor_to_image(tensor) -> PIL.Image.Image:
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


def load_img(path_to_img, resize=False) -> tf.Tensor:
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3, expand_animations=False)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:2], tf.float32)
  long_dim = tf.reduce_max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)
  if resize:
    img = tf.image.resize(img, new_shape)
  else:
    img = tf.image.resize(img, [max_dim, max_dim])
  img = img[tf.newaxis, :]
  return img


def apply_styles_to_animation(
        animation_frames: list[str], 
        style_paths: list[str]
    ) -> None:
    num_frames = len(animation_frames)
    num_styles = len(style_paths)
    style_tensors = [load_img(p, resize=False) for p in style_paths]

    hub_model = model_download()

    print("Applying styles to animation...")
    for i, frame_path in enumerate[str](animation_frames):
        # Determine which two styles we are between
        # This maps the current frame index to a float position in the style list
        style_idx_float = (i / (num_frames - 1)) * (num_styles - 1)
        idx1 = int(np.floor(style_idx_float))
        idx2 = int(np.ceil(style_idx_float))
        
        # Calculate the blend weight (0.0 to 1.0)
        fraction = style_idx_float - idx1
        
        # Interpolate between the two style images
        # This creates a 'morphed' style image for this specific frame
        interpolated_style = (1 - fraction) * style_tensors[idx1] + fraction * style_tensors[idx2]
        
        # Run Stylization
        content_img = load_img(frame_path, resize=True)
        stylized_image = hub_model(tf.constant(content_img), tf.constant(interpolated_style))[0]
        
        output_image = tensor_to_image(stylized_image)
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        output_image.save(f"{OUTPUT_DIR}/{i}.png")


def main():
    animation_frames = sorted(glob.glob(f'{ANIMATION_DIR}/*.png'))
    style_paths = imagepaths_list_from_folder(STYLE_DIR)
    apply_styles_to_animation(animation_frames, style_paths)


if __name__ == "__main__":
    main()