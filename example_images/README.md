### SD_img2img_animation1.png:
    created with `scripts/animation_scripts/SD_animation_simple.py` using
    ANIMATION_DIR = (folder with this in) `animation_frame1.png`
    STYLE_IMAGE_PATH = `sexdoll.jpg`

### controlnet_img2img_animation.png:
    created with `experiments/SD_controlnet_animation_simple.py` using
    ANIMATION_DIR = (folder with this in) `animation_frame_paler1.png`
    STYLE_IMAGE_PATH = `sexdoll.jpg`
    PROMPT = "A sex doll, detailed, realistic"

    ADAPTER_SCALE = 1.0
    TRANSFORM_STRENGTH = 0.75
    CONTROLNET_CONDITIONING_SCALE = 0.5
    NUM_INFERENCE_STEPS = 15
    GUIDANCE_SCALE = 4.0

### SD_imgtoimg.png: 
    created with `scripts/SD_img_to_img.py` using 
    STYLE_IMAGE_PATH = `sexdoll.jpg`
    CONTENT_IMAGE_PATH = `animation_frame1.png`

### stable_diffusion_prompt_to_img_test.png:
    created with `experiments/SD_prompt_image.py` using
    STYLE_PROMPT = "A sex doll, 8k, detailed, realistic"
    CONTENT_IMAGE_PATH = 'grid1.jpg'

### tensorflow_animation_frame.png:
    created with `scripts/animation_scripts/tensorflow_animation.py` using
    STYLE_IMAGE_PATH = 'sexdoll.jpg' (actually a similar image but I'm too lazy to perfectly recreate)
    CONTENT_IMAGE_PATH = 'animation_frame1.png'



