0. Install python 3.12

1. Set up a virtual environment.

```shell
python -m venv .venv
source .venv/bin/activate (Mac)
.\.venv\Scripts\Activate.ps1 (Windows)
```

2. Install dependencies

```shell
pip install -r requirements.txt
```

3. Populate Image Folders

- `data/style` should be filled with style images you want to apply.
- `data/content` should be filled with images you want to apply style to.

4. Explore the `example_images` folder and readme to see what kind of results you can get from different scripts and images.

5. Choose your script and allocate correct paths eg. STYLE_IMAGE_PATH = "data/<your_file_name>" for the images you populated in step 4.

IF ON WINDOWS- SWITCH `METHOD` to "cuda"

6. Run your script of choice eg. 
    `python scripts_that_work/SD_controlnet_animation.py`. (Mac)
    ` python .\scripts_that_work\SD_controlnet_animation.py` (Windows)
 Make sure you are running python from this directory, and expect the runtime to be VERY slow (up to half an hour) before producing any images. Downloading the AI model can take a long time.


# Common Issues

If you have issues connecting to huggingface or kagglehub,
you may need to allow SSH downloads
`/Applications/Python\ 3.12/Install\ Certificates.command`

On Windows, if you get `AssertionError: Torch not compiled with CUDA enabled` you can run
`pip uninstall torch torchvision -y; pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
to ensure you have a version of torch with cuda (for running on GPUs).
You may need to look up what version link will work with your computer's specs.