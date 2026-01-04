1. Set up a virtual environment.

```shell
python -m venv .venv
source .venv/bin/activate (mac)
.\.venv\Scripts\Activate.ps1 (windows)
```

2. Install dependencies

```shell
pip install -r requirements.txt
```

3. Allow SSH downloads

`/Applications/Python\ 3.12/Install\ Certificates.command`

4. Populate Image Folders

- `data/style` should be filled with style images you want to apply.
- `data/content` should be filled with images you want to apply style to.

5. Explore the `example_images` folder and readme to see what kind of results you can get from different scripts and images.

6. Choose your script and allocate correct paths eg. STYLE_IMAGE_PATH = "data/<your_file_name>" for the images you populated in step 4.

IF ON WINDOWS- SWITCH `METHOD` to "cuda"

7. Run your script of choice eg. `python scripts_that_work/SD_controlnet_animation.py`. Make sure you are running python from this directory, and expect the runtime to be VERY slow (up to half an hour) before producing any images. Downloading the AI model can take a long time.