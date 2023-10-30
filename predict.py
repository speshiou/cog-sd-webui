# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os, sys
sys.path.extend(['/stable-diffusion-webui'])

from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # workaround for replicate since its entrypoint may contain invalid args
        os.environ['IGNORE_CMD_ARGS_ERRORS'] = '1'
        from modules import timer
        # moved env preparation to build time
        # from modules import launch_utils

        # with launch_utils.startup_timer.subcategory("prepare environment"):
        #     launch_utils.prepare_environment()

        from modules import initialize_util
        from modules import initialize

        startup_timer = timer.startup_timer
        startup_timer.record("launcher")

        initialize.imports()

        initialize.check_versions()

        initialize.initialize()

        from fastapi import FastAPI
        app = FastAPI()
        initialize_util.setup_middleware(app)

        from modules.api.api import Api
        from modules.call_queue import queue_lock
        
        self.api = Api(app, queue_lock)

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        negative_prompt: str = Input(description="Negative Prompt", default=""),
        width: int = Input(
            description="Width of output image", ge=1, le=1024, default=512
        ),
        height: int = Input(
            description="Height of output image", ge=1, le=1024, default=512
        ),
        num_outputs: int = Input(
            description="Number of images to output", ge=1, le=4, default=1
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=['DPM++ 2M Karras', 'DPM++ SDE Karras', 'DPM++ 2M SDE Exponential', 'DPM++ 2M SDE Karras', 'Euler a', 'Euler', 'LMS', 'Heun', 'DPM2', 'DPM2 a', 'DPM++ 2S a', 'DPM++ 2M', 'DPM++ SDE', 'DPM++ 2M SDE', 'DPM++ 2M SDE Heun', 'DPM++ 2M SDE Heun Karras', 'DPM++ 2M SDE Heun Exponential', 'DPM++ 3M SDE', 'DPM++ 3M SDE Karras', 'DPM++ 3M SDE Exponential', 'DPM fast', 'DPM adaptive', 'LMS Karras', 'DPM2 Karras', 'DPM2 a Karras', 'DPM++ 2S a Karras', 'Restart', 'DDIM', 'PLMS', 'UniPC'],
            default="DPM++ SDE Karras",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=-1
        ),
        # image: Path = Input(description="Grayscale input image"),
        enable_hr: bool = Input(
            description="Hires. fix", default=False,
        ),
        scale: float = Input(
            description="Factor to scale image by", ge=1, le=4, default=2
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        payload = {
            # "init_images": [encoded_image],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": num_outputs,
            "steps": num_inference_steps,
            "cfg_scale": guidance_scale,
            "denoising_strength": 0.75,
            "seed": seed,
            "do_not_save_samples": True,
            "sampler_name": scheduler,
            "enable_hr": enable_hr,
            "denoising_strength": 0.5,
            "hr_scale": scale,
            "hr_upscaler": "R-ESRGAN 4x+ Anime6B",
        }

        from modules.api.models import StableDiffusionTxt2ImgProcessingAPI, StableDiffusionImg2ImgProcessingAPI
        req = StableDiffusionTxt2ImgProcessingAPI(**payload)
        # generate
        resp = self.api.text2imgapi(req)

        from PIL import Image
        import uuid
        import base64
        from io import BytesIO

        cnres_img = None
        if len(resp.images) > 0:
            cnres_img = resp.images[0]
        gen_bytes = BytesIO(base64.b64decode(cnres_img))
        gen_data = Image.open(gen_bytes)
        filename = "{}.png".format(uuid.uuid1())
        gen_data.save(fp=filename, format="PNG")
        print(filename)
        return Path(filename)
