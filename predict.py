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
        prompt: str = Input(description="Input prompt"),
        width: int = Input(
            description="width", ge=0, le=1024, default=512
        ),
        height: int = Input(
            description="height", ge=0, le=1024, default=512
        ),
        # image: Path = Input(description="Grayscale input image"),
        # scale: float = Input(
        #     description="Factor to scale image by", ge=0, le=10, default=1.5
        # ),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        payload = {
            # "init_images": [encoded_image],
            "prompt": prompt,
            "negative_prompt": "",
            "batch_size": 1,
            "steps": 20,
            "cfg_scale": 7,
            "denoising_strength": 0.75,
            # "seed": -1,
            "do_not_save_samples": True,
            "sampler_name": "DPM++ SDE Karras",
            "width": width,
            "height": height,
            # "enable_hr": True,
            # "denoising_strength": 0.5,
            # "hr_scale": 2,
            # "hr_upscaler": "R-ESRGAN 4x+ Anime6B",
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
