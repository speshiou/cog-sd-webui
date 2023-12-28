# cog-sd-webui

The COG template to deploy any SD models as an inference API includes the capabilities of [SD WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), such as:
* Hires. fix
* Samplers
* Scripts
* All compatible extensions (e.g. [ADetailer](https://github.com/Bing-su/adetailer))

## Setup
Put the model file into the `model` directory. Make sure there is only one model file in the directory.
## Deploy
Execute the command below to deploy your model to replicate.com.
```
cog push r8.im/${YOUR_MODEL_ID}
```
## Credits
* [SD WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
