# cog-sd-webui

[Cog](https://github.com/replicate/cog) template to deploy any SD models as an inference API endpoint with the capabilities of [SD WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), such as:
* Hires. fix
* Samplers
* Scripts
* All compatible extensions (e.g. [ADetailer](https://github.com/Bing-su/adetailer), [ControlNet](https://github.com/Mikubill/sd-webui-controlnet))

## Download Checkpoints
Put the model file (ckpt, safetensors, pth, etc.) into the `model` directory. Make sure there is only one model file in the directory. For instance:
```
wget --content-disposition -P model https://civitai.com/api/download/models/176425
```
## Build and Deploy
Execute the command below to deploy your model to [Replicate](https://replicate.com/).
```
cog push r8.im/${YOUR_MODEL_ID}
```
If you prefer to deploy to other private Docker registry, run the following command to build the image:
```
cog build -t ${IMAGE_TAG_FOR_PRIVATE_REGISTRY}
```
Then run `docker push` to deploy.
## Customize
Since the entire SD WebUI will be packaged into the image, you can utilize all functions provided by SD WebUI and it's extensions. 

As you can see in `cog.yaml` file, we've built-in ADetailer in this repo, you can achieve the same thing by adding extra scripts to bundle embedings, Lora, extensions into your image.

## Credits
* [SD WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
* [Cog](https://github.com/replicate/cog)
