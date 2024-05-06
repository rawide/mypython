import torch
from diffusers import StableDiffusionPipeline

model_id = "../../stable-diffusion-v1-5"
device = "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]