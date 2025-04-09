from typing import cast
from pathlib import Path
import argparse
import torch
import diffusers
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument("-p", "--path", type=Path, required=True)
# args = parser.parse_args()

args = argparse.Namespace()
# args.path = Path("/mnt/nas/esrh/csi_image_data/datasets/mmfi_hands_two/testset_inference_mmfi_two_cnn_vaelike_attention_768_4stepmlp_cnn_val_loss=0.7639663219451904.ckpt/")
args.path = Path(
    "/mnt/nas/esrh/csi_image_data/datasets/walking/testset_inference_walking_vaelike_512_4st_run2mlp_cnn_val_loss=5.283313751220703.ckpt"
)

p = torch.load(args.path / "all_preds.pt", mmap=True)

sd_path = next(
    (
        i
        for i in [Path("~/sd-v1-5"), args.path.parents[1] / "sd/sd-v1-5"]
        if i.exists()
    ),
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
)
ddim = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")
sd = cast(
    StableDiffusionImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline.from_pretrained(sd_path, scheduler=ddim),
).to("cuda")
sd.safety_checker = None

# ***
idx = 2
image = p[idx] * 0.18215
if len(image.shape) == 3:
    image = image.unsqueeze(0)

sd(
    "a full-length portrait photograph of a man walking in a small room, 4k, realistic, ultra high quality, detailed face",
    negative_prompt="blurry face, missing head",
    image=image,
    strength=0.6,
    guidance_scale=6.5,
    inference_steps=75,
).images[0].show()

torch.cuda.empty_cache()

# def __call__(
#         self,
#         prompt: Union[str, List[str]] = None,
#         image: PipelineImageInput = None,
#         strength: float = 0.8,
#         num_inference_steps: Optional[int] = 50,
#         timesteps: List[int] = None,
#         sigmas: List[float] = None,
#         guidance_scale: Optional[float] = 7.5,
#         negative_prompt: Optional[Union[str, List[str]]] = None,

# ***

import gc

gc.collect()
