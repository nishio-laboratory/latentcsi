from typing import cast
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderTiny
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
import torch
from PIL.Image import Image as ImageType
from torchvision.transforms.functional import to_pil_image

from demo.client.utils import DummyImageProcessor
from demo.client.webserver.models import Img2ImgParams
from src.other.types import *


def apply_sd(
    latent,
    sd: StableDiffusionImg2ImgPipeline,
    sd_params: Img2ImgParams,
) -> ImageType:
    sd.safety_checker = None
    sd.image_processor = DummyImageProcessor()
    out = cast(
        torch.Tensor,
        sd(
            image=latent.to(sd.device, sd.dtype),
            prompt=sd_params.prompt,
            neg_prompt=sd_params.negativePrompt,
            strength=sd_params.strength,
            guidance_scale=sd_params.cfg,
        ).images[0],
    )  # dummy image processor gives [3, 512, 512]
    return to_pil_image(out.clip(0, 1))


def apply_sd_lat(
    latent,
    sd: StableDiffusionImg2ImgPipeline,
    sd_params: Img2ImgParams,
    use_sd_post: bool,
) -> torch.Tensor:
    sd.safety_checker = None

    tensor = (
        sd(
            image=latent,
            prompt=sd_params.prompt,
            neg_prompt=sd_params.negativePrompt,
            strength=sd_params.strength,
            guidance_scale=sd_params.cfg,
            output_type="latent",
        )
        .images[0]
        .cpu()
        .unsqueeze(0)
    )  # pyright: ignore

    return tensor


def decode_latent_to_image(
    latent: Latent, ae: AutoencoderTiny, scale=False
) -> ImageType:
    latent_tensor = cast(torch.Tensor, latent).to(device=ae.device, dtype=ae.dtype)
    # if scale:
    #     latent_tensor *= 0.18215
    out_tensor = ae.decode(latent_tensor).sample.squeeze().cpu()
    if scale:
        out_tensor = (out_tensor + 1) / 2
    return to_pil_image(out_tensor.clip(0, 1))
