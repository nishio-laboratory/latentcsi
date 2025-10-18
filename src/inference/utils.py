from pathlib import Path
import io
from typing import Callable, cast, Optional
from diffusers import AutoencoderTiny
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from demo.client.utils import DummyImageProcessor
from src.encoder.data_utils import CSIDataset
import torch
import math
from torchvision.transforms.functional import to_pil_image
from PIL.Image import Image as PILImage
from src.other.types import *


def load_test_dataset(path: Path, aux_data=[]):
    dataset = CSIDataset(path, aux_data=aux_data)
    _, _, test = torch.utils.data.random_split(
        dataset, [0.8, 0.1, 0.1], torch.Generator().manual_seed(42)
    )
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(42)
    ).tolist()
    test_indices = indices[-int(math.floor(len(dataset) * 0.1)) :]
    return test, test_indices


def vae_decode(sd, pred) -> PILImage:
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    return to_pil_image(((sd.vae.decode(pred).sample + 1) / 2).squeeze())


def sd_load(
    path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> SDImage:
    path_candidates = [Path("~/sd-v1-5"), Path("/home/sd-v1-5")]
    if path is not None:
        path_candidates.append(path / "sd-v1-5")
    sd_path = next(
        (i for i in path_candidates if i.exists()),
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    ddim = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")
    sd = cast(
        StableDiffusionImg2ImgPipeline,
        StableDiffusionImg2ImgPipeline.from_pretrained(
            sd_path, scheduler=ddim
        ),
    )
    sd.safety_checker = None  # pyright: ignore

    if device:
        sd = sd.to(device)  # pyright: ignore
    return cast(SDImage, sd)

def sd_convert(sd: SDImage) -> SDLatent:
    sd.image_processor = DummyImageProcessor()  # pyright: ignore
    return cast(SDLatent, sd)

def sd_make_lat_tiny(sd: SDLatent, vae: AutoencoderTiny) -> SDLatentTiny:
    sd.vae = vae.to(sd.device)  # pyright: ignore
    return cast(SDLatentTiny, sd)
sd_make_im_tiny = cast(Callable[[SDImage, AutoencoderTiny], SDImageTiny], sd_make_lat_tiny)


def generate(sd, input, **kwargs) -> PILImage:
    if len(input.shape) == 3:
        input = input.unsqueeze(0)
    if "strength" in kwargs and kwargs["strength"] == 0:
        return vae_decode(sd, input)
    else:
        return sd(
            image=input * 0.18215,
            **kwargs,
        ).images[0]


def permute_color_chan(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 3:
        return t.permute(2, 1, 0)
    else:
        return t.permute(0, 3, 2, 1)

def pil_image_to_bytes(i: ImageType) -> bytes:
    buf = io.BytesIO()
    i.save(buf, format="JPEG")
    return buf.getvalue()
