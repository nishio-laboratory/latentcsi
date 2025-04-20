from pathlib import Path
from typing import cast, Optional
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from src.encoder.data_utils import CSIDataset
import torch
import math
from torchvision.transforms.functional import to_pil_image
from PIL.Image import Image as PILImage


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


def load_sd(
    path: Optional[Path] = None, device: Optional[torch.device] = None
) -> StableDiffusionImg2ImgPipeline:
    path_candidates = [Path("~/sd-v1-5"), Path("/home/sd-v1-5")]
    if path:
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
    return sd


def generate(sd, input, **kwargs) -> PILImage:
    if len(input.shape) == 3:
        input = input.unsqueeze(0)
    if kwargs["strength"] == 0:
        return vae_decode(sd, input / 0.18215)
    else:
        return sd(
            image=input,
            **kwargs,
        ).images[0]

def permute_color_chan(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 3:
        return t.permute(2, 1, 0)
    else:
        return t.permute(0, 3, 2, 1)
