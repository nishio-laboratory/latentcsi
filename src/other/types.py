from dataclasses import dataclass
from typing import Any, NewType
import typing
import re
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
import torch
from multiprocessing.synchronize import Lock as LockType
from PIL import Image

PredLatent = NewType("PredLatent", torch.Tensor)
TrueLatent = NewType("TrueLatent", torch.Tensor)
Latent = PredLatent | TrueLatent

BatchCSI = NewType("BatchCSI", torch.Tensor)
BatchTrueLatent = NewType("BatchTrueLatent", torch.Tensor)
BatchPredLatent = NewType("BatchPredLatent", torch.Tensor)
BatchLatent = BatchTrueLatent | BatchPredLatent

@dataclass
class Batch:
    csi: BatchCSI
    lat: BatchTrueLatent

VoidMessage = typing.Literal[
        "start_rec",
        "stop_rec",
        "start_train",
        "stop_train",
        "reset",
]

Message = VoidMessage | tuple[typing.Literal["chglr"], float]
def check_msg(msg: str) -> Message:
    if msg in typing.get_args(VoidMessage):
        return typing.cast(VoidMessage, msg)
    set_lr_match = re.match(r"^set_lr\((.+)\)$", msg)
    if set_lr_match:
        try:
            new_lr = float(set_lr_match.group(1))
        except ValueError:
            raise ValueError(f"Invalid float inside set_lr({set_lr_match.group(1)})")
        return ("chglr", new_lr)
    raise ValueError(f"Invalid message: {msg}")

CSI = NewType("CSI", torch.Tensor)
ImageType = Image.Image

SDImage = NewType("SDImage", StableDiffusionImg2ImgPipeline)
SDImageTiny = NewType("SDImageTiny", StableDiffusionImg2ImgPipeline)
SDLatent = NewType("SDLatent", StableDiffusionImg2ImgPipeline)
SDLatentTiny = NewType("SDLatentTiny", StableDiffusionImg2ImgPipeline)
SD = SDImage | SDImageTiny | SDLatent | SDLatentTiny
