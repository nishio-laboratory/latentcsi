from pathlib import Path
import torch
import diffusers
from typing import List
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from PIL.Image import Image as PILImage
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)


def save_pred_latent(out_path: Path, pred: torch.Tensor, sd_pipeline):
    decoded = (sd_pipeline.vae.decode(pred).sample + 1) / 2
    to_pil_image(decoded.squeeze()).save(out_path)


def test_model(
    out_path: Path,
    model: torch.nn.Module,
    sd_pipeline: StableDiffusionImg2ImgPipeline,
    photos: List[PILImage],
    inputs: torch.Tensor,
    test_idx: torch.Tensor,
):
    Image.fromarray(photos[test_idx]).save(out_path / "test_photo.png")
    sd_pipeline.safety_checker = None
    test_input = inputs[test_idx]
    pred = model(test_input.unsqueeze(0))

    save_pred_latent(out_path / "test_latent.png", pred, sd_pipeline)

    return pred * 0.18215
