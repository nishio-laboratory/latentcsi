from pathlib import Path
import argparse
import torch
import diffusers
from typing import List, cast
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from PIL.Image import Image as PILImage
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
import numpy as np

from src.encoder import training_latents, training_seg
from src.encoder.data_utils import process_csi


def save_pred_latent(out_path: Path, pred: torch.Tensor, sd_pipeline):
    decoded = (sd_pipeline.vae.decode(pred).sample + 1) / 2
    to_pil_image(decoded.squeeze()).save(out_path)


def test_model(
    out_path: Path,
    model: torch.nn.Module,
    sd_pipeline: StableDiffusionImg2ImgPipeline,
    photos: List[np.ndarray],
    inputs: np.ndarray,
    test_idx: int,
):
    Image.fromarray(photos[test_idx]).save(out_path / "test_photo.png")
    sd_pipeline.safety_checker = None
    test_input = process_csi(inputs[test_idx]).unsqueeze(0)
    test_input = test_input.to(model.device)
    pred = model(test_input).to(model.device)
    save_pred_latent(out_path / "test_latent.png", pred, sd_pipeline)
    return pred * 0.18215


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--model")
    parser.add_argument("--device", default = 0, type = int)
    args = parser.parse_args()
    if args.model == "autoencoder":
        model = training_latents.CSIAutoencoder
        model = model.load_from_checkpoint(
            args.path / "ckpts" / args.ckpt
        )
    elif args.model == "autoencoder_seg":
        model = training_seg.CSIAutoencoder
        model = model.load_from_checkpoint(
            args.path / "ckpts" / args.ckpt,
            data_path=args.path,
            layer_sizes=[1992, 2000, 1000, 500, 1000, 2000, 16384]
        )
    else:
        raise Exception("model type not valid")

    model.eval()
    model = model.to(args.device)

    sd = cast(StableDiffusionImg2ImgPipeline, StableDiffusionImg2ImgPipeline.from_pretrained(
        args.path.parents[1] / "sd/sd-v1-5"
    )).to(model.device)
    photos = np.load(args.path / "photos.npy")
    csi = np.load(args.path / "csi.npy")
    p = test_model(args.path, model, sd, photos, csi, 1445)
    img = sd(
        "photograph of a man wearing a white hoodie in a clean room, 4k, realistic, high resolution",
        p,
        strength=0.55,
        inference_steps=70,
        guidance_scale=6
    ).images[0]
    img.save(args.path / "test_out.png")


# python -m src.encoder.viz --path /data/datasets/walking --ckpt mlp_seg_1992-2000-1000-500-1000-2000-16384val_loss=14.699368476867676.ckpt --model autoencoder_seg
