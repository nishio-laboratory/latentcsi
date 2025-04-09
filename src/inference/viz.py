from pathlib import Path
import argparse
import torch
import sys
from typing import List, cast
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from PIL.Image import Image as PILImage
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
import numpy as np

from src.encoder import training_latents
from src.encoder.data_utils import process_csi
from src.inference.utils import vae_decode


def test_model(
    model: torch.nn.Module,
    input: np.ndarray,
):
    test_input = process_csi(input).unsqueeze(0)
    test_input = test_input.to(model.device)
    pred = model(test_input).to(model.device)
    return pred * 0.18215


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path, required=True)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--index", type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--photo_only", action="store_true", default=False)
    args = parser.parse_args()

    photos = np.load(args.path / "photos.npy", mmap_mode="r")
    Image.fromarray(photos[args.index]).save(args.path / "test_photo.png")
    print(f"Saved photo {args.index} / {len(photos)}")
    if args.photo_only:
        sys.exit(0)

    if args.model == "autoencoder":
        model = training_latents.CSIAutoencoder
        model = model.load_from_checkpoint(args.path / "ckpts" / args.ckpt)
    elif args.model == "mlp_cnn":
        model = training_cnn.CSIAutoencoderMLP_CNN
        model = model.load_from_checkpoint(args.path / "ckpts" / args.ckpt)
    else:
        raise Exception("model type not valid")

    model.eval()
    model = model.to(args.device)

    sd = cast(
        StableDiffusionImg2ImgPipeline,
        StableDiffusionImg2ImgPipeline.from_pretrained(
            args.path.parents[1] / "sd/sd-v1-5"
        ),
    ).to(model.device)
    sd.safety_checker = None

    csi = np.load(args.path / "csi.npy", mmap_mode="r")

    p = test_model(model, csi[args.index])
    vae_decode(sd, p / 0.18215).save(args.path / "test_latent.png")

    img = sd(
        "photograph of a man in a clean room, 4k, realistic, high resolution",
        p,
        strength=0.55,
        inference_steps=70,
        guidance_scale=6,
    ).images[0]
    img.save(args.path / "test_out.png")


# python -m src.encoder.viz --path /data/datasets/walking --ckpt mlp_seg_1992-2000-1000-500-1000-2000-16384val_loss=14.699368476867676.ckpt --model autoencoder_seg
