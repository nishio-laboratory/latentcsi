import os
from pathlib import Path
import argparse
import torch
from typing import cast
from PIL import Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
import numpy as np

from src.encoder import training_cnn_att
from src.inference.utils import vae_decode
from src.inference.utils import load_test_dataset
import math
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path, required=True)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--save", default=False, action="store_true")
    args = parser.parse_args()

    if not args.ckpt.endswith(".ckpt"):
        args.ckpt += ".ckpt"

    model = training_cnn_att.CSIAutoencoderMLP_CNN
    model = model.load_from_checkpoint(args.path / "ckpts" / args.ckpt)

    model.eval()
    model = model.to(args.device)
    torch.set_grad_enabled(False)

    if args.save:
        sd = cast(
            StableDiffusionImg2ImgPipeline,
            StableDiffusionImg2ImgPipeline.from_pretrained(
                next(
                    i
                    for i in [
                        Path("~/sd-v1-5"),
                        args.path.parents[1] / "sd/sd-v1-5",
                    ]
                    if i.exists()
                )
            ),
        ).to(model.device)
        sd.safety_checker = None
        photos = np.load(args.path / "photos.npy", mmap_mode="r")
    else:
        sd = None
        photos = None

    test, test_indices = load_test_dataset(args.path)

    inf_path = args.path / f"testset_inference_{os.path.basename(args.ckpt)}"
    inf_path.mkdir(exist_ok=True)

    test_preds = torch.zeros(len(test), 4, 64, 64).to(args.device)
    total_loss = 0
    for n, (i, (x, y)) in tqdm(
        enumerate(zip(test_indices, iter(test))), total=len(test_indices)
    ):
        x = x.to(args.device).unsqueeze(0)
        y = y.to(args.device).unsqueeze(0)
        with torch.no_grad():
            p = model(x).to(args.device)
        if len(p.shape) == 2:
            p = p.unsqueeze(0)
        # p: (1, 4, 64, 64)
        test_preds[n] = p.squeeze()
        total_loss += torch.nn.functional.mse_loss(p, y)
        if args.save and sd and photos is not None:
            vae_decode(sd, p).save(inf_path / f"{n}_l.png")
            Image.fromarray(photos[i]).save(inf_path / f"{n}_p.png")

    print(total_loss / len(test_indices))

    torch.save(test_preds, inf_path / "all_preds.pt")

# mmfi_hands_two: 11880 boundary
# python -m src.inference.viz_test_split -p /data/datasets/mmfi_hands_two/ --ckpt 'mmfi_two_cnn_vaelike_attention_768_4stepmlp_cnn_val_loss=0.7639663219451904.ckpt' --model vaelike_4step_att
