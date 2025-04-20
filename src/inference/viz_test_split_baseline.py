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
from torchvision.transforms.functional import to_pil_image

from src.encoder import baseline
from src.inference.utils import permute_color_chan, vae_decode
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

    model = baseline.CSIAutoencoderMLP_CNN
    model = model.load_from_checkpoint(args.path / "ckpts" / args.ckpt)

    model.eval()
    model = model.to(args.device)
    torch.set_grad_enabled(False)

    test, test_indices = load_test_dataset(args.path, ["photos"])

    inf_path = args.path / f"testset_inference_{os.path.basename(args.ckpt)}"
    inf_path.mkdir(exist_ok=True)

    test_preds = torch.zeros(len(test), 512, 512, 3).to(args.device)
    total_loss = 0
    for n, (i, (x, _, y)) in tqdm(
        enumerate(zip(test_indices, iter(test))), total=len(test_indices)
    ):
        x = x.to(args.device).unsqueeze(0)
        y = y.to(args.device).unsqueeze(0)
        with torch.no_grad():
            p = model(x).to(args.device)
        if len(p.shape) == 2:
            p = p.unsqueeze(0)
        # [1, 512, 512, 3]
        test_preds[n] = p.squeeze()
        total_loss += torch.nn.functional.mse_loss(p, y)
        if args.save:
            Image.fromarray(np.asarray(p.squeeze().cpu(), dtype=np.uint8)).save(inf_path / f"{n}_l.png")
            Image.fromarray(np.asarray(y.squeeze().cpu(), dtype=np.uint8)).save(inf_path / f"{n}_p.png")

    avg_loss = total_loss / len(test_indices)

    print(total_loss / len(test_indices))
    torch.save(test_preds, inf_path / "all_preds.pt")

    with open(inf_path / "stats.txt", "+a") as f:
        f.writelines(f"avg_loss = {avg_loss}")


# mmfi_hands_two: 11880 boundary
# python -m src.inference.viz_test_split -p /data/datasets/mmfi_hands_two/ --ckpt 'mmfi_two_cnn_vaelike_attention_768_4stepmlp_cnn_val_loss=0.7639663219451904.ckpt' --model vaelike_4step_att
