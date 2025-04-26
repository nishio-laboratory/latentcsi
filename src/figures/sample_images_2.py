from src.inference.utils import *
import torch.multiprocessing as mp
import argparse
from pathlib import Path
import os
from src.targets.utils import preprocess_resize
from src.other.blur_face import blur_faces_opencv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from brisque import BRISQUE


def generate_fig(photo_path: Path, p, sd, include_sd=True):
    fig, axs = plt.subplots(
        4, 3 if include_sd else 2, figsize=(7.14 / 2, 7.14)
    )
    idxs = [227, 95, 54, 91]
    for n, i in enumerate(idxs):
        photo = blur_faces_opencv(Image.open(photo_path / f"{i}_p.png"))
        latent = Image.open(photo_path / f"{i}_l.png")
        for a in axs[n]:
            a.axis("off")
        axs[n][0].imshow(photo)
        axs[n][1].imshow(latent)
        if include_sd:
            axs[n][2].imshow(
                sd(
                    image=p[i] * 0.18215,
                    prompt="photograph of a man standing in a small office room, realistic, 4k, high resolution",
                    strength=0.55,
                    inference_steps=75,
                ).images[0]
            )

    if include_sd:
        col_titles = ["Reference", "Strength = 0", "Strength = 0.6"]
    else:
        col_titles = ["Reference", "Strength = 0"]

    for col, title in enumerate(col_titles):
        axs[0, col].set_title(title, fontsize=12, pad=20)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--runs", type=int, required=False, default=1)
    parser.add_argument(
        "--sd", action="store_true", required=False, default=False
    )
    args = parser.parse_args()
    args.ckpt = "mmfi_two_cnn_vaelike_attention_512_4stepmlp_cnn_val_loss=0.7566919922828674.ckpt"

    if not args.out:
        args.out = Path(os.getcwd())
    else:
        args.out.mkdir(exist_ok=True)

    if args.sd:
        sd = load_sd(args.path.parents[1], torch.device(0))
    else:
        sd = None

    torch.set_grad_enabled(False)

    testset_path = args.path / f"testset_inference_{args.ckpt}"
    p = torch.load(
        testset_path / "all_preds.pt", mmap=True, map_location="cpu"
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "figure.autolayout": True,
        }
    )
    for i in range(args.runs):
        fig = generate_fig(testset_path, p, sd, args.sd)
        plt.tight_layout()
        fig.savefig(args.out / f"samples_{i}.pdf", backend="pgf")
