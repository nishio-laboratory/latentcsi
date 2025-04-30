from src.inference.utils import *
import sys
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


def generate_fig(prop_path, base_path):
    fig, axs = plt.subplots(4, 3, figsize=(7.14 / 2, 5))
    idxs = [227, 95, 67, 91]
    for n, i in enumerate(idxs):
        ref = blur_faces_opencv(Image.open(prop_path / f"{i}_p.png"))

        prop_img = Image.open(prop_path / f"{i}_l.png")
        base_img = Image.open(base_path / f"{i}_l.png")

        for a in axs[n]:
            a.axis("off")
        axs[n][0].imshow(ref)
        axs[n][1].imshow(prop_img)
        axs[n][2].imshow(base_img)

    col_titles = ["Reference", "Proposed model", "Baseline"]

    for col, title in enumerate(col_titles):
        axs[0, col].set_title(title, pad=10)

    return fig


def save_imgs(prop_path, base_path, out_path):
    # idxs = [227, 95, 67, 91]
    idxs = [227, 95, 15, 91]
    for n, i in enumerate(idxs):
        ref = Image.open(prop_path / f"{i}_p.png")
        prop_img = Image.open(prop_path / f"{i}_l.png")
        base_img = Image.open(base_path / f"{i}_l.png")

        ref.save(out_path / f"r_{n}.png")
        prop_img.save(out_path / f"p_{n}.png")
        base_img.save(out_path / f"b_{n}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path, required=True)
    parser.add_argument("-o", "--out", type=Path, required=False, default=None)
    parser.add_argument("-i", "--individual", action="store_true")
    args = parser.parse_args()

    prop_path = args.path / "testset_inference_final_best.ckpt"
    base_path = args.path / "testset_inference_baseline_best.ckpt"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "text.usetex": True,
            "pgf.rcfonts": False,
            "figure.autolayout": True,
        }
    )

    if args.individual:
        save_imgs(prop_path, base_path, args.out)
        sys.exit(0)

    fig = generate_fig(args.path / "reference", prop_path, base_path)
    plt.tight_layout()
    # fig.savefig(args.path / "figures" / f"samples_{i}.pdf", backend="pgf")
    fig.savefig("samples.pdf", backend="pgf", bbox_inches="tight")

    kwargs = {"backend": "pgf", "bbox_inches": "tight"}
    fig.savefig("samples.pdf", **kwargs)

    if args.out:
        fig.savefig(args.out / "samples_1.pdf", **kwargs)
    else:
        fig.savefig("samples.pdf", **kwargs)
