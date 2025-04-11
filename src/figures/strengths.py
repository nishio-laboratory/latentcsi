from src.inference.utils import *
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


def generate_fig(img, ref, sd):
    strengths = [0, 0.4, 0.5, 0.6, 0.7, 1]
    fig, axs = plt.subplots(1, len(strengths) + 1, figsize=(7.14, 2))
    axs[0].imshow(blur_faces_opencv(ref))
    axs[0].axis("off")
    axs[0].set_title(f"reference", fontsize=9)
    for ax, s in zip(axs[1:], strengths):
        img = generate(
            sd,
            image,
            strength=s,
            prompt="photograph of a man standing in a small office room, realistic, 4k, high resolution",
        )
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"strength = {s}", fontsize=9)

        ssim_score = ssim(np.asarray(ref), np.asarray(img), channel_axis=2)
        ax.text(
            0.5,
            -0.1,
            f"ssim = {ssim_score:.2f}",
            fontsize=8,
            ha="center",
            transform=ax.transAxes,
        )

    axs[-1].set_title("(baseline)\nstrength = 1", fontsize=9)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--runs", type=int, required=False, default=1)
    brisque = BRISQUE(url=False)
    args = parser.parse_args()
    args.ckpt = os.path.basename(args.ckpt)

    if not args.out:
        args.out = Path(os.getcwd())
    else:
        args.out.mkdir(exist_ok=True)

    if not args.ckpt.endswith(".ckpt"):
        args.ckpt += ".ckpt"

    test, test_indices = load_test_dataset(args.path)
    sd = load_sd(args.path.parents[1], torch.device(0))
    torch.set_grad_enabled(False)

    testset_path = args.path / f"testset_inference_{args.ckpt}"
    p = torch.load(testset_path / "all_preds.pt", mmap=True)

    idx = 2
    image = p[idx] * 0.18215
    ref = preprocess_resize(Image.open(testset_path / f"{idx}_p.png"))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "figure.autolayout": True,
        }
    )
    for i in range(args.runs):
        fig = generate_fig(image, ref, sd)
        fig.savefig(args.out / f"strengths_{i}.pdf", backend="pgf")

# python -m src.figures.strengths --path /mnt/nas/esrh/csi_image_data/datasets/walking/ --ckpt walking_vaelike_512_4st_run2mlp_cnn_val_loss=5.283313751220703.ckpt
