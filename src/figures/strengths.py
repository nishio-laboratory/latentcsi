from src.inference.utils import *
from torch import multiprocessing as mp
import argparse
from pathlib import Path
import os
from src.targets.utils import preprocess_resize
from src.other.blur_face import blur_faces_opencv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim


def generate_fig(ref, gens):
    strengths = [0, 0.4, 0.5, 0.6, 0.7, 1]
    fig, axs = plt.subplots(1, len(strengths) + 1, figsize=(7.14, 2))
    axs[0].imshow(blur_faces_opencv(ref))
    axs[0].axis("off")
    axs[0].set_title(f"reference", fontsize=9)
    for ax, s, img in zip(axs[1:], strengths, gens):
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


def worker(rank, args, latent, q):
    device = torch.device(f"cuda:{rank}")
    sd = load_sd(args.path.parents[1], device)
    torch.set_grad_enabled(False)

    latent = latent.to(device)

    strengths = [0, 0.4, 0.5, 0.6, 0.7, 1]
    for _ in range(args.runs):
        q.put(
            [
                generate(
                    sd,
                    latent,
                    strength=s,
                    prompt="photograph of a man standing in a small office room, realistic, 4k, high resolution",
                )
                for s in strengths
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--runs", type=int, required=False, default=1)
    parser.add_argument("--idx", type=int, required=False, default=2)
    args = parser.parse_args()
    args.ckpt = os.path.basename(args.ckpt)

    if not args.out:
        args.out = Path(os.getcwd())
    else:
        args.out.mkdir(exist_ok=True)

    if not args.ckpt.endswith(".ckpt"):
        args.ckpt += ".ckpt"

    torch.set_grad_enabled(False)

    p = torch.load(
        args.path / f"testset_inference_{args.ckpt}" / "all_preds.pt",
        mmap=True,
        map_location="cpu",
    )
    image = p[args.idx] * 0.18215
    ref = preprocess_resize(
        Image.open(
            args.path / f"testset_inference_{args.ckpt}" / f"{args.idx}_p.png"
        )
    )

    mp.set_start_method("spawn")
    num_gpus = torch.cuda.device_count()

    with mp.Manager() as manager:
        q = manager.Queue()
        processes = []

        for i in range(num_gpus):
            p = mp.Process(target=worker, args=(i, args, image, q))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        output = []
        while not q.empty():
            output.append(q.get())

    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "figure.autolayout": True,
        }
    )
    for n, out in enumerate(output):
        fig = generate_fig(ref, out)
        fig.savefig(args.out / f"strengths_{n}.pdf", backend="pgf")


# python -m src.figures.strengths --path /mnt/nas/esrh/csi_image_data/datasets/walking/ --ckpt walking_vaelike_512_4st_run2mlp_cnn_val_loss=5.283313751220703.ckpt
