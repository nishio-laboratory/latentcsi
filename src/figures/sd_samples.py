from typing import List, Tuple
from torch import multiprocessing as mp
from src.inference.utils import *
from torchvision.transforms.functional import to_pil_image
import numpy as np
import torch
import argparse
from pathlib import Path
import os
from src.other.blur_face import blur_faces_opencv
from PIL import Image


def worker(rank, latent, q, args, sd_kwargs):
    torch.set_grad_enabled(False)
    device = torch.device(f"cuda:{rank}")
    sd = load_sd(args.path, device)
    latent = latent.to(device)
    for _ in range(args.runs):
        q.put((sd_kwargs, generate(sd, latent, **sd_kwargs)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--idx", type=int)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    # args = argparse.Namespace(path = Path("/mnt/nas/esrh/csi_image_data"))

    dataset = (
        args.path
        / f"datasets/{args.dataset}"
        / "testset_inference_final_best.ckpt"
    )
    image_1_ref = blur_faces_opencv(Image.open(dataset / f"{args.idx}_p.png"))
    image_1_prop = torch.load(
        dataset / "all_preds.pt", mmap=True, map_location="cpu"
    )[args.idx]

    configs = [
        {
            "prompt": "a drawing of a man in a laboratory, anime, 4k",
            "strength": 0.55,
            "inference_steps": 100,
            "guidance_scale": 6.5,
        },
        {
            "prompt": "a photograph of a man in a small office room, 4k, realistic",
            "strength": 0.55,
            "inference_steps": 100,
            "guidance_scale": 6.5,
        },
    ]
    mp.set_start_method("spawn")
    num_gpus = torch.cuda.device_count()
    with mp.Manager() as manager:
        q = manager.Queue()
        processes = []

        for i in range(num_gpus):
            p = mp.Process(
                target=worker, args=(i, image_1_prop, q, args, configs[0])
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        output: List[Tuple[dict, PILImage]] = []
        while not q.empty():
            output.append(q.get())

    for n, (config, image) in enumerate(output):
        image.save(
            args.path
            / f"sd_figure/walking_samples/s_{args.dataset}_{args.idx}_{config}_{n:02}.png"
        )
