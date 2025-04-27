from torch import permute
import pickle
from typing import List
from itertools import islice
from tqdm import tqdm
import glob
from src.inference.utils import *
import argparse
from pathlib import Path
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from src.inference.fid import compute_fid_inception as fid_inception

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--crop", action="store_true")
    args = parser.parse_args()
    args.ckpt = os.path.basename(args.ckpt)

    if not args.out:
        args.out = Path(os.getcwd())
    else:
        args.out.mkdir(exist_ok=True)

    if not str(args.ckpt).endswith(".ckpt"):
        args.ckpt += ".ckpt"

    torch.set_grad_enabled(False)

    testset_path = args.path / f"testset_inference_{args.ckpt}"

    p = [
        Image.open(i)
        for i in tqdm(
            sorted(glob.glob(str(testset_path / "*_l.png"))),
            desc="loading preds",
        )
    ]
    y = [
        Image.open(i)
        for i in tqdm(
            sorted(glob.glob(str(testset_path / "*_p.png"))),
            desc="loading reals",
        )
    ]

    if args.crop:
        with open(args.path / "ts_bboxes.pkl", "rb") as f:
            bboxes: List[List[List[int]]] = pickle.load(f)
        p = [
            img.crop(
                (
                    bbox[0][0],
                    bbox[0][1],
                    bbox[0][0] + bbox[0][2],
                    bbox[0][1] + bbox[0][3],
                )
            )
            for img, bbox in zip(p, bboxes)
            if len(bbox) != 0
        ]
        y = [
            img.crop(
                (
                    bbox[0][0],
                    bbox[0][1],
                    bbox[0][0] + bbox[0][2],
                    bbox[0][1] + bbox[0][3],
                )
            )
            for img, bbox in zip(y, bboxes)
            if len(bbox) != 0
        ]

    pixel_error_sum = 0
    ssim_sum = 0

    for n, (img, ref) in tqdm(
        enumerate(zip(p, y)), total=len(p), desc="computing"
    ):
        # img = generate(sd, latent, prompt="", strength=0.6)
        img = torch.Tensor(np.asarray(img))
        ref = torch.Tensor(np.asarray(ref))

        pe = torch.sum(torch.abs(img - ref))
        rmse = np.sqrt(
            mse(
                np.asarray(img, dtype=np.uint8),
                np.asarray(ref, dtype=np.uint8),
            )
        )
        pixel_error_sum += rmse
        s = cast(
            float,
            ssim(
                np.asarray(img),
                np.asarray(ref),
                channel_axis=2,
                data_range=255,
            ),
        )
        ssim_sum += s

    pixel_error = pixel_error_sum / len(p)
    ssim = ssim_sum / len(p)
    fid = fid_inception(p, y, device="cuda")

    output_strings = [
        f"RMSE {'(crop)' if args.crop else ''}: {pixel_error}",
        f"SSIM {'(crop)' if args.crop else ''}: {ssim}",
        f"FID {'(crop)' if args.crop else ''}: {fid}"
    ]

    for i in output_strings:
        print(i)

    with open(testset_path / "stats.txt", mode="a+") as f:
        f.write("\n" + "\n".join(output_strings) + "\n")



# python -m src.figures.strengths --path /mnt/nas/esrh/csi_image_data/datasets/walking/ --ckpt walking_vaelike_512_4st_run2mlp_cnn_val_loss=5.283313751220703.ckpt
