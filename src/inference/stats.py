from torch import device, permute
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
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error as mse
from src.inference.fid import compute_fid_inception as fid_inception


def compute_stats(p: List[PILImage], y: List[PILImage]) -> dict:
    pixel_error_sum = 0
    ssim_sum = 0
    for n, (img, ref) in tqdm(
        enumerate(zip(p, y)), total=len(p), desc="computing"
    ):
        img = torch.Tensor(np.asarray(img))
        ref = torch.Tensor(np.asarray(ref))

        rmse = np.sqrt(
            mse(
                np.asarray(img, dtype=np.uint8),
                np.asarray(ref, dtype=np.uint8),
            )
        )
        pixel_error_sum += rmse
        s = cast(
            float,
            structural_similarity(
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
    return {"rmse": pixel_error, "ssim": ssim, "fid": fid}


def crop_to_bboxes(a: List[PILImage], bboxes: List[List[List[int]]]):
    return [
        img.crop(
            (
                bbox[0][0],
                bbox[0][1],
                bbox[0][0] + bbox[0][2],
                bbox[0][1] + bbox[0][3],
            )
        )
        for img, bbox in zip(a, bboxes)
        if len(bbox) != 0
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--use-images", action="store_true")
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

    if not args.use_images:
        p: List[PILImage] = [
            Image.fromarray(i.numpy())
            for i in tqdm(
                torch.load(testset_path / "all_preds.pt", map_location="cpu")
            )
        ]
    else:
        p: List[PILImage] = [
            Image.open(i)
            for i in tqdm(
                sorted(glob.glob(str(testset_path / "*_l.png"))),
                desc="loading preds",
            )
        ]

    reals_filepaths = sorted(glob.glob(str(testset_path / "*_p.png")))
    if len(reals_filepaths) == 0:
        reals_filepaths = sorted(
            glob.glob(str(testset_path.parents[0] / "reference/*.png"))
        )
    y: List[PILImage] = [
        Image.open(i)
        for i in tqdm(
            reals_filepaths,
            desc="loading reals",
        )
    ]

    if args.crop:
        with open(args.path / "ts_bboxes.pkl", "rb") as f:
            bboxes: List[List[List[int]]] = pickle.load(f)
        p = crop_to_bboxes(p, bboxes)
        y = crop_to_bboxes(y, bboxes)

    pixel_error, ssim, fid = compute_stats(p, y).items()
    output_strings = [
        f"RMSE {'(crop)' if args.crop else ''}: {pixel_error}",
        f"SSIM {'(crop)' if args.crop else ''}: {ssim}",
        f"FID {'(crop)' if args.crop else ''}: {fid}",
    ]
    for i in output_strings:
        print(i)

    if args.save:
        with open(testset_path / "stats.txt", mode="a+") as f:
            f.write("\n" + "\n".join(output_strings) + "\n")


# python -m src.figures.strengths --path /mnt/nas/esrh/csi_image_data/datasets/walking/ --ckpt walking_vaelike_512_4st_run2mlp_cnn_val_loss=5.283313751220703.ckpt
