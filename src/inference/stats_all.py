from src.inference.testset_inference import main as run_inference
from src.inference.stats import compute_stats, crop_to_bboxes
from PIL import Image
import pickle
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path
from glob import glob
import os

if __name__ == "__main__":
    walking_path = Path("/data/datasets/walking")

    d1_final_ckpts = list(
        map(
            os.path.basename,
            glob(str(walking_path / "ckpts" / "walking_final_256-*")),
        )
    )
    d1_baseline_ckpts = list(
        map(
            os.path.basename,
            glob(str(walking_path / "ckpts" / "walking_baseline_gn_inch_8-*")),
        )
    )

    mmfi_path = Path("/data/datasets/mmfi_hands_two")
    d2_final_ckpts = list(
        map(
            os.path.basename,
            glob(
                str(
                    mmfi_path
                    / "ckpts"
                    / "mmfi_two_cnn_vaelike_attention_256-*"
                )
            ),
        )
    )
    d2_baseline_ckpts = list(
        map(
            os.path.basename,
            glob(str(mmfi_path / "ckpts" / "mmfi_two_baseline_32-*")),
        )
    )

    print(d1_final_ckpts, d1_baseline_ckpts, d2_final_ckpts, d2_baseline_ckpts)

    # for ckpt in d1_final_ckpts:
    #     run_inference(
    #         path=walking_path,
    #         ckpt=ckpt,
    #         device=0,
    #         save_latent=True,
    #         save_real=False,
    #     )
    # for ckpt in d1_baseline_ckpts:
    #     run_inference(
    #         path=walking_path,
    #         ckpt=ckpt,
    #         device=0,
    #         save_latent=True,
    #         save_real=False,
    #         baseline=True,
    #     )

    # for ckpt in d2_final_ckpts:
    #     run_inference(
    #         path=mmfi_path,
    #         ckpt=ckpt,
    #         device=0,
    #         save_latent=True,
    #         save_real=False,
    #     )
    for ckpt in d2_baseline_ckpts:
        run_inference(
            path=mmfi_path,
            ckpt=ckpt,
            device=0,
            save_latent=True,
            save_real=False,
            baseline=True,
        )

    with open(walking_path / "ts_bboxes.pkl", "rb") as f:
        d1_bboxes = pickle.load(f)

    with open(mmfi_path / "ts_bboxes.pkl", "rb") as f:
        d2_bboxes = pickle.load(f)

    d1_ref = [
        Image.open(i).copy()
        for i in tqdm(
            sorted(glob(str(walking_path / "reference/*.png"))),
        )
    ]
    d1_ref_cropped = crop_to_bboxes(d1_ref, d1_bboxes)
    d2_ref = [
        Image.open(i).copy()
        for i in tqdm(
            sorted(glob(str(mmfi_path / "reference/*.png"))),
        )
    ]
    d2_ref_cropped = crop_to_bboxes(d2_ref, d2_bboxes)

    out_stats = {
        "d1_final": {
            "ssim": [],
            "rmse": [],
            "fid": [],
            "ssim_crop": [],
            "rmse_crop": [],
            "fid_crop": [],
        },
        "d1_baseline": {
            "ssim": [],
            "rmse": [],
            "fid": [],
            "ssim_crop": [],
            "rmse_crop": [],
            "fid_crop": [],
        },
        "d2_final": {
            "ssim": [],
            "rmse": [],
            "fid": [],
            "ssim_crop": [],
            "rmse_crop": [],
            "fid_crop": [],
        },
        "d2_baseline": {
            "ssim": [],
            "rmse": [],
            "fid": [],
            "ssim_crop": [],
            "rmse_crop": [],
            "fid_crop": [],
        },
    }

    # for ckpt in d1_final_ckpts:
    #     p = [
    #         Image.open(i).copy()
    #         for i in sorted(
    #             glob(
    #                 str(walking_path / f"testset_inference_{ckpt}" / "*_l.png")
    #             )
    #         )
    #     ]
    #     stats = compute_stats(p, d1_ref)
    #     crop_stats = compute_stats(crop_to_bboxes(p, d1_bboxes), d1_ref_cropped)
    #     out_stats["d1_final"]["ssim"].append(stats["ssim"])
    #     out_stats["d1_final"]["rmse"].append(stats["rmse"])
    #     out_stats["d1_final"]["fid"].append(stats["fid"])

    #     out_stats["d1_final"]["ssim_crop"].append(crop_stats["ssim"])
    #     out_stats["d1_final"]["rmse_crop"].append(crop_stats["rmse"])
    #     out_stats["d1_final"]["fid_crop"].append(crop_stats["fid"])

    # for ckpt in d1_baseline_ckpts:
    #     p = [
    #         Image.open(i).copy()
    #         for i in sorted(
    #             glob(
    #                 str(walking_path / f"testset_inference_{ckpt}" / "*_l.png")
    #             )
    #         )
    #     ]
    #     stats = compute_stats(p, d1_ref)
    #     crop_stats = compute_stats(crop_to_bboxes(p, d1_bboxes), d1_ref_cropped)
    #     out_stats["d1_baseline"]["ssim"].append(stats["ssim"])
    #     out_stats["d1_baseline"]["rmse"].append(stats["rmse"])
    #     out_stats["d1_baseline"]["fid"].append(stats["fid"])

    #     out_stats["d1_baseline"]["ssim_crop"].append(crop_stats["ssim"])
    #     out_stats["d1_baseline"]["rmse_crop"].append(crop_stats["rmse"])
    #     out_stats["d1_baseline"]["fid_crop"].append(crop_stats["fid"])

    # for ckpt in d2_final_ckpts:
    #     p = [
    #         Image.open(i).copy()
    #         for i in sorted(
    #             glob(str(mmfi_path / f"testset_inference_{ckpt}" / "*_l.png"))
    #         )
    #     ]
    #     stats = compute_stats(p, d2_ref)
    #     crop_stats = compute_stats(crop_to_bboxes(p, d2_bboxes), d2_ref_cropped)
    #     out_stats["d2_final"]["ssim"].append(stats["ssim"])
    #     out_stats["d2_final"]["rmse"].append(stats["rmse"])
    #     out_stats["d2_final"]["fid"].append(stats["fid"])

    #     out_stats["d2_final"]["ssim_crop"].append(crop_stats["ssim"])
    #     out_stats["d2_final"]["rmse_crop"].append(crop_stats["rmse"])
    #     out_stats["d2_final"]["fid_crop"].append(crop_stats["fid"])

    for ckpt in d2_baseline_ckpts:
        p = [
            Image.open(i).copy()
            for i in sorted(
                glob(str(mmfi_path / f"testset_inference_{ckpt}" / "*_l.png"))
            )
        ]
        stats = compute_stats(p, d2_ref)
        crop_stats = compute_stats(
            crop_to_bboxes(p, d2_bboxes), d2_ref_cropped
        )
        out_stats["d2_baseline"]["ssim"].append(stats["ssim"])
        out_stats["d2_baseline"]["rmse"].append(stats["rmse"])
        out_stats["d2_baseline"]["fid"].append(stats["fid"])

        out_stats["d2_baseline"]["ssim_crop"].append(crop_stats["ssim"])
        out_stats["d2_baseline"]["rmse_crop"].append(crop_stats["rmse"])
        out_stats["d2_baseline"]["fid_crop"].append(crop_stats["fid"])

    print(out_stats)
    with open("/data/final_stats_d2_baseline.pkl", mode="wb+") as f:
        pickle.dump(out_stats, f)
