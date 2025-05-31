import numpy as np
import pickle
from typing import List, Tuple

def format_row(s: dict) -> str:
    data = []
    for i in ["fid", "rmse", "ssim", "fid_crop", "rmse_crop", "ssim_crop"]:
        avg, std = float(np.mean(s[i])), float(np.std(s[i]))
        data.append(f"{avg:.2f}({std:.2f})")
    return "& " + "\n& ".join(data)

with open("/mnt/nas/esrh/csi_image_data/final_stats_merged.pkl", "rb") as f:
    stats = pickle.load(f)
    for i in stats:
        print(format_row(stats[i]))
        print("\n")
