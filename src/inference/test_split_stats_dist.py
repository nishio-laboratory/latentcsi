import os
from typing import cast
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.inference.utils import (
    generate,
    load_sd,
)  # other needed functions can be imported here
from src.targets.utils import preprocess_resize
from src.inference.utils import vae_decode  # if needed
from src.other.blur_face import blur_faces_opencv
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from torchmetrics.image.fid import FrechetInceptionDistance


def worker(rank, args, latent_arr, photos, indices, return_dict):
    sd = load_sd(args.path.parents[1], torch.device(f"cuda:{rank}"))
    sd.set_progress_bar_config(disable=True)
    torch.set_grad_enabled(False)

    pixel_error_sum = 0.0
    ssim_sum = 0.0
    fid_real_list = []
    fid_fake_list = []
    count = len(indices)

    for idx in tqdm(indices):
        latent = latent_arr[idx]
        ref = photos[idx]
        latent = latent.to(f"cuda:{rank}")
        if latent.shape[-1] == 3:
            generated_img = latent
        else:
            generated_img = vae_decode(sd, latent)
        img_tensor = torch.from_numpy(np.asarray(generated_img, copy=True))
        rmse = np.sqrt(
            mse(
                np.asarray(img_tensor, dtype=np.uint8),
                np.asarray(ref, dtype=np.uint8),
            )
        )
        pixel_error_sum += rmse

        ssim_value = ssim(
            np.asarray(img_tensor),
            np.asarray(ref),
            channel_axis=2,
            data_range=255,
        )
        ssim_value = cast(float, ssim_value)
        ssim_sum += ssim_value

        fid_real_list.append(ref.permute(2, 0, 1).unsqueeze(0))
        fid_fake_list.append(img_tensor.permute(2, 0, 1).unsqueeze(0))

    fid_real_tensor = torch.cat(fid_real_list, dim=0)
    fid_fake_tensor = torch.cat(fid_fake_list, dim=0)

    return_dict[rank] = (
        pixel_error_sum,
        ssim_sum,
        count,
        fid_real_tensor.cpu(),
        fid_fake_tensor.cpu(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--runs", type=int, required=False, default=1)
    args = parser.parse_args()

    # Prepare checkpoint filename.
    args.ckpt = os.path.basename(args.ckpt)
    if not args.out:
        args.out = Path(os.getcwd())
    else:
        args.out.mkdir(exist_ok=True)
    if not args.ckpt.endswith(".ckpt"):
        args.ckpt += ".ckpt"

    latent_arr = torch.load(
        args.path / f"testset_inference_{args.ckpt}" / "all_preds.pt",
        map_location="cpu",
    )
    photos_np = np.load(args.path / "photos_test_resized.npy").astype(np.uint8)
    photos = torch.Tensor(photos_np).type(torch.uint8)

    total_samples = len(latent_arr)
    num_processes = torch.cuda.device_count()
    chunks = [
        list(range(total_samples))[i::num_processes]
        for i in range(num_processes)
    ]

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for rank in range(num_processes):
        proc = mp.Process(
            target=worker,
            args=(rank, args, latent_arr, photos, chunks[rank], return_dict),
        )
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()

    # Aggregate results from all workers.
    total_pixel_error = 0.0
    total_ssim = 0.0
    total_count = 0
    fid_real_list = []
    fid_fake_list = []
    for key in return_dict.keys():
        pe_sum, ssim_sum, count, fid_real_tensor, fid_fake_tensor = (
            return_dict[key]
        )
        total_pixel_error += pe_sum
        total_ssim += ssim_sum
        total_count += count
        fid_real_list.append(fid_real_tensor)
        fid_fake_list.append(fid_fake_tensor)

    # Compute the average pixel error and SSIM.
    avg_pixel_error = total_pixel_error / (
        3 * total_samples
    )  # same normalization as before
    avg_ssim = total_ssim / total_samples

    # Compute the FID by updating a single metric instance with all images.
    fid_obj = FrechetInceptionDistance()
    fid_real_all = torch.cat(fid_real_list, dim=0)
    fid_fake_all = torch.cat(fid_fake_list, dim=0)
    fid_obj.update(fid_real_all, real=True)
    fid_obj.update(fid_fake_all, real=False)
    final_fid = fid_obj.compute()

    print("Average Pixel Error:", avg_pixel_error)
    print("Average SSIM:", avg_ssim)
    print("FID:", final_fid)


if __name__ == "__main__":
    # Using 'spawn' is generally more compatible for CUDA when using multiprocessing.
    mp.set_start_method("spawn", force=True)
    main()
