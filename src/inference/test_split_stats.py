from torch import permute
from tqdm import tqdm
from src.inference.utils import *
import argparse
from pathlib import Path
import os
from src.targets.utils import preprocess_resize
from src.inference.utils import vae_decode, load_sd
from src.other.blur_face import blur_faces_opencv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from torchmetrics.image.fid import FrechetInceptionDistance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--runs", type=int, required=False, default=1)
    args = parser.parse_args()
    args.ckpt = os.path.basename(args.ckpt)

    if not args.out:
        args.out = Path(os.getcwd())
    else:
        args.out.mkdir(exist_ok=True)

    if not args.ckpt.endswith(".ckpt"):
        args.ckpt += ".ckpt"

    # sd = load_sd(args.path.parents[1], torch.device(0))
    # sd.set_progress_bar_config(disable=True)

    torch.set_grad_enabled(False)

    testset_path = args.path / f"testset_inference_{args.ckpt}"
    p = torch.load(testset_path / "all_preds.pt", mmap=True, map_location="cpu").to(torch.uint8)
    photos = torch.load(
        args.path / "photos_all_resized.pt", mmap = True, map_location="cpu"
    )

    fid_obj = FrechetInceptionDistance().to("cuda")
    pixel_error_sum = 0
    ssim_sum = 0

    if p[0].shape[-1] == 3:
        preds = p
    else:
        preds = torch.zeros((5, 512, 512, 3))

    for n, (latent, ref) in tqdm(enumerate(zip(p, photos)), total=len(p)):
        # img = generate(sd, latent, prompt="", strength=0.6)
        if latent.shape[-1] == 3:
            img = latent.to(torch.uint8)
        else:
            img = np.asarray(vae_decode(sd, latent))
            img = torch.from_numpy(np.array(img))
            preds[n] = img

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

    pixel_error = pixel_error_sum / (3 * len(p))
    ssim = ssim_sum / len(p)
    print(f"RMSE: {pixel_error}")
    print(f"SSIM: {ssim}")

    # fid_obj.update(permute_color_chan(photos.to("cuda")), real=True)
    # fid_obj.update(permute_color_chan(preds.to("cuda")), real=False)
    # print(f"FID: {fid_obj.compute()}")



# python -m src.figures.strengths --path /mnt/nas/esrh/csi_image_data/datasets/walking/ --ckpt walking_vaelike_512_4st_run2mlp_cnn_val_loss=5.283313751220703.ckpt
