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

    sd = load_sd(args.path.parents[1], torch.device(0))
    sd.set_progress_bar_config(disable=True)

    torch.set_grad_enabled(False)

    testset_path = args.path / f"testset_inference_{args.ckpt}"
    p = torch.load(testset_path / "all_preds.pt", mmap=True)
    photos = np.load(
        args.path / "photos_test_resized.npy", mmap_mode="r"
    ).astype(np.uint8)
    photos = torch.Tensor(photos).type(torch.uint8)

    fid_obj = FrechetInceptionDistance()
    pixel_error_sum = 0
    ssim_sum = 0
    for n, (latent, ref) in tqdm(enumerate(zip(p, photos)), total=len(p)):
        # img = generate(sd, latent, prompt="", strength=0.6)
        img = np.asarray(vae_decode(sd, latent))
        img = torch.from_numpy(np.array(img))

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

        fid_obj.update(ref.permute(2, 0, 1).unsqueeze(0), real=True)
        fid_obj.update(img.permute(2, 0, 1).unsqueeze(0), real=False)
        # if n > 1:
        #     print(fid_obj.compute())

    pixel_error = pixel_error_sum / (3 * len(p))
    ssim = ssim_sum / len(p)
    print(pixel_error, ssim, fid_obj.compute())


# python -m src.figures.strengths --path /mnt/nas/esrh/csi_image_data/datasets/walking/ --ckpt walking_vaelike_512_4st_run2mlp_cnn_val_loss=5.283313751220703.ckpt
