import os
from pathlib import Path
import argparse
import torch
from typing import cast
from PIL import Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from src.encoder.baseline import (
    CSIAutoencoderMLP_CNN as baseline_CSIAutoencoder,
)
import numpy as np
from src.encoder import training_cnn_att
from src.inference.utils import vae_decode
from src.inference.utils import load_test_dataset
from tqdm import tqdm


def main(
    path: Path,
    ckpt: str,
    device: int,
    save_latent: bool,
    save_real: bool,
    baseline: bool = False,
):
    if not ckpt.endswith(".ckpt"):
        ckpt += ".ckpt"

    if baseline:
        model = baseline_CSIAutoencoder
    else:
        model = training_cnn_att.CSIAutoencoderMLP_CNN
    model = model.load_from_checkpoint(path / "ckpts" / ckpt)

    model.eval()
    model = model.to(device)
    torch.set_grad_enabled(False)

    if save_latent and not baseline:
        sd = cast(
            StableDiffusionImg2ImgPipeline,
            StableDiffusionImg2ImgPipeline.from_pretrained(
                next(
                    i
                    for i in [
                        Path("~/sd-v1-5"),
                        path.parents[1] / "sd/sd-v1-5",
                    ]
                    if i.exists()
                )
            ),
        ).to(model.device)
        sd.safety_checker = None
    else:
        sd = None

    if save_real:
        photos = torch.load(path / "photos.pt", mmap=True)
    else:
        photos = None

    if baseline:
        test, test_indices = load_test_dataset(path, ["photos"])
    else:
        test, test_indices = load_test_dataset(path)

    inf_path = path / f"testset_inference_{os.path.basename(ckpt)}"
    inf_path.mkdir(exist_ok=True)

    if baseline:
        test_preds = torch.zeros(len(test), 512, 512, 3).to(device)
    else:
        test_preds = torch.zeros(len(test), 4, 64, 64).to(device)

    total_loss = 0
    for n, (i, data) in tqdm(
        enumerate(zip(test_indices, iter(test))), total=len(test_indices)
    ):
        if baseline:
            x, _, y = data
        else:
            x, y = data

        x = x.to(device).unsqueeze(0)
        y = y.to(device).unsqueeze(0)
        with torch.no_grad():
            p = model(x).to(device)
        if len(p.shape) == 2:
            p = p.unsqueeze(0)
        test_preds[n] = p.squeeze()
        total_loss += torch.nn.functional.mse_loss(p, y)

        if baseline:
            if save_latent:
                Image.fromarray(
                    np.asarray(p.squeeze().cpu(), dtype=np.uint8)
                ).save(inf_path / f"{n}_l.png")
        else:
            if save_latent and sd:
                vae_decode(sd, p).save(inf_path / f"{n}_l.png")

        if save_real and photos:
            Image.fromarray(np.asarray(photos[i])).save(
                inf_path / f"{n}_p.png"
            )

    print(total_loss / len(test_indices))
    torch.save(test_preds, inf_path / "all_preds.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path, required=True)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--save-latent", default=False, action="store_true")
    parser.add_argument("--save-real", default=False, action="store_true")
    parser.add_argument("--baseline", default=False, action="store_true")
    args = parser.parse_args()
    main(**vars(args))


# python -m src.inference.viz_test_split -p /data/datasets/mmfi_hands_two/ --ckpt 'mmfi_two_cnn_vaelike_attention_768_4stepmlp_cnn_val_loss=0.7639663219451904.ckpt' --model vaelike_4step_att
