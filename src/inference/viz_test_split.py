from pathlib import Path
import argparse
import torch
from src.encoder.data_utils import CSIDataset
from typing import List, cast
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from PIL.Image import Image as PILImage
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
import numpy as np

from src.encoder import training_latents, training_cnn
from src.encoder.data_utils import process_csi
from src.inference.viz import decode
import math
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path, required=True)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--save", default=False, action="store_true")
    args = parser.parse_args()

    if args.model == "autoencoder":
        model = training_latents.CSIAutoencoder
        model = model.load_from_checkpoint(args.path / "ckpts" / args.ckpt)
    elif args.model == "mlp_cnn":
        model = training_cnn.CSIAutoencoderMLP_CNN
        model = model.load_from_checkpoint(args.path / "ckpts" / args.ckpt)
    else:
        raise Exception("model type not valid")

    model.eval()
    model = model.to(args.device)

    if args.save:
        sd = cast(
            StableDiffusionImg2ImgPipeline,
            StableDiffusionImg2ImgPipeline.from_pretrained(
                args.path.parents[1] / "sd/sd-v1-5"
            ),
        ).to(model.device)
        sd.safety_checker = None
    else:
        sd = None

    dataset = CSIDataset(args.path)
    _, _, test = torch.utils.data.random_split(
        dataset, [0.8, 0.1, 0.1], torch.Generator().manual_seed(42)
    )
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    test_indices = indices[- int(math.floor(len(dataset) * 0.1)):]

    photos = np.load(args.path / "photos.npy", mmap_mode="r")

    import os
    inf_path = (args.path / f"testset_inference_{os.path.basename(args.ckpt)}")
    inf_path.mkdir(exist_ok=True)

    test_preds = torch.zeros(len(test), 4, 64, 64).to(args.device)
    total_loss = 0
    for n, (i, (x, y)) in tqdm(enumerate(zip(test_indices, iter(test))), total=len(test_indices)):
        p = model(x.to(args.device)).to(args.device)
        test_preds[n] = p.unsqueeze(0)
        y = y.to(args.device)
        total_loss += torch.nn.functional.mse_loss(p, y)
        if args.save and sd:
            decode(sd, p.unsqueeze(0)).save(inf_path / f"l_{n}.png")
            Image.fromarray(photos[i]).save(inf_path / f"p_{n}.png")

    print(total_loss / len(test_indices))
    torch.save(test_preds, inf_path / "all_preds.pt")
# mmfi_hands_two: 11880 boundary
