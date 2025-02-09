# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "")

import sys
import gc
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import diffusers
from diffusers.image_processor import VaeImageProcessor
from pathlib import Path
import numpy as np
from PIL import Image
import os


def run_inference(rank, world_size, photos, data_path, distribution):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    vae = diffusers.AutoencoderKL().from_pretrained(
        "/data/sd/sd-v1-5",
        subfolder="vae",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    image_processor = VaeImageProcessor(vae_scale_factor=8)
    vae.to(rank)
    gen = torch.Generator(rank)
    print(f"RANK {rank} loaded model")

    chunk_size = len(photos) // world_size
    out = torch.empty(
        (chunk_size,
         8 if distribution else 4,
         photos[0].height // 8,
         photos[0].width // 8)
    ).to("cpu")

    with torch.no_grad():
        for i in range(0, chunk_size):
            idx = i + rank * chunk_size
            img = image_processor.preprocess(photos[idx]).to(
                device=rank, dtype=torch.half
            )
            if distribution:
                out[i] = vae._encode(img)
            else:
                out[i] = vae.encode(img).latent_dist.sample(gen)

    torch.save(out, data_path / "targets" / f"out_{rank}.pt")
    dist.destroy_process_group()

def preprocess_image(im: Image, left_offset = 34):
    return im.resize(
        (640, 512), resample=Image.Resampling.BICUBIC
    ).crop(
        (left_offset, 0, 512 + left_offset, 512)
    )

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    parser = argparse.ArgumentParser(
        prog="Generate latent targets from photos"
    )
    parser.add_argument(
        "-d", "--distribution",
        action="store_true",
        help="output vae distribution instead of sampled latents"
    )
    parser.add_argument("-p", "--path", required=True)
    args = parser.parse_args()

    data_path = Path(args.path)
    photos = np.load(data_path / "photos.npy")
    photos = [preprocess_image(Image.fromarray(i)) for i in photos]

    (data_path / "targets").mkdir(exist_ok=True)

    mp.spawn(
        run_inference,
        args=(world_size, photos, data_path, args.distribution),
        nprocs=world_size,
        join=True
    )

    del photos
    gc.collect()

    data = torch.concat(
        [torch.load(data_path / "targets" / f"out_{i}.pt", weights_only=False)
         for i in range(world_size)]
    )
    if args.distribution:
        torch.save(data, data_path / "targets" / "targets_dists.pt")
    else:
        torch.save(data, data_path / "targets" / "targets_latents.pt")

    for i in range(world_size):
        os.remove(data_path / "targets" / f"out_{i}.pt")
