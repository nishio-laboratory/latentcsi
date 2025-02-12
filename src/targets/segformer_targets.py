# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "")

import sys
import gc
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import SegformerImageProcessor, SegformerModel
from pathlib import Path
import numpy as np
from PIL import Image
import os
from more_itertools import batched


def run_inference(rank, world_size, photos, data_path):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    feature_extractor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    model = SegformerModel.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    model.to(rank)

    print(f"RANK {rank} loaded model")

    chunk_size = len(photos) // world_size
    out = torch.empty(
        (chunk_size, 256, photos[0].height // 32, photos[0].width // 32)
    ).to("cpu")

    with torch.no_grad():
        for i in range(0, chunk_size):
            idx = i + rank * chunk_size
            inputs = feature_extractor(
                images=photos[idx], return_tensors="pt"
            ).to(rank)
            output = model(**inputs).last_hidden_state.squeeze()
            out[i] = output

    torch.save(out, data_path / "targets" / f"out_{rank}.pt")
    dist.destroy_process_group()


def preprocess_image(im: Image, left_offset=34):
    return im.resize((640, 512), resample=Image.Resampling.BICUBIC).crop(
        (left_offset, 0, 512 + left_offset, 512)
    )


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    parser = argparse.ArgumentParser(
        prog="Generate segmentation targets from photos"
    )
    parser.add_argument("-p", "--path", required=True)
    args = parser.parse_args()

    data_path = Path(args.path)
    photos = np.load(data_path / "photos.npy")
    photos = [preprocess_image(Image.fromarray(i)) for i in photos]

    (data_path / "targets").mkdir(exist_ok=True)

    mp.spawn(
        run_inference,
        args=(world_size, photos, data_path),
        nprocs=world_size,
        join=True,
    )

    del photos
    gc.collect()

    data = torch.concat(
        [
            torch.load(
                data_path / "targets" / f"out_{i}.pt", weights_only=False
            )
            for i in range(world_size)
        ]
    )
    torch.save(data, data_path / "targets" / "targets_seg.pt")

    for i in range(world_size):
        os.remove(data_path / "targets" / f"out_{i}.pt")
