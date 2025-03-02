import time
import shutil
import gc
from tqdm import tqdm
from torch.multiprocessing.spawn import spawn
from functools import partial
from typing import Callable, Optional, Union
import torch
import argparse
from PIL import Image
from PIL.Image import Image as ImageType
from pathlib import Path
import numpy as np


def preprocess_resize(im: ImageType, left_offset=34) -> ImageType:
    return im.resize((640, 512), resample=Image.Resampling.BICUBIC).crop(
        (left_offset, 0, 512 + left_offset, 512)
    )


def tmp_file_path_formatter(
    timestamp: str, data_path: Union[str, Path], rank: int
):
    if isinstance(data_path, str):
        data_path = Path(data_path)
    return (
        data_path / "targets" / "dist_work" / f"dist_{timestamp}_gpu{rank}.pt"
    )


def _chunk_process(
    f: Callable[[torch.Tensor], torch.Tensor],
    data: list[torch.Tensor],
    rank: int,
    world_size: int,
    dims: list[int],
):
    chunk_size = len(data) // world_size
    out = torch.empty((chunk_size, *dims)).to("cpu")
    with torch.no_grad():
        for i in tqdm(range(0, chunk_size), total=chunk_size):
            idx = i + rank * chunk_size
            out[i] = f(data[idx])
    return out


def chunk_process(
    f: Callable[[torch.Tensor], torch.Tensor],
):
    return partial(_chunk_process, f)


def run_dist(
    inference_func: Callable,
    save_name: Union[str, Callable[[argparse.Namespace], str]],
    parser: Optional[argparse.ArgumentParser] = None,
    image_preprocessor: Callable[[ImageType], ImageType] = preprocess_resize,
):
    timestamp = time.strftime("%a_%X")
    world_size = torch.cuda.device_count()

    if not parser:
        parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", required=True, type=Path)
    args = parser.parse_args()

    (args.path / "targets").mkdir(exist_ok=True)
    (args.path / "targets" / "dist_work").mkdir(exist_ok=True)

    photos = np.load(args.path / "photos.npy")
    # photos = [image_preprocessor(Image.fromarray(i)) for i in photos]
    photos = [Image.fromarray(i) for i in photos]

    if photos[0].size[0] != 512:
        photos = [image_preprocessor(i) for i in photos]

    formatter = partial(tmp_file_path_formatter, timestamp)

    # TODO torch pr for mp spawn kwargs
    # https://github.com/pytorch/pytorch/issues/73902
    spawn(
        inference_func,
        args=(world_size, photos, formatter, args),
        nprocs=world_size,
        join=True,
    )
    del photos
    gc.collect()

    data = torch.concat(
        [
            torch.load(formatter(args.path, i), weights_only=False)
            for i in range(world_size)
        ]
    )

    if callable(save_name):
        save_name = save_name(args)
    torch.save(data, args.path / "targets" / (save_name + ".pt"))
    shutil.rmtree(args.path / "targets" / "dist_work")
