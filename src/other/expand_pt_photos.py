import torch
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path)
    parser.add_argument("--np", type=str)
    parser.add_argument("--pt", type=str)
    args = parser.parse_args()

    if args.np:
        a = np.load(args.path / args.np, mmap_mode="r")
    if args.pt:
        a = torch.load(args.path / args.pt, mmap=True)
    else:
        raise Exception(
            "must specify one of --np or --pt with file name (basepath)"
        )

    (args.path / "reference").mkdir(exist_ok=True)
    for n, i in tqdm(enumerate(a)):
        Image.fromarray(np.asarray(i)).save(
            args.path / "reference" / f"{n}_p.png"
        )
