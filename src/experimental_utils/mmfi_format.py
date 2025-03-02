from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from PIL import Image
from src.targets.utils import preprocess_resize

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", required=True, type=Path)
parser.add_argument("-s", "--save-path", required=True, type=Path)
parser.add_argument(
    "-e", "--env",
    type=int,
    required=True
)
parser.add_argument(
    "-a",
    "--activities",
    type=int,
    nargs="+",
    help="List of activity IDs (integers).",
    required=True
)
args = parser.parse_args()
path = args.path
subjects = list((path / f"E{args.env:02d}").glob("*"))

images = np.zeros((len(subjects), len(args.activities), 259, 512, 512, 3), dtype=np.uint8)
csi = np.zeros((len(subjects), len(args.activities), 2590, 114), dtype=complex)

for sx, subject in enumerate(subjects):
    print(f"Subject {sx}")
    for ax, activity in tqdm(enumerate(args.activities)):
        root = (subject / f"A{activity:02d}")
        rgbs = (root / "rgb").glob("*")
        for px, photo_path in enumerate((root / "rgb").glob("*")):
            img = Image.open(photo_path)
            images[sx, ax, px] = np.asanyarray(preprocess_resize(img, 64))
        for cx, csi_path in enumerate((root / "wifi-csi").glob("*")):
            raw = loadmat(csi_path)
            amps, phase = raw["CSIamp"], raw["CSIphase"]
            csi[sx, ax, cx] = amps * np.exp(1j * phase)

images = np.repeat(images, 10, axis=2).reshape((-1, 512, 512, 3))
csi = csi.reshape((-1, 114))

from typing import cast

args.save_path.mkdir(exist_ok=True)
np.save(args.save_path / "photos.npy", images)
np.save(args.save_path / "csi.npy", csi)
