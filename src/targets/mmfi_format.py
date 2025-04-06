from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from PIL import Image
from src.targets.utils import preprocess_resize

# python -m src.experimental_utils.mmfi_format -p /mmfi/MMFi_rgb_wifi/ -s /data/datasets/mmfi_hands_all -e 3 -a 13 14 17 18
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", required=True, type=Path)
parser.add_argument("-s", "--save-path", required=True, type=Path)
parser.add_argument("--subjects-trunc", type=int, default=-1)
parser.add_argument("-e", "--env", type=int, required=True)
parser.add_argument(
    "-a",
    "--activities",
    type=int,
    nargs="+",
    help="List of activity IDs (integers).",
    required=True,
)

args = parser.parse_args()
path = args.path
subjects = list((path / f"E{args.env:02d}").glob("*"))
if args.subjects_trunc != -1:
    subjects = subjects[: args.subjects_trunc]

images = np.zeros(
    (len(subjects), len(args.activities), 297, 512, 512, 3), dtype=np.uint8
)
csi = np.zeros(
    (len(subjects), len(args.activities), 2970, 3 * 114), dtype=complex
)

np.seterr(all="raise")
for sx, subject in enumerate(subjects):
    print(f"Subject {sx}")
    for ax, activity in tqdm(
        enumerate(args.activities), total=len(args.activities)
    ):
        root = subject / f"A{activity:02d}"
        rgbs = (root / "rgb").glob("*")
        for px, photo_path in enumerate((root / "rgb").glob("*")):
            img = Image.open(photo_path)
            images[sx, ax, px] = np.asanyarray(preprocess_resize(img, 64))
        for cx, csi_path in enumerate((root / "wifi-csi").glob("*")):
            raw = loadmat(csi_path)
            amps, phase = raw["CSIamp"], raw["CSIphase"]
            amps[amps == -np.inf] = 0
            csi_frame = amps * np.exp(1j * phase)  # 3x114x10
            csi_frame = csi_frame.reshape(
                (-1, 10)
            ).transpose()  # 342x10 -> 10x342
            csi[sx, ax, 10 * cx : 10 * cx + 10] = csi_frame

images = np.repeat(images, 10, axis=2).reshape((-1, 512, 512, 3))
csi = csi.reshape((-1, 342))

args.save_path.mkdir(exist_ok=True)
np.save(args.save_path / "photos.npy", images)
np.save(args.save_path / "csi.npy", csi)
