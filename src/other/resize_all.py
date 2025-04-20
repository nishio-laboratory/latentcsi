import numpy as np
from tqdm import tqdm
from PIL import Image
from src.targets.utils import preprocess_resize
from src.inference.utils import load_test_dataset
import argparse
from pathlib import Path
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path)
    parser.add_argument("--test", action="store_true", required=False)
    args = parser.parse_args()

    photos = np.load(args.path / "photos.npy").astype(np.uint8)
    print("loaded photos")
    if args.test:

        test, test_indices = load_test_dataset(args.path)

        photos = photos[test_indices]

        images = [Image.fromarray(i) for i in photos]
        resized = [preprocess_resize(i) for i in images]
        arrays = [np.asarray(i) for i in resized]
        full = np.stack(arrays, axis=0)

        np.save((args.path / "photos_test_resized.npy"), full)
    else:
        for n, i in tqdm(enumerate(photos), total=len(photos)):
            photos[n] = np.asarray(preprocess_resize(Image.fromarray(i)))
        full = np.stack(photos, axis=0, dtype=np.uint8)
        del photos
        t = torch.from_numpy(full)
        torch.save(t, (args.path / "photos_all_resized.npy"))
