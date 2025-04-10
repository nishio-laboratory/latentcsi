import numpy as np
from PIL import Image
from src.targets.utils import preprocess_resize
from src.inference.utils import load_test_dataset
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path)
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()

    photos = np.load(args.path / "photos.npy")
    print("loaded photos")
    testset_path = args.path / f"testset_inference_{args.ckpt}"
    test, test_indices = load_test_dataset(args.path)

    photos = photos[test_indices]

    images = [Image.fromarray(i) for i in photos]
    resized = [preprocess_resize(i) for i in images]
    arrays = [np.asarray(i) for i in resized]
    full = np.stack(arrays, axis=0)

    np.save((args.path / "photos_test_resized.npy"), full)
