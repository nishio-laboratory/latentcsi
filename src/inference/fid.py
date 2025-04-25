import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from tqdm import tqdm
from typing import List


class PILImageDataset(Dataset):
    def __init__(self, images: List[Image.Image], transform=None):
        self.images = images
        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),  # Converts to float32 in [0,1]
                transforms.Lambda(
                    lambda x: (x * 255).to(torch.uint8)
                ),  # Rescale and convert
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx])


def compute_fid_inception(
    images1: List[Image.Image],
    images2: List[Image.Image],
    device="cpu",
    batch_size=32,
) -> float:
    """
    Computes FID between two sets of images using InceptionV3 and torchmetrics.

    Args:
        images1: List of PIL.Image (e.g., real images)
        images2: List of PIL.Image (e.g., generated images)
        device: 'cuda' or 'cpu'
        batch_size: Batch size for processing

    Returns:
        FID score as a float
    """
    # Create datasets and dataloaders
    dataset1 = PILImageDataset(images1)
    dataset2 = PILImageDataset(images2)

    loader1 = DataLoader(
        dataset1, batch_size=batch_size, shuffle=False, num_workers=2
    )
    loader2 = DataLoader(
        dataset2, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Initialize FID metric with InceptionV3 (default)
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Process real images
    for batch in tqdm(loader1, desc="real images FID"):
        fid.update(batch.to(device), real=True)

    # Process generated images
    for batch in tqdm(loader2, desc="fake images FID"):
        fid.update(batch.to(device), real=False)

    return float(fid.compute())
