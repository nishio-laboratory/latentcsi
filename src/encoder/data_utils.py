from typing import List, Optional, Any, Tuple, Literal
import torch
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

AuxDataTypes = Literal["latent_dists", "seg_lastlayer", "seg_map", "photos"]


def process_csi(csi: np.ndarray) -> torch.Tensor:
    inputs = torch.Tensor(np.abs(csi))
    std = inputs.std()
    inputs = inputs - inputs.mean()
    inputs = inputs / std
    return inputs


class CSIDataset(Dataset):
    def __init__(
        self,
        path: Path,
        aux_data: List[AuxDataTypes] = [],
    ):
        filename_mapping = {
            "latent_dists": (path / "targets" / "targets_dists.pt"),
            "seg_lastlayer": (path / "targets" / "targets_seg.pt"),
            "seg_map": (path / "targets" / "targets_segmented.pt"),
            "photos": (path / "photos.pt"),
        }
        if (path / "csi.npy").exists():
            self.csi = process_csi(np.load(path / "csi.npy"))
        else:
            raise Exception(f"CSI data (csi.npy) not found in {path}")

        if (path / "targets" / "targets_latents.pt").exists():
            self.targets = torch.load(
                path / "targets" / "targets_latents.pt",
                weights_only=True,
                mmap=True,
            )
        else:
            raise Exception(
                f"Latent data (targets/targets_latents.pt) not found in {path}"
            )

        self.aux = []
        for i in aux_data:
            file_path = filename_mapping[i]
            if file_path.exists():
                if str(file_path).endswith(".pt"):
                    self.aux.append(
                        torch.load(file_path, weights_only=True, mmap=True)
                    )
                elif str(file_path).endswith(".npy"):
                    self.aux.append(
                        torch.load(file_path, weights_only=True, mmap=True)
                    )
            else:
                raise Exception(
                    f"Aux data {i} requires file at {file_path}, not found."
                )

    def __len__(self):
        return len(self.csi)

    def __getitem__(self, index):
        out = [self.csi[index], self.targets[index]]
        for i in self.aux:
            out.append(i[index])
        return tuple(out)
