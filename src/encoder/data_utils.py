from typing import List, Optional, Any, Tuple, Literal
import torch
from pathlib import Path
import numpy as np

AuxDataTypes = Literal["latent_dists", "seg_lastlayer", "seg_map"]


def process_csi(csi: np.ndarray) -> torch.Tensor:
    inputs = torch.Tensor(np.abs(csi))
    std = inputs.std()
    inputs = inputs - inputs.mean()
    inputs = inputs / std
    return inputs


def load_data(
    path: Path,
    aux_data: List[AuxDataTypes] = [],
    clip: Optional[int] = None,
) -> List[Tuple[Any]]:
    filename_mapping = {
        "latent_dists": (path / "targets" / "targets_dists.pt"),
        "seg_lastlayer": (path / "targets" / "targets_seg.pt"),
        "seg_map": (path / "targets" / "targets_segmented.pt"),
    }
    if (path / "csi.npy").exists():
        csi = process_csi(np.load(path / "csi.npy"))
        if clip:
            csi = csi[:clip]
    else:
        raise Exception(f"CSI data (csi.npy) not found in {path}")

    if (path / "targets" / "targets_latents.pt").exists():
        targets = torch.load(
            path / "targets" / "targets_latents.pt", weights_only=True
        )
        if clip:
            targets = targets[:clip]
    else:
        raise Exception(
            f"Latent data (targets/targets_latents.pt) not found in {path}"
        )

    aux = []
    for i in aux_data:
        file_path = filename_mapping[i]
        if file_path.exists():
            if clip:
                aux.append(torch.load(file_path, weights_only=True)[:clip])
            else:
                aux.append(torch.load(file_path, weights_only=True))
        else:
            raise Exception(
                f"Aux data {i} requires file at {file_path}, not found."
            )

    return list(zip(csi, targets, *aux))
