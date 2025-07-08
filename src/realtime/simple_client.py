from src.realtime.server import TrainPacket
from pathlib import Path
import numpy as np
import torch
import socket

path = Path("/mnt/nas/esrh/csi_image_data/datasets/mmfi_hands_two")
csi = np.load(path / "csi.npy", mmap_mode="r")
photos = torch.load(path / "targets/targets_latents.pt", mmap=True)

# ***

HOST, PORT = "0.0.0.0", 9999
with socket.create_connection((HOST, PORT)) as sock:
    for inp, lat in zip(csi[:8], photos[:8]):
        packet = TrainPacket.build({
            "length": len(inp),
            "input": abs(inp).tolist(),
            "latent": lat.contiguous().view(-1).tolist()
        })
        sock.sendall(packet)
