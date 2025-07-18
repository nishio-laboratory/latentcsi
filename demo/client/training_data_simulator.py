from pathlib import Path
import numpy as np
import torch
import socket
import struct

path = Path("/mnt/nas/esrh/csi_image_data/datasets/walking")
csi = np.load(path / "csi.npy", mmap_mode="r")
latents = torch.load(path / "targets/targets_latents.pt", mmap=True)

# ***

HOST, PORT = "192.168.1.221", 9999
for i in range(15):
    with socket.create_connection((HOST, PORT)) as sock:
        for inp, lat in zip(csi, latents):
            input_bytes = abs(inp).astype(np.float32).tobytes()
            latent_bytes = lat.contiguous().view(-1).numpy().tobytes()
            header = b"train" + (struct.pack("!I", len(input_bytes))) + (struct.pack("!I", len(latent_bytes)))
            sock.sendall(header + input_bytes + latent_bytes)
