from pathlib import Path
import numpy as np
import torch
import socket
import time
from demo.server.protocol import Data
from itertools import batched, islice

path = Path("/mnt/nas/esrh/csi_image_data/datasets/walking")
csi = np.load(path / "csi.npy", mmap_mode="r")
latents = torch.load(path / "targets/targets_latents.pt", mmap=True)

# ***
bs = 16
csi = csi[: -(len(csi) % bs)]
latents = latents[: -(len(latents) % bs)]

HOST, PORT = "192.168.1.221", 9999
with socket.create_connection((HOST, PORT)) as sock:
    for csi, latents in zip(batched(csi, bs), batched(latents, bs)):
        packet = Data.build(
            dict(
                input_shape={"dims": [bs] + list(csi[0].shape)},
                output_shape={"dims": [bs] + list(latents[0].shape)},
                batch_size=bs,
                input_bytes=b"".join(
                    abs(inp).astype(np.float32).tobytes() for inp in csi
                ),
                output_bytes=b"".join(
                    lat.contiguous().view(-1).numpy().tobytes()
                    for lat in latents
                ),
            )
        )
        sock.sendall(b"train" + len(packet).to_bytes(4, "big") + packet)
        # time.sleep(0.001)
