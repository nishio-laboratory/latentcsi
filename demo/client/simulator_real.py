from pathlib import Path
import numpy as np
import torch
import socket
import time
from demo.server.protocol import Data
from itertools import batched, islice

path = Path("/mnt/nas/esrh/csi_image_data/datasets/realtime/test")
latents = torch.load(path / "data_lat.pt", mmap=True)
csi = torch.load(path / "data_csi.pt", mmap=True)

bps = 4
HOST, PORT = "192.168.1.221", 9999
with socket.create_connection((HOST, PORT)) as sock:
    for i in range(len(csi)):
        c = csi[i]
        l = latents[i]
        packet = Data.build(
            dict(
                input_shape={"dims": list(c.shape)},
                output_shape={"dims": list(l.shape)},
                batch_size=16,
                input_bytes=c.numpy().tobytes(),
                output_bytes=l.numpy().tobytes(),
            )
        )
        sock.sendall(b"train" + len(packet).to_bytes(4, "big") + packet)
        time.sleep(1 / bps)
