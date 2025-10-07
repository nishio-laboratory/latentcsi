import socket
import struct
import numpy as np
from pathlib import Path
import threading
import io
from PIL import Image

def main():
    path = Path("/mnt/nas/esrh/csi_image_data/datasets/walking_test")
    csi = np.load(path / "csi.npy")
    photos = np.load(path / "photos.npy", mmap_mode="r")

    buf = io.BytesIO()
    Image.fromarray(photos[-1]).save(buf, format="JPEG")
    sample = (
        csi[-1],
        buf.getvalue()
    )

    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serv.bind(("localhost", 10000))
    serv.listen()
    serv.settimeout(1.0)

    print("Ready!")
    while True:
        try:
            conn, _ = serv.accept()
        except socket.timeout:
            continue
        threading.Thread(target=handle, args=(conn, sample), daemon=True).start()


def handle(conn: socket.socket, sample: tuple[np.ndarray, bytes]):
    conn.settimeout(1)
    print("Connection made!")
    with conn:
        while True:
            try:
                hdr = conn.recv(3)
            except socket.timeout:
                continue
            if hdr == b"jpg":
                print("Received request")
                csi_bytes = sample[0].astype(np.float32).tobytes()
                conn.sendall(
                    struct.pack("!II", len(sample[1]), len(csi_bytes)) +
                    sample[1] + csi_bytes
                )

if __name__ == "__main__":
    main()
