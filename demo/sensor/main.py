import argparse
from functools import partial
import socket
import sys
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Literal, Tuple
import threading
import queue
from collections import deque
import io

import numpy as np
import pyrealsense2 as rs
from PIL import Image

from demo.sensor.parse import parse
from demo.sensor.send_to_edge import send_and_recv_encode
from demo.sensor.util import Buffer

FormatType = Literal["HESU", "HT", "NOHT", "VHT"]
TRAIN_HEADER_FMT = "!5sIII"


def make_socket(address: Tuple[str, int], tcp: bool = False) -> socket.socket:
    sock = socket.socket(
        socket.AF_INET, socket.SOCK_STREAM if tcp else socket.SOCK_DGRAM
    )
    if tcp:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1_048_576)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect(address)
    return sock


class DatasetCollector:
    def __init__(
        self,
        frequency: int,
        format: FormatType,
        delay: int,
        samp_count: int,
        tx_ip: str = "192.168.2.5",
        server_ip: str = "192.168.1.221",
        start_rx_process: bool = False,
        serve_port: int = 9000
    ):
        self.rx_cmd_string = f"feitcsi -f {frequency} -w 160 -r {format}"
        self.tx_cmd_string = (
            f"feitcsi --mode inject -f {frequency} -w 160 -r {format}"
            f" --inject-delay {delay}"
        )
        self.tx_ip = tx_ip
        self.server_ip = server_ip
        self.samp_count = samp_count
        self.start_rx_process = start_rx_process
        self.serve_port = serve_port
        self.edge_addr = ("10.0.0.1", 8000)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.send_queue: queue.Queue[Tuple[bytes, bytes, int]] = queue.Queue()
        self.latest_lock = threading.Lock()
        self.latest_photo: np.ndarray | None = None
        self.latest_csi: np.ndarray | None = None
        self.server_listen_addr = ("0.0.0.0", 10000)

    def _encode(self, images: np.ndarray, req_id: int) -> np.ndarray:
        sock = make_socket(self.edge_addr, tcp=True)
        latents = send_and_recv_encode(sock, images, req_id)
        sock.close()
        return latents

    def send_batch(
        self, csis_bytes: bytes, batch_size: int, t: float, fut: Future
    ) -> None:
        self.send_queue.put((csis_bytes, fut.result().tobytes(), batch_size))

    def _sender(self) -> None:
        while True:
            csis_bytes, lat_bytes, batch_size = self.send_queue.get()
            hdr = struct.pack(
                TRAIN_HEADER_FMT,
                b"train",
                len(csis_bytes),
                len(lat_bytes),
                batch_size,
            self.server_socket.sendmsg([hdr, csis_bytes, lat_bytes])
            self.send_queue.task_done()

    def _recv_all(self, conn: socket.socket, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                break
            data += chunk
        return data

    def _server(self) -> None:
        serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        serv.bind(self.server_listen_addr)
        serv.listen()
        serv.settimeout(1.0)
        while getattr(self, "running", False):
            try:
                conn, _ = serv.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self._handle_client,
                             args=(conn,), daemon=True).start()

    def _handle_client(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        with conn:
            while getattr(self, "running", False):
                try:
                    hdr = self._recv_all(conn, 3)
                except socket.timeout:
                    continue
                if not hdr or hdr not in (b"jpg", b"raw", b"csi"):
                    break
                with self.latest_lock:
                    photo = self.latest_photo
                    csi = self.latest_csi
                if photo is None or csi is None:
                    continue
                csi_bytes = csi.tobytes()
                if hdr == b"jpg":
                    img = photo.transpose(1, 2, 0)
                    buf = io.BytesIO()
                    Image.fromarray(img).save(buf, format="JPEG")
                    img_bytes = buf.getvalue()
                    lengths = struct.pack("!II", len(img_bytes),
                                          len(csi_bytes))
                    conn.sendall(lengths + img_bytes + csi_bytes)
                elif hdr == b"raw":
                    img_bytes = photo.tobytes()
                    lengths = struct.pack("!II", len(img_bytes),
                                          len(csi_bytes))
                    conn.sendall(lengths + img_bytes + csi_bytes)
                else:
                    lengths = struct.pack("!I", len(csi_bytes))
                    conn.sendall(lengths + csi_bytes)

    def init_camera(self) -> None:
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.disable_stream(rs.stream.depth)
        self.frame_queue = rs.frame_queue(8)
        prof = self.pipe.start(cfg, self.frame_queue)
        color_cam = prof.get_device().query_sensors()[1]
        color_cam.set_option(rs.option.enable_auto_white_balance, True)

    def get_photo(self, left_offset: int = 32) -> np.ndarray:
        data = self.frame_queue.wait_for_frame().get_data()
        im = Image.frombuffer(
            "RGB", (640, 480), data, "raw", "RGB", 0, 1
        ).resize((640, 512), Image.Resampling.BICUBIC)
        im = im.crop((left_offset, 0, 512 + left_offset, 512))
        arr = np.asarray(im, dtype=np.uint8)
        return arr.transpose(2, 0, 1)

    def get_csi(self) -> np.ndarray:
        latest = None
        while True:
            try:
                latest = self.rx_socket.recv(272 + 4 * 1 * 1992)
            except BlockingIOError:
                break
        if latest is None:
            self.rx_socket.setblocking(True)
            print("no csi data... blocking!", file=sys.stderr, flush=True)
            latest = self.rx_socket.recv(272 + 4 * 1 * 1992)
            self.rx_socket.setblocking(False)
        csi = parse(latest)[0]["csi_matrix"].flatten()
        return ((np.abs(csi) - 142.76) / 78.70).astype(np.float32)

    def start(self) -> None:
        self.running = True
        self.rx_socket = make_socket(("localhost", 8008))
        self.rx_socket.setblocking(False)
        self.tx_socket = make_socket((self.tx_ip, 8008))
        self.server_socket = make_socket((self.server_ip, 9999), tcp=True)
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._server, daemon=True).start()

        if self.start_rx_process:
            self.rx_process = subprocess.Popen(
                ["sudo", "feitcsi", "-u", "-v"],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            time.sleep(2)

        if (
            subprocess.run(
                f"ssh {self.tx_ip} \"ps auxw | grep '^root.*feitcsi -u'\"",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            != 0
        ):
            raise RuntimeError("tx is not up... feitcsi -u not running")

        if (
            subprocess.run(
                "ps auxw | grep '^root.*feitcsi -u'",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            != 0
        ):
            raise RuntimeError("rx is not up... feitcsi -u not running")

        self.rx_socket.send(self.rx_cmd_string.encode())
        self.tx_socket.send(self.tx_cmd_string.encode())
        self.init_camera()

        buf_size = 8
        win_size = 4
        photo_buf = Buffer((buf_size, 3, 512, 512), np.uint8)
        csi_buf = Buffer((buf_size, 1992), np.float32)
        csi_window = deque(maxlen=win_size)

        start_time = time.time()
        try:
            for idx in range(self.samp_count):
                photo = self.get_photo()
                csi = self.get_csi()
                with self.latest_lock:
                    self.latest_photo = photo
                    self.latest_csi = csi
                photo_buf.add(photo)
                csi_window.append(csi)
                if len(csi_window) == win_size:
                    csi_buf.add(
                        np.mean(np.stack(csi_window), axis=0)
                    )
                if photo_buf.full() and len(csi_window) == win_size:
                    imgs = photo_buf.buffer.copy()
                    csis = csi_buf.buffer.copy().tobytes()
                    fut = self.executor.submit(self._encode, imgs, idx)
                    fut.add_done_callback(
                        partial(self.send_batch, csis, buf_size, time.time())
                    )
                    photo_buf.clear()
                    csi_buf.clear()
        except KeyboardInterrupt:
            print("Stopping...", file=sys.stderr)
        finally:
            self.executor.shutdown(wait=True)
            self.send_queue.join()
            elapsed = time.time() - start_time
            print(
                f"avg_time: {elapsed / self.samp_count} s",
                file=sys.stderr,
                flush=True,
            )
            self.stop()

    def stop(self) -> None:
        if not getattr(self, "running", False):
            return
        self.running = False
        self.server_socket.close()
        self.pipe.stop()
        try:
            self.tx_socket.send(b"stop")
        except Exception:
            pass
        try:
            self.rx_socket.send(b"stop")
            if self.start_rx_process:
                self.rx_process.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--frequency", type=int, default=5180)
    parser.add_argument("--inject-delay", type=int, default=10000)
    parser.add_argument("--samples", type=int, default=15000)
    parser.add_argument("-r", "--format", type=str, default="HESU")
    parser.add_argument("--start-rx-process", action="store_true")
    parser.add_argument("-tx", "--tx-ip", type=str, default="192.168.2.5")
    parser.add_argument("--server-ip", type=str, default="192.168.1.221")
    parser.add_argument("--port", type=str, default="8000")
    args = parser.parse_args()

    dc = DatasetCollector(
        args.frequency,
        args.format,
        args.inject_delay,
        args.samples,
        tx_ip=args.tx_ip,
        server_ip=args.server_ip,
        start_rx_process=args.start_rx_process,
        serve_port=args.port
    )
    dc.start()
