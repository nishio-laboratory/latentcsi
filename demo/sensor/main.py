import argparse
from functools import partial
import socket
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Literal, Tuple, List

import numpy as np
import pyrealsense2 as rs
from PIL import Image

from demo.sensor.parse import parse
from demo.sensor.send_to_edge import (
    send_and_recv_encode,
)
from demo.sensor.util import Buffer


def make_socket(address: Tuple[str, int], tcp: bool = False) -> socket.socket:
    sock = socket.socket(
        socket.AF_INET, socket.SOCK_STREAM if tcp else socket.SOCK_DGRAM
    )
    if tcp:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1_048_576)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect(address)
    return sock


TRAIN_HEADER_FMT = "!5sIII"

FormatType = Literal["HESU", "HT", "NOHT", "VHT"]

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
    ):
        self.rx_cmd_string = f"feitcsi -f {frequency} -w 160 -r {format}"
        self.tx_cmd_string = f"feitcsi --mode inject -f {frequency} -w 160 -r {format} --inject-delay {delay}"
        self.tx_ip = tx_ip
        self.server_ip = server_ip
        self.running = False
        self.start_rx_process = start_rx_process
        self.samp_count = samp_count
        self.edge_addr = ("10.0.0.1", 8000)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.pending: List[Tuple[Future, bytes, int]] = []

    def _encode_rpc(self, images: np.ndarray, req_id: int) -> np.ndarray:
        return send_and_recv_encode(
            make_socket(self.edge_addr, tcp=True), images, req_id
        )

    def check_tx_running(self) -> bool:
        out = subprocess.run(
            f"ssh {self.tx_ip} \"ps auxw | grep '^root.*feitcsi -u'\"",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return out.returncode == 0

    def check_rx_running(self) -> bool:
        out = subprocess.run(
            "ps auxw | grep '^root.*feitcsi -u'",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return out.returncode == 0

    def init_camera(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.disable_stream(rs.stream.depth)
        prof = self.pipe.start(cfg)
        color_cam = prof.get_device().query_sensors()[1]
        color_cam.set_option(rs.option.enable_auto_white_balance, True)

    def get_photo(self, left_offset: int = 32) -> np.ndarray:
        im = np.asanyarray(
            self.pipe.wait_for_frames().get_color_frame().get_data()
        ).astype(np.uint8)
        im = Image.fromarray(im).resize(
            (640, 512), resample=Image.Resampling.BICUBIC
        )
        im = im.crop((left_offset, 0, 512 + left_offset, 512))
        arr = np.asarray(im, dtype=np.uint8)
        return arr.transpose(2, 0, 1)

    def get_csi(self) -> np.ndarray:
        raw = self.rx_socket.recv(272 + 4 * 1 * 1992)
        csi = parse(raw)[0]["csi_matrix"].flatten()
        return ((np.abs(csi) - 142.76) / 78.70).astype(np.float32)

    def send_batch(self, csis_bytes: bytes, batch_size: int, fut: Future):
        try:
            lat = fut.result()
        except Exception as e:
            print(f"Encode RPC failed: {e}")
            return
        lat_bytes = lat.tobytes()
        hdr = struct.pack(
            TRAIN_HEADER_FMT,
            b"train",
            len(csis_bytes),
            len(lat_bytes),
            batch_size,
        )
        self.server_socket.sendall(hdr + csis_bytes + lat_bytes)

    def start(self):
        self.running = True
        self.rx_socket = make_socket(("localhost", 8008))
        self.tx_socket = make_socket((self.tx_ip, 8008))
        self.server_socket = make_socket((self.server_ip, 9999), tcp=True)

        if self.start_rx_process:
            self.rx_process = subprocess.Popen(
                ["sudo", "feitcsi", "-u", "-v"],
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
            time.sleep(2)

        if not self.check_tx_running():
            raise RuntimeError("tx is not up... feitcsi -u not running")
        if not self.check_rx_running():
            raise RuntimeError("rx is not up... feitcsi -u not running")

        self.rx_socket.send(self.rx_cmd_string.encode())
        self.tx_socket.send(self.tx_cmd_string.encode())
        self.init_camera()

        buf_size = 8
        csi_smoothing_window = 4
        photo_buf = Buffer((buf_size, 3, 512, 512), np.uint8)
        csi_buf = Buffer((buf_size, 1992), np.float32)
        csi_smoothing_buf = np.zeros((csi_smoothing_window, 1992), np.float32)

        start_time = time.time()
        for i in range(self.samp_count):
            photo_buf.add(self.get_photo())
            for j in range(csi_smoothing_window):
                csi_smoothing_buf[j] = self.get_csi()
            csi_buf.add(np.mean(csi_smoothing_buf, axis=0))

            if photo_buf.full():
                imgs = photo_buf.buffer.copy()
                csis = csi_buf.buffer.copy().tobytes()
                fut = self.executor.submit(self._encode_rpc, imgs, i)
                fut.add_done_callback(
                    partial(self.send_batch, csis, buf_size)
                )
                photo_buf.clear()
                csi_buf.clear()

        print("done")
        self.executor.shutdown(wait=True)
        print(f"avg_time: {(time.time() - start_time) / self.samp_count}")

        self.stop()

    def stop(self):
        if not self.running:
            return
        self.executor.shutdown(wait=True)
        self.server_socket.close()
        if self.check_tx_running():
            self.tx_socket.send(b"stop")
        if self.check_rx_running():
            self.rx_socket.send(b"stop")
            if self.start_rx_process:
                self.rx_process.terminate()
        self.running = False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--frequency", type=int, default=5180)
    p.add_argument("--inject-delay", type=int, default=10000)
    p.add_argument("--samples", type=int, default=15000)
    p.add_argument("-r", "--format", type=str, default="HESU")
    p.add_argument("--start-rx-process", action="store_true")
    p.add_argument("-tx", "--tx-ip", type=str, default="192.168.2.5")
    p.add_argument("--server-ip", type=str, default="192.168.1.221")
    args = p.parse_args()

    dc = DatasetCollector(
        args.frequency,
        args.format,
        args.inject_delay,
        args.samples,
        tx_ip=args.tx_ip,
        server_ip=args.server_ip,
        start_rx_process=args.start_rx_process,
    )
    try:
        dc.start()
    finally:
        dc.stop()
