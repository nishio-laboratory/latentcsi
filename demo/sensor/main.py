# (add-hook 'after-save-hook (lambda () (interactive) (call-process-shell-command "rsync -avP ~/prog/csi_to_image orin3-proxy:/home/esrh/ --exclude \".*\" --exclude \"env\"" nil 0)))
# (setq after-save-hook nil)

from typing import Literal, Tuple
import socket
import struct
import subprocess
import time
import numpy as np
import pyrealsense2 as rs
import argparse
from demo.sensor.parse import parse
from demo.sensor.query_inference import *
from PIL import Image


def preprocess_resize(im, left_offset=34):
    im = Image.fromarray(im)
    im = im.resize((640, 512), resample=Image.Resampling.BICUBIC).crop(
        (left_offset, 0, 512 + left_offset, 512)
    )
    im = np.asarray(im, dtype=np.uint8)
    im = im.transpose(2, 0, 1)[np.newaxis, ...]
    return im


def process_csi(csi):
    return ((np.abs(csi) - 142.76) / 78.70).astype(np.float32)


def make_socket(address: Tuple[str, int], tcp=False) -> socket.socket:
    if tcp:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, 1_048_576
        )
        sock.setsockopt(
            socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
        )
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(address)
    return sock


format_type = Literal["HESU", "HT", "NOHT", "VHT"]


class DatasetCollector:
    def __init__(
        self,
        frequency: int,
        format: format_type,
        delay: int,
        samp_count: int,
        tx_ip: str = "192.168.2.5",
        server_ip: str = "192.168.1.221",
        start_rx_process: bool = False,
    ):
        self.rx_cmd_string = f"feitcsi -f {frequency} -w 160 \
        -r {format}"
        self.tx_cmd_string = f"feitcsi --mode inject -f {frequency} \
        -w 160 -r {format} --inject-delay {delay}"
        self.tx_ip = tx_ip
        self.server_ip = server_ip
        self.running = False
        self.start_rx_process = start_rx_process
        self.samp_count = samp_count

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

    def get_photo(self) -> np.ndarray:
        photo = np.asanyarray(
            self.pipe.wait_for_frames().get_color_frame().get_data()
        ).astype(np.uint8)
        return preprocess_resize(photo)

    def get_csi(self) -> np.ndarray:
        csi = parse(self.rx_socket.recv(272 + 4 * 1 * 1992))[0][
            "csi_matrix"
        ].flatten()
        return process_csi(csi)


    def start(self):
        self.running = True

        self.rx_socket = make_socket(("localhost", 8008))
        self.tx_socket = make_socket((self.tx_ip, 8008))
        self.server_socket = make_socket((self.server_ip, 9999), True)
        self.inference_socket = make_socket(("10.0.0.1", 8000), True)

        if self.start_rx_process:
            self.rx_process = subprocess.Popen(
                ["sudo", "feitcsi", "-u", "-v"],
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
            if self.rx_process.poll() == 1:
                raise Exception(
                    f"couldn't start rx_process: {self.rx_process.stderr.read()}"
                )
            time.sleep(2)

        if not self.check_tx_running():
            raise Exception("tx is not up... feitcsi -u not running")
        if not self.check_rx_running():
            raise Exception("rx is not up... feitcsi -u not running")

        self.rx_socket.send(self.rx_cmd_string.encode())
        self.tx_socket.send(self.tx_cmd_string.encode())

        self.init_camera()

        total_times = []
        for i in range(self.samp_count):
            start = time.time()

            photo = self.get_photo()
            send_encode_request(self.inference_socket, photo, i)

            csi = self.get_csi()

            latent = receive_encode_response(self.inference_socket, i)

            input_bytes = csi.tobytes()
            latent_bytes = latent.tobytes()
            header = b"train" + struct.pack(
                "!II", len(input_bytes), len(latent_bytes)
            )
            self.server_socket.sendall(header + input_bytes + latent_bytes)

            total_times.append(time.time() - start)

        print(f"total {np.median(total_times)}")
        self.stop()

    def stop(self):
        if self.running:
            self.server_socket.close()
            self.inference_socket.close()

        self.running = False
        if self.check_tx_running():
            self.tx_socket.send("stop".encode())
        if self.check_rx_running():
            self.rx_socket.send("stop".encode())
            if self.start_rx_process:
                self.rx_process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--start-rx-process",
        action="store_true",
        help="automatically start feitcsi -u process on this machine",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        type=int,
        default=5180,
        help="transmission frequency channel",
    )
    parser.add_argument(
        "--inject-delay",
        type=int,
        default=10000,
        help="delay between packets in microseconds",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=15000,
        help="number of samples to collect",
    )
    parser.add_argument(
        "-r", "--format", type=str, default="HESU", help="packet type"
    )
    parser.add_argument(
        "-tx",
        "--tx-ip",
        type=str,
        default="192.168.2.5",
        help="TX machine ip address",
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        default="192.168.1.221",
        help="Training server ip",
    )
    args = parser.parse_args()

    dc = DatasetCollector(
        args.frequency,
        args.format,
        args.inject_delay,
        args.samples,
        start_rx_process=args.start_rx_process,
        tx_ip=args.tx_ip,
        server_ip=args.server_ip,
    )
    try:
        dc.start()
    except Exception as e:
        dc.stop()
        raise e
