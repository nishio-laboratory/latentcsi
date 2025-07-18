import socket
import struct
from enum import IntEnum
import numpy as np

class MessageType(IntEnum):
    ENCODE = 1
    DECODE = 2


HEADER_FMT = "!bIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def _recv_all(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Expected {n} bytes, got {len(buf)} before EOF"
            )
        buf.extend(chunk)
    return bytes(buf)


def send_encode_request(
    sock: socket.socket,
    image: np.ndarray,
    request_id: int = 1,
) -> None:
    payload = image.astype(np.uint8).tobytes()
    header = struct.pack(
        HEADER_FMT, MessageType.ENCODE, request_id, image.shape[0], len(payload)
    )
    sock.sendall(header + payload)


def receive_encode_response(
    sock: socket.socket,
    request_id: int = 1,
    timeout: float = 5.0,
) -> np.ndarray:
    sock.settimeout(timeout)
    try:
        hdr = _recv_all(sock, HEADER_SIZE)
    except socket.timeout:
        raise TimeoutError(f"No response from server within {timeout} seconds")
    finally:
        sock.settimeout(None)
    msg_type, resp_id, bs, size = struct.unpack(HEADER_FMT, hdr)
    if resp_id != request_id:
        raise ValueError(
            f"Mismatched request ID: sent {request_id}, received {resp_id}"
        )
    if msg_type != MessageType.ENCODE:
        raise ValueError(f"Unexpected message type: {msg_type}")
    data = _recv_all(sock, size)
    arr = np.frombuffer(data, dtype="<f4")
    expected = bs * 4 * 64 * 64
    if arr.size != expected:
        raise ValueError(f"Expected {expected} floats, got {arr.size}")
    return arr.reshape(bs, 4, 64, 64)


def send_and_recv_encode(sock: socket.socket,
                         image: np.ndarray,
                         request_id: int = 1,
                         timeout: float = 5):
    send_encode_request(sock, image, request_id)
    return receive_encode_response(sock, request_id, timeout)
