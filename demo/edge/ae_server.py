import socketserver
from pathlib import Path
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import cast
import struct
from demo.sensor.send_to_edge import HEADER_FMT, HEADER_SIZE

np.bool = np.bool_


class TRTModel:
    """
    TensorRT model wrapper for dynamic-batch inference.
    """

    def __init__(self, engine_path: Path):
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        with engine_path.open("rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_idx = None
        self.output_idx = None
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                self.input_idx = idx
            else:
                self.output_idx = idx
        if self.input_idx is None or self.output_idx is None:
            raise RuntimeError("Failed to identify input/output bindings")

        self.in_dtype = trt.nptype(
            self.engine.get_binding_dtype(self.input_idx)
        )
        self.out_dtype = trt.nptype(
            self.engine.get_binding_dtype(self.output_idx)
        )

    def infer(self, imgs: np.ndarray) -> np.ndarray:
        # imgs: numpy array of shape (N, C, H, W) matching engine's expected layout
        if imgs.dtype != self.in_dtype:
            raise ValueError(
                f"Input array must be {self.in_dtype}, got {imgs.dtype}"
            )
        imgs = np.ascontiguousarray(imgs)

        self.context.set_binding_shape(self.input_idx, imgs.shape)

        d_input = cuda.mem_alloc(imgs.nbytes)
        cuda.memcpy_htod(d_input, imgs)

        out_shape = tuple(self.context.get_binding_shape(self.output_idx))
        output_size = (
            int(np.prod(out_shape)) * np.dtype(self.out_dtype).itemsize
        )
        d_output = cuda.mem_alloc(output_size)

        bindings = [int(d_input), int(d_output)]
        self.context.execute_v2(bindings)

        output = np.empty(out_shape, dtype=self.out_dtype)
        cuda.memcpy_dtoh(output, d_output)
        return output


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        f = self.rfile
        out = self.wfile
        encoder = cast(TRTModel, server.encoder)
        decoder = cast(TRTModel, server.decoder)
        while True:
            hdr = f.read(HEADER_SIZE)
            if len(hdr) < HEADER_SIZE:
                break
            msg_type, request_id, bs, size = struct.unpack(HEADER_FMT, hdr)
            data = f.read(size)
            if msg_type == 1:
                arr = (
                    np.frombuffer(data, dtype=np.uint8)
                    .reshape(bs, 3, 512, 512)
                    .copy()
                )
                out_arr = encoder.infer(arr.astype(np.float32) / 255.0)
            elif msg_type == 2:
                arr = np.frombuffer(data, dtype=np.float32).reshape(
                    bs, 4, 64, 64
                )
                out_arr = decoder.infer(arr)
            else:
                return
            payload = out_arr.astype(np.float32).tobytes()
            out.write(
                struct.pack("!bIII", msg_type, request_id, bs, len(payload))
                + payload
            )
            out.flush()


if __name__ == "__main__":
    host, port = "0.0.0.0", 8000
    encoder = TRTModel(Path("/trt/taesd_encoder_min1o4max8.trt"))
    # decoder = TRTModel(Path("./trt/taesd_decoder.trt"))
    with socketserver.TCPServer((host, port), Handler) as server:
        server.encoder, server.decoder = encoder, encoder
        print(f"Serving on {host}:{port}...")
        server.serve_forever()
