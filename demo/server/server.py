from __future__ import annotations
from typing import ByteString, Optional
import struct
from diffusers import AutoencoderTiny, StableDiffusionImg2ImgPipeline
import numpy as np
import torch
import os
import asyncio
import torch.multiprocessing as mp
from PIL import Image
from demo.client.webserver.models import Img2ImgParams
from demo.server.decoder import apply_sd, apply_sd_lat, decode_latent_to_image
from demo.server.protocol import Data, InferLastReq
from demo.server.trainers.basic import TrainerLastReplay
from demo.server.trainers.signal_stop import TrainerStoppable
from demo.server.trainer_base import TrainerBase, LockedTensor, TrainerState
from src.inference.utils import pil_image_to_bytes
from src.other.types import *


class TrainingServerBase:
    def __init__(self, host: str, port: int, trainer: type[TrainerBase]):
        self.host = host
        self.port = port
        self.dtype = torch.float32

        self.ctx = mp.get_context("spawn")
        self.data_queue: mp.Queue[Batch] = self.ctx.Queue(maxsize=100)
        self.message_queue: mp.Queue[Message] = self.ctx.Queue(maxsize=1000)

        # self.status_queue: mp.Queue[TrainerState]  = self.ctx.Queue(maxsize=1)
        self.shared_manager = self.ctx.Manager()
        self.shared = self.shared_manager.Namespace()
        self.shared.state = TrainerState()

        self.out_tensor: LockedTensor = LockedTensor(
            torch.zeros(1, 4, 64, 64), self.ctx.Lock()
        )
        self.real_tensor = TrueLatent(torch.zeros(1, 4, 64, 64))

        self.ae_tiny = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
            torch.device("cuda:0"),
            self.dtype
        )
        self.sd = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5"
        ).to(
            torch.device("cuda:0"),
            self.dtype
        )
        self.sd.vae = self.ae_tiny
        self.trainer_class = trainer

    async def train(self, reader, writer):
        req_len = struct.unpack("!I", await reader.readexactly(4))[0]
        packet = Data.parse(await reader.readexactly(req_len))
        inputs = BatchCSI(
            torch.frombuffer(
                bytearray(packet.input_bytes), dtype=torch.float32
            ).clone().view(packet.input_shape.dims).cpu().to(self.dtype)
        )

        outputs = BatchTrueLatent(
            torch.frombuffer(
                bytearray(packet.output_bytes), dtype=torch.float32
            ).clone().view(packet.output_shape.dims).cpu().to(self.dtype)
        )
        self.real_tensor = TrueLatent(outputs[-1].unsqueeze(0))
        if self.data_queue.full():
            self.data_queue.get()
        self.data_queue.put_nowait(Batch(inputs, outputs))

    async def infer_last(self, reader, writer):
        req_len = struct.unpack("!I", await reader.readexactly(4))[0]
        req = InferLastReq.parse(await reader.readexactly(req_len))
        if self.shared.state.batches_trained == 0:
            img = Image.new("RGB", (512, 512), (0, 0, 255))
        else:
            latent_tensor = PredLatent(
                (await asyncio.to_thread(self.out_tensor.get_copy)).to(torch.device("cuda:0"), self.dtype)
            )
            if req.apply_sd and req.sd_params.prompt != "":
                img = apply_sd(
                    latent_tensor,
                    self.sd,
                    Img2ImgParams.from_construct(req.sd_params),
                )
            else:
                img = decode_latent_to_image(latent_tensor, self.ae_tiny)
        img_bytes = pil_image_to_bytes(img)
        writer.write(len(img_bytes).to_bytes(4, "big") + img_bytes)
        await writer.drain()

    async def real_latent(self, reader, writer):
        req_len = struct.unpack("!I", await reader.readexactly(4))[0]
        req = InferLastReq.parse(await reader.readexactly(req_len))
        if self.shared.state.batches_trained == 0:
            img = Image.new("RGB", (512, 512), (0, 0, 255))
        else:
            img= decode_latent_to_image(self.real_tensor, self.ae_tiny)
        img_bytes = pil_image_to_bytes(img)
        writer.write(len(img_bytes).to_bytes(4, "big") + img_bytes)
        await writer.drain()

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        addr = writer.get_extra_info("peername")
        print(f"Client connected: {addr}")
        try:
            while True:
                header = await reader.readexactly(5)
                if header == b"train":
                    await self.train(reader, writer)
                elif header == b"itrai":
                    await self.infer_last(reader, writer)
                elif header == b"truth":
                    await self.real_latent(reader, writer)
                elif header == b"messa":
                    req_len = struct.unpack("!I", await reader.readexactly(4))[
                        0
                    ]
                    msg_str = (await reader.readexactly(req_len)).decode(
                        "utf-8"
                    )
                    print(f"message recvd: {msg_str}")
                    self.message_queue.put_nowait(check_msg(msg_str))
                elif header == b"state":
                    out_bytes = self.shared.state.construct()
                    writer.write(len(out_bytes).to_bytes(4, "big") + out_bytes)
                    await writer.drain()
                else:
                    out = await self.dispatch(header, reader)
                    if out:
                        writer.write(out)
                        await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            print(f"Client disconnected: {addr}")

    async def dispatch(
        self, header: ByteString, reader: asyncio.StreamReader
    ) -> Optional[ByteString]:
        pass

    async def start(self):
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        print(f"TCP server listening on {self.host}:{self.port}")

        train_process = self.ctx.Process(
            target=self.trainer_class,
            args=(
                self.data_queue,
                self.message_queue,
                self.shared,
                self.out_tensor,
                torch.device("cuda:0"),
                self.dtype
            ),
        )
        train_process.start()

        async with server:
            await server.serve_forever()


async def main():
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    srv = TrainingServerBase(host=host, port=9000, trainer=TrainerStoppable)
    await srv.start()


if __name__ == "__main__":
    asyncio.run(main())
