from construct import Struct, Const, Int32ub, Float32l, Array
from abc import ABC, abstractmethod
import struct
from typing import ByteString, Optional
import torch
import asyncio

TRAIN_FMT = "!III"
TRAIN_SIZE = struct.calcsize(TRAIN_FMT)

class TrainableModule(torch.nn.Module, ABC):
    @abstractmethod
    def train_step(
        self, inp: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_input_dim(self) -> int:
        pass

    @abstractmethod
    def update_lr(self, lr: float):
        pass


class TrainingServerBase(ABC):
    def __init__(
        self,
        host: str,
        port: int,
        model: TrainableModule,
        batch_size: int = 8,
        max_queue_size: int = 0,  # 0 means unbounded
    ):
        self.host = host
        self.port = port
        self.model = model
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        # set maxsize if > 0
        self.queue: asyncio.Queue[tuple[torch.Tensor, torch.Tensor]] = (
            asyncio.Queue(maxsize=max_queue_size)
            if max_queue_size > 0
            else asyncio.Queue()
        )
        print(self.device)

    def train_received(self, inp, latent):
        pass

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        addr = writer.get_extra_info("peername")
        print(f"Client connected: {addr}")
        try:
            while True:
                header = await reader.readexactly(5)
                if header == b"train":
                    input_len, latent_len, bs = struct.unpack(
                        TRAIN_FMT, await reader.readexactly(TRAIN_SIZE)
                    )
                    input_bytes = await reader.readexactly(input_len)
                    latent_bytes = await reader.readexactly(latent_len)

                    inputs = torch.frombuffer(
                        input_bytes, dtype=torch.float32
                    ).view(bs, self.model.get_input_dim())
                    latents = torch.frombuffer(
                        latent_bytes, dtype=torch.float32
                    ).view(bs, 4, 64, 64)
                    for i in range(bs):
                        inp = inputs[i].unsqueeze(0)
                        lat = latents[i].unsqueeze(0)
                        if self.queue.maxsize and self.queue.full():
                            _ = await self.queue.get()
                        await self.queue.put((inp, lat))
                    self.train_received(inputs, latents)
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

    @abstractmethod
    async def dispatch(
        self, header: ByteString, reader: asyncio.StreamReader
    ) -> Optional[ByteString]:
        pass

    async def start(self):
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        print(f"TCP server listening on {self.host}:{self.port}")
        train_task = asyncio.create_task(self.train_worker())
        async with server:
            await server.serve_forever()
        train_task.cancel()
        await train_task

    @abstractmethod
    async def train_worker(self):
        pass
