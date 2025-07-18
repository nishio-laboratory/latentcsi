from construct import Struct, Const, Int32ub, Float32l, Array
from abc import ABC, abstractmethod
import struct
from typing import ByteString, Optional
import torch
import asyncio

InferPacket = Struct(
    "hdr" / Const(b"infer"),
    "length" / Int32ub,
    "input" / Array(lambda ctx: ctx.length, Float32l),
)


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
                    input_length = struct.unpack("!I", await reader.readexactly(4))[0]
                    latent_length = struct.unpack("!I", await reader.readexactly(4))[0]

                    input_bytes = await reader.readexactly(input_length)
                    latent_bytes = await reader.readexactly(latent_length)

                    inp = torch.frombuffer(input_bytes, dtype=torch.float32).view(
                        1, self.model.get_input_dim()
                    )
                    latent = torch.frombuffer(
                        latent_bytes, dtype=torch.float32
                    ).view(1, 4, 64, 64)
                    self.train_received(inp, latent)
                    if self.queue.maxsize and self.queue.full():
                        _ = await self.queue.get()
                    await self.queue.put((inp, latent))
                elif header == b"infer":
                    length_bytes = await reader.readexactly(4)
                    length = struct.unpack("!I", length_bytes)[0]
                    payload = await reader.readexactly(4 * length)
                    packet = header + length_bytes + payload
                    parsed = InferPacket.parse(packet)
                    inp = torch.tensor(parsed.input, dtype=torch.float32).view(
                        1, self.model.get_input_dim()
                    )
                    self.model.eval()
                    with torch.no_grad():
                        out = self.model(inp.to(self.device))[0]
                    writer.write(out.cpu().numpy().tobytes())
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
