from abc import ABC, abstractmethod
import asyncio
import struct
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from construct import Struct, Const, Int32ub, Float32l, Array

# --- Packet definitions using Construct ---
TrainPacket = Struct(
    "hdr"    / Const(b"TRAIN"),
    "length" / Int32ub,
    "input"  / Array(lambda ctx: ctx.length, Float32l),
    "latent" / Array(4 * 64 * 64, Float32l),
)

InferPacket = Struct(
    "hdr"    / Const(b"INFER"),
    "length" / Int32ub,
    "input"  / Array(lambda ctx: ctx.length, Float32l),
)

class SimpleNet(nn.Module):
    def __init__(self, input_dim: int, lr: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 4 * 64 * 64)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).view(x.size(0), 4, 64, 64)

    def training_step(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.train()
        self.optimizer.zero_grad()
        pred = self.forward(inp)
        loss = F.mse_loss(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss

class ReservoirBuffer:
    def __init__(self, buffer_size: int, replace_rate: float = 1.0):
        self.buffer_size = buffer_size
        self.replace_rate = replace_rate
        self.reservoir: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.count = 0

    def add(self, sample: tuple[torch.Tensor, torch.Tensor]):
        self.count += 1
        if len(self.reservoir) < self.buffer_size:
            self.reservoir.append(sample)
        else:
            if random.random() <= self.replace_rate:
                j = random.randrange(self.count)
                if j < self.buffer_size:
                    self.reservoir[j] = sample

    def sample_batch(self, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        k = min(batch_size, len(self.reservoir))
        return random.sample(self.reservoir, k)

class TrainingServerBase(ABC):
    def __init__(
        self,
        host: str,
        port: int,
        model: SimpleNet,
        batch_size: int = 8,
        buffer_size: int = 1000,
        replace_rate: float = 1.0,
        replay_fraction: float = 0.5,
        max_queue_size: int = 0,  # 0 means unbounded
    ):
        self.host = host
        self.port = port
        self.model = model
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        # set maxsize if > 0
        self.queue: asyncio.Queue[tuple[torch.Tensor, torch.Tensor]] = (
            asyncio.Queue(maxsize=max_queue_size) if max_queue_size > 0 else asyncio.Queue()
        )
        self.reservoir = ReservoirBuffer(buffer_size, replace_rate)
        self.replay_fraction = replay_fraction

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ):
        addr = writer.get_extra_info("peername")
        print(f"Client connected: {addr}")
        try:
            while True:
                header = await reader.readexactly(5)
                length_bytes = await reader.readexactly(4)
                length = struct.unpack("!I", length_bytes)[0]

                if header == b"TRAIN":
                    payload = await reader.readexactly(4 * length + 4 * 4 * 64 * 64)
                    packet = header + length_bytes + payload
                    parsed = TrainPacket.parse(packet)
                    inp = torch.tensor(parsed.input, dtype=torch.float32).view(1, self.model.input_dim)
                    latent = torch.tensor(parsed.latent, dtype=torch.float32).view(1, 4, 64, 64)
                    # drop oldest if queue full
                    if self.queue.maxsize and self.queue.full():
                        _ = await self.queue.get()
                    await self.queue.put((inp, latent))

                elif header == b"INFER":
                    payload = await reader.readexactly(4 * length)
                    packet = header + length_bytes + payload
                    parsed = InferPacket.parse(packet)
                    inp = torch.tensor(parsed.input, dtype=torch.float32).view(1, self.model.input_dim)
                    self.model.eval()
                    with torch.no_grad():
                        out = self.model(inp.to(self.device))[0]
                    writer.write(out.cpu().numpy().tobytes())
                    await writer.drain()

                else:
                    print(f"Unknown header: {header}")
                    break

        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            print(f"Client disconnected: {addr}")

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

class TrainingServer(TrainingServerBase):
    async def train_worker(self):
        print(f"Train worker started (batch={self.batch_size}, replay_frac={self.replay_fraction})")
        try:
            while True:
                fresh_samples = [await self.queue.get() for _ in range(self.batch_size)]
                # add fresh to reservoir
                for sample in fresh_samples:
                    self.reservoir.add(sample)

                num_replay = int(self.batch_size * self.replay_fraction)
                num_fresh = self.batch_size - num_replay

                replay_samples = self.reservoir.sample_batch(num_replay)
                batch = fresh_samples[:num_fresh] + replay_samples

                inputs = torch.cat([s[0] for s in batch], dim=0).to(self.device)
                targets = torch.cat([s[1] for s in batch], dim=0).to(self.device)

                loss = await asyncio.to_thread(self.model.training_step, inputs, targets)
                print(f"Batch trained! Loss: {loss.item():.6f}")
        except asyncio.CancelledError:
            print("Train worker cancelled")
            raise

async def main():
    model = SimpleNet(input_dim=342).to("cuda")
    srv = TrainingServer(
        host='0.0.0.0',
        port=9999,
        model=model,
        batch_size=16,
        buffer_size=1000,
        replace_rate=0.5,      # 50% chance to consider replacing
        replay_fraction=0.3,   # 30% of each batch from past samples
        max_queue_size=500     # drop oldest when more than 500 pending samples
    )
    await srv.start()

if __name__ == '__main__':
    asyncio.run(main())
