import asyncio
import random
import struct
import torch
import torch.nn.functional as F
from src.encoder.model import CNNDecoder
from demo.server.server_base import TrainableModule, TrainingServerBase
from typing import ByteString, Optional


class CNNDecoderTrainable(CNNDecoder, TrainableModule):
    def __init__(self, lr: float = 1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def get_input_dim(self) -> int:
        return self.input_dim

    def update_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train_step(
        self, inp: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
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

    def sample_batch(
        self, batch_size: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        k = min(batch_size, len(self.reservoir))
        return random.sample(self.reservoir, k)


class TrainingServer(TrainingServerBase):
    def __init__(
        self,
        buffer_size: int = 1000,
        replace_rate: float = 1.0,
        replay_fraction: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.last: Optional[torch.Tensor] = None
        self.reservoir = ReservoirBuffer(buffer_size, replace_rate)
        self.replay_fraction = replay_fraction
        self.batches_trained = 0

    def train_received(self, inp, latent):
        self.last = inp

    async def train_worker(self):
        print(
            f"Train worker started (inp_size={self.model.get_input_dim()}, batch={self.batch_size})"
        )
        try:
            while True:
                fresh_samples = [
                    await self.queue.get() for _ in range(self.batch_size)
                ]
                for sample in fresh_samples:
                    self.reservoir.add(sample)

                num_replay = int(self.batch_size * self.replay_fraction)
                num_fresh = self.batch_size - num_replay

                replay_samples = self.reservoir.sample_batch(num_replay)
                batch = fresh_samples[:num_fresh] + replay_samples

                inputs = torch.cat([s[0] for s in batch], dim=0).to(
                    self.device
                )
                targets = torch.cat([s[1] for s in batch], dim=0).to(
                    self.device
                )

                loss = await asyncio.to_thread(
                    self.model.train_step, inputs, targets
                )
                self.batches_trained += 1
                print(f"Batch {self.batches_trained} trained! Loss: {loss.item():.6f}")
        except asyncio.CancelledError:
            print("Train worker cancelled")
            raise

    async def dispatch(
        self, header: ByteString, reader: asyncio.StreamReader
    ) -> Optional[ByteString]:
        if header == b"ilast":
            if self.last is not None:
                self.model.eval()
                with torch.no_grad():
                    out = self.model(self.last.to(self.device))
                return out.cpu().numpy().tobytes()
        if header == b"chglr":
            lr = struct.unpack("!f", await reader.readexactly(4))[0]
            self.model.update_lr(lr)


async def main():
    model = CNNDecoderTrainable(
        input_dim=1992,
        base_channels=128,
        lr=1e-3,
    ).to(1)
    srv = TrainingServer(
        host="0.0.0.0",
        port=9999,
        model=model,
        batch_size=16,
        buffer_size=1000,
        replace_rate=0.1,
        replay_fraction=0,
        max_queue_size=500,
    )
    await srv.start()


if __name__ == "__main__":
    asyncio.run(main())
