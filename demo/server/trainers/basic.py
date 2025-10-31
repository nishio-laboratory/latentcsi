import random
import time
import torch.multiprocessing as mp
from diffusers import AutoencoderTiny, StableDiffusionImg2ImgPipeline
from pathlib import Path
import typing
import contextlib
import torch
import torch.nn.functional as F
from demo.server.protocol import *
from demo.server.trainer_base import TrainerBase
from src.encoder.model import CNNDecoder
from src.other.types import *
from src.inference.utils import sd_convert, sd_load, sd_make_lat_tiny
from enum import Enum


class CNNDecoderTrainable(CNNDecoder):
    def __init__(self, lr: float = 1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def get_input_dim(self) -> int:
        return self.input_dim

    def update_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train_step(
        self, inp: BatchCSI, target: BatchTrueLatent
    ) -> tuple[torch.Tensor, BatchPredLatent]:
        self.train()
        self.optimizer.zero_grad()
        pred = self.forward(inp)
        loss = F.mse_loss(pred, target)
        loss.backward()
        self.optimizer.step()
        return (loss, BatchPredLatent(pred.detach()))

    def as_eval(self):
        @contextlib.contextmanager
        def _ctx():
            state = self.training
            self.eval()
            try:
                yield self
            finally:
                self.train(state)

        return _ctx()


class BatchReservoir:
    class _Buffer(list):
        def __init__(self, parent: "BatchReservoir"):
            super().__init__()
            self._parent = parent

        def append(self, batch: Batch) -> None:
            super().append(self._parent._clone_batch(batch))

        def __setitem__(self, index, batch) -> None:  # type: ignore[override]
            if isinstance(index, slice):
                super().__setitem__(
                    index,
                    [self._parent._clone_batch(b) for b in batch],
                )
            else:
                super().__setitem__(index, self._parent._clone_batch(batch))

    def __init__(
        self, buffer_size: int, replace_rate: float = 0.5, uniform=True
    ):
        self.buffer_size = buffer_size
        self.replace_rate = replace_rate
        self.uniform = uniform
        self.buffer: BatchReservoir._Buffer = BatchReservoir._Buffer(self)
        self.count = 0

    def _clone_batch(self, batch: Batch) -> Batch:
        csi = BatchCSI(batch.csi.detach().clone().contiguous())
        lat = BatchTrueLatent(batch.lat.detach().clone().contiguous())
        return Batch(csi, lat)

    def add(self, batch: Batch):
        self.count += 1
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(batch)
        else:
            if random.random() <= self.replace_rate:
                if self.uniform:
                    j = random.randrange(self.count)
                    if j < self.buffer_size:
                        self.buffer[j] = batch
                else:
                    j = random.randrange(len(self.buffer))
                    self.buffer[j] = batch

    def pick(self) -> Batch:
        if self.empty():
            raise ValueError("Can't pick sample batch from empty buffer")
        return random.choice(self.buffer)

    def empty(self) -> bool:
        return len(self.buffer) == 0

    def size(self) -> int:
        return len(self.buffer)


class Trainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = CNNDecoderTrainable(
            input_dim=1992,
            base_channels=128,
            lr=1e-3,
        ).to(self.device)
        self.batch_reservoir = BatchReservoir(1000)
        self.batches_trained: int = 0
        print("Train process ready!")
        self.main_loop()

    def train_new(self) -> tuple[PredLatent, float]:
        batch = self.data_queue.get()
        self.batch_reservoir.add(batch)
        inputs = BatchCSI(batch.csi.to(self.device))
        outputs = BatchTrueLatent(batch.lat.to(self.device))
        loss, preds = self.model.train_step(inputs, outputs)
        return (
            PredLatent(preds[-1].unsqueeze(0).detach().cpu().contiguous()),
            float(loss.item()),
        )

    def train_replay(self) -> float:
        batch = self.batch_reservoir.pick()
        inputs = BatchCSI(batch.csi.to(self.device))
        outputs = BatchTrueLatent(batch.lat.to(self.device))
        loss, _ = self.model.train_step(inputs, outputs)
        return float(loss.item())

    def dispatch(self, msg: Message):
        match msg:
            case "start_rec":
                raise NotImplementedError()
            case "stop_rec":
                raise NotImplementedError()
            case "reset":
                raise NotImplementedError()
            case ("chglr", new_lr):
                pass

    def main_loop(self):
        i = 0
        times = []
        while True:
            now = time.time()

            if not self.message_queue.empty():
                msg = self.message_queue.get()

            if self.data_queue.empty():
                if self.batch_reservoir.empty():
                    continue
                loss = self.train_replay()
            else:
                pred, loss = self.train_new()
                self.latest_pred.update(pred)

            self.batches_trained += 1
            print(
                f"Batch {self.batches_trained} trained! Loss: {loss:.6f}. Qsize: {self.data_queue.qsize()}"
            )

            i += 1
            times.append(time.time() - now)
            if i % 50 == 0:
                print(f"avg time to train batch: {sum(times) / len(times)}")
                times = []


class TrainerLastReplay(TrainerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = CNNDecoderTrainable(
            input_dim=1992,
            base_channels=128,
            lr=1e-4,
        ).to(self.device)
        self.last: typing.Optional[Batch] = None
        self.batches_trained: int = 0
        print("Train process ready!")
        self.main_loop()

    def train_new(self) -> tuple[PredLatent, float]:
        batch = self.data_queue.get()
        self.last = batch
        inputs = BatchCSI(batch.csi.to(self.device))
        outputs = BatchTrueLatent(batch.lat.to(self.device))
        loss, preds = self.model.train_step(inputs, outputs)
        return (
            PredLatent(preds[-1].unsqueeze(0).detach().cpu().contiguous()),
            float(loss.item()),
        )

    def train_replay(self) -> typing.Optional[float]:
        batch = self.last
        if batch is None:
            return
        inputs = BatchCSI(batch.csi.to(self.device))
        outputs = BatchTrueLatent(batch.lat.to(self.device))
        loss, _ = self.model.train_step(inputs, outputs)
        return float(loss.item())

    def main_loop(self):
        i = 0
        times = []
        while True:
            now = time.time()

            if self.data_queue.empty():
                loss = self.train_replay()
                if loss is None:
                    continue
            else:
                pred, loss = self.train_new()
                self.latest_pred.update(pred)

            self.batches_trained += 1
            print(
                f"Batch {self.batches_trained} trained! Loss: {loss:.6f}. Qsize: {self.data_queue.qsize()}"
            )

            i += 1
            times.append(time.time() - now)
            if i % 50 == 0:
                print(f"avg time to train batch: {sum(times) / len(times)}")
                times = []
