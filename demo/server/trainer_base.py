from __future__ import annotations
from abc import ABC, abstractmethod
from multiprocessing.managers import Namespace
import torch.multiprocessing as mp
from demo.server.protocol import StatusResp
from src.other.types import *


@dataclass
class TrainerState:
    started: bool = False
    batches_trained: int = 0
    reservoir_size: int = 0
    recording: bool = False
    training: bool = True

    def construct(self) -> bytes:
        return StatusResp.build(
            {
                "training": self.training,
                "recording": self.recording,
                "reservoir_size": self.reservoir_size,
                "batches_trained": self.batches_trained,
            }
        )


class LockedTensor:
    def __init__(self, data: torch.Tensor, lock: LockType):
        tensor = data.detach().clone().cpu().contiguous()
        tensor.share_memory_()
        self.data = tensor
        self.lock = lock

    def update(self, new: torch.Tensor) -> None:
        with self.lock:
            tensor = self.data
            new_tensor = new.detach().to(device="cpu", dtype=tensor.dtype)
            if not new_tensor.is_contiguous():
                new_tensor = new_tensor.contiguous()
            if tensor.shape != new_tensor.shape:
                tensor.resize_(new_tensor.shape)
            tensor.copy_(new_tensor)

    def get_copy(self) -> torch.Tensor:
        with self.lock:
            tensor = self.data
            new_tensor = torch.zeros_like(self.data)
            new_tensor.copy_(tensor)
        return new_tensor


class TrainerBase(ABC):
    def __init__(
        self,
        data_queue: mp.Queue[Batch],
        message_queue: mp.Queue[Message],
        shared: Namespace,
        latest_pred: LockedTensor,
        device: torch.device,
        dtype: torch.dtype
    ):
        """Needs to call self.main_loop abstract method."""
        self.data_queue = data_queue
        self.shared = shared
        self.message_queue = message_queue
        self.latest_pred = latest_pred
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def main_loop(self):
        pass
