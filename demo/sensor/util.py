import numpy as np
from typing import Tuple


class Buffer:
    def __init__(self, size: Tuple, dtype: np.typing.DTypeLike):
        self.buffer = np.zeros(size, dtype=dtype)
        self.dtype = dtype
        self.size = size
        self.idx = 0

    def add(self, data: np.ndarray):
        if self.idx >= self.size[0]:
            raise Exception("buffer size exceeded")

        self.buffer[self.idx] = data.astype(self.dtype)
        self.idx += 1

    def clear(self):
        self.idx = 0

    def full(self) -> bool:
        return self.idx == self.size[0]
