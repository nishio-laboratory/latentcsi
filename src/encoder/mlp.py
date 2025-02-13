from typing import List, Tuple, Any
import torch
from torch import nn
from more_itertools import flatten


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        output_shape: Tuple[int, int, int] = (4, 64, 64),
    ):
        super().__init__()
        layers = [
            [nn.ReLU(), nn.Linear(x, y)] if n != 0 else [nn.Linear(x, y)]
            for n, (x, y) in enumerate(zip(layer_sizes, layer_sizes[1:]))
        ]
        layers = list(flatten(layers))
        self.layers = nn.Sequential(*layers)
        self.output_shape = output_shape
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        if len(x.shape) == 2:
            out = torch.reshape(out, (x.shape[0], *self.output_shape))
        else:
            out = torch.reshape(out, self.output_shape)
        return out
