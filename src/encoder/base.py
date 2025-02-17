from typing import List, Tuple, Any
from pathlib import Path
import torch
from torch import nn
from more_itertools import flatten
import lightning as L


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        output_shape: Tuple[int, int, int] = (4, 64, 64),
    ):
        super().__init__()
        self.layer_size = layer_sizes
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


class CSIAutoencoderBase(L.LightningModule):
    def __init__(self, layer_sizes, lr=5e-4):
        super().__init__()
        self.lr = lr
        self.model = MLP(layer_sizes)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, ckpts_path: Path):
        if isinstance(self.model, MLP):
            model_name = "mlp"
        else:
            raise Exception("Unmapped model type")
        save_name = model_name + "_" + "-".join(self.model.layer_sizes)
        torch.save(self.model, ckpts_path / save_name)
