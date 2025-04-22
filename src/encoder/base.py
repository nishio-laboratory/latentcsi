from typing import List, Tuple, Any
from pathlib import Path
import time
import os
import torch
from torch import nn
from more_itertools import flatten
import lightning as L
from pytorch_lightning.utilities import rank_zero_only


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        output_shape: Tuple[int, int, int] = (4, 64, 64),
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
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

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class CSIAutoencoderBase(L.LightningModule):
    def __init__(self, layer_sizes, lr=5e-4):
        super().__init__()
        self.lr = lr
        self.model = MLP(layer_sizes)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def save(self, ckpts_path: Path):
        if isinstance(self.model, MLP):
            model_name = "mlp"
        else:
            raise Exception("Unmapped model type")
        save_name = (
            model_name + "_" + "-".join(map(str, self.model.layer_sizes))
        )
        torch.save(self.model, ckpts_path / save_name)

    def forward(self, x: torch.Tensor):
        return self.model.forward(x)


class TrainingTimerCallback(L.Callback):
    def on_train_start(self, trainer, pl_module):
        pl_module._train_start_time = time.time()

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        total_time = time.time() - pl_module._train_start_time
        total_epochs = trainer.current_epoch + 1
        avg_time = total_time / total_epochs if total_epochs > 0 else 0


        out_path = Path(trainer.logger.log_dir) / "training_time.txt"

        with open(out_path, "w") as f:
            f.write(f"Total training time (s): {total_time:.3f}\n")
            f.write(f"Average time per epoch (s): {avg_time:.3f}\n")
            f.write(f"Number of epochs: {total_epochs}")

        print(f"Training timing info written to {out_path}")
