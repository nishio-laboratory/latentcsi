from typing import List, Optional
import time
from src.encoder.base import TrainingTimerCallback
from src.encoder.data_utils import CSIDataset
from src.inference.utils import permute_color_chan
import torch
from torch import nn, relu, tanh
from torch.utils.data import DataLoader
from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import argparse
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


class Generator(nn.Module):
    def __init__(self, input_dim: int, initial_channels: int):
        super(Generator, self).__init__()
        self.initial_channels = initial_channels
        self.initial_size = 64
        self.fc = nn.Linear(
            input_dim,
            self.initial_channels * self.initial_size * self.initial_size,
        )
        self.upsample_blocks = nn.ModuleList(
            [
                self._upsampling_block(
                    initial_channels, initial_channels // 2
                ),
                self._upsampling_block(
                    initial_channels // 2, initial_channels // 4
                ),
                self._upsampling_block(
                    initial_channels // 4, initial_channels // 8
                ),
            ]
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(initial_channels // 8, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def _upsampling_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1
            ),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = relu(self.fc(z))
        x = x.view(
            -1, self.initial_channels, self.initial_size, self.initial_size
        )
        for block in self.upsample_blocks:
            x = block(x)
        return self.final_conv(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.downsample_blocks = nn.ModuleList(
            [
                self._downsampling_block(3, 64),
                self._downsampling_block(64, 128),
                self._downsampling_block(128, 256),
            ]
        )
        self.fc1 = nn.Linear(256 * 8 * 8, 1)

    def _downsampling_block(
        self, in_channels: int, out_channels: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=4, padding=0
            ),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=0.25),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.downsample_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class GAN(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        initial_channels: int,
        hybrid_k: Optional[int] = None,
        name: str = "",
    ):
        super(GAN, self).__init__()
        self.generator = Generator(input_dim, initial_channels)
        self.discriminator = Discriminator()
        self.automatic_optimization = False
        self.hybrid_k = hybrid_k
        self.name = name
        self.save_hyperparameters()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def training_step(self, batch: list, batch_idx: int) -> None:
        # get optimizers
        opt_g, opt_d = self.optimizers()
        loss_fn = nn.BCEWithLogitsLoss()

        z, _, real_images = batch
        real_images = permute_color_chan(real_images).to(
            next(self.generator.parameters()).dtype
        )
        real_images = (real_images / 127.5) - 1  # 0, 255 -> -1, 1

        opt_d.zero_grad()
        opt_g.zero_grad()

        fake_images = self(z)

        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(fake_images.detach())
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        loss_d = (
            loss_fn(real_logits, real_labels)
            + loss_fn(fake_logits, fake_labels)
        ) / 2

        # backward and step D
        self.manual_backward(loss_d)
        opt_d.step()
        opt_d.zero_grad()

        if self.hybrid_k and batch_idx % self.hybrid_k == 0:
            fake_logits = self.discriminator(fake_images)
            loss_g = loss_fn(fake_logits, real_labels)
        else:
            loss_g = torch.nn.functional.mse_loss(fake_images, real_images)

        self.manual_backward(loss_g)
        opt_g.step()

        self.log("loss/generator", loss_g, prog_bar=True)
        self.log("loss/discriminator", loss_d, prog_bar=True)

    def validation_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        inputs, _, targets = batch
        targets = permute_color_chan(targets).to(
            next(self.generator.parameters()).dtype
        )
        outputs = self(inputs)
        outputs = (outputs + 1) * 127.5
        loss = torch.nn.functional.mse_loss(outputs, targets.to(outputs.dtype))
        self.log(
            "val_loss", loss, sync_dist=True, prog_bar=True, on_epoch=True
        )

        return loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        # return list only for unpacking
        return [opt_g, opt_d]

    def ckpt_name(self) -> str:
        return f"{self.name}_pixel_{{val_loss}}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-n", "--name", type=str, default="")
    parser.add_argument("-e", "--epochs", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--base-channels", default=1024, type=int)
    parser.add_argument("--hybrid-k", default=None, type=int)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    data_path = Path(args.path)
    dataset = CSIDataset(data_path, aux_data=["photos"])
    train, val, test = torch.utils.data.random_split(
        dataset, [0.8, 0.1, 0.1], torch.Generator().manual_seed(42)
    )
    train, val, test = map(
        lambda ds: DataLoader(ds, batch_size=args.batch_size, num_workers=15),
        (train, val, test),
    )
    model = GAN(
        next(iter(test))[0].size(1),
        args.base_channels,
        args.hybrid_k,
        name=args.name,
    )
    trainer = L.Trainer(
        devices=[0],
        max_epochs=args.epochs,
        accelerator="gpu",
        logger=CSVLogger(save_dir=data_path, name="logs", version=args.name),
        callbacks=[
            EarlyStopping("val_loss", patience=5),
            ModelCheckpoint(
                dirpath=data_path / "ckpts", filename=model.ckpt_name()
            ),
            TrainingTimerCallback(),
        ],
    )
    trainer.fit(model, train, val)
    #trainer.test(dataloaders=test, ckpt_path="best")
