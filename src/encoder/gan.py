from typing import cast, Union, List
import time
from src.encoder.base import TrainingTimerCallback
from src.encoder.data_utils import CSIDataset
from src.encoder.model import CNNDecoder
from src.inference.utils import permute_color_chan
import torch
from torch import nn, permute
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import ops
from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import argparse
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


class Generator(nn.Module):
    def __init__(self, input_dim, initial_channels):
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

    def _upsampling_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1
            ),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, z):
        x = self.fc(z)
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
                self._downsampling_block(3, 32),
                self._downsampling_block(32, 64),
                self._downsampling_block(64, 128),
            ]
        )
        self.fc1 = nn.Linear(128 * 64 * 64, 1)

    def _downsampling_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        for block in self.downsample_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return torch.sigmoid(x)


class GAN(L.LightningModule):
    def __init__(self, input_dim, initial_size):
        super(GAN, self).__init__()
        self.generator = Generator(input_dim, initial_size)
        self.discriminator = Discriminator()
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        print([(i.shape, i.dtype) for i in batch])
        z, _, real_images = batch
        real_images = permute_color_chan(real_images).to(
            next(self.generator.parameters()).dtype
        )
        opt_d.zero_grad()
        fake_images = self(z).detach()
        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(fake_images)

        loss_d = (
            -(
                torch.mean(torch.log(real_logits + 1e-8))
                + torch.mean(torch.log(1 - fake_logits + 1e-8))
            )
            / 2
        )

        # backprop & step D
        self.manual_backward(loss_d)
        opt_d.step()

        # ——— Train Generator ———
        opt_g.zero_grad()
        fake_images = self(z)  # fresh fakes
        fake_logits = self.discriminator(fake_images)
        loss_g = -torch.mean(torch.log(fake_logits + 1e-8))

        # backprop & step G
        self.manual_backward(loss_g)
        opt_g.step()

        # log once per step
        self.log("loss/discriminator", loss_d, prog_bar=True)
        self.log("loss/generator", loss_g, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        inputs, _, targets = batch
        targets = permute_color_chan(targets)
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log(
            "val_loss", loss, sync_dist=True, prog_bar=True, on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        return [optimizer_g, optimizer_d], []

    def ckpt_name(self):
        return "gan_pixel_{val_loss}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-n", "--name", type=str, default="")
    parser.add_argument("-e", "--epochs", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--base-channels", default=1024, type=int)
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
    print("Loaded data")
    data_dim = next(iter(test))[0].size(1)
    model = GAN(data_dim, 16)
    trainer_config = {
        "devices": [0],
        "max_epochs": args.epochs,
        "logger": CSVLogger(
            save_dir=data_path, name="logs", version=args.name
        ),
        "callbacks": [
            EarlyStopping("val_loss", patience=5),
            ModelCheckpoint(
                dirpath=data_path / "ckpts",
                filename=model.ckpt_name(),
            ),
            TrainingTimerCallback(),
        ],
    }
    if "devices" not in trainer_config or len(trainer_config["devices"]) > 1:
        trainer_config["strategy"] = "ddp_find_unused_parameters_true"
    trainer = L.Trainer(**trainer_config)
    trainer.fit(model, train, val)
    trainer.test(dataloaders=test, ckpt_path="best")


if __name__ == "__main__":
    main()
