# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")
from src.encoder.base import CSIAutoencoderBase
from typing import cast, Union, List
from src.encoder.data_utils import CSIDataset
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import ops
from pathlib import Path
import numpy as np
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import argparse
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# ***


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class UpsampleBlock(nn.Sequential):
    """
    An upsampling block that:
     - Stacks 2 residual blocks
     - Upsamples by 2x via a ConvTranspose2d (stride=2)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
            nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=2, stride=2
            ),
        )


class CNNDecoder(nn.Module):
    """
    A module that:
     1. Takes a 1D vector of length `input_dim` (e.g., 500).
     2. Linear -> reshape to (base_channels, 8, 8).
     3. UpBlock x 3 -> final res block -> final conv -> output (4, 64, 64).
    """

    def __init__(self, input_dim=500, base_channels=1024):
        super().__init__()
        self.input_dim = input_dim
        self.base_channels = base_channels

        # 1) Linear layer from input_dim -> base_channels * 8 * 8
        self.fc = nn.Linear(input_dim, base_channels * 4 * 4)

        # 2) A series of "UpBlock"s that progressively double resolution:
        #    (8 -> 16), (16 -> 32), (32 -> 64)
        self.up1 = UpsampleBlock(
            base_channels, base_channels // 2
        )  # 8x8 -> 16x16
        self.up2 = UpsampleBlock(
            base_channels // 2, base_channels // 4
        )  # 16x16 -> 32x32
        self.up3 = UpsampleBlock(
            base_channels // 4, base_channels // 8
        )  # 32x32 -> 64x64
        self.up4 = UpsampleBlock(
            base_channels // 8, base_channels // 16
        )  # 32x32 -> 64x64

        # 3) A final ResBlock and 3x3 Conv2d that go from base_channels//8 -> 4 output channels
        self.final_res = ResidualBlock(
            base_channels // 16, base_channels // 16
        )
        self.final_conv = nn.Conv2d(
            base_channels // 16, 4, kernel_size=3, padding=1
        )

    def forward(self, x):
        # x: (batch_size, input_dim)
        batch_size = x.size(0)

        # 1) Linear projection, then reshape to [batch_size, base_channels, 8, 8]
        x = self.fc(x)
        x = x.view(batch_size, self.base_channels, 4, 4)

        # 2) Apply up blocks
        x = self.up1(x)  # -> [batch, base_channels//2, 16, 16]
        x = self.up2(x)  # -> [batch, base_channels//4, 32, 32]
        x = self.up3(x)  # -> [batch, base_channels//8, 64, 64]
        x = self.up4(x)  # -> [batch, base_channels//8, 64, 64]

        # 3) Final refining ResBlock + Conv2d => [batch, 4, 64, 64]
        x = self.final_res(x)
        x = self.final_conv(x)
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class CSIAutoencoderMLP_CNN(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        mlp_layer_sizes: List[int],
        bottleneck_size: int,
        base_channels: int,
        lr=5e-4,
    ):
        super().__init__()

        if mlp_layer_sizes == []:
            self.encoder = nn.Identity()
            self.decoder = CNNDecoder(
                input_dim=input_size, base_channels=base_channels
            )
            self.model = self.decoder
        else:
            self.decoder = CNNDecoder(
                input_dim=bottleneck_size, base_channels=base_channels
            )
            self.encoder = ops.MLP(
                input_size,
                mlp_layer_sizes + [bottleneck_size],
                activation_layer=nn.ReLU,
            )
            self.model = nn.Sequential(self.encoder, nn.ReLU(), self.decoder)

        self.lr = lr
        self.input_size = input_size
        self.mlp_layer_sizes = mlp_layer_sizes
        self.bottleneck_size = bottleneck_size

        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, x: torch.Tensor):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log(
            "val_loss", loss, sync_dist=True, prog_bar=True, on_epoch=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log(
            "test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True
        )
        return {"test_loss": loss}

    def ckpt_name(self, name: Union[str, None]):
        return (
            (f"{name}_" if name else "")
            + "mlp_cnn_"
            + "-".join(map(str, self.mlp_layer_sizes))
            + f"_bottleneck={self.bottleneck_size}_"
            + "{val_loss}"
        )


# ***
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("-e", "--epochs", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--bottleneck", default=250, type=int)
    parser.add_argument("--base-channels", default=1024, type=int)
    parser.add_argument("-l", "--layer-sizes", default=[], type=int, nargs="+")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    data_path = Path(args.path)
    dataset = CSIDataset(data_path)
    train, val, test = torch.utils.data.random_split(
        dataset, [0.8, 0.1, 0.1], torch.Generator().manual_seed(42)
    )
    train, val, test = map(
        lambda ds: DataLoader(ds, batch_size=args.batch_size, num_workers=15),
        (train, val, test),
    )
    print("Loaded data")

    data_dim = next(iter(test))[0].size(1)
    model = CSIAutoencoderMLP_CNN(
        data_dim,
        args.layer_sizes,
        args.bottleneck,
        args.base_channels,
        args.lr,
    )
    print(model)
    print(sum(p.numel() for p in model.encoder.parameters()))
    print(sum(p.numel() for p in model.decoder.parameters()))

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=CSVLogger(save_dir=data_path, name="logs", version=args.name),
        strategy="ddp_find_unused_parameters_true",
        callbacks=[
            EarlyStopping("val_loss", patience=5),
            ModelCheckpoint(
                dirpath=data_path / "ckpts",
                filename=model.ckpt_name(name=args.name),
            ),
        ],
    )

    trainer.fit(model, train, val)
    trainer.test()


if __name__ == "__main__":
    main()
    # python -m src.encoder.training_latents -p /data/datasets/mmfi_hands_two/ -n final_0 -epochs 1000 -b 32
