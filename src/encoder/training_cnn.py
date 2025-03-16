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
class CNNDecoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1028 * 2 * 2)
        initial_channels = 1028
        output_channels = 4
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(initial_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Extra convolution block (preserves 8x8)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Extra convolution block (preserves 8x8)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Upsample from 8x8 to 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Extra convolution block (preserves 16x16)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Upsample from 16x16 to 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Extra convolution block (preserves 32x32)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Upsample from 32x32 to 64x64, output desired channels
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
)
    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc(x)
        x = x.view(batch_size, 1028, 2, 2)
        x = self.decoder(x)
        # x = x.permute(0, 2, 3, 1)
        return x

    
class CSIAutoencoderMLP_CNN(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        mlp_layer_sizes: List[int],
        bottleneck_size: int,
        lr=5e-4,
    ):
        super().__init__()
        self.encoder = ops.MLP(
            input_size,
            mlp_layer_sizes + [bottleneck_size],
            activation_layer=nn.ReLU,
        )
        self.decoder = CNNDecoder(bottleneck_size)
        self.model = nn.Sequential(self.encoder, self.decoder)
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
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        # Extract all loss values from outputs
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("avg_test_loss", avg_loss)  # Log aggregated loss


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
    parser.add_argument(
        "-l", "--layer-sizes", default=None, type=int, nargs="+"
    )
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

    if not args.layer_sizes:
        args.layer_sizes = [1000, 500]

    model = CSIAutoencoderMLP_CNN(
        next(iter(test))[0].size(1), args.layer_sizes, args.bottleneck, args.lr
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
