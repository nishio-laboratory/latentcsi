# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")
from typing import cast, Union, List
from src.encoder.base import TrainingTimerCallback
from src.encoder.model import CNNDecoder
from src.encoder.data_utils import CSIDataset
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import ops
from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import argparse
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


class CSIAutoencoderMLP_CNN(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        mlp_layer_sizes: List[int],
        base_channels: int,
        lr=5e-4,
        name="",
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
                input_dim=mlp_layer_sizes[-1], base_channels=base_channels
            )
            self.encoder = ops.MLP(
                input_size,
                mlp_layer_sizes,
                activation_layer=nn.ReLU,
            )
            self.model = nn.Sequential(self.encoder, nn.ReLU(), self.decoder)

        self.lr = lr
        self.input_size = input_size
        self.mlp_layer_sizes = mlp_layer_sizes
        self.name = name

        self.save_hyperparameters({"name": self.ckpt_name()})
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

    def ckpt_name(self):
        return (
            self.name
            + "mlp_cnn_"
            + "-".join(map(str, self.mlp_layer_sizes))
            + "{val_loss}"
        )

    def num_params(self):
        return sum(p.numel() for p in self.decoder.parameters())


# ***
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-n", "--name", type=str, default="")
    parser.add_argument("-e", "--epochs", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--base-channels", default=1024, type=int)
    parser.add_argument("-l", "--layer-sizes", default=[], type=int, nargs="+")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.name = "debug"
        args.path = Path("/data/datasets/walking_test")
        args.base_channels = 32

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
        data_dim, args.layer_sizes, args.base_channels, args.lr, args.name
    )
    print(sum(p.numel() for p in model.encoder.parameters()))
    print(sum(p.numel() for p in model.decoder.parameters()))

    trainer_config = {
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
    if args.debug:
        trainer_config["devices"] = [0]
    if "devices" not in trainer_config or len(trainer_config["devices"]) > 1:
        trainer_config["strategy"] = "ddp_find_unused_parameters_true"
    trainer = L.Trainer(**trainer_config)

    trainer.fit(model, train, val)
    trainer.test(dataloaders=test, ckpt_path="best")


if __name__ == "__main__":
    main()
    # python -m src.encoder.training_latents -p /data/datasets/mmfi_hands_two/ -n final_0 -epochs 1000 -b 32
