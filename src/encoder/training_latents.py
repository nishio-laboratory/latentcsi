# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")
from src.encoder.base import MLP, CSIAutoencoderBase
from typing import cast, Union
from src.encoder.data_utils import CSIDataset
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import argparse
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


class CSIAutoencoder(CSIAutoencoderBase):
    def __init__(self, layer_sizes, lr):
        super().__init__(layer_sizes, lr)

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

    def ckpt_name(self, name: Union[str, None]):
        return (
            (f"{name}_" if name else "")
            + "mlp_"
            + "-".join(map(str, self.model.layer_sizes))
            + "{val_loss}"
        )


# ***
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("-e", "--epochs", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument(
        "-l", "--layer-sizes", default=None, type=int, nargs="+"
    )
    args = parser.parse_args()

    if "H100" in torch.cuda.get_device_name(0):
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
        args.layer_sizes = [342, 1000, 500, 250, 500, 1000, 16384]

    model = CSIAutoencoder(args.layer_sizes, lr=5e-4)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=CSVLogger(save_dir=data_path, name="logs", version=args.name),
        strategy="ddp_find_unused_parameters_true",
        callbacks=[
            EarlyStopping("val_loss", patience=15),
            ModelCheckpoint(
                dirpath=data_path / "ckpts",
                filename=model.ckpt_name(name=args.name),
            ),
        ],
    )

    trainer.fit(model, train, val)


if __name__ == "__main__":
    main()
    # python -m src.encoder.training_latents -p /data/datasets/mmfi_hands_two/ -n final_0 -epochs 1000 -b 32
