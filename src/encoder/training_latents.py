# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")
from src.encoder.base import MLP, CSIAutoencoderBase
from typing import cast
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
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log("val_loss", loss)
        return loss

    def ckpt_name(self):
        return (
            "mlp_" + "-".join(map(str, self.model.layer_sizes)) + "{val_loss}"
        )


# ***
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-s", "--save", required=True, action="store_true")
    parser.add_argument("-epochs", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    args = parser.parse_args()

    data_path = Path(args.path)
    dataset = CSIDataset(data_path)
    train, val, test = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    train, val, test = map(lambda ds: DataLoader(ds, batch_size=args.batch_size), (train, val, test))
    print("Loaded data")

    model = CSIAutoencoder([342, 1000, 500, 250, 500, 1000, 16384], lr=5e-4)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=CSVLogger(save_dir=data_path / "logs"),
        strategy="ddp_find_unused_parameters_true",
        callbacks=[
            EarlyStopping("tr_pixel_loss", patience=15),
            ModelCheckpoint(
                dirpath=data_path / "ckpts", filename=model.ckpt_name()
            ),
        ],
    )

    trainer.fit(model, train, val)

    if args.save:
        (data_path / "ckpts").mkdir(exist_ok=True)
        torch.save(model, data_path / "ckpts" / "mlp_deep_64")
        print("Saved model")


if __name__ == "__main__":
    main()
