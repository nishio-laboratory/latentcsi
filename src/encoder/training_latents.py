# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")
from base import MLP, CSIAutoencoderBase
from typing import cast
from data_utils import load_data
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", required=True)
parser.add_argument("-s", "--save", required=True)
parser.add_argument("-epochs", default=1, type=int)
args = parser.parse_args()

data_path = Path(args.path)
dataset = cast(Dataset, load_data(data_path))
data = DataLoader(dataset)
print("Loaded data")
photos = np.load(data_path / "photos.npy")
print("Loaded photos")


class CSIAutoencoder(CSIAutoencoderBase):
    def __init__(self, layer_sizes, lr):
        # input: 1992*2 = 3984 or 1992
        # output: 4*60*80 = 19200
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


# ***

model = CSIAutoencoder([1992, 1000, 500, 250, 500, 1000, 16384], 5e-4)

trainer = L.Trainer(
    max_epochs=args.epochs,
    logger=CSVLogger(save_dir=data_path / "logs"),
)
trainer.fit(model, data)

if args.save:
    (data_path / "ckpts").mkdir(exist_ok=True)
    torch.save(model, data_path / "ckpts" / "mlp_deep_64")
    print("Saved model")
