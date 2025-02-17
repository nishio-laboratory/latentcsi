# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")
from base import MLP
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
args = parser.parse_args()

data_path = Path(args.path)
dataset = cast(Dataset, load_data(data_path))
data = DataLoader(dataset)
print("Loaded data")
photos = np.load(data_path / "photos.npy")
print("Loaded photos")


class CSIAutoencoder(L.LightningModule):
    def __init__(self, model):
        # input: 1992*2 = 3984 or 1992
        # output: 4*60*80 = 19200
        super().__init__()
        self.model = model

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

    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=5e-4)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ***

model = CSIAutoencoder(MLP([1992, 1000, 500, 250, 500, 1000, 16384]))
trainer = L.Trainer(
    max_epochs=1,
    logger=CSVLogger(save_dir=data_path / "logs"),
)
trainer.fit(model, data)

if args.save:
    (data_path / "ckpts").mkdir(exist_ok=True)
    torch.save(model, data_path / "ckpts" / "mlp_deep_64")
    print("Saved model")
