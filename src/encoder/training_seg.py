# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")
import sys
import gc
from base import MLP, CSIAutoencoderBase
from data_utils import load_data
from viz import test_model
import diffusers
import transformers
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import lightning as L
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", required=True)
parser.add_argument("-s", "--save", action="store_true")
parser.add_argument("-e", "--max-epochs", default=1, type=int)
parser.add_argument("-b", "--batch-size", default=16, type=int)
args = parser.parse_args()

data_path = Path(args.path)
dataset = load_data(data_path, aux_data=["seg_map"])
data = DataLoader(dataset, num_workers=15, batch_size=args.batch_size)
print("Loaded data")

# ***


class CSIAutoencoder(L.LightningModule):
    def __init__(self, layer_sizes, train_seg=False, lr=5e-4):
        super().__init__()
        # self.save_hyperparameters()
        self.train_seg = train_seg
        self.lr = lr
        self.model = MLP(layer_sizes)
        self.segformer = (
            transformers.SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
        )
        self.vae = diffusers.AutoencoderKL().from_pretrained(
            data_path.parents[1] / "sd/sd-v1-5", subfolder="vae",
        )
        self.vae.enable_tiling()

        for param in self.segformer.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False

        self.vae.eval()
        self.segformer.eval()

    def training_step(self, batch, batch_idx):
        inputs, targets, segmaps = batch
        # print(inputs.shape, targets.shape, segmaps.shape)
        outputs = self.model(inputs)
        if not self.train_seg:
            loss = torch.nn.functional.mse_loss(outputs, targets)
            self.log("tr_pixel_loss", loss)
            return loss
        else:
            decoded = ((self.vae.decode(outputs).sample + 1) / 2).clamp(0, 1)
            logits = self.segformer(decoded).logits
            loss = torch.nn.functional.cross_entropy(logits, segmaps)
            del decoded
            del logits
            gc.collect()
            torch.cuda.empty_cache()
            return loss

    # def validation_step(self, batch, batch_idx):
    #     inputs, targets = batch
    #     outputs = self.model(inputs)
    #     loss = torch.nn.functional.mse_loss(outputs, targets)
    #     self.log("val_loss", loss)
    #     return loss



# ***
model = CSIAutoencoder([1992, 2000, 1000, 500, 1000, 2000, 16384])
def make_trainer():
    return L.Trainer(
        max_epochs=args.max_epochs,
        logger=L.pytorch.loggers.CSVLogger(save_dir=data_path / "logs"),
        strategy="ddp_find_unused_parameters_true",
        precision=16,
        callbacks = [L.pytorch.callbacks.EarlyStopping("tr_pixel_loss")]
    )
trainer = make_trainer()
trainer.fit(model, data)
model.train_seg=True
trainer = make_trainer()

if args.save:
    save_path = data_path / "ckpts"
    save_path.mkdir(exist_ok=True)
    model.save(save_path)
    print("Saved model")
