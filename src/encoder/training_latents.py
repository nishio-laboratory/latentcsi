# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")
from mlp import MLP
from data_utils import load_data
from viz import test_model
import diffusers
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import lightning as L
import argparse

parser = argparse.Parser()
parser.add_argument("-p", "--path", required=True)
parser.add_argument("-d", "--dontsave", required=True)
args = parser.parse_args()

data_path = Path(args.path)
dataset = load_data(data_path)
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

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        if len(x.shape) == 2:
            out = torch.reshape(out, (x.shape[0], 4, 64, 64))
        else:
            out = torch.reshape(out, (4, 64, 64))
        return out

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
    logger=L.pytorch.loggers.CSVLogger(save_dir=data_path / "logs"),
)
trainer.fit(model, data)

# ***

if args.save:
    (data_path / "ckpts").mkdir(exist_ok=True)
    torch.save(model, data_path / "ckpts" / "mlp_deep_64")
    print("Saved model")

# ***

pipeline = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
    "/data/sd/sd-v1-5", torch_dtype=torch.half, use_safetensors=True
).to(0)

inputs = dataset[0]
test_idx = 1445
gpu_rank = 0

p = test_model(data_path, model, pipeline, photos, inputs, test_idx)

gen = torch.Generator("cuda")

test_photo = (
    torch.from_numpy(photos[test_idx])
    .to(gpu_rank, torch.half)
    .unsqueeze(0)
    .permute(0, 3, 1, 2)
)

Image.fromarray(photos[test_idx]).save(data_path / "test_photo.png")

t = (
    pipeline(
        "a photograph of a man wearing a white hoodie in a clean and empty room, 4k, realistic, high resolution",
        p,
        negative_prompts="blurry, out of focus, depth of field",
        strength=0.6,
        # num_inference_steps=70,
        # guidance_scale=6,
    )
    .images[0]
    .save(data_path / "test.png")
)
