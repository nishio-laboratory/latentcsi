# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")

import diffusers
import torch
from torchvision.transforms.functional import to_pil_image
from torch import nn
from pathlib import Path
import numpy as np
from PIL import Image
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

torch.set_default_dtype(torch.half)
data_path = Path("/data/datasets/walking")

targets = torch.load(data_path / "targets" / "targets_dists.pt",
                     weights_only=True).to(torch.half)

np_inputs = np.load(data_path / "csi.npy")
# photos = np.load(data_path / "photos.npy")

# inputs = torch.cat([torch.Tensor(np.real(inputs)),
#                     torch.Tensor(np.imag(inputs))])
inputs = torch.Tensor(np.abs(np_inputs)).to(torch.half)

inputs = inputs[:10000]
targets = targets[:10000]

shuffle_order = np.random.permutation(len(inputs))
inputs = inputs[shuffle_order]
targets = targets[shuffle_order]

std = inputs.std()
mean = inputs.mean()
inputs = inputs - inputs.mean()
inputs = inputs / std

# ***
class CSIAutoencoder(nn.Module):
    def __init__(self):
        # input: 1992*2 = 3984
        # output: 4*60*80 = 19200
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1992, 2000),
            nn.ReLU(),
            nn.Linear(2000, 38400)
        )

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = torch.reshape(x, (8, 60, 80))
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ***
model = CSIAutoencoder().to(0)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10
inputs = inputs.to(0, torch.half)
targets = targets.to(0, torch.half)
split_idx = len(inputs) - 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for x, y in zip(inputs[: split_idx], targets[: split_idx]):
        model.half()
        optimizer.zero_grad()
        p = model(x)
        loss = loss_function(p, y)
        loss.backward()
        model.float()
        optimizer.step()
        epoch_loss += loss.item()
        if torch.inf == loss.item():
            print(loss.item())
    model.half()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in zip(inputs[split_idx + 1:], targets[split_idx + 1:]):
            p = model(x)
            val_loss += loss_function(p, y).item()
    epoch_loss /= split_idx
    val_loss /= len(inputs) - split_idx
    print(f"Epoch [{epoch+1}/{num_epochs}], TL: {epoch_loss}, VL: {val_loss}")

# ***
(data_path / "ckpts").mkdir(exist_ok=True)
torch.save(model, data_path / "ckpts" / "mlp_2layer_abs")

# ***
pipeline = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
    "/data/sd/sd-v1-5",
    torch_dtype=torch.half,
    use_safetensors=True
).to("cuda:1")

pipeline.safety_checker = None

# ***
photos = np.load(data_path / "photos.npy")

gen = torch.Generator("cpu")

test_idx = 5515

test_input = inputs[test_idx]
test = model(test_input.to("cuda", torch.half)).reshape((1, 8, 60, 80))

def sample_from_dist(dist, gen):
    posterior = DiagonalGaussianDistribution(dist)
    return posterior.mode()

test_latent = sample_from_dist(test, gen)

decoded = pipeline.vae.decode(
    test_latent.to(1)
).sample

to_pil_image(decoded.squeeze()).save(data_path / "test_latent.png")

# ***

t = pipeline(
    "a photograph of a man wearing a white hoodie walking in a room, high resolution, 4k, ultra realistic",
    test_latent * 0.18215,
    strength=0.45,
    num_inference_steps=70,
    guidance_scale=6,
)
t.images[0].save(
    data_path / "test.png"
)

Image.fromarray(photos[test_idx]).save(data_path / "test_photo.png")
