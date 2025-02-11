# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")

import diffusers
from more_itertools import flatten, batched
import torch
from torchvision.transforms.functional import to_pil_image
from torch import nn
from pathlib import Path
import numpy as np
from PIL import Image

torch.set_default_dtype(torch.half)
data_path = Path("/data/datasets/walking")

targets = torch.load(
    data_path / "targets" / "targets_latents.pt", weights_only=True
).to(torch.half)

np_inputs = np.load(data_path / "csi.npy")
photos = np.load(data_path / "photos.npy")

inputs = torch.Tensor(np.abs(np_inputs)).to(torch.half)

# shuffle_order = np.random.permutation(len(inputs))
# inputs = inputs[shuffle_order]
# targets = targets[shuffle_order]

std = inputs.std()
mean = inputs.mean()
inputs = inputs - inputs.mean()
inputs = inputs / std


# ***
class CSIAutoencoder(nn.Module):
    def __init__(self):
        # input: 1992*2 = 3984 or 1992
        # output: 4*60*80 = 19200
        super().__init__()
        layer_sizes = [1992, 1000, 500, 250, 500, 1000, 16384]

        layers = [
            [nn.ReLU(), nn.Linear(x, y)] if n != 0 else [nn.Linear(x, y)]
            for n, (x, y) in enumerate(zip(layer_sizes, layer_sizes[1:]))
        ]
        layers = list(flatten(layers))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        if len(x.shape) == 2:
            out = torch.reshape(out, (x.shape[0], 4, 64, 64))
        else:
            out = torch.reshape(out, (4, 64, 64))
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ***
gpu_rank = 2
model = CSIAutoencoder().to(gpu_rank)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
num_epochs = 100
inputs = inputs.to(gpu_rank, torch.half)
targets = targets.to(gpu_rank, torch.half)
split_idx = len(inputs) - 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for x, y in zip(
        batched(inputs[:split_idx], 32), batched(targets[:split_idx], 32)
    ):
        x_batch = torch.stack(x)
        y_batch = torch.stack(y)
        model.half()
        optimizer.zero_grad()
        p = model(x_batch)
        loss = loss_function(p, y_batch)
        loss.backward()
        model.float()
        optimizer.step()
        epoch_loss += loss.item()
    model.half()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in zip(inputs[split_idx + 1 :], targets[split_idx + 1 :]):
            p = model(x.unsqueeze(0)).squeeze()
            val_loss += loss_function(p, y).item()
    epoch_loss /= split_idx
    val_loss /= len(inputs) - split_idx
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], TL: {epoch_loss}, VL: {val_loss}"
    )

# ***
(data_path / "ckpts").mkdir(exist_ok=True)
torch.save(model, data_path / "ckpts" / "mlp_deep_64")

model = torch.load(
    data_path / "ckpts" / "mlp_deep_segloss", weights_only=False
).to(0)

# ***
pipeline = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
    "/data/sd/sd-v1-5", torch_dtype=torch.half, use_safetensors=True
).to(0)

pipeline.safety_checker = None

# ***

gpu_rank = 0
test_idx = 1445

test_input = inputs[test_idx]
test = model(test_input.to(gpu_rank, torch.half).unsqueeze(0))

decoded = (pipeline.vae.decode(test.to(gpu_rank)).sample + 1) / 2
to_pil_image(decoded.squeeze()).save(data_path / "test_latent.png")

gen = torch.Generator("cuda")

test_photo = (
    torch.from_numpy(photos[test_idx])
    .to(gpu_rank, torch.half)
    .unsqueeze(0)
    .permute(0, 3, 1, 2)
)

Image.fromarray(photos[test_idx]).save(data_path / "test_photo.png")

t = pipeline(
    "a photograph of a man wearing a white hoodie in a clean and empty room, 4k, realistic, high resolution",
    test * 0.18215,
    negative_prompts="blurry, out of focus, depth of field",
    strength=0.6,
    # num_inference_steps=70,
    # guidance_scale=6,
)

t.images[0].save(data_path / "test.png")
