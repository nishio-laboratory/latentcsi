# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")

import diffusers
import gc
import transformers
from tqdm import tqdm
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

targets_seg = torch.load(
    data_path / "targets" / "targets_seg.pt", weights_only=True
).to(torch.half)

np_inputs = np.load(data_path / "csi.npy")
photos = np.load(data_path / "photos.npy")

inputs = torch.Tensor(np.abs(np_inputs)).to(torch.half)

std = inputs.std()
mean = inputs.mean()
inputs = inputs - inputs.mean()
inputs = inputs / std
# ***

pipeline = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
    "/data/sd/sd-v1-5", torch_dtype=torch.half, use_safetensors=True
)
vae = pipeline.vae.to(1)
vae.half()
vae.eval()
pipeline.safety_checker = None

# ***

feature_extractor = transformers.SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
segformer = transformers.SegformerModel.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
segformer = segformer.to(1)
segformer.half()
segformer.eval()


# ***
class CSIAutoencoder(nn.Module):
    def __init__(self):
        # input: 1992*2 = 3984 or 1992
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
model = CSIAutoencoder().to(0)

lat_loss = nn.MSELoss()
seg_loss = nn.MSELoss()

num_epochs = 40
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

inputs = inputs.to(0, torch.half)
targets = targets.to(0, torch.half)
targets_seg = targets_seg.to(0, torch.half)
split_idx = len(inputs) - 100

# ***

epoch_loss_history = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for x, t_lat, t_seg in tqdm(
        zip(
            batched(inputs[:split_idx], 32),
            batched(targets[:split_idx], 32),
            batched(targets_seg[:split_idx], 32),
        ),
        total=split_idx // 32,
    ):
        x_batch = torch.stack(x)
        t_lat_batch = torch.stack(t_lat)
        t_seg_batch = torch.stack(t_seg)
        model.half()
        optimizer.zero_grad()
        p = model(x_batch)
        loss_1 = lat_loss(p, t_lat_batch)
        with torch.no_grad():
            p = p.to(1)
            decoded = ((vae.decode(p).sample + 1) / 2).clamp(0, 1)
            del p
            p_img = feature_extractor(
                images=decoded, return_tensors="pt"
            ).pixel_values
            p_seg = segformer(decoded).last_hidden_state
            p_seg = p_seg.to(0)
        loss_2 = seg_loss(p_seg, t_seg_batch)
        loss = loss_1 + loss_2
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
            val_loss += lat_loss(p, y).item()
    epoch_loss /= split_idx
    epoch_loss_history.append(epoch_loss)

    val_loss /= len(inputs) - split_idx
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], TL: {epoch_loss}, VL: {val_loss}"
    )

# ***

(data_path / "ckpts").mkdir(exist_ok=True)
torch.save(model, data_path / "ckpts" / "mlp_deep_segloss")

# model = torch.load(data_path / "ckpts" / "mlp_deep", weights_only=False).to(0)

# ***

test_idx = 1445

test_input = inputs[test_idx]
test = model(test_input.to("cuda", torch.half).unsqueeze(0))

decoded = (pipeline.vae.decode(test.to(1)).sample + 1) / 2
to_pil_image(decoded.squeeze()).save(data_path / "test_latent.png")

gen = torch.Generator("cuda")

test_photo = (
    torch.from_numpy(photos[test_idx])
    .to(1, torch.half)
    .unsqueeze(0)
    .permute(0, 3, 1, 2)
)

to_pil_image(
    pipeline.vae.decode(
        pipeline.vae.encode(test_photo).latent_dist.mode()
    ).sample.squeeze()
).save(data_path / "test_direct.png")

Image.fromarray(photos[test_idx]).save(data_path / "test_photo.png")

t = pipeline(
    "a man in a clean and empty room, photograph, 4k, realistic, high resolution",
    test * 0.18215,
    strength=0.5,
    num_inference_steps=70,
    guidance_scale=7,
)

t.images[0].save(data_path / "test.png")
