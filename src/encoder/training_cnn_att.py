# (setq python-shell-interpreter "/home/esrh/csi_to_image/activate_docker.sh")
# (setq python-shell-intepreter-args "-p")
from typing import cast, Union, List
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


# ***
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads
        )
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, cond):
        """
        x: (B, C, H, W) spatial feature map (with C == hidden_dim)
        cond: (B, cond_dim) 1D conditioning vector
        """
        b, c, h, w = x.shape
        # Reshape spatial feature map to (B, L, C) where L = H*W.
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # Project spatial features to queries.
        q = self.to_q(x_reshaped)  # (B, L, hidden_dim)

        # Process the conditioning vector as a single token.
        cond_token = cond.unsqueeze(1)  # (B, 1, cond_dim)
        k = self.to_k(cond_token)  # (B, 1, hidden_dim)
        v = self.to_v(cond_token)  # (B, 1, hidden_dim)

        # nn.MultiheadAttention expects (sequence_length, batch, embed_dim)
        q = q.transpose(0, 1)  # (L, B, hidden_dim)
        k = k.transpose(0, 1)  # (1, B, hidden_dim)
        v = v.transpose(0, 1)  # (1, B, hidden_dim)

        # Compute attention (we ignore the attention weights).
        attn_output, _ = self.mha(q, k, v)

        # Bring output back to (B, L, hidden_dim)
        attn_output = attn_output.transpose(0, 1)
        # Final linear projection.
        attn_output = self.out_proj(attn_output)
        # Residual connection.
        x_reshaped = x_reshaped + attn_output

        # Reshape back to (B, C, H, W)
        x_out = x_reshaped.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x_out


class UpSampleBlock(nn.Module):
    def __init__(self, in_c, out_c, cond_dim, n_heads=4, apply_attention=True):
        super().__init__()
        self.res1 = ResidualBlock(in_c, out_c)
        self.res2 = ResidualBlock(out_c, out_c)
        self.apply_attention = apply_attention
        if self.apply_attention:
            self.cross_attn = CrossAttentionBlock(
                out_c, cond_dim, n_heads=n_heads
            )
        self.upsample = nn.ConvTranspose2d(
            out_c, out_c, kernel_size=2, stride=2
        )

    def forward(self, x, cond):
        x = self.res1(x)
        x = self.res2(x)
        if self.apply_attention:
            x = self.cross_attn(x, cond)
        x = self.upsample(x)
        return x


# --------------------------------------------------------
#  Main Model
# --------------------------------------------------------


class CNNDecoder(nn.Module):
    """
    A latent-space generator that:
      1) Projects a 1D input vector of size `input_dim` into a 2D tensor of shape (base_channels, 8, 8),
      2) Applies 3 upsample blocks (8x8 -> 16x16 -> 32x32 -> 64x64). Only the last two blocks use cross-attention.
      3) Uses a final convolution to produce a 4-channel output.
    """

    def __init__(self, input_dim=342, base_channels=512, n_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.base_channels = base_channels

        self.fc = nn.Linear(input_dim, base_channels * 4 * 4)

        self.up1 = UpSampleBlock(
            base_channels,
            base_channels,
            cond_dim=input_dim,
            n_heads=n_heads,
            apply_attention=False,
        )
        self.up2 = UpSampleBlock(
            base_channels,
            base_channels,
            cond_dim=input_dim,
            n_heads=n_heads,
            apply_attention=True,
        )
        self.up3 = UpSampleBlock(
            base_channels,
            base_channels,
            cond_dim=input_dim,
            n_heads=n_heads,
            apply_attention=True,
        )
        self.up4 = UpSampleBlock(
            base_channels,
            base_channels,
            cond_dim=input_dim,
            n_heads=n_heads,
            apply_attention=True,
        )

        self.out_conv = nn.Conv2d(base_channels, 4, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: (B, input_dim)
        Returns: (B, 4, 64, 64)
        """
        b = x.shape[0]
        hidden = self.fc(x)
        hidden = hidden.view(b, self.base_channels, 4, 4)
        hidden = self.up1(hidden, x)  # 8x8 -> 16x16 (no attention)
        hidden = self.up2(hidden, x)  # 16x16 -> 32x32 (with attention)
        hidden = self.up3(hidden, x)  # 32x32 -> 64x64 (with attention)
        hidden = self.up4(hidden, x)  # 32x32 -> 64x64 (with attention)
        out = self.out_conv(hidden)
        return out


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
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("-e", "--epochs", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--base-channels", default=1024, type=int)
    parser.add_argument("-l", "--layer-sizes", default=[], type=int, nargs="+")
    args = parser.parse_args()

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
    print(model)
    print(sum(p.numel() for p in model.encoder.parameters()))
    print(sum(p.numel() for p in model.decoder.parameters()))

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=CSVLogger(save_dir=data_path, name="logs", version=args.name),
        strategy="ddp_find_unused_parameters_true",
        callbacks=[
            EarlyStopping("val_loss", patience=5),
            ModelCheckpoint(
                dirpath=data_path / "ckpts",
                filename=model.ckpt_name(),
            ),
        ],
    )

    trainer.fit(model, train, val)
    trainer.test()


if __name__ == "__main__":
    main()
    # python -m src.encoder.training_latents -p /data/datasets/mmfi_hands_two/ -n final_0 -epochs 1000 -b 32
