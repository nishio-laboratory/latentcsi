import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import ops
from pathlib import Path
import lightning as L


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels < 32:
            num_groups = 8
        else:
            num_groups = 32

        self.groupnorm_1 = nn.GroupNorm(num_groups, in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        self.groupnorm_2 = nn.GroupNorm(num_groups, out_channels)
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
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        q = self.to_q(x_reshaped)  # (B, L, hidden_dim)

        cond_token = cond.unsqueeze(1)  # (B, 1, cond_dim)
        k = self.to_k(cond_token)  # (B, 1, hidden_dim)
        v = self.to_v(cond_token)  # (B, 1, hidden_dim)

        # nn.MultiheadAttention expects (sequence_length, batch, embed_dim)
        q = q.transpose(0, 1)  # (L, B, hidden_dim)
        k = k.transpose(0, 1)  # (1, B, hidden_dim)
        v = v.transpose(0, 1)  # (1, B, hidden_dim)

        attn_output, _ = self.mha(q, k, v)

        attn_output = attn_output.transpose(0, 1)
        attn_output = self.out_proj(attn_output)
        x_reshaped = x_reshaped + attn_output

        x_out = x_reshaped.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x_out


class UpsampleBlock(nn.Module):
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


class CNNDecoder(nn.Module):
    def __init__(
        self,
        input_dim=342,
        base_channels=512,
        n_heads=4,
        resolution=64,
        image=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.image = image

        assert resolution % 16 == 0
        self.initial_dim = resolution // 16
        self.fc = nn.Linear(
            input_dim, base_channels * self.initial_dim * self.initial_dim
        )

        self.up1 = UpsampleBlock(
            base_channels,
            base_channels,
            cond_dim=input_dim,
            n_heads=n_heads,
            apply_attention=False,
        )
        self.up2 = UpsampleBlock(
            base_channels,
            base_channels,
            cond_dim=input_dim,
            n_heads=n_heads,
            apply_attention=True,
        )
        self.up3 = UpsampleBlock(
            base_channels,
            base_channels,
            cond_dim=input_dim,
            n_heads=n_heads,
            apply_attention=True,
        )
        self.up4 = UpsampleBlock(
            base_channels,
            base_channels,
            cond_dim=input_dim,
            n_heads=n_heads,
            apply_attention=True,
        )

        self.out_conv = nn.Conv2d(
            base_channels, 3 if self.image else 4, kernel_size=3, padding=1
        )

    def forward(self, x):
        """
        x: (B, input_dim)
        Returns: (B, 4, 64, 64)
        """
        b = x.shape[0]
        hidden = self.fc(x)
        hidden = hidden.view(
            b, self.base_channels, self.initial_dim, self.initial_dim
        )
        hidden = self.up1(hidden, x)  # 8x8 -> 16x16 (no attention)
        hidden = self.up2(hidden, x)  # 16x16 -> 32x32 (with attention)
        hidden = self.up3(hidden, x)  # 32x32 -> 64x64 (with attention)
        hidden = self.up4(hidden, x)  # 32x32 -> 64x64 (with attention)
        out = self.out_conv(hidden)
        if self.image:
            out = out.permute(0, 3, 2, 1)
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
