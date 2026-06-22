from __future__ import annotations

import torch
from torch import nn


class LayerNormChannel(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None, None] * normalized + self.bias[:, None, None, None]


class ChannelAttention3D(nn.Module):
    def __init__(self, channels: int = 2, reduction: int = 1) -> None:
        super().__init__()
        hidden = max(channels // max(reduction, 1), 1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.net(self.pool(x))
        return x * weights


class ConvNeXtBlock3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
        self.norm = LayerNormChannel(channels)
        self.pwconv1 = nn.Conv3d(channels, 4 * channels, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout3d(dropout)
        self.pwconv2 = nn.Conv3d(4 * channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pwconv2(x)
        return x + residual


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = nn.functional.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

