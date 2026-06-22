from __future__ import annotations

import torch
from torch import nn

from hero.models.blocks import ChannelAttention3D, ConvNeXtBlock3D, UpBlock3D


class UXNetEncoder3D(nn.Module):
    def __init__(self, in_channels: int, dims: list[int], depths: list[int], kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention3D(in_channels)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dims[0]),
            nn.GELU(),
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for idx, dim in enumerate(dims):
            blocks = [ConvNeXtBlock3D(dim, kernel_size=kernel_size, dropout=dropout) for _ in range(depths[idx])]
            self.stages.append(nn.Sequential(*blocks))
            if idx < len(dims) - 1:
                self.downsamples.append(nn.Sequential(
                    nn.InstanceNorm3d(dim),
                    nn.Conv3d(dim, dims[idx + 1], kernel_size=2, stride=2),
                ))

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        x = self.channel_attention(x)
        x = self.stem(x)
        skips = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            skips.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        return skips[:-1], skips[-1]


class MultiTaskUXNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        encoder_dims: list[int],
        depths: list[int],
        kernel_size: int,
        dropout: float,
        seg_out_channels: int,
        t_stage_classes: int,
        n_stage_classes: int,
        clinical_feature_dim: int,
        staging_feature_dim: int,
        clinical_embed_dim: int,
        mlp_hidden_dim: int,
        prognosis_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = UXNetEncoder3D(
            in_channels=in_channels,
            dims=encoder_dims,
            depths=depths,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        bottleneck_dim = encoder_dims[-1]
        decoder_channels = list(reversed(encoder_dims[:-1]))
        self.decoder_blocks = nn.ModuleList([
            UpBlock3D(bottleneck_dim, decoder_channels[0], decoder_channels[0]),
            UpBlock3D(decoder_channels[0], decoder_channels[1], decoder_channels[1]),
            UpBlock3D(decoder_channels[1], decoder_channels[2], decoder_channels[2]),
        ])
        self.seg_head = nn.Conv3d(decoder_channels[-1], seg_out_channels, kernel_size=1)

        self.staging_clinical_projector = nn.Sequential(
            nn.Linear(staging_feature_dim, clinical_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prognosis_clinical_projector = nn.Sequential(
            nn.Linear(clinical_feature_dim, clinical_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.t_stage_head = nn.Sequential(
            nn.Linear(bottleneck_dim + clinical_embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, t_stage_classes),
        )
        self.n_stage_head = nn.Sequential(
            nn.Linear(bottleneck_dim + clinical_embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, n_stage_classes),
        )
        self.relapse_head = nn.Sequential(
            nn.Linear(bottleneck_dim + clinical_embed_dim, prognosis_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prognosis_hidden_dim, 1),
        )
        self.rfs_head = nn.Sequential(
            nn.Linear(bottleneck_dim + clinical_embed_dim, prognosis_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prognosis_hidden_dim, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        staging_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        skips, bottleneck = self.encoder(image)
        x = bottleneck
        for block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = block(x, skip)
        seg_logits = self.seg_head(x)

        pooled = self.pool(bottleneck).flatten(1)
        staging_embed = self.staging_clinical_projector(staging_features)
        prognosis_embed = self.prognosis_clinical_projector(clinical_features)
        staging_input = torch.cat([pooled, staging_embed], dim=1)
        prognosis_input = torch.cat([pooled, prognosis_embed], dim=1)

        return {
            "seg_logits": seg_logits,
            "t_logits": self.t_stage_head(staging_input),
            "n_logits": self.n_stage_head(staging_input),
            "relapse_logit": self.relapse_head(prognosis_input),
            "rfs_logit": self.rfs_head(prognosis_input),
        }

