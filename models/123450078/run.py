"""Vision Transformer kecil untuk empat kelas okupansi."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from src.latih_citra import argumen, jalankan


class TransformerOrang(nn.Module):
    """49 patch 32×32, dua encoder independen, mean pooling, empat kelas."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("mean", mean.reshape(1, 3, 1, 1))
        self.register_buffer("std", std.reshape(1, 3, 1, 1))
        self.patch = nn.Conv2d(3, 64, kernel_size=32, stride=32)
        self.posisi = nn.Parameter(torch.empty(1, 49, 64))
        nn.init.trunc_normal_(self.posisi, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(2)
        ])
        self.kepala = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input float RGB [0, 1]; statistik normalisasi ikut tersimpan di checkpoint.
        x = self.patch((x - self.mean) / self.std).flatten(2).transpose(1, 2)
        x = self.dropout(x + self.posisi)
        for encoder in self.encoder:
            x = encoder(x)
        return self.kepala(x.mean(dim=1))


def buat_model(mean, std, args):
    return TransformerOrang(mean, std, args.dropout)


if __name__ == "__main__":
    jalankan("123450078", "trf", argumen(epochs=10).parse_args(), buat_model)
