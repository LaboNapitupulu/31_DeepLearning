"""GRU atas urutan baris citra untuk empat kelas okupansi."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torch import nn

from src.latih_citra import Normalisasi, argumen, jalankan


class GRUBaris(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.pool = nn.AvgPool2d(4)
        self.proyeksi = nn.Sequential(nn.Linear(56 * 3, 64), nn.LayerNorm(64), nn.GELU())
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.kelas = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 4))

    def forward(self, x):
        baris = self.pool(x).permute(0, 2, 1, 3).flatten(2)
        _, keadaan = self.gru(self.proyeksi(baris))
        return self.kelas(keadaan[-1])


def buat_model(mean, std, args):
    return Normalisasi(mean, std, GRUBaris(args.dropout))


if __name__ == "__main__":
    jalankan("123450091", "rnn", argumen(lr=1e-3, dropout=0.2).parse_args(), buat_model)
