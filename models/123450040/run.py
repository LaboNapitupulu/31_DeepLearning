"""CNN dari nol untuk empat kelas okupansi."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torch import nn

from src.latih_citra import Normalisasi, argumen, jalankan


def buat_model(mean, std, args):
    cnn = nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=2, padding=2),
        nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        nn.Dropout(args.dropout), nn.Linear(64 * 4, 4),
    )
    return Normalisasi(mean, std, cnn)


if __name__ == "__main__":
    jalankan("123450040", "cnn", argumen(lr=1e-3, dropout=0.2).parse_args(), buat_model)
