"""MobileNetV2 pralatih untuk empat kelas okupansi."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from src.eksperimen import informasi_data
from src.latih_citra import Normalisasi, argumen, jalankan


class MobileNetOkupansi(nn.Module):
    def __init__(self, mobilenet, dropout, bekukan):
        super().__init__()
        self.fitur = mobilenet.features
        self.bekukan = bekukan
        self.fitur.requires_grad_(not bekukan)
        self.kelas = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(mobilenet.last_channel, 4),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.bekukan:
            self.fitur.eval()
        return self

    def forward(self, x):
        return self.kelas(self.fitur(x))


def buat_model(mean, std, args):
    if args.bobot_acak_simulasi and informasi_data()["jenis"] != "simulasi":
        raise SystemExit("Bobot acak hanya boleh dipakai untuk uji simulasi.")
    if args.bobot or args.bobot_acak_simulasi:
        model = mobilenet_v2(weights=None)
        if args.bobot:
            model.load_state_dict(torch.load(args.bobot, map_location="cpu", weights_only=True))
    else:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    return Normalisasi(mean, std, MobileNetOkupansi(model, args.dropout, not args.fine_tune))


if __name__ == "__main__":
    parser = argumen(epochs=5, lr=1e-3, dropout=0.2)
    parser.add_argument("--simulasi", action="store_true",
                        help="Uji pada data sintetis di folder hasil terpisah.")
    sumber = parser.add_mutually_exclusive_group()
    sumber.add_argument("--bobot", type=str, help="Berkas bobot MobileNetV2 ImageNet yang sudah diunduh.")
    sumber.add_argument("--bobot-acak-simulasi", action="store_true")
    parser.add_argument("--fine-tune", action="store_true", help="Latih juga ekstraktor fitur.")
    args = parser.parse_args()
    if args.simulasi:
        if args.final:
            parser.error("Simulasi tugas #13 hanya mengevaluasi validasi; jangan gunakan --final.")
        from uji_simulasi import jalankan_simulasi

        jalankan_simulasi(args)
    else:
        del args.simulasi
        jalankan("123450037", "tl", args, buat_model)
