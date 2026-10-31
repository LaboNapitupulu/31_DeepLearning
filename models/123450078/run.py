"""Transformer (masih skala kecil) untuk klasifikasi jumlah orang 0–3 dari citra RGB 224×224.

    python models/123450078/run.py --seed 1
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.common import manifest, split  # noqa: E402
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name


def citra(X: np.ndarray) -> torch.Tensor:
    """Kontrak masukan: RGB uint8, NHWC atau NCHW; tidak mengubah split."""
    X = np.asarray(X)
    if X.dtype != np.uint8 or X.ndim != 4 or len(X) == 0:
        raise ValueError("Citra harus berupa array uint8 RGB yang tidak kosong.")
    if X.shape[1:] == (224, 224, 3):
        X = X.transpose(0, 3, 1, 2)
    elif X.shape[1:] != (3, 224, 224):
        raise ValueError("Bentuk citra harus [N, 224, 224, 3] atau [N, 3, 224, 224].")
    return torch.from_numpy(np.ascontiguousarray(X))


def label(y: np.ndarray, n: int) -> torch.Tensor:
    y = np.asarray(y)
    if y.shape != (n,) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Label harus array integer [N] berisi jumlah orang 0–3.")
    if not np.isin(y, [0, 1, 2, 3]).all():
        raise ValueError("Label jumlah orang harus 0, 1, 2, atau 3.")
    return torch.from_numpy(y.astype(np.int64))


def statistik_latih(X: torch.Tensor, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Hitung statistik kanal hanya dari data latih, tanpa menyalin seluruhnya ke float."""
    jumlah = torch.zeros(3, dtype=torch.float64)
    kuadrat = torch.zeros(3, dtype=torch.float64)
    for x in X.split(batch):
        x = x.to(torch.float64) / 255
        jumlah += x.sum((0, 2, 3))
        kuadrat += x.square().sum((0, 2, 3))
    n = len(X) * X.shape[2] * X.shape[3]
    mean = jumlah / n
    std = (kuadrat / n - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    return mean.float(), std.float()


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


@torch.inference_mode()
def prediksi(model: nn.Module, X: torch.Tensor, batch: int) -> np.ndarray:
    model.eval()
    return np.concatenate([
        model(x.float() / 255).argmax(1).numpy() for x in X.split(batch)
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--perangkat", default="cpu laptop")
    ap.add_argument("--final", action="store_true", help="evaluasi uji oleh Penjaga Data, sekali")
    a = ap.parse_args()
    if min(a.epochs, a.batch_size, a.threads) < 1 or a.lr <= 0 or a.weight_decay < 0:
        ap.error("epochs, batch-size, threads, lr harus positif; weight-decay tidak negatif")
    if not 0 <= a.dropout < 1:
        ap.error("dropout harus dalam [0, 1)")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    t0 = time.perf_counter()
    try:
        # Periksa data asli sebelum membuat berkas hasil atau memulai pelatihan.
        data_hash = manifest.data_hash(f"split:{split.SEED_SPLIT}".encode())
        Xtr, ytr, Xva, yva, Xte, yte = data.muat()
    except (FileNotFoundError, NotImplementedError) as exc:
        raise SystemExit(
            "Belum bisa melatih Transformer: rekaman asli, MANIFEST.json, dan pemuat "
            "data tim dengan split kelompok harus tersedia. " + str(exc)
        ) from exc
    if data.TUGAS != "klasifikasi":
        raise SystemExit("Kasus jumlah orang 0–3 menggunakan TUGAS = 'klasifikasi'.")
    Xtr, Xva = citra(Xtr), citra(Xva)
    ytr, yva = label(ytr, len(Xtr)), label(yva, len(Xva))
    if a.final:
        if Xte is None or yte is None:
            raise SystemExit("Protokol tim belum menyediakan citra dan label uji.")
        Xte = citra(Xte)
        yte = label(yte, len(Xte))

    mean, std = statistik_latih(Xtr, a.batch_size)
    model = TransformerOrang(mean, std, a.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=a.batch_size, shuffle=True)
    riwayat = []
    for epoch in range(a.epochs):
        model.train()
        loss_total = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.float() / 255), y)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(y)
        riwayat.append(loss_total / len(Xtr))
        print(f"epoch {epoch + 1}/{a.epochs}: loss_latih={riwayat[-1]:.6f}", flush=True)

    pv = prediksi(model, Xva, a.batch_size)
    pt = prediksi(model, Xte, a.batch_size) if a.final else None
    durasi = time.perf_counter() - t0
    out = AKAR / "results" / NIM
    # Setiap seed punya direktori sendiri agar prediksi dan checkpoint tetap cocok.
    run = out / f"seed_{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    np.save(run / "pred_val.npy", pv)
    perintah = [
        sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", "trf",
        "--seed", str(a.seed), "--pred-val", str(run / "pred_val.npy"),
        "--out", str(run / "metrics.json"), "--durasi", f"{durasi:.3f}",
        "--perangkat", a.perangkat,
    ]
    tambahan = {"val_mae": float(np.abs(pv - yva.numpy()).mean())}
    if pt is not None:
        np.save(run / "pred_test.npy", pt)
        perintah += ["--pred-test", str(run / "pred_test.npy")]
        tambahan["test_mae"] = float(np.abs(pt - yte.numpy()).mean())
    subprocess.run(perintah, check=True, cwd=AKAR)
    hasil = json.loads((run / "metrics.json").read_text("utf-8"))
    if hasil["data_hash"] != data_hash:
        raise SystemExit("Data berubah selama pelatihan; hasil belum dipublikasikan ke metrics.json utama.")
    torch.save({"state_dict": model.state_dict(), "konfigurasi": vars(a)}, run / "model.pt")
    detail = {
        "seed": a.seed, "data_hash": data_hash, "metrik_tambahan": tambahan,
        "konfigurasi": vars(a), "loss_latih": riwayat,
        "jumlah_parameter": sum(p.numel() for p in model.parameters()),
        "normalisasi_mean": mean.tolist(), "normalisasi_std": std.tolist(),
    }
    (run / "detail.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(run / "pred_val.npy", out / "pred_val.npy")
    if pt is not None:
        shutil.copyfile(run / "pred_test.npy", out / "pred_test.npy")
    elif (out / "pred_test.npy").exists():
        (out / "pred_test.npy").unlink()
    shutil.copyfile(run / "metrics.json", out / "metrics.json")
    print(f"Hasil seed {a.seed}: {out / 'metrics.json'}; MAE: {tambahan}")


if __name__ == "__main__":
    main()
