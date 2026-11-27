"""Pelatihan CPU, normalisasi, dan penyimpanan artefak jalur citra tim."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.eksperimen import informasi_data, sidik_data, simpan_hasil
from src.protokol import data


def argumen(epochs=6, lr=3e-4, dropout=0.1):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=dropout)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--perangkat", default="cpu laptop")
    parser.add_argument("--final", action="store_true", help="Evaluasi uji pada sprint terakhir.")
    return parser


def citra(x):
    if x.dtype != np.uint8 or x.ndim != 4 or not len(x):
        raise ValueError("Masukan wajib berisi citra RGB uint8.")
    if x.shape[1:] == (224, 224, 3):
        x = x.transpose(0, 3, 1, 2)
    if x.shape[1:] != (3, 224, 224):
        raise ValueError("Bentuk masukan bukan citra RGB 224×224.")
    return torch.from_numpy(np.ascontiguousarray(x))


def label(y, n):
    if y.shape != (n,) or not np.issubdtype(y.dtype, np.integer) or not np.isin(y, range(4)).all():
        raise ValueError("Label harus berupa vektor kelas integer 0–3.")
    return torch.from_numpy(y.astype(np.int64, copy=False))


def statistik_latih(x, batch_size):
    total = torch.zeros(3, dtype=torch.float64)
    kuadrat = torch.zeros_like(total)
    for batch in x.split(batch_size):
        batch = batch.to(torch.float64) / 255
        total += batch.sum((0, 2, 3))
        kuadrat += batch.square().sum((0, 2, 3))
    n = len(x) * x.shape[2] * x.shape[3]
    mean = total / n
    std = (kuadrat / n - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    return mean.float(), std.float()


class Normalisasi(nn.Module):
    def __init__(self, mean, std, model):
        super().__init__()
        self.register_buffer("mean", mean.reshape(1, 3, 1, 1))
        self.register_buffer("std", std.reshape(1, 3, 1, 1))
        self.model = model

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


@torch.inference_mode()
def prediksi(model, x, batch_size):
    model.eval()
    return np.concatenate([
        model(batch.float() / 255).argmax(1).numpy() for batch in x.split(batch_size)
    ])


def jalankan(nim, jalur, args, buat_model):
    if not 0 <= args.seed < 2**32 or min(args.epochs, args.batch_size, args.threads) < 1:
        raise SystemExit("Seed harus berada di [0, 2³²). Epoch, batch dan thread minimal 1.")
    if (not np.isfinite([args.lr, args.weight_decay, args.dropout]).all()
            or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1):
        raise SystemExit("LR harus positif, weight decay tidak negatif, dan dropout di [0, 1).")
    if args.final and informasi_data()["jenis"] != "primer":
        raise SystemExit("Uji final memerlukan data primer.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    data_hash = sidik_data()
    xtr, ytr, xva, yva, xte, yte = data.muat()
    xtr, xva = citra(xtr), citra(xva)
    ytr, yva = label(ytr, len(xtr)), label(yva, len(xva))
    mean, std = statistik_latih(xtr, args.batch_size)
    model = buat_model(mean, std, args)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True)
    riwayat = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.float() / 255), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        riwayat.append(total / len(xtr))
        print(f"{jalur} epoch {epoch + 1}/{args.epochs}: loss_latih={riwayat[-1]:.6f}", flush=True)

    val = prediksi(model, xva, args.batch_size), yva.numpy()
    uji = None
    if args.final:
        xte = citra(xte)
        uji = prediksi(model, xte, args.batch_size), label(yte, len(xte)).numpy()
    detail = {
        "jumlah_parameter": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_dilatih": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "loss_latih": riwayat, "normalisasi_mean": mean.tolist(), "normalisasi_std": std.tolist(),
    }
    run = simpan_hasil(nim, jalur, vars(args), data_hash, time.perf_counter() - t0, val, uji, detail)
    torch.save({"state_dict": model.state_dict(), "konfigurasi": vars(args)}, run / "model.pt")
