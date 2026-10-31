"""Model Transfer Learning (MobileNetV2) untuk klasifikasi jumlah orang 0–3 dari citra 224x224.

Penggunaan:
    python models/123450037/run.py --seed 1
    python models/123450037/run.py --seed 1 --final
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402 (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name
JALUR = "tl"  # Transfer learning / fine-tuning (Modul 06)


def siapkan_citra(X: np.ndarray) -> torch.Tensor:
    """Format masukan citra menjadi [N, 3, 224, 224] bertipe float32."""
    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"Dimensi citra harus 4D [N, H, W, C] atau [N, C, H, W], diterima shape: {X.shape}")
    # Jika format NHWC (misal: [N, 224, 224, 3]), ubah ke NCHW
    if X.shape[-1] == 3:
        X = np.transpose(X, (0, 3, 1, 2))
    elif X.shape[1] != 3:
        raise ValueError(f"Citra harus memiliki 3 kanal (RGB), bentuk diterima: {X.shape}")

    tensor_X = torch.from_numpy(X.astype(np.float32))
    # Skalakan ke [0.0, 1.0] bila rentang nilai piksel masih uint8 [0, 255]
    if tensor_X.max() > 1.0:
        tensor_X = tensor_X / 255.0
    return tensor_X


def buat_model(jumlah_kelas: int = 4, pretrained: bool = True) -> nn.Module:
    """Membangun model Transfer Learning berbasis MobileNetV2."""
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    except Exception as e:
        print(f"[Peringatan] Gagal memuat bobot online: {e}. Menggunakan inisialisasi lokal.")
        model = models.mobilenet_v2(weights=None)

    # Ganti classifier head: Dropout(0.2) + Linear(in_features, 4)
    in_fitur = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_fitur, jumlah_kelas),
    )
    return model


def latih_dan_prediksi(
    seed: int,
    final: bool,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    perangkat: str = "cpu",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Melatih model Transfer Learning dan menghasilkan prediksi validasi (dan uji jika final)."""
    # 1. Tetapkan seed untuk reproduktibilitas
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and "gpu" in perangkat.lower() else "cpu")

    # 2. Muat data dari protokol tim
    try:
        Xtr, ytr, Xva, yva, Xte, yte = data.muat()
    except NotImplementedError:
        print("[Info] src/protokol/data.py belum diisi dengan data rekaman meja uji.")
        print("Silakan rekam/isi data primer meja uji atau jalankan demo starter.")
        raise SystemExit(0)

    # 3. Praproses: format citra dan normalisasi dari data latih saja (aturan kasus)
    Xtr_t = siapkan_citra(Xtr)
    Xva_t = siapkan_citra(Xva)
    Xte_t = siapkan_citra(Xte) if final and Xte is not None else None

    # Normalisasi channel-wise (mean dan std dihitung hanya dari X_train)
    mean = Xtr_t.mean(dim=(0, 2, 3), keepdim=True)
    std = Xtr_t.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    Xtr_t = (Xtr_t - mean) / std
    Xva_t = (Xva_t - mean) / std
    if Xte_t is not None:
        Xte_t = (Xte_t - mean) / std

    ytr_t = torch.tensor(np.asarray(ytr, dtype=np.int64))
    yva_t = torch.tensor(np.asarray(yva, dtype=np.int64))

    # 4. DataLoader
    dataset_latih = TensorDataset(Xtr_t, ytr_t)
    loader_latih = DataLoader(dataset_latih, batch_size=batch_size, shuffle=True)

    # 5. Inisialisasi model, optimizer, dan loss function
    model = buat_model(jumlah_kelas=4, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    # 6. Pelatihan
    model.train()
    for ep in range(epochs):
        kehilangan_total = 0.0
        for batch_x, batch_y in loader_latih:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            kehilangan_total += loss.item()

    # 7. Evaluasi & Inferensi
    model.eval()
    with torch.no_grad():
        Xva_dev = Xva_t.to(device)
        pred_val = model(Xva_dev).argmax(dim=1).cpu().numpy()

        pred_test = None
        if final and Xte_t is not None:
            Xte_dev = Xte_t.to(device)
            pred_test = model(Xte_dev).argmax(dim=1).cpu().numpy()

    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True, help="Seed acak untuk reproduksi")
    ap.add_argument("--final", action="store_true", help="Evaluasi data uji sekali di sprint terakhir")
    ap.add_argument("--epochs", type=int, default=5, help="Jumlah epoch pelatihan")
    ap.add_argument("--batch-size", type=int, default=16, help="Ukuran batch")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate optimizer")
    ap.add_argument("--perangkat", default="cpu laptop", help="Identitas perangkat komputasi")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(
        seed=a.seed,
        final=a.final,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        perangkat=a.perangkat,
    )
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))

    perintah = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
