# Kode dan hasil model

Jalur model menerima citra RGB dan label okupansi 0–3 dari pemuat data bersama. Skrip model menyimpan label prediksi di `pred_val.npy`. Evaluator mencatat akurasi dan macro-F1 dalam `metrics.json`. Untuk MobileNetV2, CNN, Transformer, dan GRU, paket juga memuat `detail.json` dengan MAE, konfigurasi, dan riwayat loss.

| Model | Kode | Hasil dalam paket |
|---|---|---|
| MobileNetV2 (123450037) | [run.py](123450037/run.py) | [Metrik](../results/123450037/metrics.json) · [Detail](../results/123450037/detail.json) |
| CNN (123450040) | [run.py](123450040/run.py) | [Metrik](../results/123450040/metrics.json) · [Detail](../results/123450040/detail.json) |
| Transformer (123450078) | [run.py](123450078/run.py) | [Metrik](../results/123450078/metrics.json) · [Detail](../results/123450078/detail.json) |
| GRU (123450091) | [run.py](123450091/run.py) | [Metrik](../results/123450091/metrics.json) · [Detail](../results/123450091/detail.json) |

Metrik [Random Forest](../results/baseline/metrics.json) dan seluruh jalur dari run sebelumnya tercatat di [laporan komparasi](../reports/hasil_komparasi.md).
