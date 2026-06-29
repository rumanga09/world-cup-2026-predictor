# World Cup 2026 Predictor

Project machine learning **sederhana** untuk memprediksi **juara + 2 finalis**
Piala Dunia 2026 — langsung di level tim, **tanpa simulasi per pertandingan**.

## Ide

Satu baris data = **satu tim di satu turnamen besar** (World Cup, Euro, Copa
América, Piala Afrika). Dari riwayat pertandingan internasional *sebelum*
turnamen dimulai, dihitung beberapa fitur kekuatan tim, lalu model belajar:
"tim dengan profil seperti ini, seberapa besar peluangnya lolos final / juara?"

Model lalu diterapkan ke 48 peserta WC 2026 untuk menghasilkan peluang juara
dan finalis masing-masing tim.

### Fitur per tim

| Fitur | Penjelasan |
|-------|-----------|
| `elo`, `elo_rank` | rating Elo menjelang turnamen + peringkatnya di antara peserta |
| `win_rate`, `gf_avg`, `ga_avg`, `gd_avg` | form 4 tahun terakhir (menang, gol, selisih gol) |
| `n_matches` | jumlah laga (jam terbang) |
| `is_host` | apakah tuan rumah |
| `past_finals`, `past_titles` | rekam jejak: berapa kali sampai final / juara |

### Model

Dua **logistic regression** sederhana (scikit-learn):
- `y_final` → peluang **lolos final**
- `y_champ` → peluang jadi **juara**

Dievaluasi dengan `GroupKFold` per-edisi turnamen (biar tidak bocor).
ROC-AUC ~0.79 (final) dan ~0.83 (juara).

## Cara pakai

```bash
pip install -r requirements.txt

python -m src.train      # latih model -> models/
python -m src.predict    # prediksi -> outputs/wc2026_final_prediction.json
```

## Struktur

```
data/results.csv                      # dataset historis (1872-2026)
src/features.py                       # load data + Elo + feature engineering
src/train.py                          # latih & simpan model
src/predict.py                        # prediksi juara + 2 finalis
outputs/wc2026_final_prediction.json  # hasil prediksi
```

## Data

`data/results.csv` dari dataset publik
[martj42/international_results](https://github.com/martj42/international_results).
Untuk memperbarui:

```bash
curl -L -o data/results.csv \
  https://raw.githubusercontent.com/martj42/international_results/master/results.csv
```

> Catatan: ini model statistik sederhana untuk belajar, bukan alat judi.
