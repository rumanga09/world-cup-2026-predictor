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
| **fitur relatif** | `elo_z`, `elo_gap_best`, `elo_is_top1`, `elo_is_top3`, `form_rank`, `gd_rank` — kekuatan **dibanding peserta lain** di turnamen yang sama (paling menaikkan akurasi) |

### Model

Dua **logistic regression** sederhana (scikit-learn):
- `y_final` → peluang **lolos final**
- `y_champ` → peluang jadi **juara**

Dievaluasi dengan `GroupKFold` per-edisi turnamen (biar tidak bocor), pakai
**dua metrik**:

| Target | ROC-AUC | Juara/finalis asli masuk top-8 |
|--------|---------|--------------------------------|
| Juara  | ~0.84   | **95%** |
| Finalis| ~0.81   | **93%** |

> ROC-AUC sekitar 0.84 adalah batas wajar untuk menebak juara satu turnamen
> jauh-jauh hari — sepakbola memang penuh kejutan, jadi angka yang jauh lebih
> tinggi biasanya tanda *overfitting*. Karena itu disertakan juga **hit-rate
> top-K**: shortlist 8 tim teratas model memuat juara sebenarnya ~95% edisi.

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
