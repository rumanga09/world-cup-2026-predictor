"""
Latih model ML untuk prediksi final Piala Dunia 2026.

Dua model logistic-regression sederhana:
  - model_final : peluang sebuah tim LOLOS ke final
  - model_champ : peluang sebuah tim jadi JUARA

Dilatih dari puluhan edisi turnamen besar historis, lalu disimpan ke models/.
Jalankan:  python -m src.train     (atau:  python src/train.py)
"""

import os
import sys
import json
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import roc_auc_score

from src.features import load_matches, build_training_table, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )


def main():
    print("Memuat dataset & membangun fitur historis...")
    df = load_matches()
    table, elo, finals_count, titles_count = build_training_table(df)
    print(f"  {len(table)} baris tim-per-edisi dari "
          f"{table[['tournament','year']].drop_duplicates().shape[0]} edisi turnamen")

    X = table[FEATURE_COLS].values
    groups = (table["tournament"] + table["year"].astype(str)).values

    metrics = {}
    for target in ["y_final", "y_champ"]:
        y = table[target].values
        model = _make_model()
        # CV per-edisi: jangan bocorkan tim dari turnamen yang sama
        cv = GroupKFold(n_splits=5)
        try:
            auc = cross_val_score(model, X, y, groups=groups, cv=cv,
                                  scoring="roc_auc")
            metrics[target] = {"cv_auc_mean": float(np.mean(auc)),
                               "cv_auc_std": float(np.std(auc)),
                               "positives": int(y.sum())}
            print(f"  {target}: CV ROC-AUC = {np.mean(auc):.3f} "
                  f"(+/- {np.std(auc):.3f}), positif={int(y.sum())}")
        except Exception as e:
            print(f"  {target}: CV dilewati ({e})")
        model.fit(X, y)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{target}.joblib"))

    # simpan keadaan akhir (Elo & rekam jejak) supaya predict.py konsisten
    joblib.dump(
        {"elo": elo, "finals_count": finals_count, "titles_count": titles_count},
        os.path.join(MODELS_DIR, "state.joblib"),
    )
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel tersimpan di models/. Lanjut: python -m src.predict")


if __name__ == "__main__":
    main()
