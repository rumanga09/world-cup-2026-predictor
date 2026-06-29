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
from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold

from src.features import load_matches, build_training_table, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )


def _topk_hit_rate(table, oof, target, ks=(1, 3, 5, 8)):
    """
    Seberapa sering tim sebenarnya (juara/finalis) masuk top-K prediksi
    di tiap edisi turnamen. Metrik ini lebih intuitif daripada ROC-AUC.
    """
    t = table.copy()
    t["_p"] = oof
    hits = {k: 0.0 for k in ks}
    n = 0
    for _, g in t.groupby(["tournament", "year"]):
        truth = g[g[target] == 1]
        if len(truth) == 0:
            continue
        n += 1
        order = g.sort_values("_p", ascending=False)["team"].tolist()
        truth_set = set(truth["team"])
        for k in ks:
            topk = set(order[:k])
            hits[k] += len(truth_set & topk) / len(truth_set)
    return {f"top{k}": round(hits[k] / n * 100, 1) for k in ks}, n


def main():
    print("Memuat dataset & membangun fitur historis...")
    df = load_matches()
    table, elo, finals_count, titles_count = build_training_table(df)
    print(f"  {len(table)} baris tim-per-edisi dari "
          f"{table[['tournament','year']].drop_duplicates().shape[0]} edisi turnamen")

    X = table[FEATURE_COLS].values
    groups = (table["tournament"] + table["year"].astype(str)).values

    metrics = {}
    cv = GroupKFold(n_splits=5)
    labels = {"y_final": "FINALIS", "y_champ": "JUARA "}
    for target in ["y_final", "y_champ"]:
        y = table[target].values
        # CV per-edisi: jangan bocorkan tim dari turnamen yang sama
        auc = cross_val_score(_make_model(), X, y, groups=groups, cv=cv,
                              scoring="roc_auc")
        # prediksi out-of-fold untuk hitung hit-rate top-K
        oof = cross_val_predict(_make_model(), X, y, groups=groups, cv=cv,
                                method="predict_proba")[:, 1]
        topk, n_ed = _topk_hit_rate(table, oof, target)
        metrics[target] = {
            "cv_auc_mean": round(float(np.mean(auc)), 3),
            "cv_auc_std": round(float(np.std(auc)), 3),
            "topk_hit_rate_pct": topk,
            "positives": int(y.sum()),
            "editions": n_ed,
        }
        print(f"  {labels[target]}: ROC-AUC={np.mean(auc):.3f} "
              f"(+/-{np.std(auc):.3f}) | "
              f"hit-rate top1={topk['top1']}% top3={topk['top3']}% "
              f"top5={topk['top5']}% top8={topk['top8']}%")

        model = _make_model().fit(X, y)
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
