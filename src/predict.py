"""
Prediksi JUARA + 2 FINALIS Piala Dunia 2026 pakai model yang sudah dilatih.

Jalankan:  python -m src.predict     (atau:  python src/predict.py)
Output: ringkasan di terminal + outputs/wc2026_final_prediction.json
"""

import os
import sys
import json
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (
    load_matches, build_prediction_table, wc2026_teams, FEATURE_COLS,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs",
                        "wc2026_final_prediction.json")

# Tuan rumah WC 2026 (tiga negara).
HOST_TEAMS = {"United States", "Canada", "Mexico"}


def main():
    for f in ["y_final.joblib", "y_champ.joblib", "state.joblib"]:
        if not os.path.exists(os.path.join(MODELS_DIR, f)):
            raise SystemExit("Model belum ada. Jalankan dulu: python -m src.train")

    model_final = joblib.load(os.path.join(MODELS_DIR, "y_final.joblib"))
    model_champ = joblib.load(os.path.join(MODELS_DIR, "y_champ.joblib"))
    state = joblib.load(os.path.join(MODELS_DIR, "state.joblib"))

    df = load_matches()
    teams = wc2026_teams(df)
    pred = build_prediction_table(
        df, state["elo"], state["finals_count"], state["titles_count"],
        teams, HOST_TEAMS,
    )

    X = pred[FEATURE_COLS].values
    p_final = model_final.predict_proba(X)[:, 1]
    p_champ = model_champ.predict_proba(X)[:, 1]

    # Normalisasi jadi probabilitas yang masuk akal lintas 48 tim:
    # tepat 1 juara, tepat 2 finalis.
    pred["prob_champion"] = p_champ / p_champ.sum()
    pred["prob_finalist"] = 2 * p_final / p_final.sum()
    pred = pred.sort_values("prob_champion", ascending=False).reset_index(drop=True)

    finalists = pred.sort_values("prob_finalist", ascending=False).head(2)
    champion = pred.iloc[0]

    print("\n" + "=" * 52)
    print("  PREDIKSI FINAL PIALA DUNIA 2026")
    print("=" * 52)
    print(f"  Juara    : {champion['team']:<22} "
          f"{champion['prob_champion']*100:5.1f}%")
    print(f"  Finalis  : {finalists.iloc[0]['team']} vs {finalists.iloc[1]['team']}")
    print("-" * 52)
    print("  Top 10 peluang juara:")
    for i, r in pred.head(10).iterrows():
        bar = "#" * int(r["prob_champion"] * 100)
        print(f"   {i+1:>2}. {r['team']:<22} {r['prob_champion']*100:5.1f}% {bar}")

    result = {
        "model": "logistic regression (champion + finalist), team-level",
        "predicted_champion": {
            "team": champion["team"],
            "prob": round(float(champion["prob_champion"]) * 100, 1),
        },
        "predicted_finalists": [
            {"team": r["team"], "prob": round(float(r["prob_finalist"]) * 100, 1)}
            for _, r in finalists.iterrows()
        ],
        "championship_probability": [
            {"team": r["team"], "prob": round(float(r["prob_champion"]) * 100, 1)}
            for _, r in pred.iterrows()
        ],
        "finalist_probability": [
            {"team": r["team"], "prob": round(float(r["prob_finalist"]) * 100, 1)}
            for _, r in pred.sort_values("prob_finalist", ascending=False).iterrows()
        ],
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print("=" * 52)
    print(f"  Tersimpan: outputs/wc2026_final_prediction.json")


if __name__ == "__main__":
    main()
