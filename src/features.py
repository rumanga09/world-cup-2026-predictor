"""
Feature engineering untuk prediksi final Piala Dunia 2026.

Ide intinya SEDERHANA dan TANPA simulasi per-match:
satu baris data = satu tim di satu turnamen besar (World Cup / Euro / Copa /
Piala Afrika). Dari riwayat pertandingan SEBELUM turnamen itu dimulai, kita
hitung beberapa fitur kekuatan tim (Elo, form, pengalaman, status tuan rumah,
rekam jejak juara). Label-nya: apakah tim itu sampai final, dan apakah jadi juara.

Model belajar dari puluhan edisi turnamen historis, lalu dipakai untuk
memprediksi langsung siapa juara + 2 finalis WC 2026 — bukan per pertandingan.
"""

import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.csv")

# Turnamen besar yang dipakai sebagai contoh "final" untuk training.
MAJOR_TOURNAMENTS = [
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
]

# Bobot K-factor Elo per kompetisi (makin penting, makin besar pengaruhnya).
COMP_K = {
    "FIFA World Cup": 60,
    "UEFA Euro": 50,
    "Copa América": 50,
    "African Cup of Nations": 40,
    "AFC Asian Cup": 40,
    "FIFA World Cup qualification": 30,
    "UEFA Nations League": 25,
}
DEFAULT_K = 20
DEFAULT_ELO = 1500.0

# Final yang berakhir imbang dan ditentukan adu penalti — skor saja tak cukup
# untuk menebak juara, jadi kita override manual.
PENALTY_CHAMPIONS = {
    ("FIFA World Cup", 1994): "Brazil",
    ("FIFA World Cup", 2006): "Italy",
    ("FIFA World Cup", 2022): "Argentina",
    ("UEFA Euro", 1976): "Czechoslovakia",
    ("UEFA Euro", 2020): "Italy",
    ("Copa América", 2015): "Chile",
    ("Copa América", 2016): "Chile",
    ("Copa América", 2004): "Brazil",
}

# Samakan nama lama -> nama sekarang supaya rekam jejak juara tetap nyambung.
TEAM_ALIASES = {
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Soviet Union": "Russia",
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Zaire": "DR Congo",
    "Republic of Ireland": "Ireland",
}

# Jendela form: berapa tahun ke belakang sebelum turnamen yang dihitung.
FORM_WINDOW_YEARS = 4


def _canon(name):
    return TEAM_ALIASES.get(name, name)


def load_matches(path=DATA_PATH):
    """Muat & rapikan dataset hasil pertandingan internasional."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["home_team"] = df["home_team"].map(_canon)
    df["away_team"] = df["away_team"].map(_canon)
    df["country"] = df["country"].map(_canon)
    return df.sort_values("date").reset_index(drop=True)


def _elo_expected(rh, ra):
    return 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))


def _k_factor(tournament):
    for comp, k in COMP_K.items():
        if comp.lower() in str(tournament).lower():
            return k
    return DEFAULT_K


def find_tournament_editions(df):
    """
    Temukan tiap edisi turnamen besar yang sudah selesai, beserta:
    peserta, tuan rumah, dua finalis, dan juara.
    """
    editions = []
    played = df[df["home_score"].notna()]
    for tourn in MAJOR_TOURNAMENTS:
        sub = played[played["tournament"] == tourn]
        for year, g in sub.groupby("year"):
            g = g.sort_values("date")
            teams = sorted(set(g["home_team"]) | set(g["away_team"]))
            if len(teams) < 4:
                continue
            final = g.iloc[-1]
            finalists = [final["home_team"], final["away_team"]]
            champ = PENALTY_CHAMPIONS.get((tourn, year))
            if champ is None:
                if final["home_score"] > final["away_score"]:
                    champ = final["home_team"]
                elif final["away_score"] > final["home_score"]:
                    champ = final["away_team"]
                else:
                    # final seri tanpa override -> lewati edisi ini
                    continue
            host = g["country"].mode()
            host = host.iloc[0] if len(host) else None
            editions.append({
                "tournament": tourn,
                "year": int(year),
                "start": g["date"].min(),
                "teams": teams,
                "host": host,
                "finalists": finalists,
                "champion": champ,
            })
    editions.sort(key=lambda e: e["start"])
    return editions


def _form_features(df, team, start, window_years=FORM_WINDOW_YEARS):
    """Statistik form satu tim dari pertandingan dalam jendela sebelum `start`."""
    lo = start - pd.DateOffset(years=window_years)
    m = df[(df["date"] >= lo) & (df["date"] < start) & df["home_score"].notna()
           & ((df["home_team"] == team) | (df["away_team"] == team))]
    if len(m) == 0:
        return dict(win_rate=0.0, gf_avg=0.0, ga_avg=0.0, gd_avg=0.0, n_matches=0)
    gf = np.where(m["home_team"] == team, m["home_score"], m["away_score"]).astype(float)
    ga = np.where(m["home_team"] == team, m["away_score"], m["home_score"]).astype(float)
    wins = (gf > ga).sum()
    return dict(
        win_rate=wins / len(m),
        gf_avg=gf.mean(),
        ga_avg=ga.mean(),
        gd_avg=(gf - ga).mean(),
        n_matches=len(m),
    )


FEATURE_COLS = [
    "elo", "elo_rank", "win_rate", "gf_avg", "ga_avg", "gd_avg",
    "n_matches", "is_host", "past_finals", "past_titles",
]


def build_training_table(df):
    """
    Bangun tabel fitur (satu baris per tim-per-edisi) + label final/juara,
    sambil menjalankan Elo secara kronologis melintasi seluruh sejarah.
    """
    editions = find_tournament_editions(df)
    elo = {}
    # rekam jejak kumulatif sebelum tiap edisi
    finals_count = {}
    titles_count = {}

    rows = []
    ed_idx = 0
    snapshots = {}  # id(edition) -> {team: elo saat turnamen mulai}

    played = df[df["home_score"].notna()].reset_index(drop=True)

    for _, mt in played.iterrows():
        d = mt["date"]
        # snapshot Elo untuk semua edisi yang mulai <= tanggal match ini
        while ed_idx < len(editions) and editions[ed_idx]["start"] <= d:
            ed = editions[ed_idx]
            snapshots[id(ed)] = {t: elo.get(t, DEFAULT_ELO) for t in ed["teams"]}
            ed_idx += 1

        h, a = mt["home_team"], mt["away_team"]
        rh, ra = elo.get(h, DEFAULT_ELO), elo.get(a, DEFAULT_ELO)
        exp_h = _elo_expected(rh, ra)
        sh = 1.0 if mt["home_score"] > mt["away_score"] else (
            0.5 if mt["home_score"] == mt["away_score"] else 0.0)
        k = _k_factor(mt["tournament"])
        elo[h] = rh + k * (sh - exp_h)
        elo[a] = ra + k * ((1 - sh) - (1 - exp_h))

    # snapshot sisa edisi (kalau ada yang mulai setelah match terakhir)
    while ed_idx < len(editions):
        ed = editions[ed_idx]
        snapshots[id(ed)] = {t: elo.get(t, DEFAULT_ELO) for t in ed["teams"]}
        ed_idx += 1

    # bangun baris fitur per edisi (pakai snapshot Elo + rekam jejak kumulatif)
    for ed in editions:
        snap = snapshots[id(ed)]
        elos = snap
        ranked = sorted(ed["teams"], key=lambda t: elos.get(t, DEFAULT_ELO), reverse=True)
        for t in ed["teams"]:
            form = _form_features(df, t, ed["start"])
            rows.append({
                "tournament": ed["tournament"],
                "year": ed["year"],
                "team": t,
                "elo": elos.get(t, DEFAULT_ELO),
                "elo_rank": ranked.index(t) / max(1, len(ranked) - 1),
                "is_host": 1 if t == ed["host"] else 0,
                "past_finals": finals_count.get(t, 0),
                "past_titles": titles_count.get(t, 0),
                "y_final": 1 if t in ed["finalists"] else 0,
                "y_champ": 1 if t == ed["champion"] else 0,
                **form,
            })
        # update rekam jejak SETELAH edisi ini diproses
        for t in ed["finalists"]:
            finals_count[t] = finals_count.get(t, 0) + 1
        titles_count[ed["champion"]] = titles_count.get(ed["champion"], 0) + 1

    table = pd.DataFrame(rows)
    return table, elo, finals_count, titles_count


def build_prediction_table(df, elo, finals_count, titles_count,
                           teams, host_teams, start="2026-06-01"):
    """Bangun tabel fitur untuk peserta WC 2026 (tanpa label)."""
    start = pd.to_datetime(start)
    ranked = sorted(teams, key=lambda t: elo.get(t, DEFAULT_ELO), reverse=True)
    rows = []
    for t in teams:
        form = _form_features(df, t, start)
        rows.append({
            "team": t,
            "elo": elo.get(t, DEFAULT_ELO),
            "elo_rank": ranked.index(t) / max(1, len(ranked) - 1),
            "is_host": 1 if t in host_teams else 0,
            "past_finals": finals_count.get(t, 0),
            "past_titles": titles_count.get(t, 0),
            **form,
        })
    return pd.DataFrame(rows)


def wc2026_teams(df):
    """Ambil 48 peserta WC 2026 langsung dari fixtures di dataset."""
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["year"] == 2026)]
    return sorted(set(wc["home_team"]) | set(wc["away_team"]))
