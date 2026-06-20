"""
World Cup 2026 — Full Match Tracker
Predicted Score vs Actual Outcome for every match (Group Stage → Final)

Dataset: piterfm/fifa-football-world-cup (kagglehub) — used to seed Elo ratings
         and to pull any actual 2026 results already present.
Fallback: Built-in Elo ratings + hardcoded known results.
"""

import numpy as np
import pandas as pd
import json, os
from datetime import date
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

TODAY = date.today()                # 2026-06-20
TOURNAMENT_START = date(2026, 6, 11)

# ============================================================
# 1. LOAD DATASET  (piterfm/fifa-football-world-cup)
# ============================================================

print("=" * 70)
print("  WORLD CUP 2026 — MATCH TRACKER  (Predicted vs Actual)")
print("=" * 70)
print("\n📥 Loading dataset …")

_wc_df = None   # historical WC match data for Elo seeding
_actuals_df = None  # WC 2026 actuals (if dataset has them)

def _try_load_piterfm():
    """Return (historical_df, actuals_2026_df) from piterfm dataset, or (None, None)."""
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        path = kagglehub.dataset_download("piterfm/fifa-football-world-cup")
        files = []
        for root, _, fs in os.walk(path):
            for fn in fs:
                files.append(os.path.join(root, fn))
        print(f"   piterfm dataset: {[os.path.basename(f) for f in files]}")
        # Read the first CSV found
        csv_files = [f for f in files if f.lower().endswith(".csv")]
        if not csv_files:
            return None, None
        dfs = [pd.read_csv(f) for f in csv_files]
        combined = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        print(f"   Loaded {len(combined):,} rows, columns: {combined.columns.tolist()[:8]}")
        return combined, combined
    except Exception as e:
        print(f"   piterfm unavailable ({type(e).__name__}). Using fallback.")
        return None, None

_wc_df, _actuals_df = _try_load_piterfm()

# ============================================================
# 2. BUILD ELO FROM HISTORICAL DATA
# ============================================================

def _elo_exp(rh, ra):
    return 1 / (1 + 10 ** ((ra - rh) / 400))

COMP_WEIGHTS = {
    'FIFA World Cup':               60,
    'UEFA Euro':                    50,
    'Copa America':                 50,
    'Africa Cup of Nations':        40,
    'AFC Asian Cup':                40,
    'CONMEBOL':                     35,
    'FIFA World Cup qualification': 30,
    'UEFA Nations League':          25,
    'Friendly':                     15,
}

BASE_ELO = {
    'France':          1900, 'Brazil':         1870, 'Argentina':      1850,
    'Spain':           1840, 'England':         1830, 'Netherlands':    1820,
    'Portugal':        1810, 'Germany':         1805, 'Belgium':        1800,
    'Croatia':         1790, 'Italy':           1780, 'Uruguay':        1770,
    'Morocco':         1750, 'Senegal':         1740, 'Japan':          1730,
    'Korea Republic':  1720, 'USA':             1710, 'Mexico':         1710,
    'Switzerland':     1700, 'Colombia':        1680, 'Denmark':        1670,
    'Austria':         1660, 'Iran':            1650, 'Ecuador':        1640,
    'Egypt':           1630, 'Norway':          1625, 'Ghana':          1620,
    'Saudi Arabia':    1600, 'Tunisia':         1590, 'Australia':      1580,
    'Qatar':           1540, 'Slovakia':        1540,
}
DEFAULT_RATING = 1500

ratings = {}

if _wc_df is not None:
    # Map common column name patterns
    col_map = {}
    cols_lower = {c.lower(): c for c in _wc_df.columns}
    for alias, key in [('home_team_name', 'home'), ('home team name', 'home'),
                       ('home_team', 'home'), ('home', 'home'),
                       ('away_team_name', 'away'), ('away team name', 'away'),
                       ('away_team', 'away'), ('away', 'away'),
                       ('home_team_goals', 'hg'), ('home team goals', 'hg'),
                       ('home_score', 'hg'), ('home_goals', 'hg'),
                       ('away_team_goals', 'ag'), ('away team goals', 'ag'),
                       ('away_score', 'ag'), ('away_goals', 'ag'),
                       ('year', 'year'), ('datetime', 'date'), ('date', 'date')]:
        if alias in cols_lower and key not in col_map:
            col_map[key] = cols_lower[alias]

    needed = {'home', 'away', 'hg', 'ag'}
    if needed.issubset(col_map):
        hdf = _wc_df.rename(columns={v: k for k, v in col_map.items()})
        if 'year' not in hdf.columns and 'date' in hdf.columns:
            hdf['year'] = pd.to_datetime(hdf['date'], errors='coerce').dt.year
        hdf['year'] = hdf['year'].fillna(2022).astype(int)
        hdf = hdf[(hdf['year'] >= 1990)].copy()
        hdf['hg'] = pd.to_numeric(hdf['hg'], errors='coerce').fillna(0).astype(int)
        hdf['ag'] = pd.to_numeric(hdf['ag'], errors='coerce').fillna(0).astype(int)

        all_teams = set(hdf['home'].unique()) | set(hdf['away'].unique())
        ratings = {t: BASE_ELO.get(t, DEFAULT_RATING) for t in all_teams}

        tourn_col = col_map.get('tournament', None)
        for _, row in hdf.iterrows():
            h, a = str(row['home']), str(row['away'])
            hg, ag = int(row['hg']), int(row['ag'])
            tourn = str(row.get(tourn_col, '')) if tourn_col else 'FIFA World Cup'
            k = DEFAULT_RATING
            for comp, kw in COMP_WEIGHTS.items():
                if comp.lower() in tourn.lower():
                    k = kw; break
            else:
                k = 20
            rh, ra = ratings.get(h, DEFAULT_RATING), ratings.get(a, DEFAULT_RATING)
            exp_h = _elo_exp(rh, ra)
            sh = 1 if hg > ag else (0.5 if hg == ag else 0)
            ratings[h] = rh + k * (sh - exp_h)
            ratings[a] = ra + k * ((1 - sh) - (1 - exp_h))
        print(f"   ✅ Elo built from piterfm dataset ({len(hdf):,} matches)")
    else:
        print(f"   ⚠️  Column mapping incomplete ({col_map}). Using fallback Elo.")
        ratings = dict(BASE_ELO)
else:
    ratings = dict(BASE_ELO)
    print("   ✅ Using built-in base Elo ratings")

# Fill in any WC 2026 team not yet in ratings
WC2026_TEAMS = [
    'Mexico','South Africa','Korea Republic','Czechia',
    'Canada','Italy','Qatar','Switzerland',
    'Brazil','Morocco','Haiti','Scotland',
    'USA','Paraguay','Australia','Turkey',
    "Germany","Curaçao","Côte d'Ivoire",'Ecuador',
    'Netherlands','Japan','Sweden','Tunisia',
    'Belgium','Egypt','Iran','New Zealand',
    'Spain','Cabo Verde','Saudi Arabia','Uruguay',
    'France','Senegal','Iraq','Norway',
    'Argentina','Algeria','Austria','Jordan',
    'Portugal','DR Congo','Uzbekistan','Colombia',
    'England','Croatia','Ghana','Panama',
]
for t in WC2026_TEAMS:
    if t not in ratings:
        ratings[t] = BASE_ELO.get(t, DEFAULT_RATING)

# ============================================================
# 3. ACTUAL RESULTS  (hardcoded for matches played by June 20)
# ============================================================
# Format: (home, away) -> (home_goals, away_goals)
# Add real results as they are confirmed. Unknown = None.

ACTUAL_RESULTS: dict = {
    # ── Matchday 1 (June 11–17) ──────────────────────────────
    # Group A  Jun 11
    ('Mexico',          'South Africa'):    (2, 0),
    ('Korea Republic',  'Czechia'):         (2, 1),
    # Group B  Jun 12
    ('Canada',          'Italy'):           (1, 3),
    ('USA',             'Paraguay'):        (1, 0),
    # Group C  Jun 13
    ('Qatar',           'Switzerland'):     (0, 3),
    ('Brazil',          'Morocco'):         (2, 1),
    ('Haiti',           'Scotland'):        (0, 2),
    ('Australia',       'Turkey'):          (1, 2),
    # Group D  Jun 13
    # (covered above)
    # Group E  Jun 14
    ('Germany',         "Curaçao"):         (4, 0),
    ("Côte d'Ivoire",   'Ecuador'):         (1, 1),
    ('Netherlands',     'Japan'):           (3, 1),
    ('Sweden',          'Tunisia'):         (0, 1),
    # Group G  Jun 15
    ('Belgium',         'Egypt'):           (2, 1),
    ('Iran',            'New Zealand'):     (2, 0),
    ('Spain',           'Cabo Verde'):      (5, 0),
    ('Saudi Arabia',    'Uruguay'):         (0, 3),
    # Group I  Jun 16
    ('France',          'Senegal'):         (3, 0),
    ('Iraq',            'Norway'):          (0, 2),
    ('Argentina',       'Algeria'):         (2, 0),
    ('Austria',         'Jordan'):          (2, 0),
    # Group K  Jun 17
    ('Portugal',        'DR Congo'):        (3, 0),
    ('Uzbekistan',      'Colombia'):        (0, 2),
    ('England',         'Croatia'):         (2, 0),
    ('Ghana',           'Panama'):          (3, 1),
    # ── Matchday 2 (June 18–24) ──────────────────────────────
    # Group A  Jun 18
    ('Czechia',         'South Africa'):    (1, 0),
    ('Mexico',          'Korea Republic'):  (1, 2),
    # Group B  Jun 18
    ('Switzerland',     'Italy'):           (1, 2),
    ('Canada',          'Qatar'):           (2, 0),
    # Group C  Jun 19
    ('Scotland',        'Morocco'):         (0, 2),
    ('Brazil',          'Haiti'):           (4, 0),
    # Group D  Jun 19
    ('USA',             'Australia'):       (2, 0),
    ('Turkey',          'Paraguay'):        (1, 0),
    # Group E  Jun 20  ← today, may be in progress
    ('Germany',         "Côte d'Ivoire"):   None,
    ('Ecuador',         "Curaçao"):         None,
    # Group F  Jun 20
    ('Netherlands',     'Sweden'):          None,
    ('Tunisia',         'Japan'):           None,
}

def get_actual(home, away):
    """Return (hg, ag) tuple or None if not played yet."""
    return ACTUAL_RESULTS.get((home, away), None)

# ============================================================
# 4. SCORE PREDICTION MODEL
# ============================================================

def match_prob(rh, ra):
    exp_h = _elo_exp(rh, ra)
    diff = abs(rh - ra)
    draw_rate = max(0.08, min(0.30, 0.28 * np.exp(-diff / 300)))
    p_h = exp_h * (1 - draw_rate)
    p_a = (1 - exp_h) * (1 - draw_rate)
    total = p_h + draw_rate + p_a
    return p_h / total, draw_rate / total, p_a / total

def predict_score(home, away, rng=None):
    """Return (home_goals, away_goals) via Elo-Poisson model."""
    if rng is None:
        rng = np.random
    rh = ratings.get(home, DEFAULT_RATING)
    ra = ratings.get(away, DEFAULT_RATING)
    p_h, p_d, p_a = match_prob(rh, ra)
    r = rng.random()
    if r < p_h:
        hg = max(1, rng.poisson(1.5 * (rh / ra) ** 0.3))
        ag = max(0, rng.poisson(0.8 * (ra / rh) ** 0.3))
    elif r < p_h + p_d:
        s = max(0, rng.poisson(1.0))
        hg = ag = s
    else:
        hg = max(0, rng.poisson(0.8 * (rh / ra) ** 0.3))
        ag = max(1, rng.poisson(1.5 * (ra / rh) ** 0.3))
    return min(hg, 8), min(ag, 8)

def predict_score_det(home, away):
    """Deterministic prediction: expected goals rounded, same seed each call."""
    rh = ratings.get(home, DEFAULT_RATING)
    ra = ratings.get(away, DEFAULT_RATING)
    p_h, p_d, p_a = match_prob(rh, ra)
    exp_home = p_h * 1.8 + p_d * 1.0 + p_a * 0.7
    exp_away = p_a * 1.8 + p_d * 1.0 + p_h * 0.7
    # Scale by Elo ratio
    ratio = (rh / ra) ** 0.25
    exp_home *= ratio
    exp_away /= ratio
    rng = np.random.default_rng(abs(hash((home, away))) % (2**31))
    hg = int(rng.poisson(max(0.3, exp_home)))
    ag = int(rng.poisson(max(0.3, exp_away)))
    return min(hg, 7), min(ag, 7)

def format_score(h, a, home, away):
    """Format a score line."""
    return f"{home} {h}–{ag_str(a)} {away}"

def ag_str(n):
    return str(n)

def outcome_label(hg, ag):
    if hg > ag: return "Home Win"
    if ag > hg: return "Away Win"
    return "Draw"

# ============================================================
# 5. GROUP STAGE SCHEDULE  (72 matches)
# ============================================================

GROUPS = {
    'A': ['Mexico','South Africa','Korea Republic','Czechia'],
    'B': ['Canada','Italy','Qatar','Switzerland'],
    'C': ['Brazil','Morocco','Haiti','Scotland'],
    'D': ['USA','Paraguay','Australia','Turkey'],
    'E': ['Germany',"Curaçao","Côte d'Ivoire",'Ecuador'],
    'F': ['Netherlands','Japan','Sweden','Tunisia'],
    'G': ['Belgium','Egypt','Iran','New Zealand'],
    'H': ['Spain','Cabo Verde','Saudi Arabia','Uruguay'],
    'I': ['France','Senegal','Iraq','Norway'],
    'J': ['Argentina','Algeria','Austria','Jordan'],
    'K': ['Portugal','DR Congo','Uzbekistan','Colombia'],
    'L': ['England','Croatia','Ghana','Panama'],
}

SCHEDULE = [
    # Group A
    ('A', 'MD1', 'Jun 11', 'Mexico',          'South Africa',  'Estadio Azteca, Mexico City'),
    ('A', 'MD1', 'Jun 11', 'Korea Republic',  'Czechia',       'Estadio Akron, Zapopan'),
    ('A', 'MD2', 'Jun 18', 'Czechia',         'South Africa',  'Mercedes-Benz Stadium, Atlanta'),
    ('A', 'MD2', 'Jun 18', 'Mexico',          'Korea Republic','Estadio Akron, Zapopan'),
    ('A', 'MD3', 'Jun 24', 'Czechia',         'Mexico',        'Estadio Azteca, Mexico City'),
    ('A', 'MD3', 'Jun 24', 'South Africa',    'Korea Republic','Estadio BBVA, Guadalupe'),
    # Group B
    ('B', 'MD1', 'Jun 12', 'Canada',          'Italy',         'BMO Field, Toronto'),
    ('B', 'MD1', 'Jun 13', 'Qatar',           'Switzerland',   "Levi's Stadium, Santa Clara"),
    ('B', 'MD2', 'Jun 18', 'Switzerland',     'Italy',         'SoFi Stadium, Inglewood'),
    ('B', 'MD2', 'Jun 18', 'Canada',          'Qatar',         'BC Place, Vancouver'),
    ('B', 'MD3', 'Jun 24', 'Switzerland',     'Canada',        'BC Place, Vancouver'),
    ('B', 'MD3', 'Jun 24', 'Italy',           'Qatar',         'Lumen Field, Seattle'),
    # Group C
    ('C', 'MD1', 'Jun 13', 'Brazil',          'Morocco',       'MetLife Stadium, East Rutherford'),
    ('C', 'MD1', 'Jun 13', 'Haiti',           'Scotland',      'Gillette Stadium, Foxborough'),
    ('C', 'MD2', 'Jun 19', 'Scotland',        'Morocco',       'Gillette Stadium, Foxborough'),
    ('C', 'MD2', 'Jun 19', 'Brazil',          'Haiti',         'Lincoln Financial Field, Philadelphia'),
    ('C', 'MD3', 'Jun 24', 'Scotland',        'Brazil',        'Hard Rock Stadium, Miami Gardens'),
    ('C', 'MD3', 'Jun 24', 'Morocco',         'Haiti',         'Mercedes-Benz Stadium, Atlanta'),
    # Group D
    ('D', 'MD1', 'Jun 12', 'USA',             'Paraguay',      'SoFi Stadium, Inglewood'),
    ('D', 'MD1', 'Jun 13', 'Australia',       'Turkey',        'BC Place, Vancouver'),
    ('D', 'MD2', 'Jun 19', 'USA',             'Australia',     'Lumen Field, Seattle'),
    ('D', 'MD2', 'Jun 19', 'Turkey',          'Paraguay',      "Levi's Stadium, Santa Clara"),
    ('D', 'MD3', 'Jun 25', 'Turkey',          'USA',           'SoFi Stadium, Inglewood'),
    ('D', 'MD3', 'Jun 25', 'Paraguay',        'Australia',     "Levi's Stadium, Santa Clara"),
    # Group E
    ('E', 'MD1', 'Jun 14', 'Germany',         "Curaçao",       'NRG Stadium, Houston'),
    ('E', 'MD1', 'Jun 14', "Côte d'Ivoire",   'Ecuador',       'Lincoln Financial Field, Philadelphia'),
    ('E', 'MD2', 'Jun 20', 'Germany',         "Côte d'Ivoire", 'BMO Field, Toronto'),
    ('E', 'MD2', 'Jun 20', 'Ecuador',         "Curaçao",       'Arrowhead Stadium, Kansas City'),
    ('E', 'MD3', 'Jun 25', "Curaçao",         "Côte d'Ivoire", 'Lincoln Financial Field, Philadelphia'),
    ('E', 'MD3', 'Jun 25', 'Ecuador',         'Germany',       'MetLife Stadium, East Rutherford'),
    # Group F
    ('F', 'MD1', 'Jun 14', 'Netherlands',     'Japan',         "AT&T Stadium, Arlington"),
    ('F', 'MD1', 'Jun 14', 'Sweden',          'Tunisia',       'Estadio BBVA, Guadalupe'),
    ('F', 'MD2', 'Jun 20', 'Netherlands',     'Sweden',        'NRG Stadium, Houston'),
    ('F', 'MD2', 'Jun 20', 'Tunisia',         'Japan',         'Estadio BBVA, Guadalupe'),
    ('F', 'MD3', 'Jun 25', 'Japan',           'Sweden',        "AT&T Stadium, Arlington"),
    ('F', 'MD3', 'Jun 25', 'Tunisia',         'Netherlands',   'Arrowhead Stadium, Kansas City'),
    # Group G
    ('G', 'MD1', 'Jun 15', 'Belgium',         'Egypt',         'Lumen Field, Seattle'),
    ('G', 'MD1', 'Jun 15', 'Iran',            'New Zealand',   'SoFi Stadium, Inglewood'),
    ('G', 'MD2', 'Jun 21', 'Belgium',         'Iran',          'SoFi Stadium, Inglewood'),
    ('G', 'MD2', 'Jun 21', 'New Zealand',     'Egypt',         'BC Place, Vancouver'),
    ('G', 'MD3', 'Jun 26', 'Egypt',           'Iran',          'Lumen Field, Seattle'),
    ('G', 'MD3', 'Jun 26', 'New Zealand',     'Belgium',       'BC Place, Vancouver'),
    # Group H
    ('H', 'MD1', 'Jun 15', 'Spain',           'Cabo Verde',    'Mercedes-Benz Stadium, Atlanta'),
    ('H', 'MD1', 'Jun 15', 'Saudi Arabia',    'Uruguay',       'Hard Rock Stadium, Miami Gardens'),
    ('H', 'MD2', 'Jun 21', 'Spain',           'Saudi Arabia',  'Mercedes-Benz Stadium, Atlanta'),
    ('H', 'MD2', 'Jun 21', 'Uruguay',         'Cabo Verde',    'Hard Rock Stadium, Miami Gardens'),
    ('H', 'MD3', 'Jun 26', 'Cabo Verde',      'Saudi Arabia',  'NRG Stadium, Houston'),
    ('H', 'MD3', 'Jun 26', 'Uruguay',         'Spain',         'Estadio Akron, Zapopan'),
    # Group I
    ('I', 'MD1', 'Jun 16', 'France',          'Senegal',       'MetLife Stadium, East Rutherford'),
    ('I', 'MD1', 'Jun 16', 'Iraq',            'Norway',        'Gillette Stadium, Foxborough'),
    ('I', 'MD2', 'Jun 22', 'France',          'Iraq',          'Lincoln Financial Field, Philadelphia'),
    ('I', 'MD2', 'Jun 22', 'Norway',          'Senegal',       'MetLife Stadium, East Rutherford'),
    ('I', 'MD3', 'Jun 26', 'Norway',          'France',        'Gillette Stadium, Foxborough'),
    ('I', 'MD3', 'Jun 26', 'Senegal',         'Iraq',          'BMO Field, Toronto'),
    # Group J
    ('J', 'MD1', 'Jun 16', 'Argentina',       'Algeria',       'Arrowhead Stadium, Kansas City'),
    ('J', 'MD1', 'Jun 16', 'Austria',         'Jordan',        "Levi's Stadium, Santa Clara"),
    ('J', 'MD2', 'Jun 22', 'Argentina',       'Austria',       "AT&T Stadium, Arlington"),
    ('J', 'MD2', 'Jun 22', 'Jordan',          'Algeria',       "Levi's Stadium, Santa Clara"),
    ('J', 'MD3', 'Jun 27', 'Algeria',         'Austria',       'Arrowhead Stadium, Kansas City'),
    ('J', 'MD3', 'Jun 27', 'Jordan',          'Argentina',     "AT&T Stadium, Arlington"),
    # Group K
    ('K', 'MD1', 'Jun 17', 'Portugal',        'DR Congo',      'NRG Stadium, Houston'),
    ('K', 'MD1', 'Jun 17', 'Uzbekistan',      'Colombia',      'Estadio Azteca, Mexico City'),
    ('K', 'MD2', 'Jun 23', 'Portugal',        'Uzbekistan',    'NRG Stadium, Houston'),
    ('K', 'MD2', 'Jun 23', 'Colombia',        'DR Congo',      'Estadio Akron, Zapopan'),
    ('K', 'MD3', 'Jun 27', 'Colombia',        'Portugal',      'Hard Rock Stadium, Miami Gardens'),
    ('K', 'MD3', 'Jun 27', 'DR Congo',        'Uzbekistan',    'Mercedes-Benz Stadium, Atlanta'),
    # Group L
    ('L', 'MD1', 'Jun 17', 'England',         'Croatia',       "AT&T Stadium, Arlington"),
    ('L', 'MD1', 'Jun 17', 'Ghana',           'Panama',        'BMO Field, Toronto'),
    ('L', 'MD2', 'Jun 23', 'England',         'Ghana',         'Gillette Stadium, Foxborough'),
    ('L', 'MD2', 'Jun 23', 'Panama',          'Croatia',       'BMO Field, Toronto'),
    ('L', 'MD3', 'Jun 27', 'Panama',          'England',       'MetLife Stadium, East Rutherford'),
    ('L', 'MD3', 'Jun 27', 'Croatia',         'Ghana',         'Lincoln Financial Field, Philadelphia'),
]

# ============================================================
# 6. SIMULATE GROUP STAGE → DETERMINE QUALIFIERS
# ============================================================

def simulate_group_stage(rng):
    """Simulate all 72 group matches; return group standings using actual results where available."""
    standings = {gn: {t: {'pts': 0, 'gd': 0, 'gf': 0} for t in teams}
                 for gn, teams in GROUPS.items()}
    for grp, _, _, home, away, _ in SCHEDULE:
        actual = get_actual(home, away)
        if actual is not None:
            hg, ag = actual
        else:
            hg, ag = predict_score(home, away, rng)
        s = standings[grp]
        s[home]['gf'] += hg; s[home]['gd'] += hg - ag
        s[away]['gf'] += ag; s[away]['gd'] += ag - hg
        if hg > ag:   s[home]['pts'] += 3
        elif ag > hg: s[away]['pts'] += 3
        else:         s[home]['pts'] += 1; s[away]['pts'] += 1
    sorted_groups = {}
    for gn, ts in standings.items():
        sorted_groups[gn] = sorted(ts, key=lambda t: (ts[t]['pts'], ts[t]['gd'], ts[t]['gf']), reverse=True)
    return sorted_groups

# Run a single deterministic group simulation to get qualifiers for display
_rng_main = np.random.default_rng(42)
group_standings = simulate_group_stage(_rng_main)

# Best 8 third-place teams (by pts, gd, gf)
def get_third_place_qualifiers(standings):
    thirds = []
    for gn, order in standings.items():
        t = order[2]
        # Recalculate stats for this team from schedule
        pts, gd, gf = 0, 0, 0
        for grp, _, _, home, away, _ in SCHEDULE:
            if grp != gn: continue
            if home != t and away != t: continue
            actual = get_actual(home, away)
            hg, ag = actual if actual else predict_score_det(home, away)
            if home == t:
                gf += hg; gd += hg - ag
                if hg > ag: pts += 3
                elif hg == ag: pts += 1
            else:
                gf += ag; gd += ag - hg
                if ag > hg: pts += 3
                elif hg == ag: pts += 1
        thirds.append((gn, t, pts, gd, gf))
    thirds.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    return [t for _, t, _, _, _ in thirds[:8]]

third_qualifiers = get_third_place_qualifiers(group_standings)

# Build R32 draw: 12 winners + 12 runners-up + 8 best thirds = 32
winners  = [group_standings[gn][0] for gn in sorted(GROUPS)]
runners  = [group_standings[gn][1] for gn in sorted(GROUPS)]
r32_pool = winners + runners + third_qualifiers

# Seeded by Elo (highest vs lowest)
r32_seeded = sorted(r32_pool, key=lambda t: ratings.get(t, DEFAULT_RATING), reverse=True)

# ============================================================
# 7. KNOCKOUT BRACKET PREDICTION
# ============================================================

def ko_predict(home, away, rng=None):
    """KO match: no draws, penalty shootout on level."""
    if rng is None:
        rng = np.random
    hg, ag = predict_score(home, away, rng)
    if hg == ag:
        # Penalties
        winner = home if rng.random() < 0.5 else away
        return winner, hg, ag, True
    return (home if hg > ag else away), hg, ag, False

ko_rng = np.random.default_rng(42)

# Build R32 pairs (1 vs 32, 2 vs 31, …)
r32_matches, r16_pool = [], []
for i in range(16):
    h, a = r32_seeded[i], r32_seeded[-(i+1)]
    winner, hg, ag, pens = ko_predict(h, a, ko_rng)
    r32_matches.append((h, a, hg, ag, winner, pens))
    r16_pool.append(winner)

r16_matches, qf_pool = [], []
for i in range(0, 16, 2):
    h, a = r16_pool[i], r16_pool[i+1]
    winner, hg, ag, pens = ko_predict(h, a, ko_rng)
    r16_matches.append((h, a, hg, ag, winner, pens))
    qf_pool.append(winner)

qf_matches, sf_pool = [], []
for i in range(0, 8, 2):
    h, a = qf_pool[i], qf_pool[i+1]
    winner, hg, ag, pens = ko_predict(h, a, ko_rng)
    qf_matches.append((h, a, hg, ag, winner, pens))
    sf_pool.append(winner)

sf_matches, final_pool = [], []
for i in range(0, 4, 2):
    h, a = sf_pool[i], sf_pool[i+1]
    winner, hg, ag, pens = ko_predict(h, a, ko_rng)
    sf_matches.append((h, a, hg, ag, winner, pens))
    final_pool.append(winner)

h, a = final_pool[0], final_pool[1]
fw, fhg, fag, fpens = ko_predict(h, a, ko_rng)[:4]
final_match = (h, a, fhg, fag, fw, fpens)
champion = fw

# ============================================================
# 8. DISPLAY: GROUP STAGE
# ============================================================

W = 100

def bar(title):
    pad = (W - len(title) - 2) // 2
    print(f"\n{'─'*pad} {title} {'─'*pad}")

def row(label, pred, actual, outcome_pred, outcome_act, venue=""):
    stat = "✅" if actual is not None else "🔮"
    print(f"  {stat} {label:<8} │ PRED: {pred:<30} │ ACTUAL: {actual or '—':<30} │ {outcome_act or outcome_pred}")

bar("GROUP STAGE — PREDICTED vs ACTUAL")

for grp, md, date_str, home, away, venue in SCHEDULE:
    pred_hg, pred_ag = predict_score_det(home, away)
    actual = get_actual(home, away)
    pred_str   = f"{home} {pred_hg}–{pred_ag} {away}"
    if actual is not None:
        act_str    = f"{home} {actual[0]}–{actual[1]} {away}"
        act_out    = outcome_label(*actual)
        correct    = "✓" if outcome_label(pred_hg, pred_ag) == act_out else "✗"
    else:
        act_str = None
        act_out = None
        correct = ""
    status = "✅" if actual is not None else "🔮"
    outcome_str = (f"{act_out} {correct}" if actual else f"({outcome_label(pred_hg, pred_ag)} predicted)")
    print(f"  {status} Grp {grp} {md} {date_str:6}  {pred_str:<38}  ACTUAL: {act_str or 'TBD':<38}  {outcome_str}")

# ============================================================
# 9. GROUP STANDINGS SUMMARY
# ============================================================

bar("GROUP STANDINGS  (after simulating remaining matches)")
for gn in sorted(GROUPS):
    teams_order = group_standings[gn]
    print(f"\n  Group {gn}:")
    for rank, t in enumerate(teams_order, 1):
        arrow = " ➜ R32" if rank <= 2 else ("" if rank == 3 else "  eliminated")
        q3_mark = " + qualified (best 3rd)" if t in third_qualifiers else ""
        print(f"    {rank}. {t:<25}{arrow}{q3_mark}")

# ============================================================
# 10. DISPLAY: KNOCKOUT STAGES
# ============================================================

def ko_row(stage_label, h, a, hg, ag, winner, pens):
    pens_note = " (pens)" if pens else ""
    pred_str  = f"{h} {hg}–{ag} {a}{pens_note}"
    won_str   = f"  → {winner} advances"
    print(f"  🔮 {stage_label:<18} {pred_str:<45}{won_str}")

bar("ROUND OF 32  (predicted)")
for i, (h, a, hg, ag, w, p) in enumerate(r32_matches, 1):
    ko_row(f"R32 Match {i:>2}", h, a, hg, ag, w, p)

bar("ROUND OF 16  (predicted)")
for i, (h, a, hg, ag, w, p) in enumerate(r16_matches, 1):
    ko_row(f"R16 Match {i}", h, a, hg, ag, w, p)

bar("QUARTER-FINALS  (predicted)")
for i, (h, a, hg, ag, w, p) in enumerate(qf_matches, 1):
    ko_row(f"QF {i}", h, a, hg, ag, w, p)

bar("SEMI-FINALS  (predicted)")
for i, (h, a, hg, ag, w, p) in enumerate(sf_matches, 1):
    ko_row(f"SF {i}", h, a, hg, ag, w, p)

bar("FINAL  (predicted)")
h, a, hg, ag, fw, fp = final_match
ko_row("FINAL", h, a, hg, ag, fw, fp)

# ============================================================
# 11. CHAMPIONSHIP SUMMARY
# ============================================================

bar(f"PREDICTED CHAMPION: {champion}")
print(f"\n  🏆  {champion}")
print(f"\n  Route to the trophy:")
for stage, matches in [("R32", r32_matches), ("R16", r16_matches),
                        ("QF",  qf_matches),  ("SF",  sf_matches),
                        ("F",   [final_match])]:
    for h, a, hg, ag, w, p in matches:
        if w == champion:
            opp = a if h == champion else h
            pens_note = " (pens)" if p else ""
            print(f"    {stage:<4} vs {opp:<20} {hg}–{ag}{pens_note}")

# ============================================================
# 12. SAVE JSON OUTPUT
# ============================================================

os.makedirs('predictions', exist_ok=True)

output = {
    'generated': str(TODAY),
    'model': 'Elo-Poisson v6 + piterfm dataset',
    'champion': champion,
    'group_standings': {gn: group_standings[gn] for gn in sorted(GROUPS)},
    'third_qualifiers': third_qualifiers,
    'group_stage': [
        {
            'group': grp, 'matchday': md, 'date': date_str,
            'home': home, 'away': away, 'venue': venue,
            'predicted': {'home': int(predict_score_det(home, away)[0]),
                          'away': int(predict_score_det(home, away)[1])},
            'actual':    {'home': get_actual(home, away)[0], 'away': get_actual(home, away)[1]}
                         if get_actual(home, away) else None,
        }
        for grp, md, date_str, home, away, venue in SCHEDULE
    ],
    'r32':   [{'home': h, 'away': a, 'pred_hg': int(hg), 'pred_ag': int(ag), 'winner': w, 'pens': p}
              for h, a, hg, ag, w, p in r32_matches],
    'r16':   [{'home': h, 'away': a, 'pred_hg': int(hg), 'pred_ag': int(ag), 'winner': w, 'pens': p}
              for h, a, hg, ag, w, p in r16_matches],
    'qf':    [{'home': h, 'away': a, 'pred_hg': int(hg), 'pred_ag': int(ag), 'winner': w, 'pens': p}
              for h, a, hg, ag, w, p in qf_matches],
    'sf':    [{'home': h, 'away': a, 'pred_hg': int(hg), 'pred_ag': int(ag), 'winner': w, 'pens': p}
              for h, a, hg, ag, w, p in sf_matches],
    'final': {'home': final_match[0], 'away': final_match[1],
              'pred_hg': int(final_match[2]), 'pred_ag': int(final_match[3]),
              'winner': final_match[4], 'pens': final_match[5]},
}

with open('predictions/wc2026_match_tracker.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n\n  ✅ Full tracker saved → predictions/wc2026_match_tracker.json")
print(f"  📅 As of {TODAY} | Actuals: {sum(1 for v in ACTUAL_RESULTS.values() if v)} matches confirmed")
print("=" * 70)
