"""
World Cup 2026 Predictor v6 — PROFESSIONAL GRADE FIXED
Perbaikan berdasarkan audit:
1. Draw rate dinamis (bukan fixed 25%)
2. Elo standar (1/(1+10^((ra-rh)/400)))
3. Bracket tetap (tidak direseed)
4. K-factor bervariasi per turnamen
5. Elo initialized sesuai FIFA ranking proxy
"""

import numpy as np
import json, os, pandas as pd
from tqdm import trange
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FIX 1: Standard Elo formula
# ============================================================
def elo_exp(rh, ra):
    return 1 / (1 + 10 ** ((ra - rh) / 400))

# ============================================================
# FIX 2: Match probability dengan dynamic draw rate
# ============================================================
def match_prob(rh, ra):
    exp_h = elo_exp(rh, ra)
    diff = abs(rh - ra)
    # Draw rate decline exponentially with Elo diff
    draw_rate = 0.28 * np.exp(-diff / 300)
    draw_rate = max(0.08, min(0.30, draw_rate))
    p_h = exp_h * (1 - draw_rate)
    p_a = (1 - exp_h) * (1 - draw_rate)
    total = p_h + draw_rate + p_a
    return p_h/total, draw_rate/total, p_a/total

# ============================================================
# FIX 3: Match simulation
# ============================================================
def simulate_match(home, away, ratings):
    rh = ratings.get(home, 1500)
    ra = ratings.get(away, 1500)
    p_h, p_d, p_a = match_prob(rh, ra)
    r = np.random.random()
    if r < p_h:
        hg = max(1, int(np.random.poisson(1.5 * (rh/ra)**0.3)))
        ag = max(0, int(np.random.poisson(0.8 * (ra/rh)**0.3)))
        return home, hg, ag
    elif r < p_h + p_d:
        s = max(0, int(np.random.poisson(1.0)))
        return 'draw', s, s
    else:
        hg = max(0, int(np.random.poisson(0.8 * (rh/ra)**0.3)))
        ag = max(1, int(np.random.poisson(1.5 * (ra/rh)**0.3)))
        return away, hg, ag

# ============================================================
# FIX 4: Group simulation
# ============================================================
def simulate_group(teams, ratings):
    points = {t: 0 for t in teams}
    gd = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            h, a = teams[i], teams[j]
            winner, hg, ag = simulate_match(h, a, ratings)
            gd[h] += hg - ag; gd[a] += ag - hg
            gf[h] += hg; gf[a] += ag
            if winner == h: points[h] += 3
            elif winner == a: points[a] += 3
            else: points[h] += 1; points[a] += 1
    return sorted(teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True), points, gd, gf

# ============================================================
# FIX 5: Load dataset
# ============================================================
print("📥 Loading international results...")
_df_loaded = False
try:
    import kagglehub as _kh
    _path = _kh.dataset_download("martj42/international-football-results-from-1872-to-2017")
    _csv = next(f for f in os.listdir(_path) if f.endswith('.csv'))
    df = pd.read_csv(os.path.join(_path, _csv))
    _df_loaded = True
except Exception:
    for _root, _, _files in os.walk(os.path.expanduser("~/.cache/kagglehub")):
        for _f in _files:
            if _f == 'results.csv':
                df = pd.read_csv(os.path.join(_root, _f))
                _df_loaded = True
                break
        if _df_loaded:
            break
if not _df_loaded:
    raise FileNotFoundError(
        "Dataset not found. Run: import kagglehub; kagglehub.dataset_download('martj42/international-football-results-from-1872-to-2017')"
    )
df['home'] = df['home_team']; df['away'] = df['away_team']
df['hg'] = df['home_score']; df['ag'] = df['away_score']
df['year'] = pd.to_datetime(df['date']).dt.year
df = df[(df['year'] >= 1990) & (df['year'] <= 2025)].copy()
print(f"   ✅ {len(df):,} matches (1990-2025)")

# ============================================================
# FIX 6: Weighted Elo training
# ============================================================
COMP_WEIGHTS = {
    'FIFA World Cup': 60, 'UEFA Euro': 50, 'Copa America': 50,
    'Africa Cup of Nations': 40, 'AFC Asian Cup': 40,
    'FIFA World Cup qualification': 30, 'UEFA Nations League': 25,
    'Friendly': 15,
}
DEFAULT_K = 25
DEFAULT_RATING = 1500

BASE = {
    'France': 1900, 'Brazil': 1870, 'Argentina': 1850, 'Spain': 1840,
    'England': 1830, 'Netherlands': 1820, 'Portugal': 1810, 'Germany': 1805,
    'Belgium': 1800, 'Croatia': 1790, 'Italy': 1780, 'Uruguay': 1770,
    'Morocco': 1750, 'Senegal': 1740, 'Japan': 1730, 'Korea Republic': 1720,
    'USA': 1710, 'Mexico': 1710, 'Switzerland': 1700, 'Poland': 1690,
    'Denmark': 1680, 'Sweden': 1670, 'Austria': 1660, 'Iran': 1650,
    'Colombia': 1650, 'Ecuador': 1640, 'Egypt': 1630, 'Nigeria': 1620,
    'Ghana': 1610, 'Australia': 1600, 'Saudi Arabia': 1580, 'Qatar': 1570,
}

ratings = {}
for t in set(df['home'].unique()) | set(df['away'].unique()):
    ratings[t] = BASE.get(t, DEFAULT_RATING)

for _, r in df.iterrows():
    h, a = r['home'], r['away']
    hg, ag = r['hg'], r['ag']
    tourn = str(r.get('tournament', ''))
    k = DEFAULT_K
    for comp, kw in COMP_WEIGHTS.items():
        if comp.lower() in tourn.lower(): k = kw; break
    rh, ra = ratings.get(h, DEFAULT_RATING), ratings.get(a, DEFAULT_RATING)
    exp_h = elo_exp(rh, ra)
    sh = 1 if hg > ag else (0.5 if hg == ag else 0)
    ratings[h] = rh + k * (sh - exp_h)
    ratings[a] = ra + k * ((1 - sh) - (1 - exp_h))

# ============================================================
# 2026 TEAMS
# ============================================================
official_groups = {
    'A': ['Mexico','South Africa','Korea Republic','Czechia'],
    'B': ['Canada','Italy','Qatar','Switzerland'],
    'C': ['Brazil','Morocco','Haiti','Scotland'],
    'D': ['USA','Paraguay','Australia','Slovakia'],
    'E': ['Germany','Curacao','Ivory Coast','Ecuador'],
    'F': ['Netherlands','Japan','Poland','Tunisia'],
    'G': ['Belgium','Egypt','Iran','New Zealand'],
    'H': ['Spain','Cabo Verde','Saudi Arabia','Uruguay'],
    'I': ['France','Senegal','Iraq','Norway'],
    'J': ['Argentina','Algeria','Austria','Jordan'],
    'K': ['Portugal','Jamaica','Uzbekistan','Colombia'],
    'L': ['England','Croatia','Ghana','Panama'],
}

all_wc = set()
for teams in official_groups.values():
    all_wc.update(teams)
for t in all_wc:
    if t not in ratings: ratings[t] = DEFAULT_RATING

print(f"\n{'='*60}")
print(f"  WC 2026 TEAM ELO")
print(f"{'='*60}")
wc_elo = sorted([(t, ratings[t]) for t in all_wc], key=lambda x: x[1], reverse=True)
for i, (t, r) in enumerate(wc_elo[:25], 1):
    print(f"  {i:>2}. {t:<25} {r:.0f}")
print(f"  Range: {wc_elo[0][1]:.0f} - {wc_elo[-1][1]:.0f} (diff: {wc_elo[0][1]-wc_elo[-1][1]:.0f})")

# ============================================================
# MONTE CARLO
# ============================================================
N_SIMS = 5000
np.random.seed(42)
champ_counts = {}

print(f"\n  Monte Carlo: {N_SIMS:,} simulations...")
for _ in trange(N_SIMS, desc="Simulating"):
    gw, gr, gt3 = {}, {}, []
    for gn in sorted(official_groups.keys()):
        teams = official_groups[gn]
        s, pts, gd, gf = simulate_group(teams, ratings)
        gw[gn] = s[0]; gr[gn] = s[1]
        gt3.append((gn, s[2], pts[s[2]], gd[s[2]], gf[s[2]]))
    gt3.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    q3 = [t for _, t, _, _, _ in gt3[:8]]
    
    # Knockout — R32 (no bye, official FIFA 2026 format)
    gns = sorted(official_groups.keys())
    # 12 group winners + 12 runners-up + 8 best third = 32 teams for R32
    gw_list = [gw[gn] for gn in gns]
    gr_list = [gr[gn] for gn in gns]
    r32_pool = gw_list + gr_list + q3
    
    # Pair by rating: top vs bottom
    r32_s = sorted(r32_pool, key=lambda t: ratings.get(t, 1500), reverse=True)
    curr = []
    for i in range(16):  # 32/2 = 16 R32 matches → 16 teams to R16
        h, a = r32_s[i], r32_s[-(i+1)]
        w, _, _ = simulate_match(h, a, ratings)
        if w == 'draw':
            w = h if np.random.random() < 0.5 else a  # penalty shootout
        curr.append(w)

    # R16 → QF → SF → F (fixed bracket, no reseeding)
    while len(curr) > 1:
        nxt = []
        for i in range(0, len(curr), 2):
            if i+1 < len(curr):
                w, _, _ = simulate_match(curr[i], curr[i+1], ratings)
                if w == 'draw':
                    w = curr[i] if np.random.random() < 0.5 else curr[i+1]  # penalty shootout
                nxt.append(w)
        curr = nxt
    champ_counts[curr[0]] = champ_counts.get(curr[0], 0) + 1

# Convert to probabilities
total = sum(c for _, c in champ_counts.items())
champ_sorted = sorted(champ_counts.items(), key=lambda x: x[1], reverse=True)
champ_probs = [(t, c/total*100) for t, c in champ_sorted]

print(f"\n{'='*60}")
print(f"  CHAMPIONSHIP PROBABILITY (N={N_SIMS:,})")
print(f"{'='*60}")
for i, (t, pc) in enumerate(champ_probs[:20], 1):
    bar = "█" * int(pc/2) + "░" * max(0, 50 - int(pc/2))
    print(f"  {i:>2}. {t:<25} {pc:5.1f}% {bar}")

print(f"\n  Top 1 / Top 10 ratio: {champ_probs[0][1]/champ_probs[9][1]:.1f}x")
print(f"  Top 1: {champ_probs[0][0]} ({champ_probs[0][1]:.1f}%)")
print(f"  Top 10 avg: {np.mean([pc for _, pc in champ_probs[:10]]):.1f}%")
print(f"  Total prob sum: {sum(pc for _, pc in champ_probs):.1f}%")

# Austria check
for t, pc in champ_probs:
    if 'Austria' in t or 'Austria' in str(t):
        print(f"\n  {'='*40}")
        print(f"  AUSTRIA: {pc:.1f}%")
        print(f"  Elo: {ratings.get(t, 0):.0f}")
        print(f"  {'='*40}")
        break

# ============================================================
# SAVE
# ============================================================
results = {
    'model': 'Elo v6 — professional grade',
    'dataset': f"{len(df):,} matches (1990-2025)",
    'elo_formula': 'standard Elo (1/(1+10^((ra-rh)/400)))',
    'draw_probability': 'dynamic (Elo-diff based, 8-30%)',
    'bracket': 'fixed (no reseeding per round)',
    'k_factor': 'weighted per competition (15-60)',
    'monte_carlo_runs': N_SIMS,
    'elo_range': f'{wc_elo[-1][1]:.0f} - {wc_elo[0][1]:.0f}',
    'championship_probability': [{'team': t, 'prob': round(pc, 1)} for t, pc in champ_probs[:30]],
    'team_elo': {t: round(ratings[t], 1) for t, _ in wc_elo},
}

os.makedirs('predictions', exist_ok=True)
with open('predictions/wc2026_v6_professional.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  ✅ predictions/wc2026_v6_professional.json")
print(f"{'='*60}")
print(f"  🏆 {champ_sorted[0][0]}: {champ_sorted[0][1]:.1f}%")
print(f"  🥈 {champ_sorted[1][0]}: {champ_sorted[1][1]:.1f}%")
print(f"  🥉 {champ_sorted[2][0]}: {champ_sorted[2][1]:.1f}%")
print(f"{'='*60}")