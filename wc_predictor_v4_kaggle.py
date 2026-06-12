"""World Cup 2026 Predictor v4 — Kaggle Dataset"""
import kagglehub
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json, os, sys

# ============================================================
# 1. LOAD KAGGLE DATASET
# ============================================================
print("📥 Downloading Kaggle dataset...")
path = kagglehub.dataset_download("sarazahran1/wc2026-match-probability-baseline-dataset")
print(f"✅ Path: {path}")

# List files
files = os.listdir(path)
print(f"Files: {files}")

# Load whatever CSVs are available
dfs = {}
csv_files = [f for f in files if f.endswith('.csv')]
if not csv_files:
    print("❌ No CSV files found")
    sys.exit(1)

for f in csv_files:
    df = pd.read_csv(os.path.join(path, f))
    dfs[f] = df
    print(f"  {f}: {len(df)} rows, columns: {list(df.columns)}")

# ============================================================
# 2. PREPARE FEATURES & TRAIN MODEL
# ============================================================
# Try to find match-level data
match_df = None
for name, df in dfs.items():
    cols = [c.lower() for c in df.columns]
    if 'home_team' in cols and 'away_team' in cols:
        match_df = df
        print(f"\n✅ Using {name} for training ({len(df)} matches)")
        break

if match_df is None:
    # Use first dataset with numeric columns
    match_df = dfs[csv_files[0]]
    print(f"\n⚠️  No match data found, using {csv_files[0]}")

# Show columns
print(f"\nColumns: {list(match_df.columns)}")

# ============================================================
# 3. OFFICIAL GROUP DRAW
# ============================================================
official_groups = {
    'A': ['Mexico','South Africa','Korea Republic','Czechia'],
    'B': ['Canada','Italy','Qatar','Switzerland'],
    'C': ['Brazil','Morocco','Haiti','Scotland'],
    'D': ['USA','Paraguay','Australia','Slovakia'],
    'E': ['Germany','Curaçao','Côte d\'Ivoire','Ecuador'],
    'F': ['Netherlands','Japan','Poland','Tunisia'],
    'G': ['Belgium','Egypt','Iran','New Zealand'],
    'H': ['Spain','Cabo Verde','Saudi Arabia','Uruguay'],
    'I': ['France','Senegal','Bolivia','Norway'],
    'J': ['Argentina','Algeria','Austria','Jordan'],
    'K': ['Portugal','Jamaica','Uzbekistan','Colombia'],
    'L': ['England','Croatia','Ghana','Panama'],
}

all_teams_48 = []
for gn in sorted(official_groups.keys()):
    all_teams_48.extend(official_groups[gn])
all_teams_48 = list(dict.fromkeys(all_teams_48))

print(f"\n=== 48 TEAMS ({len(official_groups)} GROUPS) ===")

# ============================================================
# 4. ELO RATINGS from dataset
# ============================================================
# Build ratings from match_df
ratings = {t: 1000 for t in all_teams_48}
for _, r in match_df.iterrows():
    try:
        h = r.get('home_team', r.get('Home Team', r.get('home')))
        a = r.get('away_team', r.get('Away Team', r.get('away')))
        hg = r.get('home_score', r.get('Home Goals', r.get('home_goals', r.get('hg', 0))))
        ag = r.get('away_score', r.get('Away Goals', r.get('away_goals', r.get('ag', 0))))
        if not h or not a: continue
        h, a = str(h).strip(), str(a).strip()
        hg, ag = float(hg), float(ag)
        rh = ratings.get(h, 950)
        ra = ratings.get(a, 950)
        exp_h = 1 / (1 + 10**((ra - rh)/400))
        exp_a = 1 - exp_h
        sh, sa = (1, 0) if hg > ag else (0, 1) if ag > hg else (0.5, 0.5)
        ratings[h] = ratings.get(h, 950) + 32 * (sh - exp_h)
        ratings[a] = ratings.get(a, 950) + 32 * (sa - exp_a)
    except:
        pass

# Print ratings
ratings_list = sorted([(t, round(ratings[t], 1)) for t in all_teams_48], key=lambda x: x[1], reverse=True)
print("\n=== TOP 15 ELO RATINGS (from Kaggle dataset) ===")
for i, (t, r) in enumerate(ratings_list[:15], 1):
    print(f"  {i:>2}. {t:<25} {r:.1f}")

# ============================================================
# 5. GROUP STAGE SIMULATION
# ============================================================
np.random.seed(42)

def predict_match(t1, t2):
    r1, r2 = ratings.get(t1, 950), ratings.get(t2, 950)
    diff = r1 - r2
    prob_t1 = max(0.1, min(0.9, 0.5 + diff/800))
    return t1 if np.random.random() < prob_t1 else t2

def simulate_group(teams, sim_id=0):
    points = {t: 0 for t in teams}
    gd = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            h, a = teams[i], teams[j]
            rh, ra = ratings.get(h, 950), ratings.get(a, 950)
            diff = rh - ra
            prob_h = max(0.1, min(0.9, 0.5 + diff/800))
            if np.random.random() < prob_h:
                points[h] += 3
                hg, ag = max(1, np.random.poisson(1.5)), max(0, np.random.poisson(1.0))
            else:
                points[a] += 3
                hg, ag = max(0, np.random.poisson(1.0)), max(1, np.random.poisson(1.5))
            gd[h] += hg - ag; gd[a] += ag - hg
            gf[h] += hg; gf[a] += ag
    return sorted(teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True), points, gd, gf

# Simulate groups
group_winners, group_runners, all_third = {}, {}, []
print(f"\n{'='*80}")
print(f"  WORLD CUP 2026 — KAGGLE DATASET SIMULATION")
print(f"{'='*80}")

for gn in sorted(official_groups.keys()):
    teams = official_groups[gn]
    sorted_t, pts, gd, gf = simulate_group(teams)
    group_winners[gn] = sorted_t[0]
    group_runners[gn] = sorted_t[1]
    all_third.append((gn, sorted_t[2], pts[sorted_t[2]], gd[sorted_t[2]], gf[sorted_t[2]]))
    print(f"\n  Group {gn}:")
    for rank, t in enumerate(sorted_t, 1):
        print(f"    {rank}. {t:<25} Pts:{pts[t]} GD:{gd[t]:+d} GF:{gf[t]}")

all_third.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
qual_third = [t for _, t, _, _, _ in all_third[:8]]
print(f"\n  Qualifying 3rd: {', '.join(qual_third)}")

# ============================================================
# 6. KNOCKOUT
# ============================================================
w_list = [group_winners[gn] for gn in sorted(official_groups.keys())]
r_list = [group_runners[gn] for gn in sorted(official_groups.keys())]
w_sorted = sorted(w_list, key=lambda t: ratings.get(t, 950), reverse=True)
bye = w_sorted[:4]
r32_pool = w_sorted[4:] + r_list + qual_third
r32_s = sorted(r32_pool, key=lambda t: ratings.get(t, 950))

print(f"\n{'='*40}")
print(f"  R32 (bye: {', '.join(bye)})")
print(f"{'='*40}")
r32_m = []
for i in range(12):
    t1, t2 = r32_s[-1-i], r32_s[i]
    w = predict_match(t1, t2)
    r32_m.append((t1, t2, w))
    print(f"  {t1:<25} vs {t2:<25} -> {w}")

r16 = bye + [w for _, _, w in r32_m]
r16_s = sorted(r16, key=lambda t: ratings.get(t, 950), reverse=True)
print(f"\n{'='*40}  R16  {'='*40}")
r16_m = []
for i in range(8):
    t1, t2 = r16_s[i*2], r16_s[i*2+1]
    w = predict_match(t1, t2)
    r16_m.append((t1, t2, w))
    print(f"  {t1:<25} vs {t2:<25} -> {w}")

# Continue
curr = [w for _, _, w in r16_m]
stages = ['QF', 'SF', 'Final']
stage_data = []
for stage in stages:
    nxt = []
    print(f"\n{'='*40}  {stage}  {'='*40}")
    for i in range(0, len(curr), 2):
        if i+1 < len(curr):
            w = predict_match(curr[i], curr[i+1])
            nxt.append(w)
            print(f"  {curr[i]:<25} vs {curr[i+1]:<25} -> {w}")
    stage_data.append(nxt)
    curr = nxt

qf_data, sf_data = stage_data[0], stage_data[1] if len(stage_data) > 1 else []
champion = stage_data[-1][0] if stage_data[-1] else 'N/A'
print(f"\n{'='*50}")
print(f"  🏆 CHAMPION: {champion}")
print(f"{'='*50}")

# ============================================================
# 7. MONTE CARLO
# ============================================================
N_SIMS = 1000
from tqdm import trange
champ_counts = {}
print(f"\n  Monte Carlo: {N_SIMS} sim...")
for _ in trange(N_SIMS):
    sw, sr, st3 = {}, {}, []
    for gn in sorted(official_groups.keys()):
        teams = official_groups[gn]
        st, _, _, _ = simulate_group(teams)
        sw[gn], sr[gn] = st[0], st[1]
        st3.append((gn, st[2], 0, 0, 0))
    st3.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    qt = [st3[i][1] for i in range(min(8, len(st3)))]
    
    sw_l = list(sw.values())
    sr_l = list(sr.values())
    sw_s = sorted(sw_l, key=lambda t: ratings.get(t, 950), reverse=True)
    bye_s = sw_s[:4]
    r32_sim = sw_s[4:] + sr_l + qt
    r32_ss = sorted(r32_sim, key=lambda t: ratings.get(t, 950))
    
    r32w = [predict_match(r32_ss[-1-i], r32_ss[i]) for i in range(12)]
    r16_all = bye_s + r32w
    r16_all_s = sorted(r16_all, key=lambda t: ratings.get(t, 950), reverse=True)
    cur = r16_all_s
    
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            if i+1 < len(cur):
                nxt.append(predict_match(cur[i], cur[i+1]))
        cur = nxt
    
    if cur:
        champ_counts[cur[0]] = champ_counts.get(cur[0], 0) + 1

# Normalize
total_mc = sum(champ_counts.values())
champ_pct = {t: c/total_mc*100 for t, c in champ_counts.items()}
champ_sorted = sorted(champ_pct.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'='*50}")
print(f"  MONTE CARLO TOP 15")
print(f"{'='*50}")
for i, (t, p) in enumerate(champ_sorted[:15], 1):
    bar = "█" * int(p/2) + "░" * max(0, 50 - int(p/2))
    print(f"  {i:>2}. {t:<25} {p:5.1f}% {bar}")

# ============================================================
# 8. SAVE
# ============================================================
out = {
    'dataset': 'kaggle/sarazahran1/wc2026-match-probability-baseline-dataset',
    'groups': official_groups,
    'r32': [(a, b, w) for a, b, w in r32_m],
    'r16': [(a, b, w) for a, b, w in r16_m],
    'qf_data': qf_data,
    'sf_data': sf_data,
    'champion': champion,
    'mc_top15': champ_sorted[:15],
}

os.makedirs('predictions', exist_ok=True)
with open('predictions/wc2026_kaggle_results.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)

print(f"\n  ✅ predictions/wc2026_kaggle_results.json")
print(f"{'='*50}")
print(f"  🏆 CHAMPION: {champion}")
print(f"{'='*50}")