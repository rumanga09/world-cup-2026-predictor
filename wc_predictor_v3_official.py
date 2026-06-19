"""
World Cup 2026 Predictor v3 — OFFICIAL GROUP DRAW
Based on actual FIFA 2026 Final Draw (5 May 2026, Washington DC)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json, os

# ============================================================
# 1. HISTORICAL MATCH DATA (same as v2)
# ============================================================
matches = [
    (2022,'Argentina','France',3,3,'final'), (2022,'France','Croatia',2,0,'semi'), (2022,'Argentina','Croatia',3,0,'semi'),
    (2022,'France','Morocco',2,0,'semi'), (2022,'Morocco','Croatia',1,0,'quarter'),
    (2022,'Argentina','Netherlands',2,2,'quarter'), (2022,'Brazil','Croatia',1,1,'quarter'),
    (2022,'England','France',1,2,'quarter'), (2022,'Japan','Croatia',1,1,'round16'),
    (2022,'Brazil','Korea Republic',4,1,'round16'), (2022,'Morocco','Spain',0,0,'round16'),
    (2022,'Netherlands','USA',3,1,'round16'), (2022,'Argentina','Australia',2,1,'round16'),
    (2022,'France','Poland',3,1,'round16'), (2022,'England','Senegal',3,0,'round16'),
    (2022,'Croatia','Japan',1,0,'group'), (2022,'Argentina','Mexico',2,0,'group'),
    (2022,'France','Australia',4,1,'group'), (2022,'Brazil','Switzerland',1,0,'group'),
    (2022,'Spain','Costa Rica',7,0,'group'), (2022,'Germany','Japan',1,2,'group'),
    (2022,'Brazil','Serbia',2,0,'group'), (2022,'Portugal','Switzerland',6,1,'round16'),
    (2022,'Portugal','Uruguay',2,0,'group'), (2022,'Belgium','Canada',1,0,'group'),
    (2022,'Croatia','Belgium',0,0,'group'), (2022,'Germany','Spain',1,1,'group'),
    (2022,'Japan','Costa Rica',0,1,'group'), (2022,'Brazil','Cameroon',1,0,'group'),
    (2022,'Switzerland','Cameroon',1,0,'group'), (2022,'South Korea','Portugal',2,1,'group'),
    (2022,'Ghana','South Korea',3,2,'group'), (2022,'Cameroon','Serbia',3,3,'group'),
    (2022,'Ecuador','Senegal',1,2,'group'), (2022,'Netherlands','Qatar',2,0,'group'),
    (2022,'Iran','Wales',2,0,'group'), (2022,'Saudi Arabia','Argentina',2,1,'group'),
    (2022,'Tunisia','France',1,0,'group'), (2022,'Morocco','Belgium',2,0,'group'),
    (2022,'Costa Rica','Germany',2,4,'group'), (2022,'Japan','Spain',2,1,'group'),
    (2022,'Croatia','Canada',4,1,'group'),
    (2018,'France','Croatia',4,2,'final'), (2018,'Croatia','England',2,1,'semi'), (2018,'France','Belgium',1,0,'semi'),
    (2018,'Russia','Croatia',2,2,'quarter'), (2018,'Brazil','Belgium',1,2,'quarter'),
    (2018,'Uruguay','France',0,2,'quarter'), (2018,'Sweden','England',0,2,'quarter'),
    (2018,'Croatia','Denmark',1,1,'round16'), (2018,'France','Argentina',4,3,'round16'),
    (2018,'Brazil','Mexico',2,0,'round16'), (2018,'Spain','Russia',1,1,'round16'),
    (2018,'Croatia','Nigeria',2,0,'group'), (2018,'France','Peru',1,0,'group'),
    (2018,'Germany','Sweden',2,1,'group'), (2018,'Brazil','Switzerland',1,1,'group'),
    (2018,'Belgium','Japan',3,2,'round16'), (2018,'Colombia','England',1,1,'round16'),
    (2018,'Belgium','England',2,0,'group'), (2018,'Switzerland','Costa Rica',2,2,'group'),
    (2018,'Uruguay','Russia',3,0,'group'), (2018,'Iran','Portugal',1,1,'group'),
    (2018,'Germany','Mexico',0,1,'group'), (2018,'Brazil','Costa Rica',2,0,'group'),
    (2018,'Nigeria','Iceland',2,0,'group'), (2018,'Argentina','Nigeria',2,1,'group'),
    (2018,'South Korea','Germany',2,0,'group'), (2018,'Denmark','France',0,0,'group'),
    (2018,'Spain','Portugal',3,3,'group'), (2018,'Tunisia','Panama',2,1,'group'),
    (2014,'Germany','Argentina',1,0,'final'), (2014,'Netherlands','Brazil',3,0,'semi'),
    (2014,'Germany','Brazil',7,1,'semi'), (2014,'Brazil','Colombia',2,1,'quarter'),
    (2014,'Germany','France',1,0,'quarter'), (2014,'Netherlands','Mexico',2,1,'round16'),
    (2014,'Brazil','Chile',1,1,'round16'), (2014,'Argentina','Switzerland',1,0,'round16'),
    (2014,'Germany','Algeria',2,1,'round16'),
    (2014,'Belgium','USA',2,1,'round16'), (2014,'Costa Rica','Greece',5,3,'round16'),
    (2014,'Argentina','Belgium',1,0,'quarter'), (2014,'Netherlands','Argentina',0,0,'semi'),
    (2014,'Brazil','Netherlands',0,3,'semi'), (2014,'Colombia','Uruguay',2,0,'round16'),
    (2014,'France','Nigeria',2,0,'round16'), (2014,'Germany','Portugal',4,0,'group'),
    (2014,'Spain','Netherlands',1,5,'group'), (2014,'Italy','England',2,1,'group'),
    (2010,'Spain','Netherlands',1,0,'final'), (2010,'Germany','Spain',0,1,'semi'),
    (2010,'Netherlands','Uruguay',2,3,'semi'), (2010,'Spain','Paraguay',1,0,'quarter'),
    (2010,'Germany','England',4,1,'round16'), (2010,'Spain','Portugal',1,0,'round16'),
    (2010,'Brazil','Netherlands',1,2,'quarter'), (2010,'Brazil','Chile',3,0,'round16'),
    (2010,'Argentina','Mexico',3,1,'round16'), (2010,'Netherlands','Slovakia',2,1,'round16'),
    (2010,'Ghana','USA',2,1,'round16'), (2010,'Uruguay','South Korea',2,1,'round16'),
    (2010,'Paraguay','Japan',5,3,'round16'), (2010,'Germany','Argentina',4,0,'quarter'),
    (2006,'Italy','France',1,1,'final'), (2006,'Germany','Italy',0,2,'semi'),
    (2006,'Portugal','France',0,1,'semi'), (2006,'Portugal','England',0,0,'round16'),
    (2006,'Brazil','France',0,1,'quarter'), (2006,'Germany','Sweden',2,0,'round16'),
    (2006,'Argentina','Germany',1,1,'quarter'), (2006,'Italy','Ukraine',3,0,'quarter'),
    (2006,'England','Ecuador',1,0,'round16'), (2006,'Netherlands','Portugal',0,1,'round16'),
    (2006,'Spain','France',1,3,'round16'), (2006,'Brazil','Ghana',3,0,'round16'),
    (2006,'Germany','Ecuador',3,0,'group'), (2006,'Argentina','Netherlands',0,0,'group'),
    (2002,'Brazil','Germany',2,0,'final'), (2002,'Turkey','Korea Republic',2,3,'semi'),
    (2002,'Brazil','Turkey',1,0,'semi'), (2002,'Brazil','England',2,1,'quarter'),
    (2002,'Germany','USA',1,0,'quarter'), (2002,'Korea Republic','Spain',5,3,'quarter'),
    (2002,'Senegal','Turkey',0,1,'quarter'), (2002,'Germany','Paraguay',1,0,'round16'),
    (2002,'Denmark','England',0,3,'round16'), (2002,'Sweden','Senegal',1,2,'round16'),
    (2002,'Spain','Ireland',3,2,'round16'), (2002,'Mexico','USA',0,2,'round16'),
    (2002,'Brazil','Belgium',2,0,'round16'), (2002,'Japan','Turkey',0,1,'round16'),
    (2002,'Korea Republic','Italy',2,1,'round16'), (2002,'Germany','Cameroon',2,0,'group'),
    (2002,'England','Argentina',1,0,'group'), (2002,'Spain','Paraguay',3,1,'group'),
    (1998,'France','Brazil',3,0,'final'), (1998,'France','Croatia',2,1,'semi'),
    (1998,'Brazil','Netherlands',1,1,'semi'), (1998,'France','Italy',0,0,'quarter'),
    (1998,'Brazil','Denmark',3,2,'quarter'), (1998,'Netherlands','Argentina',2,1,'quarter'),
    (1998,'Croatia','Germany',3,0,'quarter'), (1998,'France','Paraguay',1,0,'round16'),
    (1998,'England','Argentina',3,3,'round16'), (1998,'Italy','Norway',1,0,'round16'),
    (1998,'Brazil','Chile',4,1,'round16'), (1998,'Nigeria','Denmark',1,4,'round16'),
    (1998,'Netherlands','Yugoslavia',2,1,'round16'), (1998,'Germany','Mexico',2,1,'round16'),
    (1998,'Croatia','Romania',1,0,'round16'),
]

df = pd.DataFrame(matches, columns=['year','home','away','hg','ag','stage'])
df['winner'] = df.apply(lambda r: r['home'] if r['hg'] > r['ag'] else (r['away'] if r['ag'] > r['hg'] else 'Draw'), axis=1)
print(f"✅ {len(df)} matches loaded")

# ============================================================
# 2. OFFICIAL GROUP DRAW (FIFA, 5 May 2026)
# ============================================================
official_groups = {
    'A': ['Mexico','South Africa','Korea Republic','Czechia/Denmark/MKD/Ireland'],
    'B': ['Canada','BIH/Italy/NIR/Wales','Qatar','Switzerland'],
    'C': ['Brazil','Morocco','Haiti','Scotland'],
    'D': ['USA','Paraguay','Australia','Kosovo/Romania/Slovakia/Türkiye'],
    'E': ['Germany','Curaçao','Côte d\'Ivoire','Ecuador'],
    'F': ['Netherlands','Japan','Albania/Poland/Sweden/Ukraine','Tunisia'],
    'G': ['Belgium','Egypt','IR Iran','New Zealand'],
    'H': ['Spain','Cabo Verde','Saudi Arabia','Uruguay'],
    'I': ['France','Senegal','Bolivia/Iraq/Suriname','Norway'],
    'J': ['Argentina','Algeria','Austria','Jordan'],
    'K': ['Portugal','Congo DR/Jamaica/NCL','Uzbekistan','Colombia'],
    'L': ['England','Croatia','Ghana','Panama'],
}

# Resolve placeholder teams to specific names using most likely qualifier
resolved_groups = {
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

# Collect all teams for Elo
all_teams_48 = []
for gn in sorted(resolved_groups.keys()):
    all_teams_48.extend(resolved_groups[gn])
all_teams_48 = list(dict.fromkeys(all_teams_48))
print(f"✅ {len(resolved_groups)} groups, {len(all_teams_48)} teams (official draw)")

# ============================================================
# 3. ELO RATINGS
# ============================================================
hist_teams = sorted(set(df['home'].unique()) | set(df['away'].unique()))
all_teams = sorted(set(all_teams_48) | set(hist_teams))
ratings = {t: 1000 for t in all_teams}

for _, r in df.iterrows():
    h, a = r['home'], r['away']
    rh = ratings.get(h, 1000)
    ra = ratings.get(a, 1000)
    exp_h = 1 / (1 + 10**((ra - rh)/400))
    exp_a = 1 - exp_h
    if r['hg'] > r['ag']:
        sh, sa = 1, 0
    elif r['hg'] == r['ag']:
        sh, sa = 0.5, 0.5
    else:
        sh, sa = 0, 1
    ratings[h] = ratings.get(h, 1000) + 32 * (sh - exp_h)
    ratings[a] = ratings.get(a, 1000) + 32 * (sa - exp_a)

# Base ratings for unrated teams
all_names_48 = set(all_teams_48)
hist_names = set(hist_teams)
unrated = all_names_48 - hist_names
for t in unrated:
    ratings[t] = 950.0

# Print top ratings
ratings_list = [(t, round(ratings[t], 1)) for t in all_teams_48]
ratings_list = sorted([(t, r) for t, r in ratings_list if t in ratings], key=lambda x: x[1], reverse=True)
print("\n=== TOP 15 ELO RATINGS ===")
for i, (t, r) in enumerate(ratings_list[:15], 1):
    pstr = "[QUAL] " if t in hist_names else "[DEF]  "
    print(f"  {i:>2}. {pstr} {t:<25} {r:.1f}")

# ============================================================
# 4. GROUP STAGE SIMULATION
# ============================================================
np.random.seed(42)

def predict_knockout(t1, t2):
    r1, r2 = ratings.get(t1, 950), ratings.get(t2, 950)
    diff = r1 - r2
    prob_t1 = max(0.1, min(0.9, 0.5 + diff/800))
    return t1 if np.random.random() < prob_t1 else t2

def simulate_group(teams):
    points = {t: 0 for t in teams}
    gd = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            h, a = teams[i], teams[j]
            rh, ra = ratings.get(h, 950), ratings.get(a, 950)
            diff = rh - ra
            prob_h = max(0.1, min(0.9, 0.5 + diff/800))
            draw_rate = max(0.10, 0.28 - abs(diff) / 2000)
            r = np.random.random()
            if r < prob_h * (1 - draw_rate):
                points[h] += 3
                hg, ag = max(1, np.random.poisson(1.5)), max(0, np.random.poisson(1.0))
            elif r < prob_h * (1 - draw_rate) + draw_rate:
                s = max(0, np.random.poisson(1.1))
                hg, ag = s, s
                points[h] += 1; points[a] += 1
            else:
                points[a] += 3
                hg, ag = max(0, np.random.poisson(1.0)), max(1, np.random.poisson(1.5))
            gd[h] += hg - ag; gd[a] += ag - hg
            gf[h] += hg; gf[a] += ag
    return sorted(teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True), points, gd, gf

print(f"\n{'='*80}")
print(f"  WORLD CUP 2026 — OFFICIAL DRAW SIMULATION")
print(f"{'='*80}")

group_results = {}
group_winners = {}
group_runners = {}
all_third = []

for gn in sorted(resolved_groups.keys()):
    teams = resolved_groups[gn]
    sorted_t, pts, gd, gf = simulate_group(teams)
    group_results[gn] = {'standings': sorted_t, 'points': pts, 'gd': gd, 'gf': gf}
    group_winners[gn] = sorted_t[0]
    group_runners[gn] = sorted_t[1]
    all_third.append((gn, sorted_t[2], pts[sorted_t[2]], gd[sorted_t[2]], gf[sorted_t[2]]))
    print(f"\n  Group {gn}:")
    for rank, t in enumerate(sorted_t, 1):
        print(f"    {rank}. {t:<25} Pts:{pts[t]} GD:{gd[t]:+d} GF:{gf[t]}")

# Best 8 third-place qualify
all_third.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
qual_third = [t for _, t, _, _, _ in all_third[:8]]
print(f"\n  Qualifying 3rd-place: {', '.join(qual_third)}")

# ============================================================
# 5. KNOCKOUT — ROUND OF 32
# ============================================================
winners_sorted = [group_winners[gn] for gn in sorted(resolved_groups.keys())]
runners_sorted = [group_runners[gn] for gn in sorted(resolved_groups.keys())]

# Top 4 winners get bye
winners_by_rating = sorted(winners_sorted, key=lambda t: ratings.get(t, 950), reverse=True)
bye = winners_by_rating[:4]
r32_pool = winners_by_rating[4:] + runners_sorted + qual_third
r32_sorted = sorted(r32_pool, key=lambda t: ratings.get(t, 950))

print(f"\n{'='*40}")
print(f"  ROUND OF 32 (bye: {', '.join(bye)})")
print(f"{'='*40}")

r32_matches = []
for i in range(12):
    t1, t2 = r32_sorted[-1-i], r32_sorted[i]  # highest vs lowest
    w = predict_knockout(t1, t2)
    r32_matches.append((t1, t2, w))
    print(f"  {t1:<25} vs {t2:<25} -> {w}")

# ROUND OF 16
r16_teams = bye + [w for _, _, w in r32_matches]
r16_sorted = sorted(r16_teams, key=lambda t: ratings.get(t, 950), reverse=True)
print(f"\n{'='*40}")
print(f"  ROUND OF 16")
print(f"{'='*40}")

r16_matches = []
for i in range(8):
    t1, t2 = r16_sorted[i*2], r16_sorted[i*2+1]
    w = predict_knockout(t1, t2)
    r16_matches.append((t1, t2, w))
    print(f"  {t1:<25} vs {t2:<25} -> {w}")

# QUARTER FINAL
print(f"\n{'='*40}")
print(f"  QUARTER FINAL")
print(f"{'='*40}")
qf_matches = []
qf_in = [w for _, _, w in r16_matches]
for i in range(4):
    t1, t2 = qf_in[i*2], qf_in[i*2+1]
    w = predict_knockout(t1, t2)
    qf_matches.append((t1, t2, w))
    print(f"  {t1:<25} vs {t2:<25} -> {w}")

# SEMI FINAL
print(f"\n{'='*40}")
print(f"  SEMI FINAL")
print(f"{'='*40}")
sf_matches = []
sf_in = [w for _, _, w in qf_matches]
for i in range(2):
    t1, t2 = sf_in[i*2], sf_in[i*2+1]
    w = predict_knockout(t1, t2)
    sf_matches.append((t1, t2, w))
    print(f"  {t1:<25} vs {t2:<25} -> {w}")

# FINAL
print(f"\n{'='*40}")
print(f"  FINAL")
print(f"{'='*40}")
t1, t2 = sf_matches[0][2], sf_matches[1][2]
champion = predict_knockout(t1, t2)
print(f"  {t1:<25} vs {t2:<25}")
print(f"  {'='*40}")
print(f"  🏆 CHAMPION: {champion}")
print(f"  {'='*40}")

# ============================================================
# 6. MONTE CARLO (1,000 runs on official groups)
# ============================================================
N_SIMS = 1000
champ_counts = {}
sf_counts = {}
qf_counts = {}

print(f"\n  Monte Carlo: {N_SIMS} simulations...")
from tqdm import trange
for _ in trange(N_SIMS, desc="Simulating"):
    # group stage
    sim_w = {}; sim_r = {}; sim_t3 = []
    for gn in sorted(resolved_groups.keys()):
        teams = resolved_groups[gn]
        sorted_t, pts, gd, gf = simulate_group(teams)
        sim_w[gn] = sorted_t[0]
        sim_r[gn] = sorted_t[1]
        sim_t3.append((gn, sorted_t[2], pts[sorted_t[2]], gd[sorted_t[2]], gf[sorted_t[2]]))
    sim_t3.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    q3 = [t for _, t, _, _, _ in sim_t3[:8]]
    
    # knockouts (simplified bracket)
    sw = list(sim_w.values())
    sr = list(sim_r.values())
    sw_by_r = sorted(sw, key=lambda t: ratings.get(t, 950), reverse=True)
    bye_sim = sw_by_r[:4]
    r32_sim = sw_by_r[4:] + sr + q3
    r32_sim_s = sorted(r32_sim, key=lambda t: ratings.get(t, 950))
    
    r32w = []
    for i in range(12):
        w = predict_knockout(r32_sim_s[-1-i], r32_sim_s[i])
        r32w.append(w)
    
    r16_s = bye_sim + r32w
    r16_s_s = sorted(r16_s, key=lambda t: ratings.get(t, 950), reverse=True)
    curr = r16_s_s
    
    while len(curr) > 1:
        nxt = []
        for i in range(0, len(curr), 2):
            if i+1 < len(curr):
                w = predict_knockout(curr[i], curr[i+1])
                nxt.append(w)
        curr = nxt
    
    c = curr[0]
    champ_counts[c] = champ_counts.get(c, 0) + 1

print(f"\n{'='*50}")
print(f"  MONTE CARLO TOP 15")
print(f"{'='*50}")
champ_sorted = sorted(champ_counts.items(), key=lambda x: x[1], reverse=True)
for i, (t, c) in enumerate(champ_sorted[:15], 1):
    p = c / N_SIMS * 100
    bar = "█" * int(p/2) + "░" * max(0, 50 - int(p/2))
    print(f"  {i:>2}. {t:<25} {p:5.1f}%")
print(f"\n  ⚠️  Note: Low counts reflect small simulation size (1,000). Higher N = smoother probabilities.")

# ============================================================
# 7. SAVE
# ============================================================
out = {
    'official_groups': resolved_groups,
    'group_standings': {gn: group_results[gn]['standings'] for gn in sorted(resolved_groups.keys())},
    'bye_teams_R32': bye,
    'r32_matches': [(a, b, w) for a, b, w in r32_matches],
    'r16_matches': [(a, b, w) for a, b, w in r16_matches],
    'qf_matches': [(a, b, w) for a, b, w in qf_matches],
    'sf_matches': [(a, b, w) for a, b, w in sf_matches],
    'final': (t1, t2, champion),
    'champion_single_run': champion,
    'championship_probability_mc1000': [(t, round(c/N_SIMS*100, 1)) for t, c in champ_sorted[:20]],
}

os.makedirs('predictions', exist_ok=True)
with open('predictions/wc2026_official_draw_results.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n  ✅ predictions/wc2026_official_draw_results.json")
print(f"{'='*50}")
print(f"  🏆 OFFICIAL DRAW → Champion: {champion}")
print(f"{'='*50}")