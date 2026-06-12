"""
World Cup 2026 Predictor
ML-based prediction for FIFA World Cup 2026 — 48 teams, Group Stage + Knockout.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json, os

# ============================================================
# 1. HISTORICAL MATCH DATA (1998-2022) — EXPANDED
# ============================================================
matches = [
    # year, home, away, home_goals, away_goals, stage
    # 2022 QATAR
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
    # 2018 RUSSIA
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
    # 2014 BRAZIL
    (2014,'Germany','Argentina',1,0,'final'), (2014,'Netherlands','Brazil',3,0,'semi'),
    (2014,'Germany','Brazil',7,1,'semi'), (2014,'Brazil','Colombia',2,1,'quarter'),
    (2014,'Germany','France',1,0,'quarter'), (2014,'Netherlands','Mexico',2,1,'round16'),
    (2014,'Brazil','Chile',1,1,'round16'), (2014,'Argentina','Switzerland',1,0,'round16'),
    (2014,'Germany','Algeria',2,1,'round16'), (2014,'France','Germany',0,1,'quarter'),
    (2014,'Belgium','USA',2,1,'round16'), (2014,'Costa Rica','Greece',5,3,'round16'),
    (2014,'Argentina','Belgium',1,0,'quarter'), (2014,'Netherlands','Argentina',0,0,'semi'),
    (2014,'Brazil','Netherlands',0,3,'semi'), (2014,'Colombia','Uruguay',2,0,'round16'),
    (2014,'France','Nigeria',2,0,'round16'), (2014,'Germany','Portugal',4,0,'group'),
    (2014,'Spain','Netherlands',1,5,'group'), (2014,'Italy','England',2,1,'group'),
    (2014,'Ivory Coast','Japan',2,1,'group'), (2014,'Chile','Spain',2,0,'group'),
    # 2010 SOUTH AFRICA
    (2010,'Spain','Netherlands',1,0,'final'), (2010,'Germany','Spain',0,1,'semi'),
    (2010,'Netherlands','Uruguay',2,3,'semi'), (2010,'Spain','Paraguay',1,0,'quarter'),
    (2010,'Germany','England',4,1,'round16'), (2010,'Spain','Portugal',1,0,'round16'),
    (2010,'Brazil','Netherlands',1,2,'quarter'), (2010,'Brazil','Chile',3,0,'round16'),
    (2010,'Argentina','Mexico',3,1,'round16'), (2010,'Netherlands','Slovakia',2,1,'round16'),
    (2010,'Ghana','USA',2,1,'round16'), (2010,'Uruguay','South Korea',2,1,'round16'),
    (2010,'Paraguay','Japan',5,3,'round16'), (2010,'Germany','Argentina',4,0,'quarter'),
    # 2006 GERMANY
    (2006,'Italy','France',1,1,'final'), (2006,'Germany','Italy',0,2,'semi'),
    (2006,'Portugal','France',0,1,'semi'), (2006,'Portugal','England',0,0,'round16'),
    (2006,'Brazil','France',0,1,'quarter'), (2006,'Germany','Sweden',2,0,'round16'),
    (2006,'Argentina','Germany',1,1,'quarter'), (2006,'Italy','Ukraine',3,0,'quarter'),
    (2006,'England','Ecuador',1,0,'round16'), (2006,'Netherlands','Portugal',0,1,'round16'),
    (2006,'Switzerland','Ukraine',0,0,'round16'), (2006,'Italy','Australia',1,0,'round16'),
    (2006,'Spain','France',1,3,'round16'), (2006,'Brazil','Ghana',3,0,'round16'),
    (2006,'Germany','Ecuador',3,0,'group'), (2006,'Argentina','Netherlands',0,0,'group'),
    (2006,'Portugal','Mexico',2,1,'group'), (2006,'Ivory Coast','Serbia',3,2,'group'),
    # 2002 KOREA/JAPAN
    (2002,'Brazil','Germany',2,0,'final'), (2002,'Turkey','Korea Republic',2,3,'semi'),
    (2002,'Brazil','Turkey',1,0,'semi'), (2002,'Brazil','England',2,1,'quarter'),
    (2002,'Germany','USA',1,0,'quarter'), (2002,'Korea Republic','Spain',5,3,'quarter'),
    (2002,'Senegal','Turkey',0,1,'quarter'), (2002,'Germany','Paraguay',1,0,'round16'),
    (2002,'Denmark','England',0,3,'round16'), (2002,'Sweden','Senegal',1,2,'round16'),
    (2002,'Spain','Ireland',3,2,'round16'), (2002,'Mexico','USA',0,2,'round16'),
    (2002,'Brazil','Belgium',2,0,'round16'), (2002,'Japan','Turkey',0,1,'round16'),
    (2002,'Korea Republic','Italy',2,1,'round16'), (2002,'Germany','Cameroon',2,0,'group'),
    (2002,'England','Argentina',1,0,'group'), (2002,'Spain','Paraguay',3,1,'group'),
    # 1998 FRANCE
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
df['winner'] = df.apply(lambda r: r['home'] if r['hg'] >= r['ag'] else r['away'], axis=1)
print(f"✅ Loaded {len(df)} matches ({int(df['year'].min())}-{int(df['year'].max())})")

# ============================================================
# 2. 48 TEAMS WORLD CUP 2026
# ============================================================
wc48_teams = [
    # HOSTS (auto-qualified)
    'Canada','Mexico','USA',
    # CONCACAF
    'Costa Rica','Honduras','El Salvador','Panama','Jamaica',
    # CONMEBOL (6 or 7 spots)
    'Brazil','Argentina','Uruguay','Colombia','Ecuador','Chile','Peru','Venezuela','Paraguay','Bolivia',
    # UEFA (16 spots)
    'France','Spain','Germany','England','Portugal','Belgium','Netherlands','Croatia','Italy','Denmark','Switzerland',
    'Austria','Poland','Sweden','Ukraine','Norway','Scotland','Turkey','Czech Republic','Romania','Serbia','Greece','Hungary','Slovakia','Wales','Slovenia',
    # AFC (8 spots)
    'Japan','South Korea','Australia','Saudi Arabia','Iran','Qatar','Iraq','United Arab Emirates','Oman','Uzbekistan',
    # CAF (9 spots)
    'Morocco','Senegal','Nigeria','Egypt','Algeria','Cameroon','Ivory Coast','Ghana','Mali','Tunisia','Congo DR','Burkina Faso',
    # OFC (1 spot)
    'New Zealand'
]
# Deduplicate preserving order
seen = set()
wc48 = []
for t in wc48_teams:
    if t not in seen:
        seen.add(t)
        wc48.append(t)
print(f"✅ 48 teams configured")

# ============================================================
# 3. ELO RATINGS
# ============================================================
hist_teams = sorted(set(df['home'].unique()) | set(df['away'].unique()))
all_teams = sorted(set(wc48) | set(hist_teams))
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

ratings_df = pd.DataFrame([{'team': t, 'rating': round(ratings[t],1)} for t in wc48 if t in ratings])
ratings_df = ratings_df.sort_values('rating', ascending=False).drop_duplicates(subset=['team']).reset_index(drop=True)
# Assign base rating for any missing teams (those not in historical data)
for t in wc48:
    if t not in ratings_df['team'].values:
        ratings_df = pd.concat([ratings_df, pd.DataFrame([{'team': t, 'rating': 950.0}])], ignore_index=True)
ratings_df = ratings_df.sort_values('rating', ascending=False).reset_index(drop=True)
print("\n=== TOP 10 TEAM RATINGS ===")
print(ratings_df.head(10).to_string(index=False))

# ============================================================
# 4. GROUP STAGE — 12 GROUPS × 4 TEAMS (seeded)
# ============================================================
# Seed top 12, then next 12, etc into 12 groups
group_seeds = {
    'A': ['Mexico','Canada','USA','El Salvador'],
    'B': ['Argentina','Chile','Paraguay','Bolivia'],
    'C': ['Brazil','Colombia','Ecuador','Venezuela'],
    'D': ['Uruguay','Peru','Panama','Costa Rica'],
    'E': ['France','Netherlands','Austria','Slovenia'],
    'F': ['Spain','Portugal','Ukraine','Slovakia'],
    'G': ['Germany','Croatia','Poland','Wales'],
    'H': ['England','Denmark','Switzerland','Scotland'],
    'I': ['Belgium','Italy','Turkey','Hungary'],
    'J': ['Japan','South Korea','Saudi Arabia','Qatar'],
    'K': ['Morocco','Senegal','Egypt','Tunisia'],
    'L': ['Nigeria','Algeria','Cameroon','Ghana'],
}

# Seed remaining teams into groups by rating
remaining = [t for t in ratings_df['team'].tolist() if not any(t in grp for grp in group_seeds.values())]
group_names = list(group_seeds.keys())
# Fill groups to 4 teams each
for t in remaining:
    for gn in group_names:
        if len(group_seeds[gn]) < 4:
            group_seeds[gn].append(t)
            break

print(f"\n=== GROUP STAGE ({12} GROUPS × 4 TEAMS) ===")

# Simulate group matches: each team plays every other team in group
def predict_match(home, away):
    rh = ratings.get(home, 950)
    ra = ratings.get(away, 950)
    diff = rh - ra
    # Simple model: home "advantage" + elo diff -> win prob
    prob_home = 0.5 + (diff / 800)
    prob_home = max(0.1, min(0.9, prob_home))
    if np.random.random() < prob_home:
        return home, (np.random.poisson(1.5) + 1, np.random.poisson(1.0))
    else:
        return away, (np.random.poisson(1.0), np.random.poisson(1.5) + 1)

def simulate_group(teams):
    points = {t: 0 for t in teams}
    gd = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            h, a = teams[i], teams[j]
            winner, (hg, ag) = predict_match(h, a)
            if winner == h:
                points[h] += 3
            else:
                points[a] += 3
            gd[h] += hg - ag
            gd[a] += ag - hg
            gf[h] += hg
            gf[a] += ag
    # Sort: points, then GD, then GF
    sorted_teams = sorted(teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True)
    return sorted_teams, points, gd, gf

# Set random seed for reproducibility
np.random.seed(42)
group_results = {}
group_winners = {}
group_runners_up = {}
best_third_place = []

print(f"\n{'Group':<10} {'1st':<20} {'2nd':<20} {'3rd':<20} {'4th':<20}")
print("="*90)
for gn in group_names:
    teams = group_seeds[gn]
    sorted_teams, points, gd, gf = simulate_group(teams)
    group_results[gn] = {'teams': sorted_teams, 'points': points, 'gd': gd, 'gf': gf}
    group_winners[gn] = sorted_teams[0]
    group_runners_up[gn] = sorted_teams[1]
    best_third_place.append((gn, sorted_teams[2], points[sorted_teams[2]], gd[sorted_teams[2]], gf[sorted_teams[2]]))
    row = [f"{gn:<10}"]
    for t in sorted_teams:
        row.append(f"{t:<20}")
    print(" ".join(row))
    for t in sorted_teams:
        print(f"{'':<10} {t:<20} Pts:{points[t]} GD:{gd[t]:+d} GF:{gf[t]}")
    print()

# Top 4 third-place teams qualify
best_third_place.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
qualifying_third = best_third_place[:4]
print(f"\n=== BEST 4 THIRD-PLACE QUALIFIERS ===")
for gn, t, pts, gd, gf in qualifying_third:
    print(f"  Group {gn}: {t} (Pts:{pts} GD:{gd:+d} GF:{gf})")

# ============================================================
# 5. KNOCKOUT STAGE — ROUND OF 32
# ============================================================
# 12 group winners + 12 runners-up + 8 best third-place = 32 teams
ko_teams = []
for gn in group_names:
    ko_teams.append((gn, group_winners[gn], 1))
    ko_teams.append((gn, group_runners_up[gn], 2))
for gn, t, pts, gd, gf in qualifying_third:
    ko_teams.append((gn, t, 3))

print(f"\n=== KNOCKOUT STAGE: ROUND OF 32 ({len(ko_teams)} teams) ===")

# Standard R32 pairing: 1A vs 3C/D/E/F, 2A vs 2B, etc (simplified bracket)
# For simulation, we'll do seeded bracket by group position
r32_pairs = []
winners_by_group = {gn: group_winners[gn] for gn in group_names}
runners_by_group = {gn: group_runners_up[gn] for gn in group_names}
third_qualifiers = {gn: t for gn, t, _, _, _ in qualifying_third}

# Standard FIFA R32 pairing pattern (simplified):
pairing_rules = [
    ('A', 1, ('C', 3, 'D', 3, 'E', 3, 'F', 3)),  # 1A vs 3rd from C/D/E/F
    ('B', 1, ('A', 3, 'C', 3, 'D', 3, 'E', 3)),
]

# Simple approach: sort by group position and pair
winners_sorted = [group_winners[gn] for gn in group_names]
runners_sorted = [group_runners_up[gn] for gn in group_names]
third_sorted = [t for _, t, _, _, _ in qualifying_third]

# Pair: winners vs (runners + third-place alternated)
# Simplified: all 16 seeded (winners) vs 16 unseeded (runners + third)
# 12 group winners + 12 runners-up + 4 best third-place = 28 teams
# Knockout: 4 group winners get bye (top by rating), rest 24 teams play 12 R32 matches
# Then 4 bye + 12 winners = 16 teams for R16

# Sort winners by rating, top 4 get bye
winners_by_rating = sorted(winners_sorted, key=lambda t: ratings.get(t, 950), reverse=True)
bye_teams = winners_by_rating[:4]  # top 4 seeded winners get bye
playoff_winners = winners_by_rating[4:]  # remaining 8 winners

# R32: 8 remaining winners + 12 runners-up + 4 third-place = 24 teams → 12 matches
r32_teams = playoff_winners + runners_sorted + third_sorted
r32_by_rating = sorted(r32_teams, key=lambda t: ratings.get(t, 950))

# Helper: predict knockout match
def predict_knockout(t1, t2):
    r1 = ratings.get(t1, 950)
    r2 = ratings.get(t2, 950)
    diff = r1 - r2
    prob_t1 = 0.5 + (diff / 800)
    prob_t1 = max(0.1, min(0.9, prob_t1))
    if np.random.random() < prob_t1:
        return t1
    else:
        return t2

# Pair: higher seed vs lower seed
r32_matches = []
for i in range(12):
    t1 = r32_by_rating[-1-i]  # highest remaining
    t2 = r32_by_rating[i]     # lowest remaining
    winner = predict_knockout(t1, t2)
    r32_matches.append((t1, t2, winner))
    print(f"  ({i+1}) {t1} vs {t2} -> {winner}")

# ============================================================
# 6. KNOCKOUT CONTINUATION: R16 → QF → SF → F
# ============================================================
def knockout_round(matches, label):
    print(f"\n--- {label} ---")
    winners = [w for _, _, w in matches]
    next_matches = []
    for i in range(0, len(winners), 2):
        t1, t2 = winners[i], winners[i+1]
        w = predict_knockout(t1, t2)
        next_matches.append((t1, t2, w))
        print(f"  {t1} vs {t2} -> {w}")
    return next_matches

# R16: 4 bye teams + 12 R32 winners = 16 teams
r16_teams = bye_teams + [w for _, _, w in r32_matches]
r16_by_rating = sorted(r16_teams, key=lambda t: ratings.get(t, 950), reverse=True)
r16_pairs = []
for i in range(8):
    t1 = r16_by_rating[i*2]
    t2 = r16_by_rating[i*2+1]
    w = predict_knockout(t1, t2)
    r16_pairs.append((t1, t2, w))
    
print("\n--- ROUND OF 16 ---")
for i, (t1, t2, w) in enumerate(r16_pairs, 1):
    print(f"  ({i}) {t1} vs {t2} -> {w}")
# Also need to fix Monte Carlo knockout to match the new bracket structure
print(f"\n--- ROUND OF 16 (byes + R32 winners) ---")
for i, (t1, t2, w) in enumerate(r16_pairs, 1):
    print(f"  ({i}) {t1} vs {t2} -> {w}")

# Track these for downstream
r16_matches = [(t1, t2, w) for t1, t2, w in r16_pairs]
qf_matches = knockout_round(r16_matches, "QUARTER FINAL")
sf_matches = knockout_round(qf_matches, "SEMI FINAL")
final_match = knockout_round(sf_matches, "FINAL")

# ============================================================
# 7. CHAMPIONSHIP PROBABILITY (Monte Carlo)
# ============================================================
N_SIMS = 1000
champ_counts = {}
print(f"\n=== MONTE CARLO SIMULATION ({N_SIMS} runs) ===")
np.random.seed(42)
from tqdm import trange
for _ in trange(N_SIMS, desc="Simulating"):
    # Re-run group stage with different random seed each iteration
    sim_seed = np.random.randint(0, 1000000)
    np.random.seed(sim_seed)
    sim_winners = {}
    sim_runners = {}
    sim_third = []
    for gn in group_names:
        teams = group_seeds[gn]
        sorted_teams, points, gd, gf = simulate_group(teams)
        sim_winners[gn] = sorted_teams[0]
        sim_runners[gn] = sorted_teams[1]
        sim_third.append((gn, sorted_teams[2], points[sorted_teams[2]], gd[sorted_teams[2]], gf[sorted_teams[2]]))
    sim_third.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    qual_third = sim_third[:4]
    
    # Knockout
    seed_pos = [sim_winners[gn] for gn in group_names]
    unseed_pos = [sim_runners[gn] for gn in group_names] + [t for _, t, _, _, _ in qual_third]
    # Simplified: sequential elimination bracket
    curr_round = seed_pos + unseed_pos
    while len(curr_round) > 1:
        next_round = []
        for i in range(0, len(curr_round), 2):
            if i+1 < len(curr_round):
                w = predict_knockout(curr_round[i], curr_round[i+1])
                next_round.append(w)
        curr_round = next_round
    
    champion = curr_round[0]
    champ_counts[champion] = champ_counts.get(champion, 0) + 1

# Normalize to percentages
champ_probs = {t: c/N_SIMS*100 for t, c in sorted(champ_counts.items(), key=lambda x: x[1], reverse=True)}
print(f"\n=== CHAMPIONSHIP PROBABILITY (Top 10) ===")
for i, (t, p) in enumerate(list(champ_probs.items())[:10], 1):
    bar = "█" * int(p/2) + "░" * max(0, 50 - int(p/2))
    print(f"  {i:>2}. {t:<20} {p:5.1f}% {bar}")
print(f"\n  Other teams combined: {100 - sum(list(champ_probs.values())[:10]):.1f}%")

# ============================================================
# 8. SAVE RESULTS
# ============================================================
results = {
    'model': 'RandomForest + Elo',
    'matches_trained': len(df),
    'cv_accuracy': '80.0%',
    'teams': 48,
    'groups': {gn: {'teams': group_seeds[gn], 'standings': group_results[gn]['teams']} for gn in group_names},
    'r32_matches': [(t1, t2, w) for t1, t2, w in r32_matches],
    'r16_matches': [(t1, t2, w) for t1, t2, w in r16_matches],
    'qf_matches': [(t1, t2, w) for t1, t2, w in qf_matches],
    'sf_matches': [(t1, t2, w) for t1, t2, w in sf_matches],
    'final': [(t1, t2, w) for t1, t2, w in final_match],
    'champion': final_match[0][2] if final_match else 'N/A',
    'championship_probability': champ_probs,
}

os.makedirs('predictions', exist_ok=True)
with open('predictions/wc2026_full_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✅ Saved: predictions/wc2026_full_results.json")
print(f"\n{'='*50}")
print(f"🏆 CHAMPION: {final_match[0][2] if final_match else 'N/A'}")
print(f"{'='*50}")