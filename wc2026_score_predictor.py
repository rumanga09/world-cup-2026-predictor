"""
World Cup 2026 — Score Prediction for Every Group Stage Match
Based on official fixture schedule + historical Elo + Monte Carlo score model
"""

import numpy as np
import json, os
from datetime import datetime

# ============================================================
# ELO RATINGS (from historical data + v3)
# ============================================================
ratings = {
    # Group A
    'Mexico': 970.0, 'South Africa': 940.0, 'Korea Republic': 1036.0, 'Czechia': 970.0,
    # Group B
    'Canada': 960.0, 'Italy': 1053.7, 'Qatar': 950.0, 'Switzerland': 980.0,
    # Group C
    'Brazil': 1095.9, 'Morocco': 1016.7, 'Haiti': 920.0, 'Scotland': 950.0,
    # Group D
    'USA': 970.0, 'Paraguay': 960.0, 'Australia': 970.0, 'Turkey': 970.0,
    # Group E
    'Germany': 1096.3, 'Curaçao': 900.0, "Côte d'Ivoire": 980.0, 'Ecuador': 970.0,
    # Group F
    'Netherlands': 1083.3, 'Japan': 990.0, 'Sweden': 980.0, 'Tunisia': 1034.4,
    # Group G
    'Belgium': 1017.4, 'Egypt': 990.0, 'Iran': 1015.9, 'New Zealand': 920.0,
    # Group H
    'Spain': 1051.7, 'Cabo Verde': 910.0, 'Saudi Arabia': 1018.0, 'Uruguay': 1011.2,
    # Group I
    'France': 1148.8, 'Senegal': 1000.0, 'Iraq': 940.0, 'Norway': 980.0,
    # Group J
    'Argentina': 980.0, 'Algeria': 1000.0, 'Austria': 970.0, 'Jordan': 930.0,
    # Group K
    'Portugal': 1000.0, 'DR Congo': 940.0, 'Uzbekistan': 930.0, 'Colombia': 980.0,
    # Group L
    'England': 1005.0, 'Croatia': 1038.6, 'Ghana': 1013.8, 'Panama': 950.0,
}

# ============================================================
# ALL 72 GROUP MATCHES (from official schedule)
# ============================================================
schedule = [
    # Group A
    (11, 'Mexico', 'South Africa', 'Estadio Azteca, Mexico City'),
    (11, 'Korea Republic', 'Czechia', 'Estadio Akron, Zapopan'),
    (18, 'Czechia', 'South Africa', 'Mercedes-Benz Stadium, Atlanta'),
    (18, 'Mexico', 'Korea Republic', 'Estadio Akron, Zapopan'),
    (24, 'Czechia', 'Mexico', 'Estadio Azteca, Mexico City'),
    (24, 'South Africa', 'Korea Republic', 'Estadio BBVA, Guadalupe'),
    
    # Group B
    (12, 'Canada', 'Italy', 'BMO Field, Toronto'),
    (13, 'Qatar', 'Switzerland', "Levi's Stadium, Santa Clara"),
    (18, 'Switzerland', 'Italy', 'SoFi Stadium, Inglewood'),
    (18, 'Canada', 'Qatar', 'BC Place, Vancouver'),
    (24, 'Switzerland', 'Canada', 'BC Place, Vancouver'),
    (24, 'Italy', 'Qatar', 'Lumen Field, Seattle'),
    
    # Group C
    (13, 'Brazil', 'Morocco', 'MetLife Stadium, East Rutherford'),
    (13, 'Haiti', 'Scotland', 'Gillette Stadium, Foxborough'),
    (19, 'Scotland', 'Morocco', 'Gillette Stadium, Foxborough'),
    (19, 'Brazil', 'Haiti', 'Lincoln Financial Field, Philadelphia'),
    (24, 'Scotland', 'Brazil', 'Hard Rock Stadium, Miami Gardens'),
    (24, 'Morocco', 'Haiti', 'Mercedes-Benz Stadium, Atlanta'),
    
    # Group D
    (12, 'USA', 'Paraguay', 'SoFi Stadium, Inglewood'),
    (13, 'Australia', 'Turkey', 'BC Place, Vancouver'),
    (19, 'USA', 'Australia', 'Lumen Field, Seattle'),
    (19, 'Turkey', 'Paraguay', "Levi's Stadium, Santa Clara"),
    (25, 'Turkey', 'USA', 'SoFi Stadium, Inglewood'),
    (25, 'Paraguay', 'Australia', "Levi's Stadium, Santa Clara"),
    
    # Group E
    (14, 'Germany', 'Curaçao', 'NRG Stadium, Houston'),
    (14, "Côte d'Ivoire", 'Ecuador', 'Lincoln Financial Field, Philadelphia'),
    (20, 'Germany', "Côte d'Ivoire", 'BMO Field, Toronto'),
    (20, 'Ecuador', 'Curaçao', 'Arrowhead Stadium, Kansas City'),
    (25, 'Curaçao', "Côte d'Ivoire", 'Lincoln Financial Field, Philadelphia'),
    (25, 'Ecuador', 'Germany', 'MetLife Stadium, East Rutherford'),
    
    # Group F
    (14, 'Netherlands', 'Japan', "AT&T Stadium, Arlington"),
    (14, 'Sweden', 'Tunisia', 'Estadio BBVA, Guadalupe'),
    (20, 'Netherlands', 'Sweden', 'NRG Stadium, Houston'),
    (20, 'Tunisia', 'Japan', 'Estadio BBVA, Guadalupe'),
    (25, 'Japan', 'Sweden', "AT&T Stadium, Arlington"),
    (25, 'Tunisia', 'Netherlands', 'Arrowhead Stadium, Kansas City'),
    
    # Group G
    (15, 'Belgium', 'Egypt', 'Lumen Field, Seattle'),
    (15, 'Iran', 'New Zealand', 'SoFi Stadium, Inglewood'),
    (21, 'Belgium', 'Iran', 'SoFi Stadium, Inglewood'),
    (21, 'New Zealand', 'Egypt', 'BC Place, Vancouver'),
    (26, 'Egypt', 'Iran', 'Lumen Field, Seattle'),
    (26, 'New Zealand', 'Belgium', 'BC Place, Vancouver'),
    
    # Group H
    (15, 'Spain', 'Cabo Verde', 'Mercedes-Benz Stadium, Atlanta'),
    (15, 'Saudi Arabia', 'Uruguay', 'Hard Rock Stadium, Miami Gardens'),
    (21, 'Spain', 'Saudi Arabia', 'Mercedes-Benz Stadium, Atlanta'),
    (21, 'Uruguay', 'Cabo Verde', 'Hard Rock Stadium, Miami Gardens'),
    (26, 'Cabo Verde', 'Saudi Arabia', 'NRG Stadium, Houston'),
    (26, 'Uruguay', 'Spain', 'Estadio Akron, Zapopan'),
    
    # Group I
    (16, 'France', 'Senegal', 'MetLife Stadium, East Rutherford'),
    (16, 'Iraq', 'Norway', 'Gillette Stadium, Foxborough'),
    (22, 'France', 'Iraq', 'Lincoln Financial Field, Philadelphia'),
    (22, 'Norway', 'Senegal', 'MetLife Stadium, East Rutherford'),
    (26, 'Norway', 'France', 'Gillette Stadium, Foxborough'),
    (26, 'Senegal', 'Iraq', 'BMO Field, Toronto'),
    
    # Group J
    (16, 'Argentina', 'Algeria', 'Arrowhead Stadium, Kansas City'),
    (16, 'Austria', 'Jordan', "Levi's Stadium, Santa Clara"),
    (22, 'Argentina', 'Austria', "AT&T Stadium, Arlington"),
    (22, 'Jordan', 'Algeria', "Levi's Stadium, Santa Clara"),
    (27, 'Algeria', 'Austria', 'Arrowhead Stadium, Kansas City'),
    (27, 'Jordan', 'Argentina', "AT&T Stadium, Arlington"),
    
    # Group K
    (17, 'Portugal', 'DR Congo', 'NRG Stadium, Houston'),
    (17, 'Uzbekistan', 'Colombia', 'Estadio Azteca, Mexico City'),
    (23, 'Portugal', 'Uzbekistan', 'NRG Stadium, Houston'),
    (23, 'Colombia', 'DR Congo', 'Estadio Akron, Zapopan'),
    (27, 'Colombia', 'Portugal', 'Hard Rock Stadium, Miami Gardens'),
    (27, 'DR Congo', 'Uzbekistan', 'Mercedes-Benz Stadium, Atlanta'),
    
    # Group L
    (17, 'England', 'Croatia', "AT&T Stadium, Arlington"),
    (17, 'Ghana', 'Panama', 'BMO Field, Toronto'),
    (23, 'England', 'Ghana', 'Gillette Stadium, Foxborough'),
    (23, 'Panama', 'Croatia', 'BMO Field, Toronto'),
    (27, 'Panama', 'England', 'MetLife Stadium, East Rutherford'),
    (27, 'Croatia', 'Ghana', 'Lincoln Financial Field, Philadelphia'),
]

group_map = {
    ('Mexico','South Africa','Korea Republic','Czechia'): 'A',
    ('Canada','Italy','Qatar','Switzerland'): 'B',
    ('Brazil','Morocco','Haiti','Scotland'): 'C',
    ('USA','Paraguay','Australia','Turkey'): 'D',
    ('Germany','Curaçao',"Côte d'Ivoire",'Ecuador'): 'E',
    ('Netherlands','Japan','Sweden','Tunisia'): 'F',
    ('Belgium','Egypt','Iran','New Zealand'): 'G',
    ('Spain','Cabo Verde','Saudi Arabia','Uruguay'): 'H',
    ('France','Senegal','Iraq','Norway'): 'I',
    ('Argentina','Algeria','Austria','Jordan'): 'J',
    ('Portugal','DR Congo','Uzbekistan','Colombia'): 'K',
    ('England','Croatia','Ghana','Panama'): 'L',
}

def find_group(home, away):
    for teams, g in group_map.items():
        if home in teams and away in teams:
            return g
    return '?'

# ============================================================
# SCORE PREDICTION MODEL
# ============================================================
np.random.seed(42)

def predict_score(home, away):
    """Predict final score based on Elo difference with Poisson randomness"""
    rh = ratings.get(home, 950)
    ra = ratings.get(away, 950)
    diff = rh - ra
    
    # Base expected goals (inspired by Poisson regression on World Cup data)
    # Home team advantage: ~0.4 goals
    home_adv = 0.3
    base_home = max(0.5, 1.2 + (diff / 400) + home_adv)
    base_away = max(0.3, 0.9 - (diff / 400))
    
    # Poisson simulation
    home_goals = np.random.poisson(base_home)
    away_goals = np.random.poisson(base_away)
    
    # Cap at realistic max (WC matches rarely exceed 7)
    home_goals = min(home_goals, 7)
    away_goals = min(away_goals, 7)
    
    return home_goals, away_goals

def match_outcome(hg, ag):
    if hg > ag: return 'H'
    elif ag > hg: return 'A'
    else: return 'D'

# ============================================================
# RUN PREDICTIONS
# ============================================================
print(f"{'='*100}")
print(f"  WORLD CUP 2026 — GROUP STAGE SCORE PREDICTIONS")
print(f"  Model: Elo-based Poisson | Seed: 42")
print(f"{'='*100}")

all_predictions = []
group_predictions = {}

for day, home, away, venue in schedule:
    hg, ag = predict_score(home, away)
    outcome = match_outcome(hg, ag)
    group = find_group(home, away)
    
    pred = {
        'day': day, 'date': f"June {day}, 2026",
        'group': group,
        'home': home, 'away': away,
        'predicted_score': f"{home} {hg}-{ag} {away}",
        'home_goals': hg, 'away_goals': ag,
        'outcome': {'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'}[outcome],
        'venue': venue,
        'home_elo': round(ratings.get(home, 950), 1),
        'away_elo': round(ratings.get(away, 950), 1),
    }
    all_predictions.append(pred)
    
    if group not in group_predictions:
        group_predictions[group] = []
    group_predictions[group].append(pred)

# Print by group
for gn in sorted(group_predictions.keys()):
    print(f"\n  {'─'*80}")
    print(f"  GROUP {gn}")
    print(f"  {'─'*80}")
    for p in group_predictions[gn]:
        emoji = {'Home Win': '✅', 'Away Win': '🔴', 'Draw': '🤝'}[p['outcome']]
        print(f"  {p['date']:20s} | {emoji} {p['predicted_score']:35s} | {p['venue']}")

# ============================================================
# MONTE CARLO SIMULATION (for accurate group standings)
# ============================================================
N_SIMS = 5000
group_standings_mc = {}
points_mc = {}

print(f"\n\n{'='*80}")
print(f"  MONTE CARLO GROUP STANDINGS ({N_SIMS:,} simulations)")
print(f"{'='*80}")

for gn in sorted(group_predictions.keys()):
    teams = set()
    win_counts = {t: 0 for t in sum([list(g) for g in group_map.keys()], []) if t in [p['home'] for p in group_predictions[gn]]}
    
    # Collect team names for this group
    gt = []
    for p in group_predictions[gn]:
        if p['home'] not in gt: gt.append(p['home'])
        if p['away'] not in gt: gt.append(p['away'])
    
    # Get group tuple
    for teams_tuple, g in group_map.items():
        if g == gn:
            grp_teams = list(teams_tuple)
            break
    
    # Simulate group
    top1_counts = {t: 0 for t in grp_teams}
    top2_counts = {t: 0 for t in grp_teams}
    
    for sim in range(N_SIMS):
        points = {t: 0 for t in grp_teams}
        gd = {t: 0 for t in grp_teams}
        gf = {t: 0 for t in grp_teams}
        
        for p in group_predictions[gn]:
            hg, ag = predict_score(p['home'], p['away'])
            if hg > ag:
                points[p['home']] += 3
            elif ag > hg:
                points[p['away']] += 3
            else:
                points[p['home']] += 1
                points[p['away']] += 1
            gd[p['home']] += hg - ag
            gd[p['away']] += ag - hg
            gf[p['home']] += hg
            gf[p['away']] += ag
        
        sorted_t = sorted(grp_teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True)
        top1_counts[sorted_t[0]] += 1
        top2_counts[sorted_t[0]] += 1
        top2_counts[sorted_t[1]] += 1
    
    print(f"\n  Group {gn}:")
    for t in grp_teams:
        pct1 = top1_counts[t] / N_SIMS * 100
        pct2 = top2_counts[t] / N_SIMS * 100
        bar1 = "█" * int(pct1/2)
        bar2 = "░" * max(0, 50 - int(pct1/2))
        print(f"    {t:<20} | 1st: {pct1:5.1f}% {bar1}{bar2} | Top 2: {pct2:5.1f}%")

# ============================================================
# SAVE
# ============================================================
os.makedirs('predictions', exist_ok=True)
with open('predictions/wc2026_score_predictions.json', 'w') as f:
    json.dump(all_predictions, f, indent=2, default=str)
print(f"\n  ✅ Predictions saved to predictions/wc2026_score_predictions.json")

# Summary stats
home_wins = sum(1 for p in all_predictions if p['outcome'] == 'Home Win')
away_wins = sum(1 for p in all_predictions if p['outcome'] == 'Away Win')
draws = sum(1 for p in all_predictions if p['outcome'] == 'Draw')
total_goals = sum(p['home_goals'] + p['away_goals'] for p in all_predictions)

print(f"\n{'='*50}")
print(f"  SUMMARY")
print(f"{'='*50}")
print(f"  Total matches: {len(all_predictions)}")
print(f"  Home wins:     {home_wins} ({home_wins/len(all_predictions)*100:.0f}%)")
print(f"  Away wins:     {away_wins} ({away_wins/len(all_predictions)*100:.0f}%)")
print(f"  Draws:         {draws} ({draws/len(all_predictions)*100:.0f}%)")
print(f"  Total goals:   {total_goals}")
print(f"  Avg goals:     {total_goals/len(all_predictions):.1f} per match")
print(f"{'='*50}")