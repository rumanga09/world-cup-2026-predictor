"""
World Cup 2026 Predictor
ML-based prediction for FIFA World Cup 2026.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 1. HISTORICAL MATCH DATA (1998-2022)
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
    # 2018 RUSSIA
    (2018,'France','Croatia',4,2,'final'), (2018,'Croatia','England',2,1,'semi'), (2018,'France','Belgium',1,0,'semi'),
    (2018,'Russia','Croatia',2,2,'quarter'), (2018,'Brazil','Belgium',1,2,'quarter'),
    (2018,'Uruguay','France',0,2,'quarter'), (2018,'Sweden','England',0,2,'quarter'),
    (2018,'Croatia','Denmark',1,1,'round16'), (2018,'France','Argentina',4,3,'round16'),
    (2018,'Brazil','Mexico',2,0,'round16'), (2018,'Spain','Russia',1,1,'round16'),
    (2018,'Croatia','Nigeria',2,0,'group'), (2018,'France','Peru',1,0,'group'),
    (2018,'Germany','Sweden',2,1,'group'), (2018,'Brazil','Switzerland',1,1,'group'),
    (2018,'Belgium','Japan',3,2,'round16'), (2018,'Colombia','England',1,1,'round16'),
    # 2014 BRAZIL
    (2014,'Germany','Argentina',1,0,'final'), (2014,'Netherlands','Brazil',3,0,'semi'),
    (2014,'Germany','Brazil',7,1,'semi'), (2014,'Brazil','Colombia',2,1,'quarter'),
    (2014,'Germany','France',1,0,'quarter'), (2014,'Netherlands','Mexico',2,1,'round16'),
    (2014,'Brazil','Chile',1,1,'round16'), (2014,'Argentina','Switzerland',1,0,'round16'),
    (2014,'Germany','Algeria',2,1,'round16'), (2014,'France','Germany',0,1,'quarter'),
    # 2010 SOUTH AFRICA
    (2010,'Spain','Netherlands',1,0,'final'), (2010,'Germany','Spain',0,1,'semi'),
    (2010,'Netherlands','Uruguay',2,3,'semi'), (2010,'Spain','Paraguay',1,0,'quarter'),
    (2010,'Germany','England',4,1,'round16'), (2010,'Spain','Portugal',1,0,'round16'),
    (2010,'Brazil','Netherlands',1,2,'quarter'), (2010,'Brazil','Chile',3,0,'round16'),
    # 2006 GERMANY
    (2006,'Italy','France',1,1,'final'), (2006,'Germany','Italy',0,2,'semi'),
    (2006,'Portugal','France',0,1,'semi'), (2006,'Portugal','England',0,0,'round16'),
    (2006,'Brazil','France',0,1,'quarter'), (2006,'Germany','Sweden',2,0,'round16'),
    # 2002 KOREA/JAPAN
    (2002,'Brazil','Germany',2,0,'final'), (2002,'Turkey','Korea Republic',2,3,'semi'),
    (2002,'Brazil','Turkey',1,0,'semi'), (2002,'Brazil','England',2,1,'quarter'),
    # 1998 FRANCE
    (1998,'France','Brazil',3,0,'final'), (1998,'France','Croatia',2,1,'semi'),
    (1998,'Brazil','Netherlands',1,1,'semi'), (1998,'France','Italy',0,0,'quarter'),
    # HISTORICAL RIVALRIES
    (2018,'Germany','Mexico',0,1,'group'), (2022,'Germany','Spain',1,1,'group'),
    (2014,'Italy','Uruguay',0,1,'group'), (2018,'Germany','Sweden',2,1,'group'),
    (2022,'Croatia','Belgium',0,0,'group'), (2022,'Portugal','Uruguay',2,0,'group'),
    (2022,'Denmark','Australia',1,1,'group'), (2018,'Brazil','Costa Rica',2,0,'group'),
]

df = pd.DataFrame(matches, columns=['year','home','away','hg','ag','stage'])
df['winner'] = df.apply(lambda r: r['home'] if r['hg'] >= r['ag'] else r['away'], axis=1)
print(f"Loaded {len(df)} matches ({int(df['year'].min())}-{int(df['year'].max())})")

# 2. ELO RATINGS
wc_team_names = [
    'Mexico','Canada','USA',
    'Argentina','Brazil','Uruguay','Chile','Colombia','Ecuador',
    'England','France','Germany','Spain','Portugal','Italy','Netherlands','Belgium','Croatia','Denmark','Switzerland','Austria','Scotland','Sweden','Poland','Ukraine','Turkey',
    'Japan','South Korea','Iran','Saudi Arabia','Australia',
    'Egypt','Morocco','Senegal','Nigeria','Algeria','Cameroon','Ivory Coast',
    'New Zealand',
]
wc_teams = sorted(list(dict.fromkeys(wc_team_names)))
# seed ratings from historical data, plus any WC teams not yet mentioned at 1000
hist_teams = sorted(set(df['home'].unique()) | set(df['away'].unique()))
all_teams = sorted(set(wc_teams) | set(hist_teams))
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

ratings_df = pd.DataFrame([{'team': t, 'rating': round(ratings[t],1)} for t in sorted(set(df['home'].unique()) | set(df['away'].unique()))])
ratings_df = ratings_df.sort_values('rating', ascending=False).drop_duplicates(subset=['team']).reset_index(drop=True)
print("\n=== TOP 10 TEAM RATINGS ===")
print(ratings_df.head(10).to_string(index=False))

# 3. FEATURE ENGINEERING
def features(home, away):
    return [
        ratings.get(home, 1000), ratings.get(away, 1000),
        ratings.get(home, 1000) - ratings.get(away, 1000),
    ]

X = np.array([features(r.home, r.away) for _, r in df.iterrows()])
y = (df['home'] == df['winner']).astype(int)

clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
scores = cross_val_score(clf, X, y, cv=5)
print(f"\n=== CV ACCURACY: {scores.mean():.1%} ± {scores.std():.1%} ===")
clf.fit(X, y)

def predict_match(home, away):
    prob_home_win = clf.predict_proba([features(home, away)])[0][1]
    return home if prob_home_win >= 0.5 else away

def next_round(teams):
    winners = []
    for i in range(0, len(teams), 2):
        w = predict_match(teams[i], teams[i+1])
        winners.append((teams[i], teams[i+1], w))
    return winners

# 4. SIMULATE 2026 WC (32-team knockout)
# take top 32 by rating, deduplicated
unique_teams = list(dict.fromkeys([t for t in ratings_df['team'].tolist() if t in wc_teams] + wc_teams))
wc32 = unique_teams[:32]
print(f"\n=== 2026 WC SIMULATION ({len(wc32)} teams) ===")

r32_matches = next_round(wc32)
print("\n--- ROUND OF 32 ---")
for h, a, w in r32_matches:
    print(f"  {h} vs {a} -> {w}")

r16_teams = [w for _, _, w in r32_matches]
r16_matches = next_round(r16_teams)
print("\n--- ROUND OF 16 ---")
for h, a, w in r16_matches:
    print(f"  {h} vs {a} -> {w}")

qf_teams = [w for _, _, w in r16_matches]
qf_matches = next_round(qf_teams)
print("\n--- QUARTER FINAL ---")
for h, a, w in qf_matches:
    print(f"  {h} vs {a} -> {w}")

sf_teams = [w for _, _, w in qf_matches]
sf_matches = next_round(sf_teams)
print("\n--- SEMI FINAL ---")
for h, a, w in sf_matches:
    print(f"  {h} vs {a} -> {w}")

final_teams = [w for _, _, w in sf_matches]
final_match = next_round(final_teams)
print("\n=== FINAL ===")
for h, a, w in final_match:
    print(f"  {h} vs {a} -> 🏆 CHAMPION: {w}")
