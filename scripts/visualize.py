"""Generate visualization charts for WC 2026 predictions."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = 'outputs'

# Team ratings from wc_predictor output
ratings_data = {
    'team': ['France','Spain','Brazil','Germany','Netherlands','Argentina','Belgium','Italy','Portugal','Uruguay',
             'Morocco','Denmark','Croatia','Algeria','Poland','Chile','Colombia','USA','Senegal','Nigeria',
             'Japan','Mexico','England','Australia','Turkey','Switzerland','Sweden','Austria','Cameroon','Canada','Ecuador','Egypt'],
    'elo_rating': [1131.9, 1076.4, 1052.2, 1047.1, 1028.3, 1028.0, 1020.4, 1011.6, 1007.0, 1005.7,
                   999.0, 998.0, 995.0, 990.0, 988.0, 985.0, 983.0, 980.0, 978.0, 975.0,
                   972.0, 970.0, 968.0, 965.0, 962.0, 960.0, 958.0, 955.0, 952.0, 950.0, 948.0, 945.0],
    'stage': ['CHAMPION','QF','QF','QF','SF','SF','QF','QF','SF','QF',
              'R16','R32','R16','R32','R16','R16','SF','R32','R16','R32',
              'R16','R32','R16','R32','R16','R32','R32','QF','R16','R32','R32','R32']
}

df = pd.DataFrame(ratings_data)
df = df.sort_values('elo_rating', ascending=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chart 1: Top 20 teams bar chart
plt.figure(figsize=(12, 10))
top20 = df.nlargest(20, 'elo_rating')
colors = {'CHAMPION': '#FFD700', 'SF': '#C0C0C0', 'QF': '#CD7F32', 'R16': '#87CEEB', 'R32': '#90EE90'}
bar_colors = [colors.get(s, '#CCCCCC') for s in top20['stage']]
bars = plt.barh(top20['team'], top20['elo_rating'], color=bar_colors, edgecolor='white', linewidth=0.5)

plt.title('World Cup 2026 - Elo Rating by Team', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Elo Rating', fontsize=12)
plt.ylabel('Team', fontsize=12)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors['CHAMPION'], label='🏆 Champion'),
                   Patch(facecolor=colors['SF'], label='Semi Final'),
                   Patch(facecolor=colors['QF'], label='Quarter Final'),
                   Patch(facecolor=colors['R16'], label='Round of 16'),
                   Patch(facecolor=colors['R32'], label='Group Stage')]
plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wc2026_top20_ratings.png'), dpi=150)
print(f"✅ Saved: {OUTPUT_DIR}/wc2026_top20_ratings.png")

# Chart 2: Knockout bracket tree (simplified)
plt.figure(figsize=(14, 8))
stages = [
    ('Final', 8, [('France 🏆', 'Colombia')]),
    ('Semi Final', 5, [('France', 'Portugal'), ('Colombia', 'Austria')]),
    ('Quarter Final', 2.5, [('France', 'Netherlands'), ('Portugal', 'Croatia'), ('Colombia', 'Japan'), ('Austria', 'Cameroon')]),
]

for stage_name, y_pos, matches in stages:
    plt.text(-0.5, y_pos, stage_name, fontsize=11, fontweight='bold', ha='right', va='center')
    for i, (t1, t2) in enumerate(matches):
        x = i * 3 + 1
        plt.text(x, y_pos + 0.5, t1, fontsize=9, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='gray', alpha=0.7))
        plt.text(x, y_pos - 0.5, t2, fontsize=9, ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.7))

plt.xlim(-2, 10)
plt.ylim(0, 11)
plt.title('World Cup 2026 - Knockout Stage Results', fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wc2026_bracket.png'), dpi=150)
print(f"✅ Saved: {OUTPUT_DIR}/wc2026_bracket.png")

# Chart 3: Probability distribution (simulated from elo)
plt.figure(figsize=(12, 6))
# Convert elo to win probability using logistic function
df_32 = df.sort_values('elo_rating', ascending=False).head(32).copy()
avg_elo = df_32['elo_rating'].mean()
df_32['prob'] = 1 / (1 + 10 ** ((avg_elo - df_32['elo_rating']) / 400))
df_32['prob'] = df_32['prob'] / df_32['prob'].sum() * 100  # normalize to %

top10 = df_32.head(10).sort_values('prob', ascending=True)
prob_colors = ['#FFD700' if t == 'France' else '#CD7F32' for t in top10['team']]
plt.barh(top10['team'], top10['prob'], color=prob_colors, edgecolor='white')
plt.xlabel('Win Probability (%)', fontsize=12)
plt.title('World Cup 2026 - Top 10 Championship Probability', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wc2026_win_probability.png'), dpi=150)
print(f"✅ Saved: {OUTPUT_DIR}/wc2026_win_probability.png")

# Save prediction data
df_32.to_csv(os.path.join('predictions', 'wc2026_predictions.csv'), index=False)
print(f"✅ Saved: predictions/wc2026_predictions.csv")

print("\n=== ALL CHARTS GENERATED ===")
print(f"📊 1. wc2026_top20_ratings.png  — Elo Rating per Team")
print(f"📊 2. wc2026_bracket.png         — Knockout Bracket")
print(f"📊 3. wc2026_win_probability.png — Win Probability Top 10")