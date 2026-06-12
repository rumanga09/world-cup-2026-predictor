"""Visualisasi World Cup 2026 — 48 tim + grup stage."""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load results
with open('predictions/wc2026_full_results.json') as f:
    data = json.load(f)

# Chart 1: Championship Probability top 15
probs = data['championship_probability']
top15 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:15]
teams, pcts = zip(*top15)

plt.figure(figsize=(12, 8))
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(teams)))
bars = plt.barh(range(len(teams)), list(pcts), color=colors, edgecolor='white')
plt.yticks(range(len(teams)), teams)
plt.xlabel('Win Probability (%)', fontsize=12)
plt.title('World Cup 2026 - Championship Probability (Monte Carlo 1,000 runs)', fontsize=14, fontweight='bold')
for i, (_, p) in enumerate(top15):
    plt.text(p + 0.1, i, f'{p:.1f}%', va='center', fontsize=9)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wc2026_champ_prob_48.png'), dpi=150)
print(f"✅ {OUTPUT_DIR}/wc2026_champ_prob_48.png")

# Chart 2: Group winners heatmap
groups = data['groups']
group_names = sorted(groups.keys())
group_winners = [groups[g]['standings'][:2] for g in group_names]  # top 2

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
table_data = []
for gn in group_names:
    g = groups[gn]
    row = [f'Grup {gn}'] + g['standings'] + [f"1st: {g['standings'][0]}\n2nd: {g['standings'][1]}"]
    table_data.append(row)

col_labels = ['Group', '1st Place', '2nd Place', '3rd Place', '4th Place', 'Note']
table = ax.table(cellText=[[r[0], r[1], r[2], r[3], r[4], r[5]] for r in table_data],
                 colLabels=col_labels,
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.08, 0.25, 0.25, 0.25, 0.25, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.3)
plt.title('World Cup 2026 - Group Stage Standings', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wc2026_group_standings.png'), dpi=150)
print(f"✅ {OUTPUT_DIR}/wc2026_group_standings.png")

# Chart 3: Knockout bracket summary
champion = data['champion']
final_info = data['final'][0] if data['final'] else ('?', '?', champion)
sf_info = data['sf_matches']
qf_info = data['qf_matches']

plt.figure(figsize=(14, 6))
stages = [
    ('Final', [(final_info[0], final_info[2])]),
    ('Semi Final', [(s[0], s[2]) for s in sf_info]),
    ('Quarter Final', [(q[0], q[2]) for q in qf_info]),
]
for stage_name, matches in stages:
    y_pos_map = {'Final': 8, 'Semi Final': 5, 'Quarter Final': 2}
    y = y_pos_map.get(stage_name, 5)
    plt.text(-0.5, y, stage_name, fontsize=11, fontweight='bold', ha='right', va='center')
    for i, (t1, t2) in enumerate(matches):
        x = i * 3 + 1
        is_champ = t2 == champion
        c1 = '#FFD700' if is_champ else 'lightblue'
        c2 = 'lightblue'
        plt.text(x, y + 0.5, t1, fontsize=8, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=c2, edgecolor='gray', alpha=0.7))
        if t2 != t1:
            plt.text(x, y - 0.5, t2, fontsize=8, ha='center', va='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=c1, edgecolor='gold' if is_champ else 'gray', alpha=0.7))

plt.xlim(-2, 10)
plt.ylim(0, 11)
plt.title(f'World Cup 2026 - Knockout Stage (🏆 {champion})', fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wc2026_knockout_bracket.png'), dpi=150)
print(f"✅ {OUTPUT_DIR}/wc2026_knockout_bracket.png")

print("\n=== ALL 3 CHARTS GENERATED ===")