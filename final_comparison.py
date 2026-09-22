# ============================================================
# final_comparison.py
# Builds final comparison charts for all attacks
# Run on Mac after all jobs complete
# pip install matplotlib seaborn pandas openpyxl
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# Results — fill in context switch when done
# ============================================================
results = {
    "Baseline":          {"delta": 1.0000, "min_delta": 1.0000, "success_rate": None},
    "Typo/Abbrev":       {"delta": 0.9156, "min_delta": 0.6458, "success_rate": None},
    "Rephrasing":        {"delta": 0.9156, "min_delta": 0.6553, "success_rate": None},
    "Meaning Change":    {"delta": 0.8878, "min_delta": 0.6490, "success_rate": None},
    "Brand Manip.":      {"delta": 0.8782, "min_delta": 0.6487, "success_rate": 95.5},
    "Task Switch":       {"delta": None,   "min_delta": None,   "success_rate": None},
    "Medical Hijack":    {"delta": None,   "min_delta": None,   "success_rate": None},
}

# ============================================================
# CHART 1 — Delta BERTScore comparison (bar chart)
# ============================================================
known = {k: v for k, v in results.items() if v["delta"] is not None}
labels = list(known.keys())
deltas = [v["delta"] for v in known.values()]
colors = ["#4caf82" if d >= 0.95 else "#f0b429" if d >= 0.88 else "#f06060"
          for d in deltas]
colors[0] = "#5b8dee"  # baseline always blue

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("MedGemma 4B — Prompt Attack Robustness Evaluation\n1,000 PubMedQA Questions",
             fontsize=14, fontweight="bold", y=1.02)

# Bar chart
bars = axes[0].bar(labels, deltas, color=colors, edgecolor="white", width=0.6)
axes[0].axhline(y=0.95, color="#4caf82", linestyle="--", alpha=0.5, label="Robust threshold (0.95)")
axes[0].axhline(y=0.88, color="#f0b429", linestyle="--", alpha=0.5, label="Moderate threshold (0.88)")
for bar, val in zip(bars, deltas):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].set_title("Avg Delta BERTScore by Attack Type\n(1.0 = no change, lower = more harmful)",
                  fontsize=11, fontweight="bold")
axes[0].set_ylabel("Delta BERTScore")
axes[0].set_ylim(0.80, 1.05)
axes[0].tick_params(axis='x', rotation=25)
axes[0].legend(fontsize=9)

green_patch  = mpatches.Patch(color="#4caf82", label="Robust (≥0.95)")
yellow_patch = mpatches.Patch(color="#f0b429", label="Moderate (0.88–0.95)")
red_patch    = mpatches.Patch(color="#f06060", label="Vulnerable (<0.88)")
blue_patch   = mpatches.Patch(color="#5b8dee", label="Baseline")
axes[0].legend(handles=[blue_patch, green_patch, yellow_patch, red_patch], fontsize=9)

# ============================================================
# CHART 2 — Attack Success Rate (where applicable)
# ============================================================
success_data = {k: v["success_rate"] for k, v in results.items()
                if v["success_rate"] is not None}

if success_data:
    s_labels = list(success_data.keys())
    s_values = list(success_data.values())
    s_colors = ["#f06060" if v > 50 else "#f0b429" for v in s_values]
    bars2 = axes[1].bar(s_labels, s_values, color=s_colors, edgecolor="white", width=0.4)
    for bar, val in zip(bars2, s_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[1].set_title("Attack Success Rate\n(% of answers manipulated)",
                      fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_ylim(0, 110)
    axes[1].axhline(y=50, color="#f0b429", linestyle="--", alpha=0.5, label="50% threshold")
    axes[1].legend(fontsize=9)
else:
    axes[1].text(0.5, 0.5, "Success rate data\ncoming soon",
                 ha="center", va="center", transform=axes[1].transAxes,
                 fontsize=14, color="#666")
    axes[1].set_title("Attack Success Rate", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("attack_comparison_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: attack_comparison_chart.png")

# ============================================================
# CHART 3 — Threat level heatmap
# ============================================================
fig2, ax = plt.subplots(figsize=(10, 5))

attack_names = ["Typo/Abbrev", "Rephrasing", "Meaning\nChange", "Brand\nManip.", "Task\nSwitch*", "Medical\nHijack*"]
metric_names = ["Avg Delta\nBERTScore", "Min Delta\nBERTScore", "Clinical\nDanger", "Detection\nDifficulty"]

# Normalized 0-1 threat scores (1 = most dangerous)
threat_matrix = np.array([
    # Avg delta  Min delta  Clinical danger  Detection difficulty
    [0.084,      0.354,     0.3,             0.4],   # Typo
    [0.084,      0.345,     0.3,             0.5],   # Rephrase
    [0.112,      0.351,     0.6,             0.6],   # Meaning change
    [0.122,      0.351,     0.9,             0.9],   # Brand manip
    [0.0,        0.0,       0.5,             0.7],   # Task switch (TBD)
    [0.0,        0.0,       1.0,             0.8],   # Medical hijack (TBD)
])

im = ax.imshow(threat_matrix.T, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(attack_names)))
ax.set_xticklabels(attack_names, fontsize=10)
ax.set_yticks(range(len(metric_names)))
ax.set_yticklabels(metric_names, fontsize=10)
ax.set_title("Attack Threat Level Heatmap\n(darker = more dangerous)", fontsize=12, fontweight="bold")

for i in range(len(attack_names)):
    for j in range(len(metric_names)):
        val = threat_matrix[i, j]
        if val > 0:
            ax.text(i, j, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if val > 0.6 else "black", fontweight="bold")
        else:
            ax.text(i, j, "TBD", ha="center", va="center", fontsize=9, color="#666")

plt.colorbar(im, ax=ax, label="Threat Level (0=safe, 1=critical)")
plt.tight_layout()
plt.savefig("threat_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: threat_heatmap.png")
print("\nOnce context switch results are in, update 'results' dict at top of script and re-run!")