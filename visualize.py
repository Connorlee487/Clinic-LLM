# ============================================================
# visualize.py
# Visualizes ROUGE + BLEU + BERTScore from evaluation results
# pip install matplotlib pandas openpyxl seaborn
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# STEP 1: Load your evaluation results Excel
# ============================================================
df = pd.read_excel("medgemma_evaluation_results.xlsx")  # output from evaluate.py
print(f"✅ Loaded {len(df)} rows\n")

sns.set_theme(style="whitegrid")

# ============================================================
# CHART 1: Average Score Comparison (Bar Chart)
# ============================================================
avg_scores = {
    "ROUGE-1":   df["rouge1"].mean(),
    "ROUGE-2":   df["rouge2"].mean(),
    "ROUGE-L":   df["rougeL"].mean(),
    "BLEU":      df["bleu"].mean(),
    "BERTScore": df["bertscore_f1"].mean(),
}

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(avg_scores.keys(), avg_scores.values(),
              color=["#4C72B0", "#4C72B0", "#4C72B0", "#DD8452", "#55A868"],
              edgecolor="white", width=0.5)

# Add value labels on bars
for bar, val in zip(bars, avg_scores.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_title("MedGemma 4B – Average Evaluation Scores", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Score (0 to 1)")
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig("chart1_average_scores.png", dpi=150)
plt.show()
print("✅ Chart 1 saved: chart1_average_scores.png")

# ============================================================
# CHART 2: Per-Question Score Heatmap
# ============================================================
heat_df = df[["Question", "rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]].copy()
heat_df["Question"] = heat_df["Question"].str[:40]  # truncate long questions
heat_df = heat_df.set_index("Question")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlGnBu",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Score"})
ax.set_title("Per-Question Evaluation Scores", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Metric")
ax.set_ylabel("Question")
plt.xticks(rotation=0)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("chart2_heatmap.png", dpi=150)
plt.show()
print("✅ Chart 2 saved: chart2_heatmap.png")

# ============================================================
# CHART 3: Score Distribution (Box Plot)
# ============================================================
plot_df = df[["rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]].copy()
plot_df.columns = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore"]

fig, ax = plt.subplots(figsize=(8, 5))
plot_df.boxplot(ax=ax, patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", color="white"),
                medianprops=dict(color="yellow", linewidth=2),
                whiskerprops=dict(color="gray"),
                capprops=dict(color="gray"))

ax.set_title("Score Distribution Across 20 Questions", fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Score (0 to 1)")
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig("chart3_boxplot.png", dpi=150)
plt.show()
print("✅ Chart 3 saved: chart3_boxplot.png")