# ============================================================
# run_bert_embedding_analysis.py
# BERT Embedding Analysis for UMLS Graph Attack Detection
#
# Two goals:
#   1. Similarity analysis — cosine similarity between
#      original vs attacked question embeddings
#   2. Detection classifier — can a simple classifier
#      distinguish attacked from non-attacked questions?
#
# Model: BioBERT (dmis-lab/biobert-base-cased-v1.2)
# Input: umls_graph_attack_results.xlsx
# Output: bert_embedding_analysis_results.xlsx
#         bert_embedding_pca.png
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# ============================================================
# CONFIG
# ============================================================
MODEL_ID   = "dmis-lab/biobert-base-cased-v1.2"
INPUT_FILE = "umls_graph_attack_results.xlsx"
BATCH_SIZE = 32
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Loading BioBERT: {MODEL_ID}\n")

# ============================================================
# STEP 1: Load UMLS attack results
# ============================================================
print("Loading UMLS attack results...")
df = pd.read_excel(INPUT_FILE)
print(f"Loaded {len(df)} rows")
print(f"UMLS substitutions found: {df['umls_substitution'].notna().sum()}")
print(f"No substitution (fallback): {df['umls_substitution'].isna().sum()}\n")

original_questions = df["original_question"].tolist()
attacked_questions  = df["attacked_question"].tolist()

# Binary label: 1 = UMLS substitution applied, 0 = no change
labels = df["umls_substitution"].notna().astype(int).tolist()

# ============================================================
# STEP 2: Load BioBERT
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
print("BioBERT loaded.\n")

# ============================================================
# STEP 3: Embed questions
# Mean-pool over token embeddings (CLS token alone is weaker
# for sentence-level similarity in BERT-base models)
# ============================================================
def embed_texts(texts: list, batch_size: int = BATCH_SIZE) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            output = model(**encoded)
        # Mean pooling over token dimension (ignore padding)
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        token_embeddings = output.last_hidden_state
        mean_pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        all_embeddings.append(mean_pooled.cpu().numpy())

        if (i // batch_size + 1) % 5 == 0:
            done = min(i + batch_size, len(texts))
            print(f"  Embedded {done}/{len(texts)}")

    return np.vstack(all_embeddings)

print("Embedding original questions...")
original_embs = embed_texts(original_questions)
print(f"  Done. Shape: {original_embs.shape}\n")

print("Embedding attacked questions...")
attacked_embs = embed_texts(attacked_questions)
print(f"  Done. Shape: {attacked_embs.shape}\n")

# ============================================================
# STEP 4: Cosine similarity analysis
# Per-pair cosine similarity between original and attacked
# ============================================================
print("Computing pairwise cosine similarities...")
cos_sims = np.array([
    cosine_similarity(original_embs[i].reshape(1, -1),
                      attacked_embs[i].reshape(1, -1))[0, 0]
    for i in range(len(original_embs))
])
df["cosine_similarity"] = cos_sims.round(4)

# Split by whether UMLS substitution was actually applied
attacked_mask     = np.array(labels, dtype=bool)
not_attacked_mask = ~attacked_mask

sim_attacked     = cos_sims[attacked_mask]
sim_not_attacked = cos_sims[not_attacked_mask]

print(f"\n{'='*60}")
print(f"  COSINE SIMILARITY ANALYSIS")
print(f"{'='*60}")
print(f"  Questions with UMLS substitution (n={attacked_mask.sum()}):")
print(f"    Mean:   {sim_attacked.mean():.4f}")
print(f"    Std:    {sim_attacked.std():.4f}")
print(f"    Min:    {sim_attacked.min():.4f}")
print(f"    Max:    {sim_attacked.max():.4f}")
print(f"\n  Questions without substitution (n={not_attacked_mask.sum()}):")
print(f"    Mean:   {sim_not_attacked.mean():.4f}")
print(f"    Std:    {sim_not_attacked.std():.4f}")
print(f"    Min:    {sim_not_attacked.min():.4f}")
print(f"    Max:    {sim_not_attacked.max():.4f}")
print(f"\n  Delta (non-attacked - attacked): {(sim_not_attacked.mean() - sim_attacked.mean()):.4f}")
print(f"  (Lower similarity in attacked questions = embeddings drifted)")
print(f"{'='*60}\n")

# ============================================================
# STEP 5: Detection classifier
# Feature: difference vector (attacked_emb - original_emb)
# A perfect UMLS attack should produce near-zero diff for
# unmodified questions and a non-zero diff for modified ones.
# ============================================================
print("Building detection classifier...")

# Feature: L2-normalised difference vector + cosine sim as scalar feature
diff_features = attacked_embs - original_embs
scaler        = StandardScaler()
X             = scaler.fit_transform(
    np.hstack([diff_features, cos_sims.reshape(-1, 1)])
)
y = np.array(labels)

clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

# 5-fold cross-validation (stratified to handle class imbalance)
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs  = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
accs  = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

print(f"\n{'='*60}")
print(f"  DETECTION CLASSIFIER (Logistic Regression, 5-fold CV)")
print(f"{'='*60}")
print(f"  ROC-AUC:  {aucs.mean():.4f}  ± {aucs.std():.4f}")
print(f"  Accuracy: {accs.mean():.4f}  ± {accs.std():.4f}")
print(f"\n  Interpretation:")
if aucs.mean() >= 0.85:
    print(f"  ✅ Strong detection signal — UMLS attacks ARE detectable via BERT embeddings")
elif aucs.mean() >= 0.70:
    print(f"  ⚠️  Moderate signal — partial detectability")
else:
    print(f"  ✅ Weak signal — UMLS attacks are largely UNDETECTABLE via BERT embeddings")
    print(f"     (This is actually the best-case finding for the attack's stealth)")
print(f"{'='*60}\n")

# Full fit for classification report
clf.fit(X, y)
y_pred = clf.predict(X)
print("  Classification report (on full training set):")
print(classification_report(y, y_pred, target_names=["Not Attacked", "UMLS Attacked"]))

# ============================================================
# STEP 6: PCA visualisation
# Project embeddings to 2D to show separation (or lack of it)
# ============================================================
print("Running PCA for visualisation...")
pca = PCA(n_components=2, random_state=42)

# Stack original + attacked, fit PCA on original, transform both
all_embs   = np.vstack([original_embs, attacked_embs])
pca.fit(original_embs)
orig_2d    = pca.transform(original_embs)
attacked_2d = pca.transform(attacked_embs)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("BioBERT Embedding Analysis — UMLS Graph Attack Detection", fontsize=14, fontweight="bold")

# --- Plot 1: PCA scatter ---
ax = axes[0]
ax.set_title("PCA: Original vs Attacked Question Embeddings", fontsize=11)
# Unmodified questions (label=0)
ax.scatter(orig_2d[~attacked_mask, 0], orig_2d[~attacked_mask, 1],
           c="#4C72B0", alpha=0.4, s=15, label="Original (no substitution)")
# Modified original questions (label=1)
ax.scatter(orig_2d[attacked_mask, 0], orig_2d[attacked_mask, 1],
           c="#55A868", alpha=0.5, s=15, label="Original (UMLS substituted)")
# Attacked versions
ax.scatter(attacked_2d[attacked_mask, 0], attacked_2d[attacked_mask, 1],
           c="#C44E52", alpha=0.5, s=15, marker="x", label="Attacked version")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Plot 2: Cosine similarity distribution ---
ax2 = axes[1]
ax2.set_title("Cosine Similarity: Original vs Attacked\n(per question pair)", fontsize=11)
bins = np.linspace(cos_sims.min() - 0.01, 1.01, 40)
ax2.hist(sim_not_attacked, bins=bins, color="#4C72B0", alpha=0.6,
         label=f"No substitution (n={not_attacked_mask.sum()})")
ax2.hist(sim_attacked, bins=bins, color="#C44E52", alpha=0.6,
         label=f"UMLS attacked (n={attacked_mask.sum()})")
ax2.axvline(sim_attacked.mean(),     color="#C44E52", linestyle="--", linewidth=1.5,
            label=f"Attacked mean: {sim_attacked.mean():.3f}")
ax2.axvline(sim_not_attacked.mean(), color="#4C72B0", linestyle="--", linewidth=1.5,
            label=f"Unmodified mean: {sim_not_attacked.mean():.3f}")
ax2.set_xlabel("Cosine Similarity")
ax2.set_ylabel("Number of Questions")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("bert_embedding_pca.png", dpi=150, bbox_inches="tight")
print("  Saved: bert_embedding_pca.png\n")

# ============================================================
# STEP 7: Per-relation breakdown
# How well do embeddings separate by UMLS relation type?
# ============================================================
print(f"{'='*60}")
print(f"  PER-RELATION COSINE SIMILARITY BREAKDOWN")
print(f"{'='*60}")
for rel in df["umls_relation"].dropna().unique():
    mask = df["umls_relation"] == rel
    mean_sim = df.loc[mask, "cosine_similarity"].mean()
    n = mask.sum()
    print(f"  {rel:<35} n={n:<5} avg_cos_sim={mean_sim:.4f}")
print(f"{'='*60}\n")

# ============================================================
# STEP 8: Save full results
# ============================================================
df["bert_label"]          = labels
df["original_emb_norm"]   = np.linalg.norm(original_embs, axis=1).round(4)
df["attacked_emb_norm"]   = np.linalg.norm(attacked_embs,  axis=1).round(4)
df["emb_diff_l2"]         = np.linalg.norm(attacked_embs - original_embs, axis=1).round(4)

df.to_excel("bert_embedding_analysis_results.xlsx", index=False)
df.to_csv("bert_embedding_analysis_results.csv",    index=False)

print("✅ Saved: bert_embedding_analysis_results.xlsx")
print("✅ Saved: bert_embedding_analysis_results.csv")
print("\nDone.")
