# ============================================================
# prepare_medqa.py
# Downloads MedQA USMLE test split (1,273 questions) from
# HuggingFace and formats it to match pubmedqa structure.
#
# Source: GBaker/MedQA-USMLE-4-options (standard Parquet)
# Run on Delta LOGIN NODE (has internet access):
#   python prepare_medqa.py
#
# Output: medqa_1273_questions.xlsx
#         medqa_1273_questions.csv
# ============================================================

import pandas as pd
from datasets import load_dataset

print("Downloading MedQA USMLE test split...")
ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
print(f"Downloaded {len(ds)} questions\n")

rows = []
for i, item in enumerate(ds):
    question    = item["question"]
    options     = item["options"]        # dict: {'A': '...', 'B': '...', ...}
    answer_idx  = item["answer_idx"]     # e.g. 'B'
    ground_truth = item["answer"]        # full text of correct answer

    # Format options as a string for inclusion in prompts
    options_str = " | ".join(f"{k}: {v}" for k, v in options.items())

    rows.append({
        "qid":          f"medqa_{i:04d}",
        "question":     question,
        "options":      options_str,
        "answer_idx":   answer_idx,
        "ground_truth": ground_truth,
        "meta_info":    item.get("meta_info", ""),
    })

df = pd.DataFrame(rows)
print(f"Built dataframe: {df.shape}")
print(f"\nSample question:\n  {df.iloc[0]['question'][:120]}...")
print(f"\nSample options:\n  {df.iloc[0]['options'][:120]}")
print(f"\nSample answer:\n  {df.iloc[0]['ground_truth']}")

df.to_excel("medqa_1273_questions.xlsx", index=False)
df.to_csv("medqa_1273_questions.csv",    index=False)

print(f"\n✅ Saved: medqa_1273_questions.xlsx  ({len(df)} rows)")
print(f"✅ Saved: medqa_1273_questions.csv")
print(f"\nDone.")