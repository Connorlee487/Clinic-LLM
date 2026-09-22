# ============================================================
# evaluate.py
# Computes ROUGE + BERTScore + BLEU from your existing Excel file
# Install dependencies:
# pip install rouge-score bert-score pandas openpyxl nltk
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import sacrebleu

# ============================================================
# STEP 1: Load your Excel file
# ============================================================
df = pd.read_excel("pubmedqa_1000_answers.xlsx") # 👈 replace with your actual filename

questions         = df["question"].tolist()
reference_answers = df["ground_truth"].tolist()
generated_answers = df["medgemma_answer"].tolist()

print(f"✅ Loaded {len(questions)} rows from Excel\n")

# ============================================================
# STEP 2: Compute ROUGE Scores
# ============================================================
print("Computing ROUGE scores...")
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

for ref, gen in zip(reference_answers, generated_answers):
    ref = str(ref) if pd.notna(ref) else ""
    gen = str(gen) if pd.notna(gen) else ""
    s = scorer.score(ref, gen)
    rouge1_scores.append(round(s["rouge1"].fmeasure, 4))
    rouge2_scores.append(round(s["rouge2"].fmeasure, 4))
    rougeL_scores.append(round(s["rougeL"].fmeasure, 4))

print("✅ ROUGE done\n")

# ============================================================
# STEP 3: Compute BLEU Scores
# ============================================================
print("Computing BLEU scores...")
bleu_scores = []

for ref, gen in zip(reference_answers, generated_answers):
    ref = str(ref) if pd.notna(ref) else ""
    gen = str(gen) if pd.notna(gen) else ""
    bleu = sacrebleu.sentence_bleu(gen, [ref]).score / 100  # sacrebleu returns 0-100, normalize to 0-1
    bleu_scores.append(round(bleu, 4))

print("✅ BLEU done\n")

# ============================================================
# STEP 4: Compute BERTScore
# ============================================================
print("Computing BERTScore (may take a minute)...")
P, R, F1 = bert_score(
    generated_answers,
    reference_answers,
    lang="en",
    model_type="distilbert-base-uncased",
    verbose=False
)
bert_f1_scores = [round(f, 4) for f in F1.tolist()]
print("✅ BERTScore done\n")

# ============================================================
# STEP 5: Print Summary
# ============================================================
avg = lambda lst: round(sum(lst) / len(lst), 4)

print("=" * 55)
print("📊 EVALUATION SUMMARY (averages across all questions)")
print("=" * 55)
print(f"  ROUGE-1   : {avg(rouge1_scores)}")
print(f"  ROUGE-2   : {avg(rouge2_scores)}")
print(f"  ROUGE-L   : {avg(rougeL_scores)}")
print(f"  BLEU      : {avg(bleu_scores)}")
print(f"  BERTScore : {avg(bert_f1_scores)}")
print("=" * 55)

# ============================================================
# STEP 6: Save Results to Excel
# ============================================================
df["rouge1"]       = rouge1_scores
df["rouge2"]       = rouge2_scores
df["rougeL"]       = rougeL_scores
df["bleu"]         = bleu_scores
df["bertscore_f1"] = bert_f1_scores

output_file = "medgemma_evaluation_results.xlsx"
df.to_excel(output_file, index=False)
print(f"\n✅ Full results saved to: {output_file}")

# ============================================================
# STEP 7: Per-question breakdown
# ============================================================
print("\n📋 Per-Question Scores:")
print(df[["question", "rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1"]].to_string(index=False))