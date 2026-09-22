# ============================================================
# run_attack_mistakes.py
# 1. Loads 1000 baseline questions + answers
# 2. Generates typo/abbreviation version of each question
# 3. Runs MedGemma on tweaked questions
# 4. Computes delta score = how much answer changed
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import random
import re
import pandas as pd
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login
import torch

import os; login(token=os.environ.get("HF_TOKEN"))

# ============================================================
# STEP 1: Load baseline results
# ============================================================
print("Loading baseline answers...")
df = pd.read_excel("pubmedqa_1000_answers.xlsx")
questions        = df["question"].tolist()
baseline_answers = df["medgemma_answer"].tolist()
ground_truths    = df["ground_truth"].tolist()
print(f"Loaded {len(df)} questions\n")

# ============================================================
# STEP 2: Typo + abbreviation attack
# ============================================================

# Medical abbreviations map
ABBREV = {
    "hypertension":     "HTN",
    "diabetes":         "DM",
    "diabetes mellitus":"T2DM",
    "myocardial infarction": "MI",
    "blood pressure":   "BP",
    "heart failure":    "HF",
    "chronic obstructive pulmonary disease": "COPD",
    "shortness of breath": "SOB",
    "history":          "Hx",
    "diagnosis":        "Dx",
    "treatment":        "Tx",
    "patient":          "pt",
    "patients":         "pts",
    "significant":      "sig",
    "approximately":    "approx",
    "increase":         "incr",
    "decrease":         "decr",
    "without":          "w/o",
    "with":             "w/",
    "bilateral":        "bilat",
}

# Common typo patterns
def introduce_typos(text: str, typo_rate: float = 0.15) -> str:
    words = text.split()
    result = []
    for word in words:
        if random.random() < typo_rate and len(word) > 3:
            typo_type = random.choice(["swap", "drop", "double"])
            if typo_type == "swap" and len(word) > 2:
                i = random.randint(0, len(word) - 2)
                word = word[:i] + word[i+1] + word[i] + word[i+2:]
            elif typo_type == "drop":
                i = random.randint(1, len(word) - 1)
                word = word[:i] + word[i+1:]
            elif typo_type == "double":
                i = random.randint(0, len(word) - 1)
                word = word[:i] + word[i] + word[i:]
        result.append(word)
    return " ".join(result)

def apply_abbreviations(text: str) -> str:
    for full, abbr in ABBREV.items():
        text = re.sub(re.escape(full), abbr, text, flags=re.IGNORECASE)
    return text

def generate_mistake_question(question: str) -> str:
    # Apply abbreviations first then typos
    q = apply_abbreviations(question)
    q = introduce_typos(q, typo_rate=0.15)
    return q

random.seed(42)  # reproducibility

print("Generating typo/abbreviation attack questions...")
mistake_questions = [generate_mistake_question(q) for q in questions]

# Show sample
print(f"\nSample:")
print(f"  Original : {questions[0]}")
print(f"  Attacked : {mistake_questions[0]}\n")

# ============================================================
# STEP 3: Load MedGemma
# ============================================================
MODEL_ID = "google/medgemma-4b-it"
print("Loading MedGemma 4B...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="cuda"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(f"Model loaded! GPU: {torch.cuda.memory_allocated()//1024**2} MB\n")

def ask_medgemma(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a biomedical research assistant. Answer in 1-2 sentences only. Be direct and concise. No bullet points, no headers, no lists."},
        {"role": "user",   "content": question}
    ]
    out = pipe(messages, max_new_tokens=80)
    return out[0]["generated_text"][-1]["content"].strip()

# ============================================================
# STEP 4: Run MedGemma on attacked questions
# ============================================================
print("Running MedGemma on attacked questions...")
attacked_answers = []
total = len(mistake_questions)

for i, q in enumerate(mistake_questions):
    print(f"[{i+1}/{total}] {q[:65]}...")
    ans = ask_medgemma(q)
    attacked_answers.append(ans)

    if (i + 1) % 100 == 0:
        pd.DataFrame({
            "question":         questions[:i+1],
            "mistake_question": mistake_questions[:i+1],
            "baseline_answer":  baseline_answers[:i+1],
            "attacked_answer":  attacked_answers,
        }).to_csv("attack_checkpoint.csv", index=False)
        print(f"  💾 Checkpoint saved: {i+1}/{total}")

# ============================================================
# STEP 5: Compute delta scores
# ============================================================
print("\nComputing delta BERTScore...")
print("(Comparing baseline answer vs attacked answer)")

_, _, F1 = bert_score(
    attacked_answers,
    baseline_answers,
    lang="en",
    model_type="distilbert-base-uncased",
    verbose=False
)
delta_scores = F1.tolist()

# ============================================================
# STEP 6: Save results
# ============================================================
result_df = pd.DataFrame({
    "qid":              df["qid"].tolist(),
    "original_question": questions,
    "mistake_question":  mistake_questions,
    "ground_truth":      ground_truths,
    "baseline_answer":   baseline_answers,
    "attacked_answer":   attacked_answers,
    "delta_bertscore":   [round(s, 4) for s in delta_scores],
})

result_df.to_excel("attack_mistakes_results.xlsx", index=False)
result_df.to_csv("attack_mistakes_results.csv",   index=False)

# ============================================================
# STEP 7: Summary
# ============================================================
avg_delta  = sum(delta_scores) / len(delta_scores)
min_delta  = min(delta_scores)
max_delta  = max(delta_scores)

# Questions where attack caused biggest change
result_df_sorted = result_df.sort_values("delta_bertscore")

print(f"\n{'='*60}")
print(f"📊 ATTACK IMPACT SUMMARY — MISTAKES (1000 questions)")
print(f"{'='*60}")
print(f"  Avg delta BERTScore : {avg_delta:.4f}")
print(f"  Min delta BERTScore : {min_delta:.4f}  ← most affected")
print(f"  Max delta BERTScore : {max_delta:.4f}  ← least affected")
print(f"\n  Interpretation:")
print(f"  Score = 1.0 → attack had NO effect on answer")
print(f"  Score < 0.8 → attack significantly changed answer")
print(f"\n  Questions most affected by typo attack:")
for _, row in result_df_sorted.head(5).iterrows():
    print(f"  [{row['delta_bertscore']:.3f}] {row['original_question'][:70]}...")
print(f"{'='*60}")
print(f"\n✅ Saved to attack_mistakes_results.xlsx")