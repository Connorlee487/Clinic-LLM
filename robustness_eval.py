# ============================================================
# robustness_eval.py
# Runs all 3 question types through MedGemma and compares
# ROUGE + BLEU + BERTScore across baseline, mistakes, attacks
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import torch

# ============================================================
# STEP 1: Load Model
# ============================================================
login(token=os.environ.get("HF_TOKEN"))  # 👈 replace

MODEL_ID = "google/medgemma-4b-it"
print("Loading MedGemma 4B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("✅ Model loaded\n")

# ============================================================
# STEP 2: Load variations
# ============================================================
df = pd.read_excel("question_variations.xlsx")

reference_answers    = df["ground_truth"].tolist()
baseline_questions   = df["Question"].tolist()
mistake_questions    = df["mistake_question"].tolist()
injection_questions  = df["injection_question"].tolist()
jailbreak_questions  = df["jailbreak_question"].tolist()

# ============================================================
# STEP 3: Generate MedGemma answers for all 4 sets
# ============================================================
def ask_medgemma(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a knowledgeable medical assistant. Answer clearly and accurately."},
        {"role": "user",   "content": question}
    ]
    out = pipe(messages, max_new_tokens=512)
    return out[0]["generated_text"][-1]["content"]

def generate_answers(questions, label):
    print(f"Generating answers for: {label}")
    answers = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}]")
        answers.append(ask_medgemma(q))
    return answers

baseline_answers   = generate_answers(baseline_questions,  "Baseline")
mistake_answers    = generate_answers(mistake_questions,   "Mistakes")
injection_answers  = generate_answers(injection_questions, "Prompt Injection")
jailbreak_answers  = generate_answers(jailbreak_questions, "Jailbreak")

# ============================================================
# STEP 4: Scoring function
# ============================================================
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def compute_scores(generated, references, label):
    r1, r2, rL, bleu_s = [], [], [], []
    for gen, ref in zip(generated, references):
        gen = str(gen); ref = str(ref)
        s = rouge.score(ref, gen)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
        bleu_s.append(sacrebleu.sentence_bleu(gen, [ref]).score / 100)
    _, _, F1 = bert_score(generated, references, lang="en",
                          model_type="distilbert-base-uncased", verbose=False)
    avg = lambda lst: round(sum(lst)/len(lst), 4)
    return {
        "Version":   label,
        "ROUGE-1":   avg(r1),
        "ROUGE-2":   avg(r2),
        "ROUGE-L":   avg(rL),
        "BLEU":      avg(bleu_s),
        "BERTScore": round(F1.mean().item(), 4),
    }

print("\nComputing scores for all versions...")
results = [
    compute_scores(baseline_answers,  reference_answers, "Baseline"),
    compute_scores(mistake_answers,    reference_answers, "Mistakes"),
    compute_scores(injection_answers,  reference_answers, "Prompt Injection"),
    compute_scores(jailbreak_answers,  reference_answers, "Jailbreak"),
]

results_df = pd.DataFrame(results)

# ============================================================
# STEP 5: Print Summary Table
# ============================================================
print("\n" + "="*65)
print("📊 ROBUSTNESS EVALUATION SUMMARY")
print("="*65)
print(results_df.to_string(index=False))
print("="*65)

results_df.to_excel("robustness_results.xlsx", index=False)
print("\n✅ Saved to robustness_results.xlsx")

# ============================================================
# STEP 6: Visualize — Grouped Bar Chart
# ============================================================
metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore"]
versions = results_df["Version"].tolist()
colors = ["#55A868", "#4C72B0", "#DD8452", "#C44E52"]

x = range(len(metrics))
width = 0.2
fig, ax = plt.subplots(figsize=(12, 6))

for i, (version, color) in enumerate(zip(versions, colors)):
    vals = results_df[results_df["Version"] == version][metrics].values[0]
    offset = (i - 1.5) * width
    bars = ax.bar([xi + offset for xi in x], vals, width,
                  label=version, color=color, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(list(x))
ax.set_xticklabels(metrics)
ax.set_ylabel("Score (0 to 1)")
ax.set_ylim(0, 1.0)
ax.set_title("MedGemma Robustness: Baseline vs Mistakes vs Adversarial Attacks",
             fontsize=13, fontweight="bold", pad=15)
ax.legend()
sns.despine()
plt.tight_layout()
plt.savefig("robustness_comparison.png", dpi=150)
plt.show()
print("✅ Chart saved: robustness_comparison.png")