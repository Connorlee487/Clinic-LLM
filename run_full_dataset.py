# ============================================================
# run_pubmedqa_1000.py
# Downloads PubMedQA, runs MedGemma on all 1000 questions
# Saves question + ground_truth + medgemma_answer to Excel
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login
import torch

# ============================================================
# STEP 1: Login
# ============================================================
import os; login(token=os.environ.get("HF_TOKEN"))

# ============================================================
# STEP 2: Load PubMedQA — all 1000 labeled questions
# ============================================================
print("Loading PubMedQA dataset...")
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

questions     = []
ground_truths = []

for item in dataset:
    q   = item.get("question", "").strip()
    ans = item.get("long_answer", "").strip()
    if q and ans:
        questions.append(q)
        ground_truths.append(ans)

print(f"✅ Loaded {len(questions)} questions")
print(f"   Sample Q:  {questions[0][:80]}")
print(f"   Sample GT: {ground_truths[0][:80]}\n")

# ============================================================
# STEP 3: Load MedGemma with 4-bit quantization
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
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda",
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(f"✅ Model loaded! GPU memory: {torch.cuda.memory_allocated()//1024**2} MB\n")

# ============================================================
# STEP 4: Generate answers
# ============================================================
def ask_medgemma(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical research assistant. "
                "Answer in 1-2 sentences only. "
                "Be direct and concise. "
                "No bullet points, no headers, no lists."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]
    out = pipe(messages, max_new_tokens=80)  # strict token limit
    return out[0]["generated_text"][-1]["content"].strip()

# ============================================================
# STEP 5: Run on all questions with checkpointing
# ============================================================
results = []
total   = len(questions)

for i, (q, gt) in enumerate(zip(questions, ground_truths)):
    print(f"[{i+1}/{total}] {q[:70]}...")
    ans = ask_medgemma(q)
    print(f"         → {ans[:80]}...")

    results.append({
        "qid":            f"PQ_{str(i).zfill(4)}",
        "question":       q,
        "ground_truth":   gt,
        "medgemma_answer": ans,
    })

    # Save checkpoint every 100 questions
    if (i + 1) % 100 == 0:
        pd.DataFrame(results).to_csv("checkpoint.csv", index=False)
        print(f"\n💾 Checkpoint saved — {i+1}/{total} done\n")

# ============================================================
# STEP 6: Save final output
# ============================================================
df = pd.DataFrame(results)
df.to_excel("pubmedqa_1000_answers.xlsx", index=False)
df.to_csv("pubmedqa_1000_answers.csv",    index=False)

print(f"\n{'='*60}")
print(f"✅ DONE! Saved {len(df)} rows to pubmedqa_1000_answers.xlsx")
print(f"   Columns: qid | question | ground_truth | medgemma_answer")
print(f"\n📊 Avg answer length:")
print(f"   Ground truth:    {df['ground_truth'].str.split().str.len().mean():.0f} words")
print(f"   MedGemma answer: {df['medgemma_answer'].str.split().str.len().mean():.0f} words")
print(f"{'='*60}")