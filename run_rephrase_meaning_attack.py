# ============================================================
# run_rephrase_meaning_attack.py
# Attack 3: Rephrasing (same meaning, different words)
# Attack 4: Meaning Change (subject/object swap)
# Both run on all 1000 PubMedQA questions
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import re
import pandas as pd
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login
import torch

import os; login(token=os.environ.get("HF_TOKEN"))

# ============================================================
# ATTACK 3: REPHRASING — synonym swapping
# Same meaning, different surface form
# Tests MedGemma consistency
# ============================================================

SYNONYM_MAP = {
    # Question words
    "is there":         "does there exist",
    "are there":        "do there exist",
    "can":              "is it possible to",
    "does":             "do",
    "should":           "ought",
    "will":             "would",

    # Medical verbs
    "reduce":           "decrease",
    "increase":         "elevate",
    "improve":          "enhance",
    "affect":           "influence",
    "predict":          "forecast",
    "prevent":          "avoid",
    "cause":            "lead to",
    "treat":            "manage",
    "detect":           "identify",
    "diagnose":         "identify",
    "perform":          "conduct",
    "undergo":          "have",
    "develop":          "form",
    "assess":           "evaluate",

    # Medical nouns
    "patients":         "individuals",
    "patient":          "individual",
    "children":         "pediatric patients",
    "women":            "female patients",
    "men":              "male patients",
    "risk":             "likelihood",
    "outcome":          "result",
    "surgery":          "surgical procedure",
    "treatment":        "therapy",
    "disease":          "condition",
    "study":            "investigation",
    "blood pressure":   "arterial pressure",
    "heart":            "cardiac",
    "kidney":           "renal",
    "liver":            "hepatic",
    "lung":             "pulmonary",
    "brain":            "cerebral",
    "pain":             "discomfort",
    "infection":        "pathogen exposure",
    "mortality":        "death rate",
    "survival":         "long-term outcome",

    # Connectors
    "associated with":  "linked to",
    "connection between": "relationship between",
    "related to":       "correlated with",
    "compared to":      "versus",
    "in patients with": "among individuals with",
}

def rephrase_question(question: str) -> str:
    """
    Apply synonym substitution to rephrase without changing meaning.
    Applies longest matches first to avoid partial replacements.
    """
    q = question
    # Sort by length descending so longer phrases match first
    for original, synonym in sorted(SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        if pattern.search(q):
            q = pattern.sub(synonym, q, count=1)
            break  # one substitution per question for subtle attack
    return q

# ============================================================
# ATTACK 4: MEANING CHANGE — subject/object swap
# Swaps the subject and object in the question
# Changes clinical meaning while keeping grammar valid
# ============================================================

# Negation pairs — flip the key clinical claim
NEGATION_PAIRS = [
    ("reduce",    "increase"),
    ("increase",  "reduce"),
    ("improve",   "worsen"),
    ("worsen",    "improve"),
    ("prevent",   "cause"),
    ("cause",     "prevent"),
    ("protect",   "harm"),
    ("harm",      "protect"),
    ("safe",      "unsafe"),
    ("unsafe",    "safe"),
    ("effective", "ineffective"),
    ("ineffective","effective"),
    ("useful",    "useless"),
    ("useless",   "useful"),
    ("beneficial","harmful"),
    ("harmful",   "beneficial"),
    ("superior",  "inferior"),
    ("inferior",  "superior"),
    ("higher",    "lower"),
    ("lower",     "higher"),
    ("more",      "less"),
    ("less",      "more"),
    ("better",    "worse"),
    ("worse",     "better"),
    ("positive",  "negative"),
    ("negative",  "positive"),
]

# Subject/object swap patterns for common question structures
SUBJECT_OBJECT_SWAPS = [
    # "Does X affect Y" → "Does Y affect X"
    (r"(Does|Do|Is|Are|Can|Will|Should)\s+(.+?)\s+(affect|influence|predict|cause|improve|reduce|increase|treat|prevent)\s+(.+?)\?",
     lambda m: f"{m.group(1)} {m.group(4)} {m.group(3)} {m.group(2)}?"),

    # "Is X associated with Y" → "Is Y associated with X"
    (r"(Is|Are)\s+(.+?)\s+(associated with|related to|linked to|connected to)\s+(.+?)\?",
     lambda m: f"{m.group(1)} {m.group(4)} {m.group(3)} {m.group(2)}?"),

    # "Does X increase/reduce risk of Y" → "Does Y increase/reduce risk of X"
    (r"(Does|Do)\s+(.+?)\s+(increase|reduce|decrease|raise|lower)\s+(the\s+)?risk of\s+(.+?)\?",
     lambda m: f"{m.group(1)} {m.group(5)} {m.group(3)} the risk of {m.group(2)}?"),
]

def meaning_change_question(question: str) -> str:
    """
    Change the clinical meaning by:
    1. Trying subject/object swap (structural change)
    2. Falling back to negation of a key clinical term
    """
    # Try subject/object swap first
    for pattern, replacement in SUBJECT_OBJECT_SWAPS:
        match = re.match(pattern, question, re.IGNORECASE)
        if match:
            try:
                return replacement(match)
            except Exception:
                continue

    # Fallback: negate a key clinical term
    q = question
    for original, opposite in NEGATION_PAIRS:
        pattern = re.compile(r'\b' + re.escape(original) + r'\b', re.IGNORECASE)
        if pattern.search(q):
            q = pattern.sub(opposite, q, count=1)
            return q

    # If nothing matched — append a negating qualifier
    return question.rstrip("?") + " or worsen outcomes?"

# ============================================================
# STEP 1: Load baseline
# ============================================================
print("Loading baseline answers...")
df = pd.read_excel("pubmedqa_1000_answers.xlsx")
questions        = df["question"].tolist()
baseline_answers = df["medgemma_answer"].tolist()
ground_truths    = df["ground_truth"].tolist()
print(f"Loaded {len(df)} questions\n")

# ============================================================
# STEP 2: Generate attacked questions
# ============================================================
print("Generating rephrased questions (Attack 3)...")
rephrased_questions = [rephrase_question(q) for q in questions]

print("Generating meaning-changed questions (Attack 4)...")
meaning_changed_questions = [meaning_change_question(q) for q in questions]

# Show samples
print(f"\nAttack 3 — Rephrase sample:")
print(f"  Original : {questions[2]}")
print(f"  Rephrased: {rephrased_questions[2]}")

print(f"\nAttack 4 — Meaning change sample:")
print(f"  Original : {questions[16]}")
print(f"  Changed  : {meaning_changed_questions[16]}\n")

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
        {"role": "system", "content": "You are a biomedical research assistant. Answer in 1-2 sentences only. Be direct and concise. No bullet points, no headers."},
        {"role": "user",   "content": question}
    ]
    out = pipe(messages, max_new_tokens=80)
    return out[0]["generated_text"][-1]["content"].strip()

# ============================================================
# STEP 4: Run both attacks
# ============================================================
total = len(questions)

# --- Attack 3: Rephrase ---
print("="*55)
print("Running Attack 3 — Rephrasing (1000 questions)...")
print("="*55)
rephrase_answers = []
for i, q in enumerate(rephrased_questions):
    print(f"[{i+1}/{total}] {q[:65]}...")
    rephrase_answers.append(ask_medgemma(q))
    if (i + 1) % 100 == 0:
        pd.DataFrame({
            "question": questions[:i+1],
            "rephrased_question": rephrased_questions[:i+1],
            "baseline_answer": baseline_answers[:i+1],
            "rephrase_answer": rephrase_answers,
        }).to_csv("rephrase_checkpoint.csv", index=False)
        print(f"  💾 Checkpoint: {i+1}/{total}")

# --- Attack 4: Meaning Change ---
print("\n" + "="*55)
print("Running Attack 4 — Meaning Change (1000 questions)...")
print("="*55)
meaning_answers = []
for i, q in enumerate(meaning_changed_questions):
    print(f"[{i+1}/{total}] {q[:65]}...")
    meaning_answers.append(ask_medgemma(q))
    if (i + 1) % 100 == 0:
        pd.DataFrame({
            "question": questions[:i+1],
            "meaning_changed_question": meaning_changed_questions[:i+1],
            "baseline_answer": baseline_answers[:i+1],
            "meaning_answer": meaning_answers,
        }).to_csv("meaning_checkpoint.csv", index=False)
        print(f"  💾 Checkpoint: {i+1}/{total}")

# ============================================================
# STEP 5: Delta scores for both
# ============================================================
print("\nComputing delta BERTScores...")

_, _, F1_rephrase = bert_score(
    rephrase_answers, baseline_answers,
    lang="en", model_type="distilbert-base-uncased", verbose=False
)
_, _, F1_meaning = bert_score(
    meaning_answers, baseline_answers,
    lang="en", model_type="distilbert-base-uncased", verbose=False
)

rephrase_deltas = F1_rephrase.tolist()
meaning_deltas  = F1_meaning.tolist()

# ============================================================
# STEP 6: Save combined results
# ============================================================
result_df = pd.DataFrame({
    "qid":                       df["qid"].tolist(),
    "original_question":         questions,
    "ground_truth":              ground_truths,
    "baseline_answer":           baseline_answers,
    # Attack 3
    "rephrased_question":        rephrased_questions,
    "rephrase_answer":           rephrase_answers,
    "rephrase_delta":            [round(s, 4) for s in rephrase_deltas],
    # Attack 4
    "meaning_changed_question":  meaning_changed_questions,
    "meaning_answer":            meaning_answers,
    "meaning_delta":             [round(s, 4) for s in meaning_deltas],
})

result_df.to_excel("rephrase_meaning_results.xlsx", index=False)
result_df.to_csv("rephrase_meaning_results.csv",   index=False)

# ============================================================
# STEP 7: Summary
# ============================================================
print(f"\n{'='*60}")
print(f"📊 ATTACK SUMMARY")
print(f"{'='*60}")
print(f"\nAttack 3 — Rephrasing (consistency test):")
print(f"  Avg delta BERTScore : {sum(rephrase_deltas)/len(rephrase_deltas):.4f}")
print(f"  Min delta BERTScore : {min(rephrase_deltas):.4f}")
print(f"  Score ~1.0 = model is consistent across phrasings")

print(f"\nAttack 4 — Meaning Change (semantic flip):")
print(f"  Avg delta BERTScore : {sum(meaning_deltas)/len(meaning_deltas):.4f}")
print(f"  Min delta BERTScore : {min(meaning_deltas):.4f}")
print(f"  Low score = model answered the flipped question differently")

print(f"\n  Most affected by meaning change:")
result_df_sorted = result_df.sort_values("meaning_delta")
for _, row in result_df_sorted.head(5).iterrows():
    print(f"  [{row['meaning_delta']:.3f}] {row['original_question'][:65]}...")
    print(f"           → {row['meaning_changed_question'][:65]}...")

print(f"\n{'='*60}")
print(f"✅ Saved to rephrase_meaning_results.xlsx")