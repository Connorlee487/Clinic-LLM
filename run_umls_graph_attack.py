# ============================================================
# run_umls_graph_attack.py
# Attack 8: Real UMLS Knowledge Graph Attack
# Modes: sub | add | add_k | hop2 | new_q
# Use --questions-only to sample attacked questions without
# answering or computing BERTScore.
# Use --from-csv + --eval-n to answer a sample of valid rows
# from a prior questions-only CSV.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

import pandas as pd

from umls_substitution import (
    ATTACK_MODES,
    EXTRACTORS,
    apply_umls_mode,
    rewrite_for_fluency,
)

# ============================================================
# CLI FLAGS
# ============================================================
parser = argparse.ArgumentParser(description="UMLS knowledge graph attack")
parser.add_argument("--extractor", choices=EXTRACTORS, default="hybrid",
                    help="entity extractor: keyword list, scispaCy, or scispaCy + keyword fallback")
parser.add_argument("--mode", choices=ATTACK_MODES, default="sub",
                    help="sub = 1-hop substitute; add = LLM sentence with top-1 neighbour; "
                         "add_k = LLM sentence with top-k neighbours; hop2 = 2-hop substitute; "
                         "new_q = LLM writes a new question from top-k neighbours")
parser.add_argument("--k", type=int, default=5,
                    help="number of UMLS neighbours for add_k / new_q (add forces k=1)")
parser.add_argument("--n", type=int, default=0,
                    help="number of questions to run (0 = all)")
parser.add_argument("--questions-only", action="store_true",
                    help="build attacked questions and write CSV; skip answers and BERTScore")
parser.add_argument("--from-csv", type=str, default=None,
                    help="evaluate an existing questions CSV instead of generating attacks")
parser.add_argument("--eval-n", type=int, default=0,
                    help="with --from-csv: sample this many valid rows (0 = all valid)")
parser.add_argument("--eval-seed", type=int, default=42,
                    help="random seed for --eval-n sampling")
parser.add_argument("--fluency-rewrite", action="store_true",
                    help="for sub: LLM fluency rewrite after substitution "
                         "(also applied in --from-csv eval when mode is sub)")
args = parser.parse_args()

effective_k = 1 if args.mode == "add" else args.k
needs_k_tag = args.mode in ("add_k", "new_q")
TAG = f"{args.extractor}_{args.mode}" + (f"_k{effective_k}" if needs_k_tag else "")
RESULTS_STEM    = f"umls_graph_attack_results_{TAG}"
CHECKPOINT_PATH = f"umls_checkpoint_{TAG}.csv"
LLM_MODES = ("add", "add_k", "new_q")

# ============================================================
# Helpers
# ============================================================
def _truthy(val) -> bool:
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return str(val).strip().lower() in ("true", "1", "yes")


def _has_text(val) -> bool:
    return isinstance(val, str) and bool(val.strip()) and val.strip().lower() != "nan"


def is_valid_attack_row(row: pd.Series, mode: str) -> bool:
    """Valid = llm_ok for LLM modes; successful sub with no fail_reason otherwise."""
    fail = row.get("fail_reason")
    if _has_text(fail):
        return False
    attacked = row.get("attacked_question")
    original = row.get("question", row.get("original_question"))
    if not _has_text(attacked) or not _has_text(original):
        return False
    if attacked.strip() == original.strip():
        return False
    if mode in LLM_MODES:
        return _truthy(row.get("llm_ok"))
    terms = row.get("terms_used", row.get("umls_substitution"))
    return _has_text(terms)


def tag_from_csv_path(path: str, mode: str, extractor: str, k: int) -> str:
    stem = Path(path).stem
    prefix = "umls_graph_attack_results_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    needs_k = mode in ("add_k", "new_q")
    return f"{extractor}_{mode}" + (f"_k{k}" if needs_k else "")


# ============================================================
# Eval path: answer valid rows from an existing CSV
# ============================================================
if args.from_csv:
    if args.questions_only:
        raise SystemExit("--from-csv cannot be combined with --questions-only")

    print(f"Loading attacks from {args.from_csv}...")
    eval_df = pd.read_csv(args.from_csv)
    mode_col = eval_df["mode"].iloc[0] if "mode" in eval_df.columns and len(eval_df) else args.mode
    mode = str(mode_col) if pd.notna(mode_col) else args.mode
    args.mode = mode
    effective_k = 1 if mode == "add" else args.k
    TAG = tag_from_csv_path(args.from_csv, mode, args.extractor, effective_k)

    # Normalize column names from questions-only vs full-result CSVs
    if "question" not in eval_df.columns and "original_question" in eval_df.columns:
        eval_df = eval_df.rename(columns={"original_question": "question"})
    if "terms_used" not in eval_df.columns and "umls_substitution" in eval_df.columns:
        eval_df = eval_df.rename(columns={"umls_substitution": "terms_used"})
    if "entity" not in eval_df.columns and "original_entity" in eval_df.columns:
        eval_df = eval_df.rename(columns={"original_entity": "entity"})
    if "relation" not in eval_df.columns and "umls_relation" in eval_df.columns:
        eval_df = eval_df.rename(columns={"umls_relation": "relation"})

    valid = eval_df[eval_df.apply(lambda r: is_valid_attack_row(r, mode), axis=1)].copy()
    print(f"  Valid rows: {len(valid)}/{len(eval_df)} (mode={mode})")
    if valid.empty:
        raise SystemExit("No valid attack rows to evaluate.")

    if args.eval_n and args.eval_n < len(valid):
        valid = valid.sample(n=args.eval_n, random_state=args.eval_seed)
        print(f"  Sampled eval-n={args.eval_n} (seed={args.eval_seed})")
    valid = valid.reset_index(drop=True)
    N = len(valid)
    EVAL_STEM = f"umls_eval_{TAG}_n{N}"
    print(f"Run config: from-csv eval mode={mode} n={N} fluency_rewrite={args.fluency_rewrite}")
    print(f"Outputs: {EVAL_STEM}.xlsx / .csv\n")

    import torch
    from huggingface_hub import login
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

    login(token=os.environ.get("HF_TOKEN"))
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

    def generate_fn(messages: list) -> str:
        out = pipe(messages, max_new_tokens=100, do_sample=False)
        return out[0]["generated_text"][-1]["content"]

    def ask_medgemma(question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a biomedical research assistant. Answer in 1-2 sentences only. Be direct and concise. No bullet points, no headers."},
            {"role": "user",   "content": question}
        ]
        out = pipe(messages, max_new_tokens=80)
        return out[0]["generated_text"][-1]["content"].strip()

    REFUSAL_PATTERN = re.compile(
        r"\b(i|i'm|i am)\b[^.]{0,40}?\b(cannot|can't|can not|unable|not able|do not have|don't have)\b"
        r"|\bas an ai\b|\[error",
        re.IGNORECASE,
    )

    def is_refusal(text) -> str:
        if not isinstance(text, str):
            return None
        m = REFUSAL_PATTERN.search(text)
        return m.group(0) if m else None

    questions = valid["question"].tolist()
    qids = valid["qid"].tolist() if "qid" in valid.columns else list(range(N))
    attacked_questions = valid["attacked_question"].tolist()
    entities = valid["entity"].tolist() if "entity" in valid.columns else [None] * N
    terms_list = valid["terms_used"].tolist() if "terms_used" in valid.columns else [None] * N
    relations = valid["relation"].tolist() if "relation" in valid.columns else [None] * N
    ground_truths = valid["ground_truth"].tolist() if "ground_truth" in valid.columns else [None] * N

    fluency_ok_list = []
    fluency_rewrite_list = []
    fluency_fail_list = []
    raw_attacked_list = list(attacked_questions)

    if args.fluency_rewrite and mode == "sub":
        print("Applying fluency rewrite to sub attacks...")
        for i, (q, aq, ent, terms) in enumerate(
            zip(questions, attacked_questions, entities, terms_list)
        ):
            sub_term = str(terms).split(";")[0].strip() if _has_text(terms) else None
            if not sub_term or not _has_text(ent):
                fluency_ok_list.append(False)
                fluency_rewrite_list.append(None)
                fluency_fail_list.append("missing entity/term")
                continue
            fluent, ok, reason = rewrite_for_fluency(
                q, aq, str(ent), sub_term, generate_fn
            )
            fluency_ok_list.append(ok)
            fluency_rewrite_list.append(fluent)
            fluency_fail_list.append(reason)
            if ok and fluent:
                attacked_questions[i] = fluent
        print(f"  Fluency ok: {sum(1 for v in fluency_ok_list if v)}/{N}\n")
    else:
        fluency_ok_list = [None] * N
        fluency_rewrite_list = [None] * N
        fluency_fail_list = [None] * N

    print("Answering original + attacked questions with MedGemma 4B...")
    baseline_answers = []
    attacked_answers = []
    for i, (q, aq) in enumerate(zip(questions, attacked_questions)):
        print(f"[{i+1}/{N}] {str(q)[:60]}...")
        baseline_answers.append(ask_medgemma(q))
        attacked_answers.append(ask_medgemma(aq))

    from bert_score import score as bert_score

    print("\nComputing delta BERTScores...")
    _, _, F1 = bert_score(
        attacked_answers, baseline_answers,
        lang="en", model_type="distilbert-base-uncased", verbose=False
    )
    raw_scores = F1.tolist()

    baseline_matches = [is_refusal(a) for a in baseline_answers]
    attacked_matches = [is_refusal(a) for a in attacked_answers]
    baseline_refusal = [m is not None for m in baseline_matches]
    attacked_refusal = [m is not None for m in attacked_matches]
    refusal_match = [
        " | ".join(f"{side}: {m}" for side, m in (("baseline", b), ("attacked", a)) if m) or None
        for b, a in zip(baseline_matches, attacked_matches)
    ]
    delta_scores = [float("nan") if b or a else s
                    for s, b, a in zip(raw_scores, baseline_refusal, attacked_refusal)]
    valid_scores = [s for s in delta_scores if s == s]

    out = pd.DataFrame({
        "qid":               qids,
        "question":          questions,
        "attacked_question": attacked_questions,
        "raw_attacked_question": raw_attacked_list,
        "entity":            entities,
        "terms_used":        terms_list,
        "relation":          relations,
        "mode":              mode,
        "fluency_ok":        fluency_ok_list,
        "fluency_rewrite":   fluency_rewrite_list,
        "fluency_fail_reason": fluency_fail_list,
        "ground_truth":      ground_truths,
        "baseline_answer":   baseline_answers,
        "attacked_answer":   attacked_answers,
        "baseline_refusal":  baseline_refusal,
        "attacked_refusal":  attacked_refusal,
        "refusal_match":     refusal_match,
        "delta_bertscore":   [round(s, 4) for s in delta_scores],
    })
    # Carry through useful metadata columns when present
    for col in ("entity_source", "cui", "n_neighbors", "neighbors", "llm_ok",
                "fail_reason", "llm_sentence", "llm_terms_used_count",
                "llm_original_term_present", "llm_question_form", "entity_attempts"):
        if col in valid.columns:
            out[col] = valid[col].tolist()

    out.to_excel(f"{EVAL_STEM}.xlsx", index=False)
    out.to_csv(f"{EVAL_STEM}.csv", index=False)

    avg_delta = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")
    min_delta = min(valid_scores) if valid_scores else float("nan")
    print(f"\n{'='*65}")
    print(f"EVAL SUMMARY — {N:,} valid pairs ({EVAL_STEM})")
    print(f"{'='*65}")
    print(f"  Avg delta BERTScore: {avg_delta:.4f}  (n={len(valid_scores)}, refusals excluded)")
    print(f"  Min delta BERTScore: {min_delta:.4f}")
    print(f"  Saved to {EVAL_STEM}.csv / .xlsx")
    print(f"{'='*65}")
    raise SystemExit(0)

# ============================================================
# Generate path
# ============================================================
print(f"Run config: extractor={args.extractor} mode={args.mode} k={effective_k} "
      f"n={args.n or 'all'} questions_only={args.questions_only} "
      f"fluency_rewrite={args.fluency_rewrite}")
print(f"Outputs: {RESULTS_STEM}.xlsx / .csv\n")

need_model = (
    (args.mode in LLM_MODES)
    or (args.fluency_rewrite and args.mode == "sub")
    or (not args.questions_only)
)

# ============================================================
# STEP 1: Load baseline
# ============================================================
print("Loading baseline answers...")
df = pd.read_excel("pubmedqa_1000_answers.xlsx")
if args.n:
    df = df.head(args.n)
N = len(df)
questions        = df["question"].tolist()
qids             = df["qid"].tolist()
baseline_answers = df["medgemma_answer"].tolist()
ground_truths    = df["ground_truth"].tolist()
print(f"Loaded {N} questions\n")

# ============================================================
# STEP 2: Load MedGemma (when answering or LLM attack modes)
# ============================================================
pipe = None
generate_fn = None

if need_model:
    import torch
    from huggingface_hub import login
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

    login(token=os.environ.get("HF_TOKEN"))
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

    def generate_fn(messages: list) -> str:
        out = pipe(messages, max_new_tokens=100, do_sample=False)
        return out[0]["generated_text"][-1]["content"]

def ask_medgemma(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a biomedical research assistant. Answer in 1-2 sentences only. Be direct and concise. No bullet points, no headers."},
        {"role": "user",   "content": question}
    ]
    out = pipe(messages, max_new_tokens=80)
    return out[0]["generated_text"][-1]["content"].strip()

# First-person refusals only; "There is insufficient evidence..." is a real answer.
REFUSAL_PATTERN = re.compile(
    r"\b(i|i'm|i am)\b[^.]{0,40}?\b(cannot|can't|can not|unable|not able|do not have|don't have)\b"
    r"|\bas an ai\b|\[error",
    re.IGNORECASE,
)

def is_refusal(text) -> str:
    """Return the matched refusal phrase, or None if the answer is not a refusal."""
    if not isinstance(text, str):
        return None
    m = REFUSAL_PATTERN.search(text)
    return m.group(0) if m else None

# ============================================================
# STEP 3: Generate UMLS graph attacks
# ============================================================
print(f"Building UMLS attacks (mode={args.mode})...")
print("(This may take a few minutes due to API rate limiting)\n")

attacked_questions   = []
original_entities    = []
terms_used_list      = []
relation_types       = []
entity_sources       = []
cuis                 = []
neighbor_counts      = []
neighbor_lists       = []
llm_ok_list          = []
fail_reasons         = []
llm_sentences        = []
llm_terms_used_counts = []
llm_original_term_flags = []
llm_question_form_flags = []
entity_attempts_list = []
fluency_ok_list      = []
fluency_rewrite_list = []
fluency_fail_list    = []
raw_attacked_list    = []
neighbor_labels      = Counter()
umls_success_count   = 0

gen_for_mode = generate_fn if (
    args.mode in LLM_MODES or (args.fluency_rewrite and args.mode == "sub")
) else None

for i, q in enumerate(questions):
    attacked_q, entity, terms_used, rel, meta = apply_umls_mode(
        q, args.mode, k=effective_k,
        generate_fn=gen_for_mode,
        extractor=args.extractor,
        fluency_rewrite=args.fluency_rewrite and args.mode == "sub",
    )

    neighbors = meta.get("neighbors") or []
    # neighbors may be list of (name, label) or list of names (hop2)
    if neighbors and isinstance(neighbors[0], tuple):
        neighbor_str = "; ".join(f"{n} ({l})" for n, l in neighbors[:effective_k])
        neighbor_labels.update(l for _, l in neighbors)
        n_count = meta.get("n_neighbors", len(neighbors))
    else:
        neighbor_str = "; ".join(neighbors) if neighbors else None
        n_count = meta.get("n_neighbors", len(neighbors))

    success = attacked_q != q and (terms_used or rel)
    if success:
        umls_success_count += 1
        if umls_success_count <= 5:
            print(f"  ✅ [{rel}] llm_ok={meta.get('llm_ok')} fluency_ok={meta.get('fluency_ok')}")
            print(f"     Original : {q[:80]}")
            print(f"     Attacked : {attacked_q[:80]}")
            if meta.get("llm_sentence"):
                print(f"     Sentence : {meta['llm_sentence'][:90]}")
            print()

    attacked_questions.append(attacked_q)
    original_entities.append(entity)
    terms_used_list.append("; ".join(terms_used) if terms_used else None)
    relation_types.append(rel)
    entity_sources.append(meta.get("entity_source"))
    cuis.append(meta.get("cui"))
    neighbor_counts.append(n_count)
    neighbor_lists.append(neighbor_str)
    llm_ok_list.append(meta.get("llm_ok"))
    fail_reasons.append(meta.get("fail_reason"))
    llm_sentences.append(meta.get("llm_sentence"))
    llm_terms_used_counts.append(meta.get("llm_terms_used_count", len(terms_used)))
    llm_original_term_flags.append(meta.get("llm_original_term_present"))
    llm_question_form_flags.append(meta.get("llm_question_form"))
    entity_attempts_list.append(meta.get("entity_attempts", 0))
    fluency_ok_list.append(meta.get("fluency_ok"))
    fluency_rewrite_list.append(meta.get("fluency_rewrite"))
    fluency_fail_list.append(meta.get("fluency_fail_reason"))
    raw_attacked_list.append(meta.get("raw_attacked"))

    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{N} | UMLS successes: {umls_success_count}")

print(f"\nUMLS attack generation complete!")
print(f"  Successfully attacked: {umls_success_count}/{N} questions")
print(f"  Unchanged (fallback):  {N - umls_success_count}/{N}")

if args.mode in LLM_MODES:
    attempted = [v for v in llm_ok_list if v is not None]
    print(f"  LLM ok: {sum(1 for v in attempted if v)}/{len(attempted)}")
    for reason, count in Counter(r for r in fail_reasons if r).most_common():
        print(f"    {reason:<28} {count}")

if args.fluency_rewrite and args.mode == "sub":
    attempted = [v for v in fluency_ok_list if v is not None]
    print(f"  Fluency ok: {sum(1 for v in attempted if v)}/{len(attempted)}")
    for reason, count in Counter(r for r in fluency_fail_list if r).most_common():
        print(f"    {reason:<28} {count}")

source_counts = Counter(s for s in entity_sources if s)
cui_counts    = [n for n, c in zip(neighbor_counts, cuis) if c]
print(f"\n  UMLS coverage ({args.extractor} extractor):")
print(f"    Entity found:          {sum(source_counts.values())}/{N} "
      f"(scispaCy {source_counts['scispacy']}, keyword {source_counts['keyword']})")
print(f"    UMLS CUI found:        {sum(1 for c in cuis if c)}/{N}")
print(f"    >= 1 neighbour:        {sum(n >= 1 for n in neighbor_counts)}/{N}")
print(f"    >= {effective_k} neighbours:     {sum(n >= effective_k for n in neighbor_counts)}/{N}")
retried = sum(1 for a in entity_attempts_list if a and a > 1)
print(f"    Entity retries (>1 try): {retried}/{N}")
if cui_counts:
    print(f"    Neighbours per CUI:    mean {mean(cui_counts):.1f}, median {median(cui_counts):g}")
if neighbor_labels:
    print(f"    Neighbour relations:   RO {neighbor_labels['RO']}, RB {neighbor_labels['RB']}")
print()

# ============================================================
# Questions-only: save and exit
# ============================================================
if args.questions_only:
    result_df = pd.DataFrame({
        "qid":               qids,
        "question":          questions,
        "attacked_question": attacked_questions,
        "raw_attacked_question": raw_attacked_list,
        "entity":            original_entities,
        "entity_source":     entity_sources,
        "cui":               cuis,
        "n_neighbors":       neighbor_counts,
        "neighbors":         neighbor_lists,
        "terms_used":        terms_used_list,
        "relation":          relation_types,
        "mode":              args.mode,
        "llm_ok":            llm_ok_list,
        "fail_reason":       fail_reasons,
        "llm_sentence":      llm_sentences,
        "llm_terms_used_count": llm_terms_used_counts,
        "llm_original_term_present": llm_original_term_flags,
        "llm_question_form": llm_question_form_flags,
        "entity_attempts":   entity_attempts_list,
        "fluency_ok":        fluency_ok_list,
        "fluency_rewrite":   fluency_rewrite_list,
        "fluency_fail_reason": fluency_fail_list,
        "ground_truth":      ground_truths,
    })
    result_df.to_excel(f"{RESULTS_STEM}.xlsx", index=False)
    result_df.to_csv(f"{RESULTS_STEM}.csv", index=False)
    print(f"{'='*65}")
    print(f"QUESTIONS-ONLY — {N:,} questions ({TAG})")
    print(f"  Attacked: {umls_success_count}/{N}")
    print(f"  Saved to {RESULTS_STEM}.csv / .xlsx")
    print(f"{'='*65}")
    raise SystemExit(0)

# ============================================================
# STEP 4: Run MedGemma on UMLS attacked questions
# ============================================================
print("Running MedGemma on UMLS graph-attacked questions...")
attacked_answers = []
total = len(attacked_questions)

for i, q in enumerate(attacked_questions):
    print(f"[{i+1}/{total}] {questions[i][:60]}...")
    attacked_answers.append(ask_medgemma(q))

    if (i + 1) % 100 == 0:
        pd.DataFrame({
            "qid":               qids[:i+1],
            "question":          questions[:i+1],
            "original_entity":   original_entities[:i+1],
            "terms_used":        terms_used_list[:i+1],
            "relation":          relation_types[:i+1],
            "attacked_question": attacked_questions[:i+1],
            "baseline_answer":   baseline_answers[:i+1],
            "attacked_answer":   attacked_answers,
        }).to_csv(CHECKPOINT_PATH, index=False)
        print(f"  Checkpoint: {i+1}/{total}")

# ============================================================
# STEP 5: Compute delta scores
# ============================================================
from bert_score import score as bert_score

print("\nComputing delta BERTScores...")
_, _, F1 = bert_score(
    attacked_answers, baseline_answers,
    lang="en", model_type="distilbert-base-uncased", verbose=False
)
raw_scores = F1.tolist()

baseline_matches = [is_refusal(a) for a in baseline_answers]
attacked_matches = [is_refusal(a) for a in attacked_answers]
baseline_refusal = [m is not None for m in baseline_matches]
attacked_refusal = [m is not None for m in attacked_matches]
refusal_match = [
    " | ".join(f"{side}: {m}" for side, m in (("baseline", b), ("attacked", a)) if m) or None
    for b, a in zip(baseline_matches, attacked_matches)
]
delta_scores = [float("nan") if b or a else s
                for s, b, a in zip(raw_scores, baseline_refusal, attacked_refusal)]
valid_scores = [s for s in delta_scores if s == s]

# ============================================================
# STEP 6: Save results
# ============================================================
result_df = pd.DataFrame({
    "qid":               qids,
    "original_question": questions,
    "ground_truth":      ground_truths,
    "original_entity":   original_entities,
    "entity_source":     entity_sources,
    "cui":               cuis,
    "n_neighbors":       neighbor_counts,
    "neighbors":         neighbor_lists,
    "umls_substitution": terms_used_list,
    "umls_relation":     relation_types,
    "mode":              args.mode,
    "llm_ok":            llm_ok_list,
    "fail_reason":       fail_reasons,
    "llm_sentence":      llm_sentences,
    "llm_terms_used_count": llm_terms_used_counts,
    "llm_original_term_present": llm_original_term_flags,
    "llm_question_form": llm_question_form_flags,
    "entity_attempts":   entity_attempts_list,
    "fluency_ok":        fluency_ok_list,
    "fluency_rewrite":   fluency_rewrite_list,
    "fluency_fail_reason": fluency_fail_list,
    "raw_attacked_question": raw_attacked_list,
    "attacked_question": attacked_questions,
    "baseline_answer":   baseline_answers,
    "attacked_answer":   attacked_answers,
    "baseline_refusal":  baseline_refusal,
    "attacked_refusal":  attacked_refusal,
    "refusal_match":     refusal_match,
    "delta_bertscore":   [round(s, 4) for s in delta_scores],
})

result_df.to_excel(f"{RESULTS_STEM}.xlsx", index=False)
result_df.to_csv(f"{RESULTS_STEM}.csv",   index=False)

refusal_rows = result_df["baseline_refusal"] | result_df["attacked_refusal"]
refusals_df = result_df.loc[refusal_rows, [
    "qid", "original_question", "attacked_question", "refusal_match",
    "baseline_answer", "attacked_answer",
]].copy()
refusals_df.insert(3, "refused", [
    "both" if b and a else "baseline" if b else "attacked"
    for b, a in zip(result_df.loc[refusal_rows, "baseline_refusal"],
                    result_df.loc[refusal_rows, "attacked_refusal"])
])
refusals_df["delta_bertscore_raw"] = [round(s, 4) for s, r in zip(raw_scores, refusal_rows) if r]
refusals_df.to_csv(f"{RESULTS_STEM}_refusals.csv", index=False)

# ============================================================
# STEP 7: Summary
# ============================================================
avg_delta = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")
min_delta = min(valid_scores) if valid_scores else float("nan")

rel_counts = Counter(r for r in relation_types if r)
print(f"\n{'='*65}")
print(f"UMLS GRAPH ATTACK SUMMARY — {N:,} questions ({TAG})")
print(f"{'='*65}")
print(f"  UMLS attacks:        {umls_success_count}/{N}")
print(f"  Excluded {len(refusals_df)} refusals (baseline {sum(baseline_refusal)}, "
      f"attacked {sum(attacked_refusal)}, both {sum(b and a for b, a in zip(baseline_refusal, attacked_refusal))})"
      f" → {RESULTS_STEM}_refusals.csv")
print(f"  Avg delta BERTScore: {avg_delta:.4f}  (n={len(valid_scores)}, refusals excluded)")
print(f"  Min delta BERTScore: {min_delta:.4f}")
print(f"\n  Relations used (UMLS graph edges):")
for rel, count in rel_counts.most_common():
    rows = result_df[result_df["umls_relation"] == rel]
    avg_d = rows["delta_bertscore"].mean()
    print(f"  {rel:<30} n={count:<5} avg_delta={avg_d:.4f}")
print(f"\n  Most affected questions:")
for _, row in result_df.dropna(subset=["delta_bertscore"]).nsmallest(5, "delta_bertscore").iterrows():
    print(f"  [{row['delta_bertscore']:.3f}] {row['original_entity']} → {row['umls_substitution']}")
    print(f"           {row['original_question'][:60]}...")
print(f"{'='*65}")
print(f"\nSaved to {RESULTS_STEM}.xlsx")
