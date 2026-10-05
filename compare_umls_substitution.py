# ============================================================
# compare_umls_substitution.py
# Runs only the entity-extraction + UMLS substitution step of
# the UMLS graph attack (no LLM) and compares the scispaCy
# UMLS linker against the keyword list, question by question.
#
# Usage:
#   python compare_umls_substitution.py                  # first 50 questions
#   python compare_umls_substitution.py --n 200 --start 100
#   python compare_umls_substitution.py --no-umls        # extraction only, no UMLS API calls
#
# Output: umls_substitution_comparison.csv (one row per question)
# ============================================================

import argparse
from collections import Counter

import pandas as pd

from umls_substitution import (
    extract_medical_entity,
    load_scispacy_pipeline,
    scispacy_candidates,
    substitute_entity,
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="pubmedqa_1000_answers.xlsx",
    help="Excel or CSV file with a 'question' column")
parser.add_argument("--n", type=int, default=50,
    help="How many questions to compare (0 = all)")
parser.add_argument("--start", type=int, default=0,
    help="Index of the first question")
parser.add_argument("--no-umls", action="store_true",
    help="Compare entity extraction only; skip UMLS relation lookups")
parser.add_argument("--out", default="umls_substitution_comparison.csv")
parser.add_argument("--examples", type=int, default=5,
    help="How many disagreement examples to print")
args = parser.parse_args()

# ============================================================
# LOAD QUESTIONS
# ============================================================
if args.input.endswith(".csv"):
    df = pd.read_csv(args.input)
else:
    df = pd.read_excel(args.input)
end = len(df) if args.n == 0 else min(len(df), args.start + args.n)
df = df.iloc[args.start:end].reset_index(drop=True)
qids = df["qid"].tolist() if "qid" in df.columns else list(range(args.start, end))
print(f"Comparing {len(df)} questions (rows {args.start}..{end - 1}) from {args.input}\n")

scispacy_ok = load_scispacy_pipeline()
if not scispacy_ok:
    print("!" * 65)
    print("  scispaCy did not load — every scispaCy column will be empty.")
    print("  See the WARNING above for the reason.")
    print("!" * 65 + "\n")

# ============================================================
# COMPARE
# ============================================================
def format_candidates(cands: list) -> str:
    parts = []
    for c in cands:
        if c["cui"] is None:
            parts.append(f"{c['mention']} [rejected: {c['reason']}]")
            continue
        status = "kept" if c["kept"] else f"rejected: {c['reason']}"
        parts.append(f"{c['mention']} -> {c['name']} "
                     f"({c['score']}, {'/'.join(c['types'])}) [{status}]")
    return " | ".join(parts)

rows = []
reject_reasons = Counter()
total_mentions = 0

for i, (qid, q) in enumerate(zip(qids, df["question"])):
    cands = scispacy_candidates(q)
    total_mentions += len(cands)
    reject_reasons.update(c["reason"] for c in cands if not c["kept"])
    kept = [c for c in cands if c["kept"]]
    best = max(kept, key=lambda c: (c["score"], len(c["mention"]))) if kept else None

    sci_entity = best["mention"] if best else None
    sci_cui    = best["cui"] if best else None
    kw_entity  = extract_medical_entity(q)

    row = {
        "qid":            qid,
        "question":       q,
        "sci_entity":     sci_entity,
        "sci_cui":        sci_cui,
        "sci_concept":    best["name"] if best else None,
        "sci_score":      best["score"] if best else None,
        "sci_candidates": format_candidates(cands),
        "kw_entity":      kw_entity,
        "same_entity":    bool(sci_entity and kw_entity and
                               sci_entity.lower() == kw_entity.lower()),
        "pipeline_source": "scispacy" if sci_entity else ("keyword" if kw_entity else None),
    }

    if not args.no_umls:
        for prefix, entity, cui in [("sci", sci_entity, sci_cui), ("kw", kw_entity, None)]:
            if entity:
                attacked, sub, rel = substitute_entity(q, entity, cui)
            else:
                attacked, sub, rel = q, None, None
            row[f"{prefix}_substitution"] = sub
            row[f"{prefix}_relation"]     = rel
            row[f"{prefix}_attacked"]     = attacked if sub else None

    rows.append(row)
    if (i + 1) % 25 == 0:
        print(f"  Processed {i + 1}/{len(df)}")

out_df = pd.DataFrame(rows)
out_df.to_csv(args.out, index=False)

# ============================================================
# SUMMARY
# ============================================================
n        = len(out_df)
has_sci  = out_df["sci_entity"].notna()
has_kw   = out_df["kw_entity"].notna()
both     = has_sci & has_kw

print(f"\n{'='*65}")
print(f"  UMLS SUBSTITUTION COMPARISON — {n} questions")
print(f"{'='*65}")
print(f"  Entity found")
print(f"    scispaCy:            {has_sci.sum():>5}/{n}")
print(f"    keyword list:        {has_kw.sum():>5}/{n}")
print(f"    both:                {both.sum():>5}")
print(f"    scispaCy only:       {(has_sci & ~has_kw).sum():>5}")
print(f"    keyword only:        {(~has_sci & has_kw).sum():>5}")
print(f"    neither:             {(~has_sci & ~has_kw).sum():>5}")
if both.sum():
    print(f"    same entity (when both found): "
          f"{out_df.loc[both, 'same_entity'].sum()}/{both.sum()}")

if scispacy_ok:
    print(f"\n  scispaCy mentions: {total_mentions} total, "
          f"{sum(reject_reasons.values())} rejected")
    for reason, count in reject_reasons.most_common():
        print(f"    {reason:<30} {count}")

if not args.no_umls:
    sci_subs = out_df["sci_substitution"].notna().sum()
    kw_subs  = out_df["kw_substitution"].notna().sum()
    print(f"\n  UMLS substitution succeeded")
    print(f"    scispaCy entity:     {sci_subs:>5}/{has_sci.sum()}")
    print(f"    keyword entity:      {kw_subs:>5}/{has_kw.sum()}")
    print(f"\n  Relations used")
    for prefix, label in [("sci", "scispaCy"), ("kw", "keyword")]:
        counts = Counter(out_df[f"{prefix}_relation"].dropna())
        print(f"    {label:<10} " + ", ".join(f"{r}={c}" for r, c in counts.most_common()))

disagree = out_df[both & ~out_df["same_entity"]]
if len(disagree) and args.examples:
    print(f"\n  Disagreements (first {min(args.examples, len(disagree))} of {len(disagree)})")
    for _, r in disagree.head(args.examples).iterrows():
        print(f"    Q: {r['question'][:80]}")
        sci_sub = r.get("sci_substitution")
        kw_sub  = r.get("kw_substitution")
        print(f"       scispaCy: {r['sci_entity']} ({r['sci_concept']})"
              + (f" -> {sci_sub}" if pd.notna(sci_sub) else ""))
        print(f"       keyword:  {r['kw_entity']}"
              + (f" -> {kw_sub}" if pd.notna(kw_sub) else ""))
print(f"{'='*65}")
print(f"\nSaved per-question comparison to {args.out}")
