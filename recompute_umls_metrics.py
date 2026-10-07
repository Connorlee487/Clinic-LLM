# ============================================================
# recompute_umls_metrics.py
# Recompute the UMLS graph attack summary from saved results CSVs
# (no model, GPU, or UMLS API needed), and compare runs side by side.
#
# Usage:
#   python recompute_umls_metrics.py                         # all umls_graph_attack_results_*.csv
#   python recompute_umls_metrics.py a.csv b.csv --k 5
#   python recompute_umls_metrics.py --out umls_metrics_comparison.csv
# ============================================================

import argparse
import glob
import os
import re

import pandas as pd

# Same pattern as run_umls_graph_attack.py; only used for older CSVs
# that were saved before the refusal columns existed.
REFUSAL_PATTERN = re.compile(
    r"\b(i|i'm|i am)\b[^.]{0,40}?\b(cannot|can't|can not|unable|not able|do not have|don't have)\b"
    r"|\bas an ai\b|\[error",
    re.IGNORECASE,
)

parser = argparse.ArgumentParser(description="Recompute UMLS attack metrics from results CSVs")
parser.add_argument("csvs", nargs="*",
                    help="results CSVs (default: umls_graph_attack_results*.csv in this folder)")
parser.add_argument("--k", type=int, default=5,
                    help="threshold for the '>= k neighbours' coverage line")
parser.add_argument("--top", type=int, default=5,
                    help="how many most-affected questions to print per run")
parser.add_argument("--out", default=None,
                    help="optional path to save the comparison table as CSV")
args = parser.parse_args()

paths = args.csvs or sorted(
    p for p in glob.glob("umls_graph_attack_results*.csv")
    if "refusals" not in os.path.basename(p)
)
if not paths:
    raise SystemExit("No results CSVs found.")


def to_bool(series: pd.Series) -> pd.Series:
    """CSV round-trips turn True/False/blank into strings or objects."""
    return series.map(lambda v: {"true": True, "false": False}.get(str(v).strip().lower()))


def run_tag(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("umls_graph_attack_results_", "").replace("umls_graph_attack_results", "original") or name


def summarize(path: str) -> dict:
    df = pd.read_csv(path)
    tag = run_tag(path)
    N = len(df)
    attacked = df["umls_substitution"].notna()

    # Refusals: use saved columns, or detect them for older CSVs
    if "baseline_refusal" in df.columns:
        base_ref = to_bool(df["baseline_refusal"]).fillna(False).astype(bool)
        att_ref  = to_bool(df["attacked_refusal"]).fillna(False).astype(bool)
        refusal_source = "saved columns"
    else:
        base_ref = df["baseline_answer"].map(lambda t: isinstance(t, str) and bool(REFUSAL_PATTERN.search(t)))
        att_ref  = df["attacked_answer"].map(lambda t: isinstance(t, str) and bool(REFUSAL_PATTERN.search(t)))
        refusal_source = "re-detected (older CSV)"
    excluded = base_ref | att_ref
    delta = df["delta_bertscore"].where(~excluded)

    print(f"\n{'='*65}")
    print(f"📊 {tag}  —  {N:,} questions  ({path})")
    print(f"{'='*65}")
    print(f"  UMLS substitutions:   {attacked.sum()}/{N}")
    print(f"  Excluded {excluded.sum()} refusals (baseline {base_ref.sum()}, attacked {att_ref.sum()}, "
          f"both {(base_ref & att_ref).sum()}) [{refusal_source}]")
    print(f"  Avg delta BERTScore:  {delta.mean():.4f}  (n={delta.notna().sum()}, all non-refusal rows)")
    print(f"    attacked rows:      {delta[attacked].mean():.4f}  (n={delta[attacked].notna().sum()})")
    print(f"    unattacked rows:    {delta[~attacked].mean():.4f}  (n={delta[~attacked].notna().sum()}, noise floor)")
    print(f"  Min delta BERTScore:  {delta.min():.4f}")
    print(f"  Median delta (attacked): {delta[attacked].median():.4f}")

    print(f"\n  Relations used (UMLS graph edges):")
    for rel, rows in df[attacked].groupby("umls_relation"):
        print(f"  {rel:<30} n={len(rows):<5} avg_delta={delta[rows.index].mean():.4f}")

    row = {
        "run": tag, "n": N, "substituted": int(attacked.sum()),
        "refusals_excluded": int(excluded.sum()),
        "avg_delta_all": round(delta.mean(), 4),
        "avg_delta_attacked": round(delta[attacked].mean(), 4),
        "avg_delta_unattacked": round(delta[~attacked].mean(), 4),
        "median_delta_attacked": round(delta[attacked].median(), 4),
        "min_delta": round(delta.min(), 4),
    }

    if "rewrite_valid" in df.columns and df["rewrite_valid"].notna().any():
        valid = to_bool(df["rewrite_valid"]).dropna()
        print(f"\n  LLM rewrites valid:   {int(valid.sum())}/{len(valid)}")
        for reason, count in df["rewrite_reason"].value_counts().items():
            print(f"    {reason:<28} {count}")
        if len(valid):
            v = to_bool(df["rewrite_valid"])
            print(f"    avg delta, valid rewrites:     {delta[v == True].mean():.4f}")   # noqa: E712
            print(f"    avg delta, static fallbacks:   {delta[v == False].mean():.4f}")  # noqa: E712
        row["rewrite_valid_rate"] = round(valid.mean(), 3) if len(valid) else None

    if "n_neighbors" in df.columns:
        sources = df["entity_source"].value_counts()
        has_cui = df["cui"].notna()
        nn = df["n_neighbors"]
        labels = df["neighbors"].dropna().str.findall(r"\((RO|RB)\)").explode().value_counts()
        print(f"\n  UMLS coverage:")
        print(f"    Entity found:          {df['entity_source'].notna().sum()}/{N} "
              f"(scispaCy {sources.get('scispacy', 0)}, keyword {sources.get('keyword', 0)})")
        print(f"    UMLS CUI found:        {has_cui.sum()}/{N}")
        print(f"    >= 1 RO/RB neighbour:  {(nn >= 1).sum()}/{N}")
        print(f"    >= {args.k} RO/RB neighbours: {(nn >= args.k).sum()}/{N}")
        if has_cui.any():
            print(f"    Neighbours per CUI:    mean {nn[has_cui].mean():.1f}, median {nn[has_cui].median():g}")
        print(f"    Relations in saved top-k neighbours: RO {labels.get('RO', 0)}, RB {labels.get('RB', 0)}")
        row.update({
            "entity_found": int(df["entity_source"].notna().sum()),
            "scispacy": int(sources.get("scispacy", 0)),
            "keyword": int(sources.get("keyword", 0)),
            "cui_found": int(has_cui.sum()),
            "ge1_neighbor": int((nn >= 1).sum()),
            f"ge{args.k}_neighbors": int((nn >= args.k).sum()),
        })

    print(f"\n  Most affected questions:")
    worst = df.assign(_delta=delta).dropna(subset=["_delta"]).nsmallest(args.top, "_delta")
    for _, r in worst.iterrows():
        print(f"  [{r['_delta']:.3f}] {r['original_entity']} → {r['umls_substitution']}")
        print(f"           {str(r['original_question'])[:60]}...")
    return row


rows = [summarize(p) for p in paths]

comparison = pd.DataFrame(rows)
print(f"\n{'='*65}")
print("📋 COMPARISON (delta = BERTScore F1 vs baseline answer; lower = bigger change)")
print(f"{'='*65}")
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(comparison.to_string(index=False))

if args.out:
    comparison.to_csv(args.out, index=False)
    print(f"\n✅ Saved comparison to {args.out}")
