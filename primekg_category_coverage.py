# ============================================================
# primekg_category_coverage.py
# Measure which PrimeKG node categories appear in PubMedQA.
#
# Requires: python build_primekg_type_index.py
#
# Usage:
#   python primekg_category_coverage.py
#   python primekg_category_coverage.py --n 500
# ============================================================

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "kg" / "processed"
DEFAULT_QUESTIONS = ROOT / "umls_graph_attack_results_scispacy_none.csv"
DEFAULT_OUT = PROCESSED / "primekg_pubmedqa_category_coverage.csv"
DEFAULT_SUMMARY = PROCESSED / "primekg_pubmedqa_category_summary.csv"


def find_names_in_text(text, name_to_id, id_to_name, max_tokens=8):
    """
    Fast longest-match over whitespace tokens.
    For each start token, try spans of length max_tokens..1 and dict-lookup.
    O(#tokens * max_tokens) per question — not O(#dictionary).
    """
    if not text or not name_to_id:
        return []
    # Keep original char spans roughly via regex tokens on lowercased text
    hay = text.lower()
    token_matches = list(re.finditer(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", hay))
    if not token_matches:
        return []
    tokens = [m.group(0) for m in token_matches]
    occupied = [False] * len(tokens)
    hits = []
    i = 0
    while i < len(tokens):
        if occupied[i]:
            i += 1
            continue
        found = None
        upper = min(len(tokens), i + max_tokens)
        for j in range(upper, i, -1):
            span = " ".join(tokens[i:j])
            # Also try hyphenated join for names stored without spaces oddly
            if span not in name_to_id and j == i + 1:
                span = tokens[i]
            if span in name_to_id:
                # Prefer multi-word / longer spans (we iterate long→short)
                node_id = name_to_id[span]
                char_start = token_matches[i].start()
                char_end = token_matches[j - 1].end()
                found = {
                    "mention": text[char_start:char_end],
                    "node_id": node_id,
                    "name": id_to_name.get(node_id, span),
                    "_i": i,
                    "_j": j,
                }
                break
        if found:
            for k in range(found["_i"], found["_j"]):
                occupied[k] = True
            hits.append({k: v for k, v in found.items() if not k.startswith("_")})
            i = found["_j"]
        else:
            i += 1
    return hits


def load_questions(path: Path, n: int = 0):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    qcol = "question" if "question" in rows[0] else "original_question"
    if qcol not in rows[0]:
        raise SystemExit(f"No question column in {path}")
    out = [{"qid": r.get("qid"), "question": r.get(qcol) or ""} for r in rows]
    if n:
        out = out[:n]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="PrimeKG category coverage on PubMedQA"
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument(
        "--types", type=str, default="",
        help="comma-separated types to include (default: all)",
    )
    args = parser.parse_args()

    names_path = PROCESSED / "primekg_names_by_type.json"
    if not names_path.is_file():
        raise SystemExit(
            f"Missing {names_path}\nRun: python build_primekg_type_index.py"
        )

    print(f"Loading {names_path} ...")
    names_by_type = json.loads(names_path.read_text())
    if args.types.strip():
        keep = {t.strip() for t in args.types.split(",") if t.strip()}
        names_by_type = {t: m for t, m in names_by_type.items() if t in keep}
        missing = keep - set(names_by_type)
        if missing:
            print(f"WARNING: unknown types ignored: {sorted(missing)}")

    # Filter short names once; build id→name
    id_to_name_by_type = {}
    filtered = {}
    for t, name_to_id in names_by_type.items():
        m = {n: i for n, i in name_to_id.items() if len(n) >= args.min_len}
        filtered[t] = m
        rev = {}
        for name, nid in m.items():
            rev.setdefault(nid, name)
        id_to_name_by_type[t] = rev
    names_by_type = filtered

    questions = load_questions(args.questions, args.n)
    N = len(questions)
    types = sorted(names_by_type.keys())
    print(f"Categories: {len(types)} | Questions: {N}")
    print("(fast token matcher; progress every 10 questions)")
    for t in types:
        print(f"  {len(names_by_type[t]):6d} names  {t}", flush=True)

    type_hit_questions = Counter()
    type_name_freq = defaultdict(Counter)
    any_hit = 0
    rows_out = []

    for i, item in enumerate(questions):
        q = item["question"]
        row = {"qid": item["qid"], "question": q}
        categories_hit = []
        all_hits_str = []

        for t in types:
            hits = find_names_in_text(
                q, names_by_type[t], id_to_name_by_type[t]
            )
            # dedupe by node_id
            seen = {}
            for h in hits:
                seen.setdefault(h["node_id"], h)
            uniq = list(seen.values())
            flag = len(uniq) > 0
            # CSV-safe type key
            key = re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")
            row[f"has_{key}"] = flag
            row[f"n_{key}"] = len(uniq)
            row[f"hits_{key}"] = "; ".join(
                f"{h['name']} ({h['node_id']})" for h in uniq
            )
            if flag:
                type_hit_questions[t] += 1
                categories_hit.append(t)
                for h in uniq:
                    type_name_freq[t][h["name"]] += 1
                    all_hits_str.append(f"{t}:{h['name']}")

        row["n_categories"] = len(categories_hit)
        row["categories"] = "; ".join(categories_hit)
        row["has_any"] = len(categories_hit) > 0
        if row["has_any"]:
            any_hit += 1
        rows_out.append(row)

        if (i + 1) % 10 == 0 or (i + 1) == N:
            print(f"  {i+1}/{N} | any-category so far: {any_hit}", flush=True)

    # Write per-question coverage
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["qid", "question", "has_any", "n_categories", "categories"]
    for t in types:
        key = re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")
        fieldnames += [f"has_{key}", f"n_{key}", f"hits_{key}"]

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    # Summary table
    with args.summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "type", "n_names_in_index", "n_questions_hit",
                "pct_questions", "top_names",
            ],
        )
        w.writeheader()
        for t in types:
            top = ", ".join(
                f"{name} ({c})" for name, c in type_name_freq[t].most_common(8)
            )
            w.writerow({
                "type": t,
                "n_names_in_index": len(names_by_type[t]),
                "n_questions_hit": type_hit_questions[t],
                "pct_questions": round(100 * type_hit_questions[t] / N, 2),
                "top_names": top,
            })
        w.writerow({
            "type": "ANY",
            "n_names_in_index": "",
            "n_questions_hit": any_hit,
            "pct_questions": round(100 * any_hit / N, 2),
            "top_names": "",
        })

    print()
    print(f"Questions scanned: {N}")
    print(f"With ≥1 category:  {any_hit}/{N} ({100*any_hit/N:.1f}%)")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary}")
    print()
    print(f"{'type':<28} {'hit':>6} {'pct':>7}")
    for t in sorted(types, key=lambda x: -type_hit_questions[x]):
        c = type_hit_questions[t]
        print(f"{t:<28} {c:6d} {100*c/N:6.1f}%")
    print(f"{'ANY':<28} {any_hit:6d} {100*any_hit/N:6.1f}%")


if __name__ == "__main__":
    main()
