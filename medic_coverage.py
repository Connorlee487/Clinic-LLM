# ============================================================
# medic_coverage.py
# How many questions mention a MeDIC drug?
#
# Usage:
#   python build_medic_index.py
#   python medic_coverage.py --questions umls_graph_attack_results_scispacy_none.csv
#   python medic_coverage.py --questions medqa_1273_questions.csv \
#       --out data/kg/MeDIC/processed/medic_medqa_drug_coverage.csv
# ============================================================

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "kg" / "MeDIC" / "processed"
DEFAULT_QUESTIONS = ROOT / "umls_graph_attack_results_scispacy_none.csv"
DEFAULT_OUT = PROCESSED / "medic_pubmedqa_drug_coverage.csv"


def load_name_map():
    path = PROCESSED / "medic_drug_name_to_id.json"
    drugs_path = PROCESSED / "medic_drugs.csv"
    if not path.is_file() or not drugs_path.is_file():
        raise SystemExit(
            "Missing processed MeDIC index. Run:\n"
            "  python build_medic_index.py"
        )
    name_to_id = json.loads(path.read_text())
    id_to_name = {}
    with drugs_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_name[row["drug_id"]] = row["name"]
    return name_to_id, id_to_name


def find_drugs_in_text(text, name_to_id, id_to_name, min_len=4):
    """Longest-first, non-overlapping name matches (word-boundary-ish)."""
    if not text:
        return []
    hay = text.lower()
    names = sorted(
        (n for n in name_to_id if len(n) >= min_len),
        key=len,
        reverse=True,
    )
    occupied = [False] * len(hay)
    hits = []
    for name in names:
        parts = [re.escape(p) for p in name.split() if p]
        if not parts:
            continue
        pattern = r"(?<![a-z0-9])" + r"\s+".join(parts) + r"(?![a-z0-9])"
        for m in re.finditer(pattern, hay):
            start, end = m.start(), m.end()
            if any(occupied[start:end]):
                continue
            for i in range(start, end):
                occupied[i] = True
            drug_id = name_to_id[name]
            hits.append({
                "mention": text[start:end],
                "drug_id": drug_id,
                "name": id_to_name.get(drug_id, name),
            })
    return hits


def load_questions(path: Path, n: int = 0):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    qcol = "question" if "question" in rows[0] else "original_question"
    if qcol not in rows[0]:
        raise SystemExit(f"No question column in {path}. Columns: {list(rows[0])}")
    out = [{"qid": r.get("qid"), "question": r.get(qcol) or ""} for r in rows]
    if n:
        out = out[:n]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="MeDIC drug-mention coverage on QA questions"
    )
    parser.add_argument(
        "--questions", type=Path, default=DEFAULT_QUESTIONS,
        help="CSV with qid + question (or original_question)",
    )
    parser.add_argument("--n", type=int, default=0, help="0 = all rows")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-len", type=int, default=4,
                        help="ignore drug names shorter than this")
    args = parser.parse_args()

    name_to_id, id_to_name = load_name_map()
    questions = load_questions(args.questions, args.n)
    N = len(questions)
    print(f"Name map: {len(name_to_id):,} aliases → {len(id_to_name):,} drugs")
    print(f"Scanning {N} questions from {args.questions.name} ...")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_with = 0
    drug_freq = Counter()
    rows_out = []

    for i, item in enumerate(questions):
        hits = find_drugs_in_text(
            item["question"], name_to_id, id_to_name, min_len=args.min_len
        )
        seen = {}
        for h in hits:
            seen.setdefault(h["drug_id"], h)
        uniq = list(seen.values())
        has = len(uniq) > 0
        if has:
            n_with += 1
            for h in uniq:
                drug_freq[h["name"]] += 1

        rows_out.append({
            "qid": item["qid"],
            "question": item["question"],
            "n_drugs": len(uniq),
            "drugs": "; ".join(f"{h['name']} ({h['drug_id']})" for h in uniq),
            "has_drug": has,
        })

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{N} | with drug so far: {n_with}")

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["qid", "question", "n_drugs", "drugs", "has_drug"]
        )
        w.writeheader()
        w.writerows(rows_out)

    print()
    print(f"Questions scanned:    {N}")
    print(f"With ≥1 MeDIC drug:   {n_with}/{N} ({100 * n_with / N:.1f}%)")
    print(f"Wrote {args.out}")
    print("Top drugs mentioned:")
    for name, c in drug_freq.most_common(15):
        print(f"  {c:4d}  {name}")


if __name__ == "__main__":
    main()
