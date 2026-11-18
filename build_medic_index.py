# ============================================================
# build_medic_index.py
# Build name → id maps from the MeDIC Drug List CSV.
#
# Usage:
#   python build_medic_index.py
#   python build_medic_index.py --drugs "data/kg/MeDIC/MeDIC Drug List.csv"
# ============================================================

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DRUGS = ROOT / "data" / "kg" / "MeDIC" / "MeDIC Drug List.csv"
OUT_DIR = ROOT / "data" / "kg" / "MeDIC" / "processed"


def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _label_aliases(curie_label: str) -> list:
    """Split curie_label on ';' and drop parenthetical noise tails lightly."""
    aliases = []
    for part in (curie_label or "").split(";"):
        part = part.strip()
        if not part:
            continue
        # Drop trailing ", (R,S)-rel-" style stereo tags for matching
        core = re.sub(r",\s*\([^)]*\)\s*-?(?:rel-?)?$", "", part).strip()
        aliases.append(core if core else part)
    return aliases


def build_index(drugs_path: Path):
    name_to_id = {}
    id_to_name = {}
    rows_out = []
    n_alias = 0

    with drugs_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            curie = (row.get("curie") or "").strip()
            drug_name = (row.get("drug_name") or "").strip()
            curie_label = (row.get("curie_label") or "").strip()
            if not curie:
                continue

            # Preferred display name: first curie_label piece, else drug_name
            aliases = _label_aliases(curie_label)
            display = aliases[0] if aliases else drug_name
            if not display:
                continue
            id_to_name[curie] = display
            rows_out.append({
                "drug_id": curie,
                "name": display,
                "drug_name": drug_name,
                "curie_label": curie_label,
                "combination_therapy": row.get("combination_therapy", ""),
            })

            candidates = [drug_name] + aliases
            for cand in candidates:
                key = _norm_name(cand)
                if not key or len(key) < 3:
                    continue
                # First name wins if collisions (stable CSV order)
                if key not in name_to_id:
                    name_to_id[key] = curie
                    n_alias += 1

    return name_to_id, id_to_name, rows_out, n_alias


def main():
    ap = argparse.ArgumentParser(description="Build MeDIC drug name index")
    ap.add_argument("--drugs", type=Path, default=DEFAULT_DRUGS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if not args.drugs.is_file():
        raise SystemExit(f"Missing MeDIC drug list: {args.drugs}")

    name_to_id, id_to_name, rows_out, n_alias = build_index(args.drugs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    drugs_csv = args.out_dir / "medic_drugs.csv"
    name_json = args.out_dir / "medic_drug_name_to_id.json"

    with drugs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "drug_id", "name", "drug_name", "curie_label", "combination_therapy",
            ],
        )
        w.writeheader()
        w.writerows(rows_out)

    name_json.write_text(json.dumps(name_to_id, indent=0, ensure_ascii=False) + "\n")

    print(f"Drugs:           {len(id_to_name):,}")
    print(f"Name aliases:    {n_alias:,}")
    print(f"Wrote {drugs_csv}")
    print(f"Wrote {name_json}")


if __name__ == "__main__":
    main()
