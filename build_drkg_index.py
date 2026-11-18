# ============================================================
# build_drkg_index.py
# Extract Compound entities from DRKG and resolve DrugBank IDs
# to surface names (via PrimeKG drug list, optional MeDIC).
#
# Download first (into data/kg/DRKG/):
#   mkdir -p data/kg/DRKG && cd data/kg/DRKG
#   wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz
#   tar -xzf drkg.tar.gz
#   # expect data/kg/DRKG/drkg.tsv (or nested; pass --drkg)
#
# Usage:
#   python build_drkg_index.py
# ============================================================

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DRKG = ROOT / "data" / "kg" / "DRKG" / "drkg.tsv"
DEFAULT_OUT = ROOT / "data" / "kg" / "DRKG" / "processed"
PRIMEKG_DRUGS = ROOT / "data" / "kg" / "processed" / "primekg_drugs.csv"
MEDIC_DRUGS = ROOT / "data" / "kg" / "MeDIC" / "MeDIC Drug List.csv"

csv.field_size_limit(min(2**31 - 1, 10_000_000))


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"\s+", " ", s)


def load_drugbank_names() -> dict:
    """Map DrugBank id (DB#####) → preferred display name."""
    db_to_name = {}

    if PRIMEKG_DRUGS.is_file():
        with PRIMEKG_DRUGS.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                did = (row.get("drug_id") or "").strip()
                name = (row.get("name") or "").strip()
                if did.startswith("DB") and name and did not in db_to_name:
                    db_to_name[did] = name
        print(f"  PrimeKG DrugBank names: {len(db_to_name):,}")

    if MEDIC_DRUGS.is_file():
        n_before = len(db_to_name)
        with MEDIC_DRUGS.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                alts = row.get("alternate_ids") or ""
                for m in re.finditer(r"DRUGBANK:(DB\d+)", alts):
                    dbid = m.group(1)
                    if dbid in db_to_name:
                        continue
                    label = (row.get("curie_label") or "").split(";")[0].strip()
                    name = label or (row.get("drug_name") or "").strip()
                    if name:
                        db_to_name[dbid] = name
        print(f"  + MeDIC DrugBank names: {len(db_to_name) - n_before:,} new "
              f"(total {len(db_to_name):,})")

    if not db_to_name:
        raise SystemExit(
            "No DrugBank name map found. Need either:\n"
            f"  {PRIMEKG_DRUGS}\n"
            f"  or {MEDIC_DRUGS}"
        )
    return db_to_name


def find_drkg_tsv(explicit: Path) -> Path:
    if explicit.is_file():
        return explicit
    candidates = [
        ROOT / "data" / "kg" / "DRKG" / "drkg.tsv",
        ROOT / "data" / "kg" / "DRKG" / "drkg" / "drkg.tsv",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise SystemExit(
        "Missing drkg.tsv. Download:\n"
        "  mkdir -p data/kg/DRKG && cd data/kg/DRKG\n"
        "  wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz\n"
        "  tar -xzf drkg.tar.gz\n"
        f"Looked for: {explicit}"
    )


def collect_compounds(drkg_path: Path) -> set:
    """Unique Compound::* entity strings from triplets."""
    compounds = set()
    n = 0
    with drkg_path.open(encoding="utf-8") as f:
        for line in f:
            n += 1
            if n % 1_000_000 == 0:
                print(f"  scanned {n:,} triplets | compounds {len(compounds):,}")
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            for ent in (parts[0], parts[2]):
                if ent.startswith("Compound::"):
                    compounds.add(ent)
    print(f"  scanned {n:,} triplets | compounds {len(compounds):,}")
    return compounds


def main():
    ap = argparse.ArgumentParser(description="Build DRKG compound name index")
    ap.add_argument("--drkg", type=Path, default=DEFAULT_DRKG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    drkg_path = find_drkg_tsv(args.drkg)
    print(f"DRKG file: {drkg_path}")
    print("Loading DrugBank → name map ...")
    db_to_name = load_drugbank_names()

    print("Collecting Compound entities ...")
    compounds = collect_compounds(drkg_path)

    name_to_id = {}
    rows_out = []
    n_named = 0
    n_unnamed = 0

    for ent in sorted(compounds):
        # Compound::DB00001 or Compound::CHEMBL...
        local = ent.split("::", 1)[1] if "::" in ent else ent
        name = None
        if local.startswith("DB") and local in db_to_name:
            name = db_to_name[local]
        if not name:
            n_unnamed += 1
            continue
        n_named += 1
        rows_out.append({
            "drug_id": ent,
            "drugbank_id": local,
            "name": name,
        })
        key = _norm(name)
        if key and key not in name_to_id:
            name_to_id[key] = ent

    args.out_dir.mkdir(parents=True, exist_ok=True)
    drugs_csv = args.out_dir / "drkg_drugs.csv"
    name_json = args.out_dir / "drkg_drug_name_to_id.json"

    with drugs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["drug_id", "drugbank_id", "name"])
        w.writeheader()
        w.writerows(rows_out)

    name_json.write_text(json.dumps(name_to_id, indent=0, ensure_ascii=False) + "\n")

    print()
    print(f"Compound entities:     {len(compounds):,}")
    print(f"With resolvable name:  {n_named:,}")
    print(f"Skipped (no name):     {n_unnamed:,}")
    print(f"Name aliases:          {len(name_to_id):,}")
    print(f"Wrote {drugs_csv}")
    print(f"Wrote {name_json}")


if __name__ == "__main__":
    main()
