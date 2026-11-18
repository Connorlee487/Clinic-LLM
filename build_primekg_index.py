# ============================================================
# build_primekg_index.py
# Step 1 only: turn data/kg/primekg/kg.csv into small lookup files
# under data/kg/processed/ so attack code never rescans ~1GB CSV.
#
# Why: kg.csv has millions of edges (mostly non-drug). Attacks only
# need (1) drug name → id and (2) drug–drug pairs + relation label.
#
# Uses stdlib csv only (avoids pandas segfaults in some envs).
# ============================================================

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_KG = ROOT / "data" / "kg" / "primekg" / "kg.csv"
DEFAULT_OUT = ROOT / "data" / "kg" / "processed"

# Raise CSV field size for long PrimeKG rows
csv.field_size_limit(min(2**31 - 1, 10_000_000))


def main():
    parser = argparse.ArgumentParser(description="Build PrimeKG drug/DDI index")
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.kg.is_file():
        raise SystemExit(f"Missing PrimeKG file: {args.kg}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {args.kg} ...")

    drugs = {}       # id -> name
    ddi_path = args.out / "primekg_ddi_pairs.csv"
    n_rows = 0
    n_ddi = 0

    # Stream DDI rows straight to disk (2.6M+ edges — don't hold all in RAM)
    with args.kg.open(newline="", encoding="utf-8") as fin, \
         ddi_path.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "drug_a_id", "drug_a_name",
                "drug_b_id", "drug_b_name",
                "display_relation",
            ],
        )
        writer.writeheader()

        for row in reader:
            n_rows += 1
            if n_rows % 1_000_000 == 0:
                print(f"  scanned {n_rows:,} | drugs {len(drugs):,} | ddi {n_ddi:,}")

            xt, yt = row.get("x_type"), row.get("y_type")
            if xt == "drug":
                xid, xname = row.get("x_id"), (row.get("x_name") or "").strip()
                if xid and xname and xid not in drugs:
                    drugs[xid] = xname
            if yt == "drug":
                yid, yname = row.get("y_id"), (row.get("y_name") or "").strip()
                if yid and yname and yid not in drugs:
                    drugs[yid] = yname

            if row.get("relation") != "drug_drug":
                continue

            n_ddi += 1
            writer.writerow({
                "drug_a_id": row.get("x_id") or "",
                "drug_a_name": (row.get("x_name") or "").strip(),
                "drug_b_id": row.get("y_id") or "",
                "drug_b_name": (row.get("y_name") or "").strip(),
                "display_relation": (row.get("display_relation") or "").strip(),
            })

    print(f"Done scan: {n_rows:,} rows | {len(drugs):,} drugs | {n_ddi:,} ddi edges")

    drugs_path = args.out / "primekg_drugs.csv"
    with drugs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["drug_id", "name"])
        w.writeheader()
        for drug_id, name in sorted(drugs.items()):
            w.writerow({"drug_id": drug_id, "name": name})

    name_to_id = {}
    for drug_id, name in drugs.items():
        key = name.lower().strip()
        if key and key not in name_to_id:
            name_to_id[key] = drug_id

    map_path = args.out / "primekg_drug_name_to_id.json"
    map_path.write_text(json.dumps(name_to_id, indent=0, sort_keys=True))

    print(f"Wrote {drugs_path} ({len(drugs):,} rows)")
    print(f"Wrote {ddi_path} ({n_ddi:,} rows)")
    print(f"Wrote {map_path} ({len(name_to_id):,} names)")
    print("Step 1 complete. Next: smoke-test name → DDI partner lookups.")


if __name__ == "__main__":
    main()
