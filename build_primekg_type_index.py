# ============================================================
# build_primekg_type_index.py
# Build per-category name → id maps from PrimeKG kg.csv.
# Why: measure which PrimeKG node types appear in PubMedQA
# (not just drugs).
#
# Output under data/kg/processed/:
#   primekg_types.json              — list of types + counts
#   primekg_names_by_type.json      — {type: {name_lower: id, ...}}
#   primekg_nodes_by_type/<type>.csv — id,name per type
#
# Usage:
#   python build_primekg_type_index.py
# ============================================================

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_KG = ROOT / "data" / "kg" / "primekg" / "kg.csv"
DEFAULT_OUT = ROOT / "data" / "kg" / "processed"

csv.field_size_limit(min(2**31 - 1, 10_000_000))

# Safe filename for a type like "gene/protein" or "effect/phenotype"
def type_slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser(description="Build PrimeKG per-type name indexes")
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.kg.is_file():
        raise SystemExit(f"Missing PrimeKG file: {args.kg}")

    args.out.mkdir(parents=True, exist_ok=True)
    nodes_dir = args.out / "primekg_nodes_by_type"
    nodes_dir.mkdir(parents=True, exist_ok=True)

    # type -> id -> name
    nodes = defaultdict(dict)
    type_counts = Counter()
    n_rows = 0

    print(f"Scanning {args.kg} ...")
    with args.kg.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            if n_rows % 1_000_000 == 0:
                print(f"  scanned {n_rows:,} | types {len(nodes)}")

            for side in ("x", "y"):
                t = (row.get(f"{side}_type") or "").strip()
                i = (row.get(f"{side}_id") or "").strip()
                n = (row.get(f"{side}_name") or "").strip()
                if not t or not i or not n:
                    continue
                type_counts[t] += 1
                if i not in nodes[t]:
                    nodes[t][i] = n

    print(f"Done scan: {n_rows:,} rows | {len(nodes)} types")

    names_by_type = {}
    types_meta = []
    for t, id_to_name in sorted(nodes.items(), key=lambda kv: -len(kv[1])):
        name_to_id = {}
        for node_id, name in id_to_name.items():
            key = name.lower().strip()
            if key and key not in name_to_id:
                name_to_id[key] = node_id
        names_by_type[t] = name_to_id
        types_meta.append({
            "type": t,
            "n_nodes": len(id_to_name),
            "n_names": len(name_to_id),
            "side_mentions_in_edges": type_counts[t],
        })

        slug = type_slug(t)
        csv_path = nodes_dir / f"{slug}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fout:
            w = csv.DictWriter(fout, fieldnames=["node_id", "name"])
            w.writeheader()
            for node_id, name in sorted(id_to_name.items(), key=lambda kv: kv[1].lower()):
                w.writerow({"node_id": node_id, "name": name})

    types_path = args.out / "primekg_types.json"
    names_path = args.out / "primekg_names_by_type.json"
    types_path.write_text(json.dumps(types_meta, indent=2))
    # Compact JSON (large); no indent
    names_path.write_text(json.dumps(names_by_type))

    print(f"Wrote {types_path}")
    print(f"Wrote {names_path}")
    print(f"Wrote per-type CSVs under {nodes_dir}/")
    print("Types (by node count):")
    for row in types_meta:
        print(f"  {row['n_nodes']:6d} nodes  {row['type']}")


if __name__ == "__main__":
    main()
