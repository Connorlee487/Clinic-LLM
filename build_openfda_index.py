# ============================================================
# build_openfda_index.py
# Build name → id maps from the openFDA NDC Directory bulk dump
# (brand_name + generic_name + active ingredients).
#
# Download (curl; no wget needed):
#   mkdir -p data/kg/openfda
#   cd data/kg/openfda
#   curl -L -o drug-ndc-0001-of-0001.json.zip \
#     https://download.open.fda.gov/drug/ndc/drug-ndc-0001-of-0001.json.zip
#   cd ../../..
#
# Or let this script download for you:
#   python build_openfda_index.py --download
#
# Usage:
#   python build_openfda_index.py
#   python build_openfda_index.py --ndc-zip data/kg/openfda/drug-ndc-0001-of-0001.json.zip
# ============================================================

import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "kg" / "openfda"
DEFAULT_ZIP = RAW_DIR / "drug-ndc-0001-of-0001.json.zip"
DEFAULT_JSON = RAW_DIR / "drug-ndc-0001-of-0001.json"
OUT_DIR = ROOT / "data" / "kg" / "openfda" / "processed"
NDC_ZIP_URL = (
    "https://download.open.fda.gov/drug/ndc/drug-ndc-0001-of-0001.json.zip"
)


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"\s+", " ", s)


def _aliases_from_record(rec: dict) -> list:
    """Collect brand, generic, and active-ingredient surface forms."""
    names = []
    brand = (rec.get("brand_name") or "").strip()
    if brand:
        names.append(brand)
    generic = (rec.get("generic_name") or "").strip()
    if generic:
        # Sometimes "A; B" style lists
        for part in re.split(r"[;]", generic):
            part = part.strip()
            if part:
                names.append(part)
    for ing in rec.get("active_ingredients") or []:
        if isinstance(ing, dict):
            n = (ing.get("name") or "").strip()
            if n:
                names.append(n)
    # Deduplicate preserving order
    seen = set()
    out = []
    for n in names:
        key = _norm(n)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def load_results(ndc_zip: Path = None, ndc_json: Path = None) -> list:
    if ndc_json and ndc_json.is_file():
        print(f"Reading {ndc_json} ...")
        data = json.loads(ndc_json.read_text(encoding="utf-8"))
        return data.get("results") or []

    if ndc_zip and ndc_zip.is_file():
        print(f"Reading {ndc_zip} ...")
        with zipfile.ZipFile(ndc_zip) as zf:
            members = [n for n in zf.namelist() if n.endswith(".json")]
            if not members:
                raise SystemExit(f"No .json inside {ndc_zip}")
            with zf.open(members[0]) as fh:
                raw = io.TextIOWrapper(fh, encoding="utf-8")
                data = json.load(raw)
        return data.get("results") or []

    raise SystemExit(
        "Missing openFDA NDC dump. Download:\n"
        f"  mkdir -p {RAW_DIR}\n"
        f"  cd {RAW_DIR}\n"
        f"  curl -L -o drug-ndc-0001-of-0001.json.zip \\\n"
        f"    {NDC_ZIP_URL}\n"
        "  cd ../../..\n"
        "Or re-run with:  python build_openfda_index.py --download"
    )


def build_index(results: list):
    name_to_id = {}
    rows_out = []
    n_alias = 0

    for rec in results:
        ndc = (rec.get("product_ndc") or "").strip()
        if not ndc:
            continue
        aliases = _aliases_from_record(rec)
        if not aliases:
            continue
        display = aliases[0]
        rows_out.append({
            "drug_id": ndc,
            "name": display,
            "brand_name": (rec.get("brand_name") or "").strip(),
            "generic_name": (rec.get("generic_name") or "").strip(),
        })
        for alias in aliases:
            key = _norm(alias)
            if len(key) < 3:
                continue
            if key not in name_to_id:
                name_to_id[key] = ndc
                n_alias += 1

    return name_to_id, rows_out, n_alias


def main():
    ap = argparse.ArgumentParser(description="Build openFDA NDC drug name index")
    ap.add_argument("--ndc-zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--ndc-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--download", action="store_true",
        help="Download the NDC bulk zip into data/kg/openfda/ first",
    )
    args = ap.parse_args()

    if args.download:
        args.ndc_zip.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {NDC_ZIP_URL} ...")
        urlretrieve(NDC_ZIP_URL, args.ndc_zip)
        print(f"Saved {args.ndc_zip}")

    # Prefer unzipped JSON if present, else zip
    ndc_json = args.ndc_json if args.ndc_json.is_file() else None
    ndc_zip = args.ndc_zip if args.ndc_zip.is_file() else None
    results = load_results(ndc_zip=ndc_zip, ndc_json=ndc_json)
    print(f"NDC products: {len(results):,}")

    name_to_id, rows_out, n_alias = build_index(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    drugs_csv = args.out_dir / "openfda_drugs.csv"
    name_json = args.out_dir / "openfda_drug_name_to_id.json"

    with drugs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["drug_id", "name", "brand_name", "generic_name"]
        )
        w.writeheader()
        w.writerows(rows_out)

    name_json.write_text(json.dumps(name_to_id, indent=0, ensure_ascii=False) + "\n")

    print(f"Products indexed:  {len(rows_out):,}")
    print(f"Name aliases:      {n_alias:,}")
    print(f"Wrote {drugs_csv}")
    print(f"Wrote {name_json}")


if __name__ == "__main__":
    main()
