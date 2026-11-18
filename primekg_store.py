# ============================================================
# primekg_store.py
# Minimal lookups over data/kg/processed/* from build_primekg_index.py.
# Why a separate module: keep UMLS code untouched; attack modes can
# import this later without rescanning kg.csv.
# ============================================================

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "kg" / "processed"

_name_to_id = None
_id_to_name = None
_interactions = None  # drug_id -> list of (partner_id, partner_name, display_relation)


def _ensure_names():
    global _name_to_id, _id_to_name
    if _name_to_id is not None:
        return
    path = PROCESSED / "primekg_drug_name_to_id.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python build_primekg_index.py"
        )
    _name_to_id = json.loads(path.read_text())
    drugs_path = PROCESSED / "primekg_drugs.csv"
    _id_to_name = {}
    with drugs_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            _id_to_name[row["drug_id"]] = row["name"]


def _ensure_interactions():
    global _interactions
    if _interactions is not None:
        return
    _ensure_names()
    path = PROCESSED / "primekg_ddi_pairs.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python build_primekg_index.py"
        )
    _interactions = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a, b = row["drug_a_id"], row["drug_b_id"]
            rel = row.get("display_relation") or ""
            _interactions[a].append((b, row["drug_b_name"], rel))
            _interactions[b].append((a, row["drug_a_name"], rel))


def find_drug(mention: str):
    """Return (drug_id, canonical_name) or (None, None). Exact lowercased name match."""
    _ensure_names()
    if not mention:
        return None, None
    key = mention.lower().strip()
    drug_id = _name_to_id.get(key)
    if not drug_id:
        return None, None
    return drug_id, _id_to_name.get(drug_id, mention)


def interactions(drug_id: str, limit: int = 5):
    """Return up to `limit` (partner_id, partner_name, display_relation)."""
    _ensure_interactions()
    return list(_interactions.get(drug_id, []))[:limit]


def has_ddi(drug_a_id: str, drug_b_id: str) -> bool:
    """True if a drug_drug edge exists between the two ids."""
    _ensure_interactions()
    return any(p == drug_b_id for p, _, _ in _interactions.get(drug_a_id, []))
