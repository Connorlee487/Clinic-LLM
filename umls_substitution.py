# ============================================================
# umls_substitution.py
# Entity extraction + UMLS graph substitution, shared by
# run_umls_graph_attack.py and compare_umls_substitution.py.
# Importing this module does not load any LLM.
# ============================================================

import os
import re
import time
import requests

# ============================================================
# UMLS API CONFIG
# ============================================================
UMLS_API_KEY = os.environ.get("UMLS_API_KEY")
UMLS_BASE    = "https://uts-ws.nlm.nih.gov/rest"
UMLS_VERSION = "current"

# ============================================================
# UMLS GRAPH TRAVERSAL
# ============================================================


def find_umls_substitution(term: str, cui: str = None) -> tuple:
    """
    Full UMLS graph traversal:
    1. Get CUI for the term (skipped if cui is already known)
    2. Get related concepts via UMLS relations API
    3. Filter for clean English substitutions
    Returns (substitution_name, relation_type, cui)
    """
    try:
        # Step 1: Get CUI
        if cui is None:
            r = requests.get(
                f"{UMLS_BASE}/search/{UMLS_VERSION}",
                params={"string": term, "apiKey": UMLS_API_KEY,
                        "pageSize": 1, "searchType": "exact"},
                timeout=10
            )
            results = r.json()["result"]["results"]
            if not results or results[0]["ui"] == "NONE":
                return None, None, None
            cui = results[0]["ui"]

        # Step 2: Get related concepts
        r2 = requests.get(
            f"{UMLS_BASE}/content/{UMLS_VERSION}/CUI/{cui}/relations",
            params={"apiKey": UMLS_API_KEY, "pageSize": 25},
            timeout=10
        )
        relations = r2.json().get("result", [])

        # Step 3: Pick best substitution
        # Use RO (related other) and RB (related broader) — clinically linked
        for rel in relations:
            name  = rel.get("relatedIdName", "")
            label = rel.get("relationLabel", "")
            if (label in ["RO", "RB"] and
                name and
                len(name.split()) <= 4 and
                term.lower() not in name.lower() and
                all(ord(c) < 128 for c in name)):  # English only
                return name, label, cui

        return None, None, cui

    except Exception:
        return None, None, None


# ============================================================
# ENTITY EXTRACTION
# Medical entity keywords to look for in questions
# ============================================================

MEDICAL_ENTITIES = [
    # Conditions — longer/specific first to avoid partial matches
    "myocardial infarction", "atrial fibrillation", "heart failure",
    "blood pressure", "chronic obstructive pulmonary disease",
    "rheumatoid arthritis", "type 2 diabetes", "breast cancer",
    "lung cancer", "prostate cancer", "colorectal cancer",
    "spinal cord", "bone marrow", "lymph node",
    "hypertension", "diabetes", "cancer", "tumor", "tumour",
    "asthma", "pneumonia", "hepatitis", "migraine", "epilepsy",
    "depression", "arthritis", "pancreatitis", "sepsis", "obesity",
    "anemia", "fibrosis", "carcinoma", "lymphoma", "leukemia",
    "melanoma", "cirrhosis", "cholesterol", "thyroid", "parkinson",
    "alzheimer", "osteoporosis", "schizophrenia", "dementia",
    "stroke", "angina", "arrhythmia", "thrombosis", "embolism",
    "fracture", "infection", "inflammation", "ulcer", "polyp",
    "cyst", "abscess", "stenosis", "insufficiency", "dysfunction",
    # Drugs
    "warfarin", "aspirin", "metformin", "insulin", "heparin",
    "statin", "methotrexate", "tamoxifen", "lithium", "morphine",
    "ibuprofen", "paracetamol", "amoxicillin", "vancomycin",
    # Procedures
    "surgery", "biopsy", "chemotherapy", "radiotherapy", "dialysis",
    "transplant", "angioplasty", "endoscopy", "laparoscopy",
    "cholecystectomy", "appendectomy", "mastectomy", "colectomy",
    # Anatomy
    "kidney", "liver", "heart", "lung", "brain", "colon",
    "prostate", "breast", "pancreas", "ovary", "uterus",
    "bladder", "spleen", "gallbladder", "appendix", "tonsil",
]

def extract_medical_entity(question: str) -> str:
    """
    Extract the most prominent medical entity from a question.
    Tries longest match first to avoid partial matches.
    """
    q_lower = question.lower()
    # Sort by length descending — match longer phrases first
    for entity in sorted(MEDICAL_ENTITIES, key=len, reverse=True):
        if entity in q_lower:
            return entity
    return None

# ============================================================
# SCISPACY UMLS ENTITY LINKER
# Install the model:
#   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
# First use downloads ~1GB of UMLS linker files to ~/.scispacy
# (run once on a node with internet before SLURM compute jobs).
# ============================================================

SCISPACY_MIN_SCORE = 0.85

# UMLS semantic types (TUIs) worth substituting
CLINICAL_TUIS = {
    "T047",  # Disease or Syndrome
    "T191",  # Neoplastic Process
    "T184",  # Sign or Symptom
    "T046",  # Pathologic Function
    "T037",  # Injury or Poisoning
    "T121",  # Pharmacologic Substance
    "T200",  # Clinical Drug
    "T195",  # Antibiotic
    "T061",  # Therapeutic or Preventive Procedure
    "T060",  # Diagnostic Procedure
    "T023",  # Body Part, Organ, or Organ Component
}

_NLP           = None
_LINKER        = None
_NLP_AVAILABLE = None  # None = not yet attempted

def load_scispacy_pipeline() -> bool:
    """Load en_core_sci_sm + abbreviation detector + UMLS linker once."""
    global _NLP, _LINKER, _NLP_AVAILABLE
    if _NLP_AVAILABLE is not None:
        return _NLP_AVAILABLE
    try:
        import spacy
        from scispacy.abbreviation import AbbreviationDetector  # noqa: F401 (registers pipe)
        from scispacy.linking import EntityLinker               # noqa: F401 (registers pipe)

        print("Loading scispaCy pipeline (en_core_sci_sm + UMLS linker)...")
        _NLP = spacy.load("en_core_sci_sm")
        _NLP.add_pipe("abbreviation_detector")
        _NLP.add_pipe("scispacy_linker",
                      config={"resolve_abbreviations": True, "linker_name": "umls"})
        _LINKER = _NLP.get_pipe("scispacy_linker")
        _NLP_AVAILABLE = True
        print("scispaCy pipeline loaded.\n")
    except Exception as e:
        print(f"WARNING: scispaCy pipeline unavailable ({e}) — "
              f"using keyword extraction for all questions.\n")
        _NLP_AVAILABLE = False
    return _NLP_AVAILABLE

def scispacy_candidates(question: str) -> list:
    """
    Every entity scispaCy finds in the question, with its top UMLS link and
    whether it passes the score and semantic-type filters.
    Each item: {mention, cui, name, score, types, kept, reason}.
    """
    if not load_scispacy_pipeline():
        return []

    candidates = []
    for ent in _NLP(question).ents:
        item = {"mention": ent.text, "cui": None, "name": None,
                "score": None, "types": [], "kept": False, "reason": None}
        if not ent._.kb_ents:
            item["reason"] = "no UMLS link"
            candidates.append(item)
            continue
        cui, score = ent._.kb_ents[0]
        concept = _LINKER.kb.cui_to_entity.get(cui)
        item.update(cui=cui, score=round(float(score), 3),
                    name=concept.canonical_name if concept else None,
                    types=list(concept.types) if concept else [])
        if score < SCISPACY_MIN_SCORE:
            item["reason"] = f"score < {SCISPACY_MIN_SCORE}"
        elif not CLINICAL_TUIS.intersection(item["types"]):
            item["reason"] = "non-clinical semantic type"
        else:
            item["kept"] = True
        candidates.append(item)
    return candidates

def extract_medical_entity_scispaCy(question: str) -> tuple:
    """
    Extract the most confident clinical entity using scispaCy's UMLS linker.
    Returns (mention_text, cui), or (None, None) if nothing qualifies.
    """
    kept = [c for c in scispacy_candidates(question) if c["kept"]]
    if not kept:
        return None, None
    best = max(kept, key=lambda c: (c["score"], len(c["mention"])))
    return best["mention"], best["cui"]


def extract_entity(question: str) -> tuple:
    """
    Pick the entity to attack: scispaCy UMLS linker first, keyword list as fallback.
    Returns (entity, cui, source); source is "scispacy", "keyword", or None.
    """
    entity, cui = extract_medical_entity_scispaCy(question)
    if entity:
        return entity, cui, "scispacy"
    entity = extract_medical_entity(question)
    if entity:
        return entity, None, "keyword"
    return None, None, None


# ============================================================
# SUBSTITUTION
# ============================================================

def substitute_entity(question: str, entity: str, cui: str = None) -> tuple:
    """
    Look up a UMLS neighbour for the entity and swap it into the question.
    Returns (attacked_question, substitution, relation); substitution and
    relation are None when no usable neighbour was found.
    """
    substitution, relation, cui = find_umls_substitution(entity, cui)
    if not substitution:
        return question, None, None

    pattern = re.compile(re.escape(entity), re.IGNORECASE)
    attacked = pattern.sub(lambda _: substitution, question, count=1)
    if attacked == question:
        return question, None, None

    time.sleep(0.3)  # rate limiting — be nice to UMLS API
    return attacked, substitution, relation


def generate_umls_attack(question: str) -> tuple:
    """
    Generate UMLS graph-based attack for a question.
    Returns (attacked_question, original_entity, substituted_entity, relation, entity_source)
    where entity_source is "scispacy", "keyword", or None.
    """
    entity, cui, source = extract_entity(question)
    if not entity:
        return question, None, None, None, None

    attacked, substitution, relation = substitute_entity(question, entity, cui)
    return attacked, entity, substitution, relation, source
