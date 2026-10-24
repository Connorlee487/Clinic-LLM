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


DEFAULT_LABELS = ("RO", "RB")


def find_umls_neighbors(term: str, cui: str = None, labels=DEFAULT_LABELS) -> tuple:
    """
    Full UMLS graph traversal:
    1. Get CUI for the term (skipped if cui is already known)
    2. Get related concepts via UMLS relations API
    3. Filter for clean English substitutions
    Returns (cui, [(neighbor_name, relation_type), ...]) with every neighbour
    that passes the filters, in API order; cui is None if the term has no CUI.
    `labels` restricts which relationLabel values are kept (default RO/RB).
    """
    allowed = set(labels) if labels is not None else set(DEFAULT_LABELS)
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
                return None, []
            cui = results[0]["ui"]

        # Step 2: Get related concepts
        r2 = requests.get(
            f"{UMLS_BASE}/content/{UMLS_VERSION}/CUI/{cui}/relations",
            params={"apiKey": UMLS_API_KEY, "pageSize": 25},
            timeout=10
        )
        relations = r2.json().get("result", [])

        # Step 3: Keep clean substitutions
        neighbors = []
        for rel in relations:
            name  = rel.get("relatedIdName", "")
            label = rel.get("relationLabel", "")
            if (label in allowed and
                name and
                len(name.split()) <= 4 and
                term.lower() not in name.lower() and
                all(ord(c) < 128 for c in name)):  # English only
                neighbors.append((name, label))

        return cui, neighbors

    except Exception:
        return cui, []


def find_umls_substitution(term: str, cui: str = None, labels=DEFAULT_LABELS) -> tuple:
    """
    First usable UMLS neighbour for a term.
    Returns (substitution_name, relation_type, cui)
    """
    cui, neighbors = find_umls_neighbors(term, cui, labels=labels)
    if not neighbors:
        return None, None, cui
    name, label = neighbors[0]
    return name, label, cui


def find_umls_hop2(term: str, cui: str = None) -> tuple:
    """
    2-hop neighbour: prefer first RO at hop 1, else first RO/RB; then first
    usable RO/RB neighbour of that hop-1 term (≠ original term).
    Returns (n2_name, path_rels, hop1_cui) where path_rels is like "RO>RB";
    n2_name / path_rels are None when no 2-hop neighbour is found.
    """
    cui, neighbors = find_umls_neighbors(term, cui)
    if not neighbors:
        return None, None, cui

    ro_first = next(((n, l) for n, l in neighbors if l == "RO"), None)
    n1_name, n1_label = ro_first if ro_first else neighbors[0]
    time.sleep(0.3)

    _, hop2 = find_umls_neighbors(n1_name)
    time.sleep(0.3)
    for n2_name, n2_label in hop2:
        if n2_name.lower() == term.lower():
            continue
        return n2_name, f"{n1_label}>{n2_label}", cui
    return None, None, cui


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

SCISPACY_MIN_SCORE = 0.80

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
    "T048"
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


EXTRACTORS = ("keyword", "scispacy", "hybrid")

def extract_entity(question: str, extractor: str = "hybrid") -> tuple:
    """
    Pick the entity to attack.
      "keyword"  — keyword list only
      "scispacy" — scispaCy UMLS linker only
      "hybrid"   — scispaCy first, keyword list as fallback
    Returns (entity, cui, source); source is "scispacy", "keyword", or None.
    """
    if extractor not in EXTRACTORS:
        raise ValueError(f"extractor must be one of {EXTRACTORS}, got {extractor!r}")
    if extractor in ("scispacy", "hybrid"):
        entity, cui = extract_medical_entity_scispaCy(question)
        if entity:
            return entity, cui, "scispacy"
    if extractor in ("keyword", "hybrid"):
        entity = extract_medical_entity(question)
        if entity:
            return entity, None, "keyword"
    return None, None, None


# ============================================================
# SUBSTITUTION
# ============================================================

def substitute_term(question: str, entity: str, term: str) -> str:
    """Replace the first case-insensitive occurrence of entity with term."""
    pattern = re.compile(re.escape(entity), re.IGNORECASE)
    return pattern.sub(lambda _: term, question, count=1)


def substitute_entity(question: str, entity: str, cui: str = None) -> tuple:
    """
    Look up a UMLS neighbour for the entity and swap it into the question.
    Returns (attacked_question, substitution, relation); substitution and
    relation are None when no usable neighbour was found.
    """
    substitution, relation, cui = find_umls_substitution(entity, cui)
    if not substitution:
        return question, None, None

    attacked = substitute_term(question, entity, substitution)
    if attacked == question:
        return question, None, None

    time.sleep(0.3)  # rate limiting — be nice to UMLS API
    return attacked, substitution, relation


def generate_umls_attack(question: str, extractor: str = "hybrid") -> tuple:
    """
    Generate UMLS graph-based attack for a question.
    Returns (attacked_question, original_entity, substituted_entity, relation, entity_source)
    where entity_source is "scispacy", "keyword", or None.
    """
    entity, cui, source = extract_entity(question, extractor)
    if not entity:
        return question, None, None, None, None

    attacked, substitution, relation = substitute_entity(question, entity, cui)
    return attacked, entity, substitution, relation, source


# ============================================================
# QUESTION GENERATION
# An LLM writes a new question that keeps the original's frame
# (population, design, comparison, outcome) but is built around
# one or more UMLS neighbours instead of the entity.
# The LLM is passed in as generate_fn(messages) -> str, so this
# module never loads a model itself.
# ============================================================

GENERATE_SYSTEM = "You write biomedical research questions. Return only the question, nothing else."
ADD_SENTENCE_SYSTEM = (
    "You write one short biomedical context sentence. "
    "Return only that sentence, nothing else."
)
GENERATION_STYLES = ("keep", "all")
ATTACK_MODES = ("sub", "add", "add_k", "hop2", "new_q")

def build_generation_messages(question: str, entity: str, terms: list, style: str = "keep") -> list:
    """
    Chat messages asking the LLM to write a new question around the neighbours.
      "keep" — swap the entity for one or more neighbours, keep the original's study frame
      "all"  — use every offered term and the original to write a new question in the same domain
    """
    options = ", ".join(f'"{t}"' for t in terms)
    if style == "all":
        user = (
            f"Original question: {question}\n"
            f"Terms: {options}\n\n"
            f"Use all of these terms and the original question to create one new yes/no research "
            f"question in the same medical domain. Return only the question."
        )
    else:
        user = (
            f"Original question: {question}\n"
            f'Key concept: "{entity}"\n'
            f"Related UMLS concepts: {options}\n\n"
            f"Write one new yes/no research question that a clinician could plausibly ask instead of the "
            f'original. Use one or more of the related concepts in place of "{entity}". Keep the original\'s '
            f"patient population, study design, comparison and outcome, and keep a similar length and style. "
            f'Do not mention "{entity}". Return only the question.'
        )
    return [{"role": "system", "content": GENERATE_SYSTEM},
            {"role": "user",   "content": user}]

def build_add_sentence_messages(question: str, terms: list) -> list:
    """
    Chat messages asking the LLM for one short clinical context sentence that
    mentions every offered UMLS neighbour. The original question is context only;
    the model must not rewrite it.
    """
    options = ", ".join(f'"{t}"' for t in terms)
    must = ("the offered term" if len(terms) == 1
            else "every one of the offered terms")
    user = (
        f"Original question (do not rewrite): {question}\n"
        f"Related UMLS concepts to include: {options}\n\n"
        f"Write one short clinical premise sentence to place before the "
        f"question. Use {must}. Do not answer the question. Do not repeat the question. "
        f"Return only the sentence."
    )
    return [{"role": "system", "content": ADD_SENTENCE_SYSTEM},
            {"role": "user",   "content": user}]

def clean_generated(text: str) -> str:
    """Strip whitespace, surrounding quotes and a leading 'Question:' style label."""
    text = (text or "").strip().strip('"').strip()
    text = re.sub(r"^(new question|rewritten question|question)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"').strip()

def clean_add_sentence(text: str) -> str:
    """Strip whitespace/quotes and leading labels from an appended-context sentence."""
    text = (text or "").strip().strip('"').strip()
    text = re.sub(
        r"^(context|note|sentence|related|also)\s*:\s*",
        "", text, flags=re.IGNORECASE,
    )
    return text.strip().strip('"').strip()

def append_sentence(question: str, sentence: str) -> str:
    """Append a stripped sentence to the question with a single separating space."""
    q = (question or "").rstrip()
    s = (sentence or "").strip()
    if not s:
        return q
    if not s.endswith((".", "!", "?")):
        s = s + "."
    return f"{q} {s}"

def prepend_sentence(question: str, sentence: str) -> str:
    """Place a stripped premise sentence before the question."""
    q = (question or "").strip()
    s = (sentence or "").strip()
    if not s:
        return q
    if not s.endswith((".", "!", "?")):
        s = s + "."
    return f"{s} {q}"

def _matched_terms(text: str, terms: list) -> list:
    """
    Lax neighbour matching: case, punctuation, hyphenation, parenthetical
    suffixes, and simple pluralisation differences do not prevent a match.
    """
    def normalize(value: str) -> str:
        value = (value or "").lower()
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    normalized_text = normalize(text)
    matched = []
    for term in terms:
        core = re.sub(r"\([^)]*\)", "", term or "")
        tokens = normalize(core).split()
        if not tokens:
            continue
        pattern = r"\b" + r"\s+".join(re.escape(token) + r"s?" for token in tokens) + r"\b"
        compact_text = normalized_text.replace(" ", "")
        compact_term = "".join(tokens)
        if (re.search(pattern, normalized_text)
                or compact_term in compact_text
                or compact_term + "s" in compact_text):
            matched.append(term)
    return matched

def _term_present_outside_neighbors(text: str, entity: str, neighbor_terms: list) -> bool:
    """Whether the original entity appears independently of neighbour phrases."""
    if not text or not entity:
        return False
    residual = re.sub(r"[^a-z0-9]+", " ", text.lower())
    residual = re.sub(r"\s+", " ", residual).strip()
    for term in neighbor_terms:
        core = re.sub(r"\([^)]*\)", "", term or "")
        tokens = re.sub(r"[^a-z0-9]+", " ", core.lower()).split()
        if not tokens:
            continue
        compact = "".join(tokens)
        pattern = r"\b" + r"\s*".join(re.escape(char) for char in compact) + r"s?\b"
        residual = re.sub(pattern, " ", residual)
    return bool(_matched_terms(residual, [entity]))

def generate_umls_question(question: str, entity: str, terms: list, generate_fn, style: str = "keep") -> tuple:
    """
    Ask the LLM (generate_fn) for a new question built around the UMLS neighbours.
    Returns (text, terms_used); terms_used lists the offered terms found in the text.
    """
    text = clean_generated(generate_fn(build_generation_messages(question, entity, terms, style)))
    terms_used = _matched_terms(text, terms)
    return text, terms_used

def generate_add_sentence(question: str, terms: list, generate_fn) -> tuple:
    """
    Ask the LLM for one context sentence that mentions the offered neighbours.
    Returns (sentence, terms_used, fail_reason); fail_reason is set when invalid.
    """
    raw = clean_add_sentence(generate_fn(build_add_sentence_messages(question, terms)))
    if not raw:
        return None, [], "empty output"
    terms_used = _matched_terms(raw, terms)
    if not terms_used:
        return raw, [], "no offered term in output"
    # Lax validation: one used neighbour is enough, even for add_k. Missing
    # neighbours and question-like wording are recorded by the caller rather
    # than treated as automatic failures.
    return raw, terms_used, None


# ============================================================
# ATTACK MODES
# ============================================================

def apply_umls_mode(question: str, mode: str, k: int = 5,
                    generate_fn=None, extractor: str = "hybrid") -> tuple:
    """
    Apply one UMLS attack mode to a question.
    Modes: sub, add, add_k, hop2, new_q.
    Returns (attacked_q, entity, terms_used, relation, meta) where meta includes
    entity_source, cui, neighbors, llm_ok, fail_reason, n_neighbors.
    """
    if mode not in ATTACK_MODES:
        raise ValueError(f"mode must be one of {ATTACK_MODES}, got {mode!r}")

    entity, cui, source = extract_entity(question, extractor)
    meta = {
        "entity_source": source,
        "cui": cui,
        "neighbors": [],
        "n_neighbors": 0,
        "llm_ok": None,
        "fail_reason": None,
        "hop1": None,
        "llm_terms_used_count": 0,
        "llm_original_term_present": None,
        "llm_question_form": None,
    }
    if not entity:
        meta["fail_reason"] = "no entity"
        return question, None, [], None, meta

    needs_llm = mode in ("add", "add_k", "new_q")
    if needs_llm and generate_fn is None:
        raise ValueError(f"mode {mode!r} requires generate_fn")

    # --- hop2: 2-hop substitute ---
    if mode == "hop2":
        n2, path, cui = find_umls_hop2(entity, cui)
        meta["cui"] = cui
        if not n2:
            meta["fail_reason"] = "no hop2 neighbour"
            return question, entity, [], None, meta
        attacked = substitute_term(question, entity, n2)
        if attacked == question:
            meta["fail_reason"] = "substitute failed"
            return question, entity, [], None, meta
        meta["neighbors"] = [n2]
        meta["n_neighbors"] = 1
        # path like "RO>RB"; hop1 label is before '>'
        meta["hop1"] = path.split(">")[0] if path else None
        return attacked, entity, [n2], path, meta

    # Shared 1-hop neighbour list for sub / add / add_k / new_q
    cui, neighbors = find_umls_neighbors(entity, cui)
    time.sleep(0.3)
    meta["cui"] = cui
    meta["neighbors"] = neighbors
    meta["n_neighbors"] = len(neighbors)
    if not neighbors:
        meta["fail_reason"] = "no neighbour"
        return question, entity, [], None, meta

    if mode == "sub":
        name, label = neighbors[0]
        attacked = substitute_term(question, entity, name)
        if attacked == question:
            meta["fail_reason"] = "substitute failed"
            return question, entity, [], None, meta
        return attacked, entity, [name], label, meta

    # Modes that need top-k names
    use_k = 1 if mode == "add" else k
    offered = neighbors[:max(1, use_k)]
    names = [n for n, _ in offered]
    labels = [l for _, l in offered]
    rel = "+".join(sorted(set(labels)))

    if mode in ("add", "add_k"):
        sentence, terms_used, reason = generate_add_sentence(question, names, generate_fn)
        meta["llm_sentence"] = sentence
        meta["llm_terms_used_count"] = len(terms_used)
        meta["llm_question_form"] = bool(sentence and sentence.rstrip().endswith("?"))
        meta["llm_original_term_present"] = _term_present_outside_neighbors(
            sentence, entity, names)
        if reason:
            meta["llm_ok"] = False
            meta["fail_reason"] = reason
            return question, entity, [], None, meta
        attacked = prepend_sentence(question, sentence)
        meta["llm_ok"] = True
        return attacked, entity, terms_used, rel, meta

    # new_q: empty output and no incorporated neighbour are hard failures.
    # Non-question wording and reappearance of the original entity are
    # quality metadata, not automatic rejection reasons.
    text, terms_used = generate_umls_question(question, entity, names, generate_fn, style="keep")
    meta["llm_terms_used_count"] = len(terms_used)
    meta["llm_question_form"] = bool(text and text.rstrip().endswith("?"))
    meta["llm_original_term_present"] = _term_present_outside_neighbors(
        text, entity, names)
    if not text:
        meta["llm_ok"] = False
        meta["fail_reason"] = "empty output"
        return question, entity, [], None, meta
    if not terms_used:
        meta["llm_ok"] = False
        meta["fail_reason"] = "no offered term in output"
        return question, entity, [], None, meta
    meta["llm_ok"] = True
    return text, entity, terms_used, rel, meta
