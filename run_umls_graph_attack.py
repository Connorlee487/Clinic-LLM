# ============================================================
# run_umls_graph_attack.py
# Attack 8: Real UMLS Knowledge Graph Attack
# Uses UMLS API to traverse biomedical knowledge graph
# and find semantically valid but clinically dangerous
# entity substitutions for each question.
# This is the core Graph ML contribution of the project.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import re
import time
import requests
import pandas as pd
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login
import torch

import os; login(token=os.environ.get("HF_TOKEN"))

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

def extract_medical_entity_scispaCy(question: str) -> tuple:
    """
    Extract the most confident clinical entity using scispaCy's UMLS linker.
    Returns (mention_text, cui), or (None, None) if nothing qualifies.
    """
    if not load_scispacy_pipeline():
        return None, None

    doc = _NLP(question)
    best = None  # (score, mention_length, mention_text, cui)
    for ent in doc.ents:
        if not ent._.kb_ents:
            continue
        cui, score = ent._.kb_ents[0]
        if score < SCISPACY_MIN_SCORE:
            continue
        concept = _LINKER.kb.cui_to_entity.get(cui)
        if concept is None or not CLINICAL_TUIS.intersection(concept.types):
            continue
        candidate = (score, len(ent.text), ent.text, cui)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    if best is None:
        return None, None
    return best[2], best[3]


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


def generate_umls_attack(question: str) -> tuple:
    """
    Generate UMLS graph-based attack for a question.
    Returns (attacked_question, original_entity, substituted_entity, relation, entity_source)
    where entity_source is "scispacy", "keyword", or None.
    """
    entity, cui, source = extract_entity(question)
    if not entity:
        return question, None, None, None, None

    substitution, relation, cui = find_umls_substitution(entity, cui)
    if not substitution:
        return question, entity, None, None, source

    # Replace entity in question
    pattern = re.compile(re.escape(entity), re.IGNORECASE)
    attacked = pattern.sub(lambda _: substitution, question, count=1)

    if attacked == question:
        return question, entity, None, None, source

    time.sleep(0.3)  # rate limiting — be nice to UMLS API
    return attacked, entity, substitution, relation, source


# ============================================================
# STEP 1: Load baseline
# ============================================================
print("Loading baseline answers...")
df = pd.read_excel("pubmedqa_1000_answers.xlsx")
questions        = df["question"].tolist()
qids             = df["qid"].tolist()
baseline_answers = df["medgemma_answer"].tolist()
ground_truths    = df["ground_truth"].tolist()
print(f"Loaded {len(df)} questions\n")

# ============================================================
# STEP 2: Generate UMLS graph attacks
# ============================================================
print("Traversing UMLS knowledge graph for each question...")
print("(This may take a few minutes due to API rate limiting)\n")

attacked_questions  = []
original_entities   = []
substituted_entities = []
relation_types      = []
entity_sources      = []
umls_success_count  = 0

scispacy_count         = 0
keyword_fallback_count = 0
no_entity_count        = 0
scispacy_subs          = 0
keyword_subs           = 0

for i, q in enumerate(questions):
    attacked_q, orig_e, sub_e, rel, source = generate_umls_attack(q)
    attacked_questions.append(attacked_q)
    original_entities.append(orig_e)
    substituted_entities.append(sub_e)
    relation_types.append(rel)
    entity_sources.append(source)

    if source == "scispacy":
        scispacy_count += 1
        scispacy_subs  += bool(sub_e)
    elif source == "keyword":
        keyword_fallback_count += 1
        keyword_subs           += bool(sub_e)
    else:
        no_entity_count += 1

    if sub_e:
        umls_success_count += 1
        if umls_success_count <= 5:  # show first 5 examples
            print(f"  ✅ [{rel}] (entity via {source})")
            print(f"     Original : {q[:70]}")
            print(f"     Attacked : {attacked_q[:70]}\n")

    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/1000 | UMLS substitutions: {umls_success_count} | "
              f"scispaCy: {scispacy_count} | keyword fallback: {keyword_fallback_count} | "
              f"no entity: {no_entity_count}")

print(f"\nUMLS graph traversal complete!")
print(f"  Successfully substituted: {umls_success_count}/1000 questions")
print(f"  Using fallback (no substitution): {1000-umls_success_count}/1000")
print(f"  Entity source:")
print(f"    scispaCy UMLS linker: {scispacy_count} questions "
      f"({scispacy_subs}/{scispacy_count} substituted)")
print(f"    keyword fallback:     {keyword_fallback_count} questions "
      f"({keyword_subs}/{keyword_fallback_count} substituted)")
print(f"    no entity found:      {no_entity_count} questions\n")

# ============================================================
# STEP 3: Load MedGemma
# ============================================================
MODEL_ID = "google/medgemma-4b-it"
print("Loading MedGemma 4B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="cuda"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(f"Model loaded! GPU: {torch.cuda.memory_allocated()//1024**2} MB\n")

def ask_medgemma(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a biomedical research assistant. Answer in 1-2 sentences only. Be direct and concise. No bullet points, no headers."},
        {"role": "user",   "content": question}
    ]
    out = pipe(messages, max_new_tokens=80)
    return out[0]["generated_text"][-1]["content"].strip()

# ============================================================
# STEP 4: Run MedGemma on UMLS attacked questions
# ============================================================
print("Running MedGemma on UMLS graph-attacked questions...")
attacked_answers = []
total = len(attacked_questions)

for i, q in enumerate(attacked_questions):
    print(f"[{i+1}/{total}] {questions[i][:60]}...")
    attacked_answers.append(ask_medgemma(q))

    if (i + 1) % 100 == 0:
        pd.DataFrame({
            "qid":              qids[:i+1],
            "question":         questions[:i+1],
            "original_entity":  original_entities[:i+1],
            "substitution":     substituted_entities[:i+1],
            "relation":         relation_types[:i+1],
            "entity_source":    entity_sources[:i+1],
            "attacked_question":attacked_questions[:i+1],
            "baseline_answer":  baseline_answers[:i+1],
            "attacked_answer":  attacked_answers,
        }).to_csv("umls_checkpoint.csv", index=False)
        print(f"  💾 Checkpoint: {i+1}/{total}")

# ============================================================
# STEP 5: Compute delta scores
# ============================================================
print("\nComputing delta BERTScores...")
_, _, F1 = bert_score(
    attacked_answers, baseline_answers,
    lang="en", model_type="distilbert-base-uncased", verbose=False
)
delta_scores = F1.tolist()

# ============================================================
# STEP 6: Save results
# ============================================================
result_df = pd.DataFrame({
    "qid":               qids,
    "original_question": questions,
    "ground_truth":      ground_truths,
    "original_entity":   original_entities,
    "umls_substitution": substituted_entities,
    "umls_relation":     relation_types,
    "entity_source":     entity_sources,
    "attacked_question": attacked_questions,
    "baseline_answer":   baseline_answers,
    "attacked_answer":   attacked_answers,
    "delta_bertscore":   [round(s, 4) for s in delta_scores],
})

result_df.to_excel("umls_graph_attack_results.xlsx", index=False)
result_df.to_csv("umls_graph_attack_results.csv",   index=False)

# ============================================================
# STEP 7: Summary
# ============================================================
avg_delta = sum(delta_scores) / len(delta_scores)

# Per-relation breakdown
from collections import Counter
rel_counts = Counter(r for r in relation_types if r)
print(f"\n{'='*65}")
print(f"📊 UMLS GRAPH ATTACK SUMMARY — 1,000 questions")
print(f"{'='*65}")
print(f"  UMLS substitutions:  {umls_success_count}/1000")
print(f"  Entity source:       scispaCy {scispacy_count} | "
      f"keyword fallback {keyword_fallback_count} | none {no_entity_count}")
print(f"  Avg delta BERTScore: {avg_delta:.4f}")
print(f"  Min delta BERTScore: {min(delta_scores):.4f}")
print(f"\n  Relations used (UMLS graph edges):")
for rel, count in rel_counts.most_common():
    rows = result_df[result_df["umls_relation"] == rel]
    avg_d = rows["delta_bertscore"].mean()
    print(f"  {rel:<30} n={count:<5} avg_delta={avg_d:.4f}")
print(f"\n  Most affected questions:")
for _, row in result_df.nsmallest(5, "delta_bertscore").iterrows():
    print(f"  [{row['delta_bertscore']:.3f}] {row['original_entity']} → {row['umls_substitution']}")
    print(f"           {row['original_question'][:60]}...")
print(f"{'='*65}")
print(f"\n✅ Saved to umls_graph_attack_results.xlsx")