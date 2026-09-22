# ============================================================
# expand_umls_lookup.py
# Expands the UMLS substitution lookup table from 40 entities
# to cover ALL unique medical entities found in the dataset.
#
# Run on Delta LOGIN NODE (has internet access):
#   python expand_umls_lookup.py
#
# Input:  umls_graph_attack_results.xlsx (original 1000 questions)
#         pubmedqa_1000_answers.xlsx
#         umls_substitutions.csv (existing 40-entity lookup)
# Output: umls_substitutions_expanded.csv
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import re
import time
import requests
import pandas as pd
from collections import defaultdict

# ============================================================
# UMLS API CONFIG
# ============================================================
UMLS_API_KEY = os.environ.get("UMLS_API_KEY")
UMLS_BASE     = "https://uts-ws.nlm.nih.gov/rest"
UMLS_VERSION  = "current"

# ============================================================
# FULL MEDICAL ENTITY LIST
# Expanded from original 40 to cover broader clinical vocabulary
# ============================================================
MEDICAL_ENTITIES = [
    # Conditions — multi-word first
    "myocardial infarction", "atrial fibrillation", "heart failure",
    "blood pressure", "chronic obstructive pulmonary disease",
    "rheumatoid arthritis", "type 2 diabetes", "type 1 diabetes",
    "breast cancer", "lung cancer", "prostate cancer", "colorectal cancer",
    "ovarian cancer", "cervical cancer", "bladder cancer", "skin cancer",
    "spinal cord injury", "bone marrow transplant", "lymph node biopsy",
    "multiple sclerosis", "amyotrophic lateral sclerosis",
    "systemic lupus erythematosus", "inflammatory bowel disease",
    "irritable bowel syndrome", "crohn disease", "ulcerative colitis",
    "chronic kidney disease", "acute kidney injury",
    "coronary artery disease", "peripheral artery disease",
    "deep vein thrombosis", "pulmonary embolism",
    "sleep apnea", "restless leg syndrome",
    "attention deficit hyperactivity disorder",
    "post traumatic stress disorder",
    "bipolar disorder", "major depressive disorder",
    "generalized anxiety disorder", "panic disorder",
    # Single-word conditions
    "hypertension", "diabetes", "cancer", "tumor", "tumour",
    "asthma", "pneumonia", "hepatitis", "migraine", "epilepsy",
    "depression", "arthritis", "pancreatitis", "sepsis", "obesity",
    "anemia", "fibrosis", "carcinoma", "lymphoma", "leukemia",
    "melanoma", "cirrhosis", "cholesterol", "thyroid", "parkinson",
    "alzheimer", "osteoporosis", "schizophrenia", "dementia",
    "stroke", "angina", "arrhythmia", "thrombosis", "embolism",
    "fracture", "infection", "inflammation", "ulcer", "polyp",
    "cyst", "abscess", "stenosis", "insufficiency", "dysfunction",
    "neuropathy", "retinopathy", "nephropathy", "cardiomyopathy",
    "encephalopathy", "myopathy", "vasculitis", "sarcoidosis",
    "endometriosis", "polycystic", "hyperthyroidism", "hypothyroidism",
    "hyperlipidemia", "hyperglycemia", "hypoglycemia",
    "tachycardia", "bradycardia", "hypertrophy", "atrophy",
    "ischemia", "infarction", "hemorrhage", "hematoma",
    "edema", "ascites", "jaundice", "cyanosis",
    # Drugs
    "warfarin", "aspirin", "metformin", "insulin", "heparin",
    "statin", "methotrexate", "tamoxifen", "lithium", "morphine",
    "ibuprofen", "paracetamol", "amoxicillin", "vancomycin",
    "prednisone", "dexamethasone", "hydrochlorothiazide",
    "amlodipine", "lisinopril", "atorvastatin", "simvastatin",
    "omeprazole", "pantoprazole", "clopidogrel", "rivaroxaban",
    "dabigatran", "levothyroxine", "sertraline", "fluoxetine",
    "citalopram", "escitalopram", "venlafaxine", "duloxetine",
    "quetiapine", "olanzapine", "risperidone", "haloperidol",
    "lorazepam", "diazepam", "alprazolam", "clonazepam",
    "gabapentin", "pregabalin", "carbamazepine", "valproate",
    "levodopa", "donepezil", "memantine", "rivastigmine",
    "adalimumab", "infliximab", "rituximab", "trastuzumab",
    "bevacizumab", "pembrolizumab", "nivolumab",
    "cyclophosphamide", "doxorubicin", "cisplatin", "carboplatin",
    "paclitaxel", "docetaxel", "vincristine", "bleomycin",
    # Procedures
    "surgery", "biopsy", "chemotherapy", "radiotherapy", "dialysis",
    "transplant", "angioplasty", "endoscopy", "laparoscopy",
    "cholecystectomy", "appendectomy", "mastectomy", "colectomy",
    "bypass", "stenting", "catheterization", "intubation",
    "ventilation", "resection", "amputation", "arthroscopy",
    "colonoscopy", "bronchoscopy", "cystoscopy", "hysterectomy",
    "prostatectomy", "nephrectomy", "splenectomy", "thyroidectomy",
    # Anatomy
    "kidney", "liver", "heart", "lung", "brain", "colon",
    "prostate", "breast", "pancreas", "ovary", "uterus",
    "bladder", "spleen", "gallbladder", "appendix", "tonsil",
    "aorta", "artery", "vein", "lymph node", "bone marrow",
    "retina", "cornea", "cochlea", "thyroid", "adrenal",
    "pituitary", "hypothalamus", "cerebellum", "hippocampus",
    "trachea", "esophagus", "stomach", "duodenum", "jejunum",
    "ileum", "rectum", "sigmoid", "cecum", "peritoneum",
    # Lab values / biomarkers
    "hemoglobin", "creatinine", "albumin", "bilirubin",
    "troponin", "natriuretic peptide", "procalcitonin",
    "interleukin", "cytokine", "antibody", "antigen",
    "glucose", "insulin resistance", "cortisol", "testosterone",
    "estrogen", "progesterone", "thyroid hormone", "vitamin d",
]

# Remove duplicates while preserving order
seen = set()
MEDICAL_ENTITIES_UNIQUE = []
for e in MEDICAL_ENTITIES:
    if e not in seen:
        seen.add(e)
        MEDICAL_ENTITIES_UNIQUE.append(e)

print(f"Total unique entities to look up: {len(MEDICAL_ENTITIES_UNIQUE)}")

# ============================================================
# UMLS API FUNCTIONS
# ============================================================
def get_cui(term: str) -> tuple:
    """Get CUI and canonical name for a term."""
    try:
        r = requests.get(
            f"{UMLS_BASE}/search/{UMLS_VERSION}",
            params={"string": term, "apiKey": UMLS_API_KEY,
                    "pageSize": 1, "searchType": "bestMatch"},
            timeout=10
        )
        results = r.json()["result"]["results"]
        if not results or results[0]["ui"] == "NONE":
            return None, None
        return results[0]["ui"], results[0].get("name", term)
    except Exception as e:
        return None, None


def get_best_substitution(cui: str, original_term: str) -> tuple:
    """
    Get the best substitution for a CUI by traversing graph edges.
    Prefers RO (related other) and RB (related broader) relations.
    Returns (substitution_name, relation_type)
    """
    try:
        r = requests.get(
            f"{UMLS_BASE}/content/{UMLS_VERSION}/CUI/{cui}/relations",
            params={"apiKey": UMLS_API_KEY, "pageSize": 25},
            timeout=10
        )
        relations = r.json().get("result", [])

        # Priority order: RO first, then RB, then RN
        for priority_rel in ["RO", "RB", "RN"]:
            for rel in relations:
                name   = rel.get("relatedIdName", "")
                label  = rel.get("relationLabel", "")
                if (label == priority_rel and
                    name and
                    len(name.split()) <= 5 and
                    original_term.lower() not in name.lower() and
                    all(ord(c) < 128 for c in name) and
                    len(name) > 3):
                    return name, label

        return None, None
    except Exception:
        return None, None


# ============================================================
# LOAD EXISTING LOOKUP (preserve what we already have)
# ============================================================
existing = pd.read_csv("umls_substitutions.csv")
existing_lookup = dict(zip(
    existing["entity"].str.lower(),
    zip(existing["substitution"], existing["relation"])
))
print(f"Existing lookup entries: {len(existing_lookup)}")

# ============================================================
# EXPAND LOOKUP
# ============================================================
new_rows = []
skipped  = []
failed   = []

# Start with existing entries
for _, row in existing.iterrows():
    new_rows.append({
        "entity":       row["entity"],
        "cui":          row.get("cui", ""),
        "substitution": row["substitution"],
        "relation":     row["relation"],
    })

entities_to_process = [
    e for e in MEDICAL_ENTITIES_UNIQUE
    if e.lower() not in existing_lookup
]
print(f"\nNew entities to process: {len(entities_to_process)}")
print("Starting UMLS API queries (this will take ~15-20 minutes)...\n")

for i, entity in enumerate(entities_to_process):
    # Step 1: Get CUI
    cui, canonical = get_cui(entity)
    time.sleep(0.3)

    if not cui:
        skipped.append(entity)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(entities_to_process)}] Processed | "
                  f"Added: {len(new_rows) - len(existing_lookup)} | "
                  f"Skipped: {len(skipped)} | Failed: {len(failed)}")
        continue

    # Step 2: Get substitution
    substitution, relation = get_best_substitution(cui, entity)
    time.sleep(0.3)

    if not substitution:
        failed.append(entity)
    else:
        new_rows.append({
            "entity":       entity,
            "cui":          cui,
            "substitution": substitution,
            "relation":     relation,
        })
        print(f"  ✅ [{relation}] {entity} → {substitution}")

    if (i + 1) % 20 == 0:
        # Save checkpoint
        checkpoint_df = pd.DataFrame(new_rows)
        checkpoint_df.to_csv("umls_substitutions_expanded.csv", index=False)
        print(f"\n  [{i+1}/{len(entities_to_process)}] Checkpoint saved | "
              f"Total entries: {len(new_rows)} | "
              f"Skipped: {len(skipped)} | Failed: {len(failed)}\n")

# ============================================================
# SAVE FINAL EXPANDED LOOKUP
# ============================================================
final_df = pd.DataFrame(new_rows)
final_df.to_csv("umls_substitutions_expanded.csv", index=False)

print(f"\n{'='*60}")
print(f"  EXPANSION COMPLETE")
print(f"{'='*60}")
print(f"  Original entries:  {len(existing_lookup)}")
print(f"  New entries added: {len(new_rows) - len(existing_lookup)}")
print(f"  Total entries:     {len(new_rows)}")
print(f"  Skipped (no CUI):  {len(skipped)}")
print(f"  Failed (no sub):   {len(failed)}")
print(f"\n✅ Saved: umls_substitutions_expanded.csv")
print(f"\nNext step: rerun GNN detector with expanded lookup")
print(f"  python run_gnn_detector_v2.py")
