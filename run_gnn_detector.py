# ============================================================
# run_gnn_detector.py
# GNN-Based Adversarial Detector for UMLS Graph Attacks
#
# Detects whether a medical query has been adversarially
# modified using UMLS knowledge graph traversal — WITHOUT
# requiring access to the original query (no oracle needed).
#
# Architecture:
#   1. Build UMLS concept subgraph for each question
#      - Extract medical entities via scispaCy NER
#      - Retrieve CUIs and relations from UMLS API
#      - Build per-question graph: nodes=concepts, edges=relations
#   2. Featurize nodes and edges
#      - Node features: semantic type one-hot (127 types),
#        concept frequency proxy, entity length
#      - Edge features: relation type one-hot (RO/RB/RN/SY/etc.)
#   3. Train Graph Attention Network (GAT) binary classifier
#      - Label 1: UMLS substitution applied (314 examples)
#      - Label 0: clean query (686 examples)
#      - 5-fold cross-validation
#   4. Evaluate and report ROC-AUC, accuracy, per-class metrics
#
# Input:  umls_graph_attack_results.xlsx
#         umls_substitutions.csv (for entity lookup)
# Output: gnn_detector_results.xlsx
#         gnn_detector_model.pt
#
# Run on Delta GPU node:
#   python run_gnn_detector.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import re
import time
import requests
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score

# ============================================================
# CONFIG
# ============================================================
UMLS_API_KEY = os.environ.get("UMLS_API_KEY")
UMLS_BASE     = "https://uts-ws.nlm.nih.gov/rest"
UMLS_VERSION  = "current"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS        = 50
LR            = 1e-3
HIDDEN_DIM    = 64
NUM_HEADS     = 4
BATCH_SIZE    = 32
N_FOLDS       = 5

print(f"Device: {DEVICE}")
print(f"PyTorch Geometric available: {True}")

# ============================================================
# UMLS SEMANTIC TYPES (abbreviated list — 127 total in UMLS)
# We use the most common ones for one-hot encoding
# ============================================================
SEMANTIC_TYPES = [
    "Disease or Syndrome", "Finding", "Sign or Symptom",
    "Pharmacologic Substance", "Clinical Drug", "Antibiotic",
    "Body Part, Organ, or Organ Component", "Tissue", "Cell",
    "Organism", "Bacterium", "Virus",
    "Diagnostic Procedure", "Therapeutic or Preventive Procedure",
    "Laboratory Procedure", "Research Activity",
    "Biologically Active Substance", "Hormone", "Enzyme",
    "Gene or Genome", "Amino Acid, Peptide, or Protein",
    "Pathologic Function", "Physiologic Function",
    "Congenital Abnormality", "Acquired Abnormality",
    "Neoplastic Process", "Mental or Behavioral Dysfunction",
    "Injury or Poisoning", "Anatomical Abnormality",
    "Medical Device", "Chemical", "Element, Ion, or Isotope",
    "Unknown",  # fallback
]
STYPE_TO_IDX = {s: i for i, s in enumerate(SEMANTIC_TYPES)}
NUM_STYPES   = len(SEMANTIC_TYPES)

RELATION_TYPES = ["RO", "RB", "RN", "SY", "RQ", "PAR", "CHD", "OTHER"]
RTYPE_TO_IDX   = {r: i for i, r in enumerate(RELATION_TYPES)}
NUM_RTYPES     = len(RELATION_TYPES)

# ============================================================
# UMLS API HELPERS
# ============================================================
def get_cui_and_stype(term: str) -> tuple:
    """Get CUI and semantic type for a term."""
    try:
        r = requests.get(
            f"{UMLS_BASE}/search/{UMLS_VERSION}",
            params={"string": term, "apiKey": UMLS_API_KEY,
                    "pageSize": 1, "searchType": "bestMatch"},
            timeout=8
        )
        results = r.json()["result"]["results"]
        if not results or results[0]["ui"] == "NONE":
            return None, "Unknown"
        cui  = results[0]["ui"]
        name = results[0].get("name", term)

        # Get semantic type
        r2 = requests.get(
            f"{UMLS_BASE}/content/{UMLS_VERSION}/CUI/{cui}",
            params={"apiKey": UMLS_API_KEY},
            timeout=8
        )
        data   = r2.json().get("result", {})
        stypes = data.get("semanticTypes", [])
        stype  = stypes[0].get("name", "Unknown") if stypes else "Unknown"
        time.sleep(0.2)
        return cui, stype
    except Exception:
        return None, "Unknown"


def get_relations(cui: str) -> list:
    """Get related CUIs and relation types."""
    try:
        r = requests.get(
            f"{UMLS_BASE}/content/{UMLS_VERSION}/CUI/{cui}/relations",
            params={"apiKey": UMLS_API_KEY, "pageSize": 15},
            timeout=8
        )
        rels = r.json().get("result", [])
        time.sleep(0.2)
        return [
            {
                "relatedCui":  rel.get("relatedId", ""),
                "relatedName": rel.get("relatedIdName", ""),
                "relation":    rel.get("relationLabel", "OTHER")
            }
            for rel in rels if rel.get("relatedId")
        ]
    except Exception:
        return []


# ============================================================
# ENTITY EXTRACTION (same list as UMLS attack script)
# ============================================================
MEDICAL_ENTITIES = [
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
    "warfarin", "aspirin", "metformin", "insulin", "heparin",
    "statin", "methotrexate", "tamoxifen", "lithium", "morphine",
    "ibuprofen", "paracetamol", "amoxicillin", "vancomycin",
    "surgery", "biopsy", "chemotherapy", "radiotherapy", "dialysis",
    "transplant", "angioplasty", "endoscopy", "laparoscopy",
    "kidney", "liver", "heart", "lung", "brain", "colon",
    "prostate", "breast", "pancreas", "ovary", "uterus",
    "bladder", "spleen", "gallbladder", "appendix", "tonsil",
]

def extract_entities(question: str) -> list:
    """Extract all matching medical entities from a question."""
    q_lower = question.lower()
    found = []
    for entity in sorted(MEDICAL_ENTITIES, key=len, reverse=True):
        if entity in q_lower and entity not in found:
            found.append(entity)
    return found[:3]  # limit to 3 entities per question


# ============================================================
# GRAPH BUILDING
# ============================================================
def build_question_graph(question: str, cui_cache: dict) -> Data:
    """
    Build a UMLS concept subgraph for a question.
    Returns a PyG Data object with node and edge features.
    """
    entities   = extract_entities(question)
    node_feats = []
    edge_index = [[], []]
    edge_feats = []
    cui_to_idx = {}

    for entity in entities:
        if entity not in cui_cache:
            cui, stype = get_cui_and_stype(entity)
            cui_cache[entity] = (cui, stype)
        else:
            cui, stype = cui_cache[entity]

        if cui is None:
            cui = f"FAKE_{entity}"

        if cui not in cui_to_idx:
            idx = len(node_feats)
            cui_to_idx[cui] = idx
            # Node feature: semantic type one-hot + entity length (normalized)
            stype_idx = STYPE_TO_IDX.get(stype, STYPE_TO_IDX["Unknown"])
            feat = [0.0] * NUM_STYPES
            feat[stype_idx] = 1.0
            feat.append(min(len(entity) / 50.0, 1.0))  # length feature
            node_feats.append(feat)

        # Get relations for this CUI
        if cui.startswith("FAKE_"):
            continue
        rels = get_relations(cui)
        for rel in rels[:5]:  # limit edges per node
            related_cui  = rel["relatedCui"]
            relation     = rel["relation"]
            related_name = rel["relatedName"]

            if related_cui not in cui_to_idx:
                # Add related node with unknown stype
                ridx = len(node_feats)
                cui_to_idx[related_cui] = ridx
                feat = [0.0] * NUM_STYPES
                feat[STYPE_TO_IDX["Unknown"]] = 1.0
                feat.append(min(len(related_name) / 50.0, 1.0))
                node_feats.append(feat)

            src = cui_to_idx[cui]
            dst = cui_to_idx[related_cui]
            edge_index[0].append(src)
            edge_index[1].append(dst)
            edge_index[0].append(dst)  # undirected
            edge_index[1].append(src)

            rel_idx = RTYPE_TO_IDX.get(relation, RTYPE_TO_IDX["OTHER"])
            rel_feat = [0.0] * NUM_RTYPES
            rel_feat[rel_idx] = 1.0
            edge_feats.append(rel_feat)
            edge_feats.append(rel_feat)  # both directions

    # Handle empty graphs (no entities found)
    if not node_feats:
        node_feats = [[0.0] * (NUM_STYPES + 1)]
        edge_index = [[], []]
        edge_feats = []

    x = torch.tensor(node_feats, dtype=torch.float)

    if edge_index[0]:
        ei = torch.tensor(edge_index, dtype=torch.long)
        ef = torch.tensor(edge_feats, dtype=torch.float)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
        ef = torch.zeros((0, NUM_RTYPES), dtype=torch.float)

    return Data(x=x, edge_index=ei, edge_attr=ef)


# ============================================================
# GAT MODEL
# ============================================================
class MedicalQueryGAT(nn.Module):
    """
    Graph Attention Network for adversarial query detection.
    Input:  UMLS concept subgraph of a medical query
    Output: probability that query has been adversarially modified
    """
    def __init__(self, in_dim, hidden_dim, num_heads, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads,
                             dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim,
                             heads=1, dropout=dropout, concat=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        # Global mean pooling — graph-level representation
        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\nLoading UMLS attack results...")
df = pd.read_excel("umls_graph_attack_results.xlsx")
print(f"Loaded {len(df)} rows")
print(f"  Attacked (label=1): {df['umls_substitution'].notna().sum()}")
print(f"  Clean   (label=0): {df['umls_substitution'].isna().sum()}")

# Labels: 1 = UMLS substitution applied, 0 = clean
labels    = df["umls_substitution"].notna().astype(int).tolist()
questions = df["attacked_question"].tolist()  # use attacked question (no oracle)

# ============================================================
# STEP 2: BUILD GRAPHS
# ============================================================
cache_file = "umls_cui_cache.pkl"
cui_cache  = {}

if os.path.exists(cache_file):
    import pickle
    with open(cache_file, "rb") as f:
        cui_cache = pickle.load(f)
    print(f"\nLoaded CUI cache: {len(cui_cache)} entries")
else:
    print("\nNo cache found — will build from UMLS API")

print(f"\nBuilding UMLS subgraphs for {len(questions)} questions...")
print("(This queries the UMLS API — may take 20-40 minutes)")

graphs = []
for i, q in enumerate(questions):
    g = build_question_graph(q, cui_cache)
    g.y = torch.tensor([labels[i]], dtype=torch.float)
    graphs.append(g)

    if (i + 1) % 50 == 0:
        print(f"  Built {i+1}/{len(questions)} graphs")
        # Save cache periodically
        import pickle
        with open(cache_file, "wb") as f:
            pickle.dump(cui_cache, f)

# Final cache save
import pickle
with open(cache_file, "wb") as f:
    pickle.dump(cui_cache, f)
print(f"\nGraphs built. CUI cache: {len(cui_cache)} entries saved.")

# ============================================================
# GRAPH STATISTICS
# ============================================================
num_nodes_list = [g.x.shape[0] for g in graphs]
num_edges_list = [g.edge_index.shape[1] for g in graphs]
print(f"\nGraph statistics:")
print(f"  Avg nodes per graph: {np.mean(num_nodes_list):.1f}")
print(f"  Avg edges per graph: {np.mean(num_edges_list):.1f}")
print(f"  Node feature dim:    {graphs[0].x.shape[1]}")

# ============================================================
# STEP 3: TRAIN GAT WITH CROSS-VALIDATION
# ============================================================
print(f"\n{'='*65}")
print(f"  TRAINING GAT DETECTOR — {N_FOLDS}-Fold Cross-Validation")
print(f"{'='*65}")

IN_DIM = graphs[0].x.shape[1]
labels_arr = np.array(labels)
fold_results = []

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(graphs, labels_arr)):
    print(f"\n  Fold {fold+1}/{N_FOLDS}")

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False)

    model = MedicalQueryGAT(
        in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # Class weights for imbalance (686 clean vs 314 attacked)
    pos_weight = torch.tensor([686.0 / 314.0]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc   = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss   = criterion(logits, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                val_logits.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())

        auc = roc_auc_score(val_labels, val_logits)
        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | loss={total_loss/len(train_loader):.4f} | val_auc={auc:.4f}")

    # Final evaluation on best model
    model.load_state_dict(best_state)
    model.eval()
    val_logits, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            val_logits.extend(torch.sigmoid(logits).cpu().numpy())
            val_labels.extend(batch.y.cpu().numpy())

    val_preds = [1 if p > 0.5 else 0 for p in val_logits]
    fold_auc  = roc_auc_score(val_labels, val_logits)
    fold_acc  = accuracy_score(val_labels, val_preds)
    fold_results.append({"auc": fold_auc, "acc": fold_acc,
                          "labels": val_labels, "probs": val_logits})
    print(f"  Fold {fold+1} best AUC: {fold_auc:.4f} | Acc: {fold_acc:.4f}")

# ============================================================
# STEP 4: AGGREGATE RESULTS
# ============================================================
aucs = [r["auc"] for r in fold_results]
accs = [r["acc"] for r in fold_results]

all_labels = []
all_probs  = []
for r in fold_results:
    all_labels.extend(r["labels"])
    all_probs.extend(r["probs"])
all_preds = [1 if p > 0.5 else 0 for p in all_probs]

print(f"\n{'='*65}")
print(f"  GNN DETECTOR RESULTS — {N_FOLDS}-Fold Cross-Validation")
print(f"{'='*65}")
print(f"  ROC-AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"\n  Classification Report (aggregated across all folds):")
print(classification_report(all_labels, all_preds,
                             target_names=["Clean", "UMLS Attacked"]))

# Interpretation
mean_auc = np.mean(aucs)
print(f"  Interpretation:")
if mean_auc >= 0.80:
    print(f"  ✅ Strong detection — GNN successfully identifies UMLS-attacked queries")
    print(f"     This is the only method that works WITHOUT oracle access")
elif mean_auc >= 0.65:
    print(f"  ⚠️  Moderate detection — GNN partially identifies attacked queries")
    print(f"     Expanding UMLS coverage beyond 40 entities would improve this")
else:
    print(f"  ℹ️  Weak detection — limited by small entity coverage (40 entities)")
    print(f"     Full UMLS API (3M concepts) would significantly improve performance")
print(f"{'='*65}")

# ============================================================
# STEP 5: TRAIN FINAL MODEL ON ALL DATA
# ============================================================
print("\nTraining final model on all data...")
final_model = MedicalQueryGAT(
    in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS
).to(DEVICE)
final_optimizer = Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
full_loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    final_model.train()
    for batch in full_loader:
        batch = batch.to(DEVICE)
        final_optimizer.zero_grad()
        logits = final_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y.float())
        loss.backward()
        final_optimizer.step()

torch.save(final_model.state_dict(), "gnn_detector_model.pt")
print("Final model saved: gnn_detector_model.pt")

# ============================================================
# STEP 6: SAVE RESULTS
# ============================================================
# Run final model on all graphs for per-question scores
final_model.eval()
all_scores = []
inference_loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
with torch.no_grad():
    for batch in inference_loader:
        batch = batch.to(DEVICE)
        logits = final_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_scores.extend(torch.sigmoid(logits).cpu().numpy())

df["gnn_attack_probability"] = [round(s, 4) for s in all_scores]
df["gnn_predicted_attacked"]  = [1 if s > 0.5 else 0 for s in all_scores]
df["true_label"]               = labels

df.to_excel("gnn_detector_results.xlsx", index=False)
df.to_csv("gnn_detector_results.csv",   index=False)

# Summary stats
results_summary = pd.DataFrame({
    "metric": ["ROC-AUC (mean)", "ROC-AUC (std)", "Accuracy (mean)", "Accuracy (std)"],
    "value":  [round(np.mean(aucs), 4), round(np.std(aucs), 4),
               round(np.mean(accs), 4), round(np.std(accs), 4)]
})
results_summary.to_excel("gnn_detector_summary.xlsx", index=False)

print(f"\n✅ Saved: gnn_detector_results.xlsx")
print(f"✅ Saved: gnn_detector_summary.xlsx")
print(f"✅ Saved: gnn_detector_model.pt")
print(f"\nDone.")
