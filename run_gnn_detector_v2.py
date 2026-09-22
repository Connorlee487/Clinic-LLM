# ============================================================
# run_gnn_detector_v2.py
# GNN Detector with expanded UMLS entity coverage
#
# Same architecture as v1 but uses umls_substitutions_expanded.csv
# which covers ~200+ entities vs original 40.
# More entities = denser subgraphs = better GAT classification.
#
# Run on Delta GPU node (submit via SLURM):
#   python run_gnn_detector_v2.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import re
import time
import pickle
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
EPOCHS        = 80
LR            = 5e-4
HIDDEN_DIM    = 128
NUM_HEADS     = 4
BATCH_SIZE    = 32
N_FOLDS       = 5
DROPOUT       = 0.3

print(f"Device: {DEVICE}")

# ============================================================
# SEMANTIC TYPES & RELATION TYPES
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
    "Unknown",
]
STYPE_TO_IDX = {s: i for i, s in enumerate(SEMANTIC_TYPES)}
NUM_STYPES   = len(SEMANTIC_TYPES)

RELATION_TYPES = ["RO", "RB", "RN", "SY", "RQ", "PAR", "CHD", "OTHER"]
RTYPE_TO_IDX   = {r: i for i, r in enumerate(RELATION_TYPES)}
NUM_RTYPES     = len(RELATION_TYPES)

# Node feature dimension: semantic type one-hot + entity length + is_substituted flag
NODE_FEAT_DIM = NUM_STYPES + 2

# ============================================================
# LOAD EXPANDED UMLS LOOKUP
# ============================================================
lookup_file = "umls_substitutions_expanded.csv"
if not os.path.exists(lookup_file):
    print(f"WARNING: {lookup_file} not found, falling back to original")
    lookup_file = "umls_substitutions.csv"

umls_df = pd.read_csv(lookup_file)
UMLS_LOOKUP   = dict(zip(umls_df["entity"].str.lower(), umls_df["substitution"]))
UMLS_RELATION = dict(zip(umls_df["entity"].str.lower(), umls_df["relation"]))
print(f"Loaded {len(UMLS_LOOKUP)} UMLS substitutions from {lookup_file}")

# ============================================================
# ENTITY EXTRACTION
# ============================================================
def extract_entities(question: str) -> list:
    """Extract all matching medical entities — longest match first."""
    q_lower = question.lower()
    found   = []
    for entity in sorted(UMLS_LOOKUP.keys(), key=len, reverse=True):
        if entity in q_lower and entity not in found:
            found.append(entity)
            if len(found) >= 5:  # max 5 entities per question
                break
    return found

# ============================================================
# UMLS API — for getting semantic types of related concepts
# ============================================================
def get_stype(cui: str, cache: dict) -> str:
    if cui in cache:
        return cache[cui]
    try:
        r = requests.get(
            f"{UMLS_BASE}/content/{UMLS_VERSION}/CUI/{cui}",
            params={"apiKey": UMLS_API_KEY},
            timeout=6
        )
        data   = r.json().get("result", {})
        stypes = data.get("semanticTypes", [])
        stype  = stypes[0].get("name", "Unknown") if stypes else "Unknown"
        cache[cui] = stype
        time.sleep(0.15)
        return stype
    except Exception:
        return "Unknown"


def get_relations_for_entity(entity: str, cui_cache: dict, rel_cache: dict) -> list:
    """Get UMLS relations for an entity using API."""
    if entity in rel_cache:
        return rel_cache[entity]
    try:
        # Get CUI first
        r = requests.get(
            f"{UMLS_BASE}/search/{UMLS_VERSION}",
            params={"string": entity, "apiKey": UMLS_API_KEY,
                    "pageSize": 1, "searchType": "bestMatch"},
            timeout=8
        )
        results = r.json()["result"]["results"]
        if not results or results[0]["ui"] == "NONE":
            rel_cache[entity] = []
            return []
        cui = results[0]["ui"]
        time.sleep(0.15)

        # Get relations
        r2 = requests.get(
            f"{UMLS_BASE}/content/{UMLS_VERSION}/CUI/{cui}/relations",
            params={"apiKey": UMLS_API_KEY, "pageSize": 10},
            timeout=8
        )
        rels = r2.json().get("result", [])
        time.sleep(0.15)

        result = [
            {"relatedCui": rel.get("relatedId", ""),
             "relatedName": rel.get("relatedIdName", ""),
             "relation": rel.get("relationLabel", "OTHER")}
            for rel in rels if rel.get("relatedId")
        ]
        rel_cache[entity] = result
        return result
    except Exception:
        rel_cache[entity] = []
        return []

# ============================================================
# GRAPH BUILDING — enhanced with substitution flag
# ============================================================
def build_graph(question: str, is_attacked: bool,
                cui_cache: dict, rel_cache: dict) -> Data:
    """
    Build UMLS concept subgraph for a question.
    Node features include a flag indicating if the entity
    was a known UMLS substitution target.
    """
    entities   = extract_entities(question)
    node_feats = []
    edge_index = [[], []]
    edge_feats = []
    entity_to_idx = {}

    for entity in entities:
        if entity not in entity_to_idx:
            idx = len(node_feats)
            entity_to_idx[entity] = idx

            # Semantic type feature
            stype_idx = STYPE_TO_IDX.get("Unknown")
            feat = [0.0] * NUM_STYPES
            feat[stype_idx] = 1.0
            feat.append(min(len(entity) / 50.0, 1.0))  # length
            # Is this entity in our substitution lookup?
            feat.append(1.0 if entity.lower() in UMLS_LOOKUP else 0.0)
            node_feats.append(feat)

        # Get relations for this entity
        rels = get_relations_for_entity(entity, cui_cache, rel_cache)
        for rel in rels[:4]:
            related_name = rel["relatedName"].lower()
            relation     = rel["relation"]

            if not related_name or len(related_name) < 2:
                continue

            if related_name not in entity_to_idx:
                ridx = len(node_feats)
                entity_to_idx[related_name] = ridx
                feat = [0.0] * NUM_STYPES
                feat[STYPE_TO_IDX.get("Unknown")] = 1.0
                feat.append(min(len(related_name) / 50.0, 1.0))
                feat.append(0.0)  # related concept, not a lookup entity
                node_feats.append(feat)

            src = entity_to_idx[entity]
            dst = entity_to_idx[related_name]
            rel_idx = RTYPE_TO_IDX.get(relation, RTYPE_TO_IDX["OTHER"])
            rel_feat = [0.0] * NUM_RTYPES
            rel_feat[rel_idx] = 1.0

            edge_index[0].extend([src, dst])
            edge_index[1].extend([dst, src])
            edge_feats.extend([rel_feat, rel_feat])

    if not node_feats:
        node_feats = [[0.0] * NODE_FEAT_DIM]

    x = torch.tensor(node_feats, dtype=torch.float)

    if edge_index[0]:
        ei = torch.tensor(edge_index, dtype=torch.long)
        ef = torch.tensor(edge_feats,  dtype=torch.float)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
        ef = torch.zeros((0, NUM_RTYPES), dtype=torch.float)

    return Data(x=x, edge_index=ei, edge_attr=ef)

# ============================================================
# GAT MODEL — deeper than v1
# ============================================================
class MedicalQueryGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads,
                             dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim,
                             heads=num_heads, dropout=dropout, concat=True)
        self.conv3 = GATConv(hidden_dim * num_heads, hidden_dim,
                             heads=1, dropout=dropout, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\nLoading data...")
df     = pd.read_excel("umls_graph_attack_results.xlsx")
labels = df["umls_substitution"].notna().astype(int).tolist()
questions = df["attacked_question"].tolist()
print(f"Loaded {len(df)} questions | Attacked: {sum(labels)} | Clean: {len(labels)-sum(labels)}")

# ============================================================
# STEP 2: BUILD GRAPHS WITH CACHING
# ============================================================
cache_file_cui = "umls_cui_cache_v2.pkl"
cache_file_rel = "umls_rel_cache_v2.pkl"

cui_cache = pickle.load(open(cache_file_cui, "rb")) if os.path.exists(cache_file_cui) else {}
rel_cache = pickle.load(open(cache_file_rel, "rb")) if os.path.exists(cache_file_rel) else {}
print(f"CUI cache: {len(cui_cache)} | Rel cache: {len(rel_cache)}")

print(f"\nBuilding subgraphs for {len(questions)} questions...")
graphs = []
for i, (q, label) in enumerate(zip(questions, labels)):
    g      = build_graph(q, bool(label), cui_cache, rel_cache)
    g.y    = torch.tensor([label], dtype=torch.float)
    graphs.append(g)

    if (i + 1) % 100 == 0:
        print(f"  Built {i+1}/{len(questions)} | "
              f"Avg nodes: {np.mean([g.x.shape[0] for g in graphs[-100:]]):.1f}")
        pickle.dump(cui_cache, open(cache_file_cui, "wb"))
        pickle.dump(rel_cache, open(cache_file_rel, "wb"))

pickle.dump(cui_cache, open(cache_file_cui, "wb"))
pickle.dump(rel_cache, open(cache_file_rel, "wb"))

# Graph statistics
num_nodes = [g.x.shape[0] for g in graphs]
num_edges = [g.edge_index.shape[1] for g in graphs]
print(f"\nGraph statistics:")
print(f"  Avg nodes: {np.mean(num_nodes):.1f} (v1 had ~2-3)")
print(f"  Avg edges: {np.mean(num_edges):.1f}")
print(f"  Questions with >3 nodes: {sum(1 for n in num_nodes if n > 3)}")

# ============================================================
# STEP 3: TRAIN WITH CROSS-VALIDATION
# ============================================================
print(f"\n{'='*65}")
print(f"  TRAINING GAT v2 — {N_FOLDS}-Fold Cross-Validation")
print(f"{'='*65}")

IN_DIM     = NODE_FEAT_DIM
labels_arr = np.array(labels)
fold_results = []
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(graphs, labels_arr)):
    print(f"\n  Fold {fold+1}/{N_FOLDS}")
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False)

    model     = MedicalQueryGAT(IN_DIM, HIDDEN_DIM, NUM_HEADS, DROPOUT).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    pos_weight = torch.tensor([(len(labels_arr) - sum(labels_arr)) / sum(labels_arr)]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_state = 0.0, None

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss   = criterion(logits, batch.y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        val_logits, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                val_logits.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels_list.extend(batch.y.cpu().numpy())

        try:
            auc = roc_auc_score(val_labels_list, val_logits)
            if auc > best_auc:
                best_auc   = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        except Exception:
            pass

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d} | val_auc={auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    val_logits, val_labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            val_logits.extend(torch.sigmoid(logits).cpu().numpy())
            val_labels_list.extend(batch.y.cpu().numpy())

    val_preds = [1 if p > 0.5 else 0 for p in val_logits]
    fold_auc  = roc_auc_score(val_labels_list, val_logits)
    fold_acc  = accuracy_score(val_labels_list, val_preds)
    fold_results.append({"auc": fold_auc, "acc": fold_acc,
                         "labels": val_labels_list, "probs": val_logits})
    print(f"  Fold {fold+1} best AUC: {fold_auc:.4f} | Acc: {fold_acc:.4f}")

# ============================================================
# STEP 4: FINAL RESULTS
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
print(f"  GNN DETECTOR v2 RESULTS")
print(f"{'='*65}")
print(f"  ROC-AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  (v1 was 0.6160)")
print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"\n  Classification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=["Clean", "UMLS Attacked"]))

improvement = np.mean(aucs) - 0.6160
print(f"  Improvement over v1: {improvement:+.4f}")
print(f"{'='*65}")

# ============================================================
# STEP 5: SAVE FINAL MODEL
# ============================================================
final_model = MedicalQueryGAT(IN_DIM, HIDDEN_DIM, NUM_HEADS, DROPOUT).to(DEVICE)
full_loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
final_optimizer = Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
pos_weight = torch.tensor([(len(labels_arr) - sum(labels_arr)) / sum(labels_arr)]).to(DEVICE)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

for epoch in range(EPOCHS):
    final_model.train()
    for batch in full_loader:
        batch = batch.to(DEVICE)
        final_optimizer.zero_grad()
        logits = final_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss   = criterion(logits, batch.y.float())
        loss.backward()
        final_optimizer.step()

torch.save(final_model.state_dict(), "gnn_detector_v2_model.pt")

# Per-question scores
final_model.eval()
all_scores = []
with torch.no_grad():
    for batch in DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False):
        batch = batch.to(DEVICE)
        logits = final_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_scores.extend(torch.sigmoid(logits).cpu().numpy())

df["gnn_v2_probability"] = [round(s, 4) for s in all_scores]
df["gnn_v2_predicted"]   = [1 if s > 0.5 else 0 for s in all_scores]
df["true_label"]          = labels
df.to_excel("gnn_detector_v2_results.xlsx", index=False)

print(f"\n✅ Saved: gnn_detector_v2_results.xlsx")
print(f"✅ Saved: gnn_detector_v2_model.pt")
print(f"\nDone.")
