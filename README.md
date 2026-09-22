# Prompt Injection Attacks on Medical LLMs

This repository contains the anonymized implementation and experimental artifacts accompanying a research submission investigating the vulnerability of medical large language models to prompt injection attacks. It also presents a graph-based detection approach using Unified Medical Language System (UMLS) knowledge graphs.

## Overview

Medical LLMs are increasingly deployed in clinical decision-support applications, yet their robustness against adversarial manipulation remains understudied.

This project evaluates eight prompt injection attack types across four large language models and two biomedical question-answering datasets. It also proposes a Graph Attention Network (GAT)-based detector that uses UMLS biomedical knowledge-graph structure to identify potentially attacked queries.

## Models Evaluated

| Model                    | Type               | Parameters |
| ------------------------ | ------------------ | ---------: |
| MedGemma 4B              | Medical fine-tuned |         4B |
| BioMistral 7B            | Medical fine-tuned |         7B |
| Mistral 7B Instruct v0.2 | General purpose    |         7B |
| MedAlpaca 7B             | Medical fine-tuned |         7B |

## Datasets

* **PubMedQA:** 1,000 open-ended biomedical research question-answer pairs.
* **MedQA USMLE:** 1,273 multiple-choice clinical exam questions with A/B/C/D answer options.

## Attack Types

| Attack             | Description                                                            |
| ------------------ | ---------------------------------------------------------------------- |
| Typo/Abbreviation  | Introduces medical abbreviations and typographical errors              |
| Rephrasing         | Produces semantically equivalent rewrites                              |
| Meaning Change     | Applies subtle negations and factual changes                           |
| Brand Manipulation | Adds references to fictional pharmaceutical brands                     |
| Task Switch        | Redirects the model toward an unrelated task                           |
| Medical Hijack     | Replaces benign clinical concepts with dangerous alternatives          |
| Stealth            | Combines multiple subtle instruction-injection strategies              |
| UMLS Graph         | Substitutes entities using relationships from the UMLS knowledge graph |

## Key Results

### Attack Effectiveness

Delta BERTScore results on PubMedQA are shown below. Lower scores indicate greater attack-induced response drift.

| Attack             | MedGemma | BioMistral | Mistral | MedAlpaca |
| ------------------ | -------: | ---------: | ------: | --------: |
| Brand Manipulation |   0.8795 |     0.8801 |  0.8617 |    0.9310 |
| Task Switch        |   0.8067 |     0.7897 |  0.8337 |    0.8383 |
| UMLS Graph         |   0.9049 |     0.9631 |  0.8942 |    0.9746 |
| Average            |   0.8929 |     0.9305 |  0.8843 |    0.9533 |

### Brand Attack Success Rate

| Model      | PubMedQA | MedQA USMLE |
| ---------- | -------: | ----------: |
| MedGemma   |    76.4% |        0.3% |
| BioMistral |    70.1% |        1.3% |
| Mistral    |    75.5% |        5.7% |
| MedAlpaca  |    24.9% |        0.0% |

### Detectability

* Seven of the eight evaluated attacks were rarely detected by simple output-level monitoring.
* The UMLS graph attack had a detection rate of 0.6%.
* Medical Hijack and Stealth had detection rates below 0.1%.

### GNN Detector Performance

| Version | Entity Coverage |       ROC-AUC | Accuracy | Clean F1 | Attacked F1 |
| ------- | --------------: | ------------: | -------: | -------: | ----------: |
| v1      |     40 entities | 0.616 ± 0.027 |    0.620 |     0.71 |        0.44 |
| v2      |    208 entities | 0.631 ± 0.038 |    0.688 |     0.79 |        0.43 |

## GNN Detector Architecture

```text
Medical Query
     ↓
Entity Extraction using scispaCy NER
     ↓
UMLS Subgraph Construction
G_q = (V_q, E_q)
     ↓
Featurization
  • Node features:
    Semantic-type one-hot encoding plus entity length
  • Edge features:
    Relation-type one-hot encoding
     ↓
Three-Layer Graph Attention Network
  • Four attention heads
  • Hidden dimension: 128
  • Batch normalization
  • Dropout: 0.3
     ↓
Global Mean Pooling
     ↓
MLP Classifier
     ↓
0: Clean Query
1: Attacked Query
```

Training configuration:

* Adam optimizer
* Learning rate: `5e-4`
* Weighted binary cross-entropy
* Positive-class weight: `2.18`
* Five-fold cross-validation
* 80 epochs per fold

## Project Structure

```text
├── run_all_attacks.py              # Main attack runner
├── run_brand_attack.py             # Brand-manipulation attack
├── run_context_switch_attack.py    # Task-switch attack
├── run_rephrase_meaning_attack.py  # Rephrasing and meaning-change attacks
├── run_stealth_attack.py           # Stealth injection attack
├── run_umls_graph_attack.py        # UMLS knowledge-graph attack
├── run_attack_mistakes.py          # Typo and abbreviation attack
├── run_full_dataset.py             # Full-dataset evaluation
├── run_gnn_detector.py             # GNN detector v1
├── run_gnn_detector_v2.py          # GNN detector v2
├── run_bert_embedding_analysis.py  # BERT embedding analysis
├── run_output_intent_monitor.py    # Output-intent monitoring
├── run_comparison_analysis.py      # Cross-model comparison
├── expand_umls_lookup.py           # UMLS entity-lookup expansion
├── prepare_medqa.py                # MedQA dataset preparation
├── evaluate.py                     # Evaluation metrics
├── visualize.py                    # Result visualization
├── config.py                       # Local API configuration; not committed
├── .gitignore
└── job_*.sb                        # SLURM job scripts
```

## Setup

### Requirements

```bash
pip install transformers torch torch_geometric \
    sentence-transformers rouge-score nltk \
    pandas openpyxl requests scispacy
```

## Configuration

Create a `config.py` file in the project root. This file should remain excluded through `.gitignore`.

```python
import os

UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
```

Environment variables may alternatively be configured directly:

```bash
export HF_TOKEN="your-token"
export UMLS_API_KEY="your-key"
```

A UMLS API key can be obtained through the official UMLS Terminology Services portal.

Do not commit API keys, access tokens, credentials, personal usernames, email addresses, or institution-specific configuration.

## Running Experiments

### Run all attacks for one model and dataset

```bash
python run_all_attacks.py \
    --model medgemma \
    --dataset pubmedqa
```

### Run the UMLS graph attack

```bash
python run_umls_graph_attack.py
```

### Train the GNN detector

```bash
python run_gnn_detector_v2.py
```

### Run using a SLURM environment

```bash
sbatch job_medgemma_pubmedqa_final.sb
sbatch job_gnn_v2.sb
```

The provided SLURM scripts may require modification based on the compute environment, allocation, partition, storage paths, and available GPU resources.

## Key Findings

* Brand manipulation was highly effective on open-ended questions for three of the four evaluated models.
* The constrained multiple-choice format reduced the success of brand-manipulation attacks.
* Domain-specific medical fine-tuning did not eliminate prompt-injection vulnerability.
* MedAlpaca demonstrated the highest overall robustness among the evaluated models.
* Most evaluated attacks evaded simple output-level detection.
* The UMLS graph attack caused meaningful response drift while producing only a small input-space perturbation.
* Increasing medical-entity coverage improved the GNN detector’s overall discrimination performance.

## Infrastructure

Experiments were conducted using NVIDIA A40 GPUs with 40 GB of GPU memory in a SLURM-managed high-performance computing environment.

Institution-specific account names, allocation identifiers, usernames, email addresses, and local storage paths have been omitted from this anonymized repository.

## Reproducibility

The repository contains the scripts used to:

* Generate attacked medical queries
* Run inference across the evaluated models
* Calculate response-similarity and attack-success metrics
* Perform output-level attack detection
* Construct UMLS query subgraphs
* Train and evaluate the GNN-based detector
* Generate comparison tables and visualizations

Some experiments require access to gated model weights, biomedical datasets, UMLS services, and suitable GPU resources.
