# HyperSynergyX

**Synergistic Drug Combination Prediction via Hypergraph Modeling and Knowledge Graph-Enhanced Retrieval-Augmented Generation**

HyperSynergyX is an integrated framework designed to predict and explain higher-order (three-drug) synergistic combinations. It combines a hypergraph-based predictive model (DBRWH) with a KG-RAG system to provide both accurate synergy scores and mechanistically grounded explanations.

## Overview

Drug combination therapy is a cornerstone of modern oncology, but identifying effective three-drug regimens is computationally challenging due to the combinatorial explosion. HyperSynergyX addresses this by:

1.  **Predicting Synergy:** Using a **Dual-Biased Random Walk on Hypergraphs (DBRWH)** to model higher-order interactions directly on a three-drug hypergraph.
2.  **Uncovering Latent Patterns:** Applying tensor decomposition to synergy scores to identify reproducible, biologically meaningful "modes" of drug combinations.
3.  **Explaining Mechanisms:** Leveraging a **Knowledge Graph-Enhanced Retrieval-Augmented Generation (KG-RAG)** module to generate plausible, literature-backed mechanistic hypotheses for predicted combinations.

## Repository Structure

This repository contains the core implementation of the HyperSynergyX framework:

*   **`dbrwh.py`**: Implementation of the Dual-Biased Random Walk algorithm for synergy scoring.
*   **`run_cv_eval.py`**: Script to run 5-fold cross-validation and evaluate predictive performance (AUROC, AUPRC).
*   **`run_ablation.py`**: Ablation studies to validate the contribution of the dual-bias mechanism.
*   **`models_registry.py`**: Registry for managing and calling different models (DBRWH and baselines).
*   **`rag_explain.py`**: The KG-RAG module for generating mechanistic explanations using LLMs and Neo4j.
*   **`rag_neo4j_loader.py`**: Utilities for connecting to the drug knowledge graph stored in Neo4j.
*   **`eval_metrics.py`**: Core evaluation metrics (AUROC, AUPRC, F1-score, etc.).
*   **`compute_nlp_metrics_v2.py`**: Evaluation of generated explanations (ROUGE scores, Factuality Score).
*   **`run_pathway_enrichment.py`**: Analysis script for biological pathway enrichment of latent tensor modes.

## Requirements

The code is written in Python. Main dependencies include:

*   Python 3.8+
*   `numpy`
*   `pandas`
*   `scikit-learn`
*   `networkx`
*   `neo4j` (for the KG-RAG module)
*   `torch` (for baseline comparisons)

You can install the necessary packages using pip:

```bash
pip install numpy pandas scikit-learn networkx neo4j torch
```

*Note: To run the KG-RAG module, you will need access to a Neo4j database instance containing the drug knowledge graph.*

## Usage

### 1. Synergy Prediction (Cross-Validation)

To evaluate the model's performance on a dataset using 5-fold cross-validation:

```bash
python run_cv_eval.py --dataset breast --folds 5
```

### 2. Ablation Study

To verify the effectiveness of the dual-bias mechanism (comparing full model vs. restart-only vs. attraction-only):

```bash
python run_ablation.py
```

### 3. Generating Explanations (KG-RAG)

To generate mechanistic explanations for a specific drug triplet (requires Neo4j connection):

```bash
python rag_explain.py --drugs "DrugA,DrugB,DrugC"
```

## License

This project is licensed under the MIT License.
