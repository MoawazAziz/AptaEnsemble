# AptaEnsemble

**Computational Prediction of Aptamer–Protein Interactions by Integrated k-mer and Physicochemical Descriptors with Multi-Architecture Ensemble Learning**

## 📖 Overview

**AptaEnsemble** is a rigorous, leakage-controlled computational framework for predicting aptamer–protein interactions (APIs). It integrates engineered sequence descriptors and pretrained transformer embeddings across a comprehensive panel of Machine Learning (ML), Deep Learning (DL), Ensemble, and ML-DL Hybrid architectures.

This repository contains the complete, reproducible source code, data processing pipelines, and evaluation scripts for aptamer-protein interaction prediction and candidate prioritization.

### 🌟 Key Features

- **Strict Leakage-Controlled Validation**: Implements a rigorous M1/M2/M3 cohort hierarchy with out-of-fold (OOF) threshold optimization to prevent data leakage.
- **Comprehensive Model Panel**: Evaluates ExtraTrees, RandomForest, XGBoost, LightGBM, MLPs, CNN1D, Stacking Ensembles, and 15 ML-DL Hybrid configurations.
- **Pretrained Transformer Ablation**: Benchmarks engineered descriptors against frozen DNABERT6 (aptamer) and ProtBERT (protein) concatenated embeddings.
- **Imbalance-Robust Metrics**: Optimizes and reports Matthews Correlation Coefficient (MCC), PR-AUC, and Balanced Accuracy alongside standard metrics.
- **Translational Application**: Includes scripts for protein-integrated candidate prioritization for therapeutic targets (CTGF, DKK1, BCMA).

## 🔒 Entity-Disjoint Validation Framework

Random splits — including the M1/M2/M3 design above, which is itself an unconstrained stratified split — can allow the same aptamer or protein entity to appear in both training and test, inflating apparent performance through entity memorisation rather than genuine sequence-interaction learning. To quantify this directly, the repository includes a five-design validation framework (`splits/`):

| # | Design | Entity control |
|---|--------|-----------------|
| 1 | Pair-level (random) | None (baseline; statistically equivalent to the M1/M2/pair-level holdout cohort design above) |
| 2 | Protein-disjoint | Test proteins absent from training |
| 3 | Aptamer-disjoint | Test aptamers absent from training |
| 4 | Cluster-disjoint | Similarity clusters (cosine ≥ 0.90) disjoint between train and test |
| 5 | Doubly-disjoint | Neither aptamer nor protein present in training |

```bash
python build_entity_disjoint_splits.py --data clean.csv --out splits/
python doubly_disjoint_analysis.py --data clean.csv --out splits/doubly_disjoint_detail
python kmer_label_correction.py --fix --data clean.csv train.csv M2.csv M3.csv
python run_all_validation_designs.py --data clean.csv --out splits/all_designs_results
```

## 🧬 k=4 Feature Label Correction

The k=4 column labels in `clean.csv`, `train.csv`, `M2.csv`, and `M3.csv` have positions 3–4 transposed relative to the sequence whose frequency they store (e.g. column `aptamer_frequency_GGGC` holds the frequency of `GGCG`). 192 of 256 possible 4-mer labels are affected; feature values are correct, only column names require renaming. Corrected copies are provided at the repository root, named by data role: `clean_kmer_corrected.csv`, `train_kmer_corrected.csv`, `val_kmer_corrected.csv` (from `M2.csv`), and `test_kmer_corrected.csv` (from `M3.csv`). `kmer4_correction_map.csv` documents every stored-to-correct label pair. Details: [`splits/README.md`](splits/README.md).

## 📂 Repository Structure

```text
AptaEnsemble/
│
├── main_pipeline.py                    # Main descriptor-based ML/DL/Hybrid pipeline
├── pretrained_ablation.py              # DNABERT6/ProtBERT embedding ablation
├── candidate_ranking.py                # Protein-integrated candidate prioritization
├── feature_extraction.py               # k-mer + PseAAC descriptor generation
│
├── build_entity_disjoint_splits.py     # Builds all 5 entity-disjoint validation designs
├── doubly_disjoint_analysis.py         # Doubly-disjoint: build, verify, train, evaluate
├── kmer_label_correction.py            # Detects and corrects k=4 column-label transposition
├── run_all_validation_designs.py       # Comparative run across all 5 designs with controls
│
├── clean.csv / clean_kmer_corrected.csv       # Full feature table (3,003 pairs, 641 cols)
├── train.csv / train_kmer_corrected.csv       # Development partition (2,402 samples) — identical to M1.csv
├── M1.csv                                     # Seen diagnostic cohort (= train.csv)
├── M2.csv / val_kmer_corrected.csv            # Semi-unseen diagnostic cohort (600 samples)
├── M3.csv / test_kmer_corrected.csv           # External unseen validation cohort (601 samples)
│
├── splits/                             # Entity-disjoint split datasets
│   ├── README.md                       # Full documentation, methodology, and results
│   ├── split_summary.csv
│   ├── kmer4_correction_map.csv
│   ├── <design>_train.csv / <design>_test.csv   (5 designs)
│   ├── doubly_disjoint_detail/         # Audit trail and results for the doubly-disjoint design
│   └── all_designs_results/            # Comparative results table across all 5 designs
│
├── sequence_libraries/
│   ├── CT-20.txt                       # CTGF aptamer library
│   ├── DK-30.txt                       # DKK1 aptamer library
│   └── BC-6.txt                        # BCMA aptamer library
│
├── output/                             # Generated outputs (tables, figures, models)
├── .gitignore
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
└── README.md                           # This file
