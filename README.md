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

## 📊 Performance Highlights

The descriptor-based **ExtraTrees** classifier achieved state-of-the-art performance on the independent M3 validation cohort:

| Metric | Value |
|--------|-------|
| **Accuracy** | 95.01% |
| **Balanced Accuracy** | 91.52% |
| **Precision** | 97.14% |
| **Recall** | 83.95% |
| **Specificity** | 99.09% |
| **F1-Score** | 0.9007 |
| **MCC** | 0.8717 |
| **ROC-AUC** | 0.9495 |
| **PR-AUC** | 0.9256 |

## 📂 Repository Structure

```text
AptaEnsemble/
│
├── main_pipeline.py                    # Main descriptor-based ML/DL/Hybrid pipeline
├── pretrained_ablation.py              # DNABERT6/ProtBERT embedding ablation
├── candidate_ranking.py                # Protein-integrated candidate prioritization
│
├── data/
│   ├── train.csv                       # Development partition (2,402 samples)
│   ├── M1.csv                          # Seen diagnostic cohort (2,402 samples)
│   ├── M2.csv                          # Semi-unseen diagnostic cohort (600 samples)
│   └── M3.csv                          # External unseen validation cohort (601 samples)
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
