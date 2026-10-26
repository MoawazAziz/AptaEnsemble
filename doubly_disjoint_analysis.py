#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Doubly-Disjoint Validation Analysis for AptaEnsemble.

Builds the doubly-disjoint split (neither aptamer nor protein entity shared
between training and test), verifies it is free of entity leakage, trains an
ExtraTrees classifier with an MCC-optimal threshold selected by cross-
validation, and evaluates once on the held-out test partition with bootstrap
confidence intervals.

Usage
-----
    python doubly_disjoint_analysis.py --data clean.csv --out output/doubly_disjoint
"""

import argparse
import json
import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")

LABEL_COL = "Class"
KMER_PREFIX = "aptamer_frequency_"
PROTEIN_PREFIX = "protein_"
SEED = 42
N_ESTIMATORS = 500
CV_FOLDS = 5
N_BOOTSTRAP = 400
TEST_FRAC = 0.20


def assign_entity_ids(df: pd.DataFrame) -> pd.DataFrame:
    kmer_cols = [c for c in df.columns if c.startswith(KMER_PREFIX)]
    protein_cols = [c for c in df.columns if c.startswith(PROTEIN_PREFIX)]

    def _vec_id(mat: np.ndarray) -> np.ndarray:
        rows = [tuple(np.round(r, 6)) for r in mat]
        mapping: Dict[tuple, int] = {}
        ids = []
        for r in rows:
            if r not in mapping:
                mapping[r] = len(mapping)
            ids.append(mapping[r])
        return np.array(ids, dtype=np.int32)

    df = df.copy()
    df["aptamer_id"] = _vec_id(df[kmer_cols].values.astype(np.float32))
    df["protein_id"] = _vec_id(df[protein_cols].values.astype(np.float32))
    return df


def build_doubly_disjoint(df: pd.DataFrame,
                           test_frac: float = TEST_FRAC,
                           seed: int = SEED
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition unique aptamers and unique proteins each into a development
    pool and a test pool. A pair is TEST only if both entities fall in the
    test pool, DEV only if both fall in the development pool; mixed pairs
    are discarded to avoid leaking an entity across the boundary.
    """
    rng = np.random.default_rng(seed)

    unique_apts = df["aptamer_id"].unique().copy()
    unique_pros = df["protein_id"].unique().copy()
    rng.shuffle(unique_apts)
    rng.shuffle(unique_pros)

    def _try_frac(frac):
        n_a = int(len(unique_apts) * frac)
        n_p = int(len(unique_pros) * frac)
        test_a = set(unique_apts[:n_a])
        test_p = set(unique_pros[:n_p])
        is_test = df["aptamer_id"].isin(test_a) & df["protein_id"].isin(test_p)
        is_dev = ~df["aptamer_id"].isin(test_a) & ~df["protein_id"].isin(test_p)
        n_used = is_test.sum() + is_dev.sum()
        return is_test, is_dev, (is_test.sum() / max(1, n_used))

    frac = 0.45
    is_test = is_dev = None
    for _ in range(20):
        is_test, is_dev, achieved = _try_frac(frac)
        if abs(achieved - test_frac) < 0.01:
            break
        frac = frac + 0.02 if achieved < test_frac else frac - 0.02
        frac = max(0.10, min(0.70, frac))

    apt_arr = df["aptamer_id"].values
    pro_arr = df["protein_id"].values
    assignments = np.where(is_test, "test", np.where(is_dev, "dev", "discarded"))

    audit_df = pd.DataFrame({
        "pair_index": np.arange(len(df)),
        "aptamer_id": apt_arr,
        "protein_id": pro_arr,
        "label": df[LABEL_COL].values,
        "assignment": assignments,
    })

    dev_df = df[is_dev].drop(columns=["aptamer_id", "protein_id"]).reset_index(drop=True)
    test_df = df[is_test].drop(columns=["aptamer_id", "protein_id"]).reset_index(drop=True)

    return dev_df, test_df, audit_df


def verify_doubly_disjoint(audit_df: pd.DataFrame) -> Dict:
    dev = audit_df[audit_df["assignment"] == "dev"]
    test = audit_df[audit_df["assignment"] == "test"]

    dev_aptamers = set(dev["aptamer_id"].unique())
    dev_proteins = set(dev["protein_id"].unique())

    leaking_apt = test[test["aptamer_id"].isin(dev_aptamers)]
    leaking_pro = test[test["protein_id"].isin(dev_proteins)]

    n_leak_apt = len(leaking_apt)
    n_leak_pro = len(leaking_pro)

    return {
        "n_dev": len(dev),
        "n_test": len(test),
        "n_dev_aptamers": dev["aptamer_id"].nunique(),
        "n_dev_proteins": dev["protein_id"].nunique(),
        "n_test_aptamers": test["aptamer_id"].nunique(),
        "n_test_proteins": test["protein_id"].nunique(),
        "leaking_aptamers": n_leak_apt,
        "leaking_proteins": n_leak_pro,
        "doubly_disjoint_ok": (n_leak_apt == 0) and (n_leak_pro == 0),
    }


def mcc_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray,
                           n_thresholds: int = 100) -> float:
    thresholds = np.linspace(y_score.min(), y_score.max(), n_thresholds)
    best_mcc, best_thr = -2.0, 0.5
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        m = matthews_corrcoef(y_true, y_pred)
        if m > best_mcc:
            best_mcc, best_thr = m, thr
    return best_thr


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "specificity": round(float(tn / (tn + fp + 1e-9)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": round(float(threshold), 4),
    }


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, threshold: float,
                  n_resamples: int = N_BOOTSTRAP, seed: int = SEED) -> Dict:
    rng = np.random.default_rng(seed)
    mccs, aucs = [], []
    n = len(y_true)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        yp = (ys >= threshold).astype(int)
        if len(np.unique(yp)) < 2:
            continue
        mccs.append(matthews_corrcoef(yt, yp))
        aucs.append(roc_auc_score(yt, ys))

    def _ci(vals):
        vals = np.array(vals)
        return (round(float(np.percentile(vals, 2.5)), 4),
                round(float(np.percentile(vals, 97.5)), 4))

    return {
        "mcc_ci": _ci(mccs) if mccs else (None, None),
        "roc_auc_ci": _ci(aucs) if aucs else (None, None),
    }


def run_cv_and_train(dev_df: pd.DataFrame, seed: int = SEED
                      ) -> Tuple[ExtraTreesClassifier, float, float]:
    feature_cols = [c for c in dev_df.columns if c != LABEL_COL]
    X = dev_df[feature_cols].values
    y = dev_df[LABEL_COL].values

    model = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    oof_pred = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]

    threshold = mcc_optimal_threshold(y, oof_pred)
    oof_mcc = matthews_corrcoef(y, (oof_pred >= threshold).astype(int))

    model.fit(X, y)
    return model, threshold, float(oof_mcc)


def main():
    parser = argparse.ArgumentParser(description="Doubly-disjoint split analysis.")
    parser.add_argument("--data", default="clean.csv")
    parser.add_argument("--out", default="output/doubly_disjoint")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df_raw = pd.read_csv(args.data)
    print(f"Loaded {len(df_raw):,} pairs | "
          f"pos={int((df_raw[LABEL_COL]==1).sum())} "
          f"neg={int((df_raw[LABEL_COL]==0).sum())}")

    df = assign_entity_ids(df_raw)
    print(f"Unique aptamers: {df['aptamer_id'].nunique():,}  |  "
          f"Unique proteins: {df['protein_id'].nunique():,}")

    dev_df, test_df, audit_df = build_doubly_disjoint(df, seed=args.seed)
    print(f"Dev: {len(dev_df):,} pairs (pos={int((dev_df[LABEL_COL]==1).sum())})")
    print(f"Test: {len(test_df):,} pairs (pos={int((test_df[LABEL_COL]==1).sum())})")

    v = verify_doubly_disjoint(audit_df)
    print(f"\nEntity leakage check: "
          f"{'PASS' if v['doubly_disjoint_ok'] else 'FAIL'}")
    print(f"  Dev aptamers/proteins:  {v['n_dev_aptamers']} / {v['n_dev_proteins']}")
    print(f"  Test aptamers/proteins: {v['n_test_aptamers']} / {v['n_test_proteins']}")
    print(f"  Leaking aptamers: {v['leaking_aptamers']}  |  "
          f"Leaking proteins: {v['leaking_proteins']}")

    print(f"\nTraining ExtraTrees (n_estimators={N_ESTIMATORS})...")
    model, threshold, oof_mcc = run_cv_and_train(dev_df, seed=args.seed)
    print(f"  OOF MCC: {oof_mcc:.4f}  |  Threshold: {threshold:.4f}")

    feat_cols = [c for c in test_df.columns if c != LABEL_COL]
    X_test = test_df[feat_cols].values
    y_test = test_df[LABEL_COL].values
    scores = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, scores, threshold)
    ci = bootstrap_ci(y_test, scores, threshold, n_resamples=N_BOOTSTRAP, seed=args.seed)

    mcc_lo, mcc_hi = ci["mcc_ci"]
    auc_lo, auc_hi = ci["roc_auc_ci"]

    print(f"\nTest set evaluation (single held-out pass):")
    print(f"  MCC:         {metrics['mcc']:.4f}  [{mcc_lo:.3f}, {mcc_hi:.3f}]")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}  [{auc_lo:.3f}, {auc_hi:.3f}]")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")

    dev_path = os.path.join(args.out, "doubly_disjoint_dev.csv")
    test_path = os.path.join(args.out, "doubly_disjoint_test.csv")
    audit_path = os.path.join(args.out, "doubly_disjoint_audit.csv")

    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    results = {
        **metrics,
        "mcc_ci_lo": mcc_lo, "mcc_ci_hi": mcc_hi,
        "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
        "oof_mcc": round(oof_mcc, 4),
        "threshold": round(threshold, 4),
        "n_dev": len(dev_df),
        "n_test": len(test_df),
        "verification": v,
    }

    results_path = os.path.join(args.out, "doubly_disjoint_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nOutputs written to: {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
