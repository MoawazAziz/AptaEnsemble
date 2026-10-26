#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entity-Disjoint Validation Split Builder for AptaEnsemble.

Builds five validation designs from a feature table of aptamer-protein pairs,
ranging from a standard random split to a doubly-disjoint split in which
neither the aptamer nor the protein in any test pair appears in training.

Designs
-------
1. pair_level        Random stratified split. Statistically equivalent to
                     the original M1/M2/M3 cohort construction, which is
                     itself an unconstrained stratified split.
2. protein_disjoint  Test proteins absent from training.
3. aptamer_disjoint  Test aptamers absent from training.
4. cluster_disjoint  Aptamer similarity clusters (cosine >= 0.90) disjoint
                     between train and test.
5. doubly_disjoint   Neither aptamer nor protein entity shared between
                     train and test.

Since the feature table stores only descriptors (no sequence identifiers),
entity identity is recovered by fingerprinting each row's k-mer sub-vector
(aptamer) and PseAAC sub-vector (protein); identical fingerprints indicate
the same physical entity.

Usage
-----
    python build_entity_disjoint_splits.py --data clean.csv --out splits/
"""

import argparse
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

LABEL_COL = "Class"
KMER_PREFIX = "aptamer_frequency_"
PROTEIN_PREFIX = "protein_"

DEFAULT_DATA = "clean.csv"
DEFAULT_OUT_DIR = "splits"
DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.20
COSINE_THRESHOLD = 0.90


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def get_kmer_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith(KMER_PREFIX)]


def get_protein_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith(PROTEIN_PREFIX)]


def assign_entity_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Recover aptamer_id and protein_id by hashing feature sub-vectors."""
    kmer_cols = get_kmer_cols(df)
    protein_cols = get_protein_cols(df)

    apt_mat = df[kmer_cols].values.astype(np.float32)
    pro_mat = df[protein_cols].values.astype(np.float32)

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
    df["aptamer_id"] = _vec_id(apt_mat)
    df["protein_id"] = _vec_id(pro_mat)
    return df


def build_pair_level(df: pd.DataFrame, test_frac: float, seed: int
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df, test_size=test_frac, random_state=seed, stratify=df[LABEL_COL])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_protein_disjoint(df: pd.DataFrame, test_frac: float, seed: int
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    unique_proteins = df["protein_id"].unique()
    rng.shuffle(unique_proteins)

    test_proteins = set()
    n_total = len(df)
    for pid in unique_proteins:
        test_proteins.add(pid)
        if df["protein_id"].isin(test_proteins).sum() / n_total >= test_frac:
            break

    is_test = df["protein_id"].isin(test_proteins)
    train = df[~is_test].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)
    return train, test


def build_aptamer_disjoint(df: pd.DataFrame, test_frac: float, seed: int
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    unique_aptamers = df["aptamer_id"].unique()
    rng.shuffle(unique_aptamers)

    test_aptamers = set()
    n_total = len(df)
    for aid in unique_aptamers:
        test_aptamers.add(aid)
        if df["aptamer_id"].isin(test_aptamers).sum() / n_total >= test_frac:
            break

    is_test = df["aptamer_id"].isin(test_aptamers)
    train = df[~is_test].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)
    return train, test


def build_cluster_disjoint(df: pd.DataFrame, test_frac: float, seed: int,
                            threshold: float = COSINE_THRESHOLD
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single-linkage clustering on aptamer k-mer profiles, cosine >= threshold."""
    kmer_cols = get_kmer_cols(df)
    apt_ids = df["aptamer_id"].values
    unique_ids = np.unique(apt_ids)
    id_to_idx = {aid: np.where(apt_ids == aid)[0][0] for aid in unique_ids}

    apt_mat = df[kmer_cols].values.astype(np.float32)
    unique_vecs = np.array([apt_mat[id_to_idx[aid]] for aid in unique_ids])

    n = len(unique_ids)
    cluster_map = np.full(n, -1, dtype=np.int32)
    next_cluster = 0

    for i in range(n):
        if cluster_map[i] >= 0:
            continue
        cluster_map[i] = next_cluster
        sim = _cosine_sim(unique_vecs[i:i + 1], unique_vecs).flatten()
        for j in range(i + 1, n):
            if cluster_map[j] < 0 and sim[j] >= threshold:
                cluster_map[j] = next_cluster
        next_cluster += 1

    aid_to_cluster = {unique_ids[i]: int(cluster_map[i]) for i in range(n)}
    df = df.copy()
    df["_cluster"] = df["aptamer_id"].map(aid_to_cluster)

    rng = np.random.default_rng(seed)
    unique_clusters = df["_cluster"].unique()
    rng.shuffle(unique_clusters)

    test_clusters = set()
    n_total = len(df)
    for cid in unique_clusters:
        test_clusters.add(cid)
        if df["_cluster"].isin(test_clusters).sum() / n_total >= test_frac:
            break

    is_test = df["_cluster"].isin(test_clusters)
    train = df[~is_test].drop(columns=["_cluster"]).reset_index(drop=True)
    test = df[is_test].drop(columns=["_cluster"]).reset_index(drop=True)
    return train, test


def build_doubly_disjoint(df: pd.DataFrame, test_frac: float, seed: int
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entity-partition construction: unique aptamers and unique proteins are
    each partitioned into a development pool and a test pool. A pair is
    assigned to test only if both its aptamer and protein fall in the test
    pool, and to development only if both fall in the development pool.
    Mixed pairs are discarded, since assigning them to either side would
    leak one entity into the opposite partition.
    """
    rng = np.random.default_rng(seed)

    unique_apts = df["aptamer_id"].unique().copy()
    unique_pros = df["protein_id"].unique().copy()
    rng.shuffle(unique_apts)
    rng.shuffle(unique_pros)

    def _try_frac(frac_a: float, frac_p: float):
        n_a = int(len(unique_apts) * frac_a)
        n_p = int(len(unique_pros) * frac_p)
        test_a = set(unique_apts[:n_a])
        test_p = set(unique_pros[:n_p])
        is_test = df["aptamer_id"].isin(test_a) & df["protein_id"].isin(test_p)
        is_dev = ~df["aptamer_id"].isin(test_a) & ~df["protein_id"].isin(test_p)
        n_used = is_test.sum() + is_dev.sum()
        return is_test, is_dev, (is_test.sum() / max(1, n_used))

    frac = 0.45
    for _ in range(20):
        is_test, is_dev, achieved = _try_frac(frac, frac)
        if abs(achieved - test_frac) < 0.01:
            break
        frac = frac + 0.02 if achieved < test_frac else frac - 0.02
        frac = max(0.10, min(0.70, frac))

    train = df[is_dev].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)
    return train, test


def _drop_entity_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in ["aptamer_id", "protein_id"]
                             if c in df.columns], errors="ignore")


def save_splits(splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                 out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for name, (train, test) in splits.items():
        train_path = os.path.join(out_dir, f"{name}_train.csv")
        test_path = os.path.join(out_dir, f"{name}_test.csv")

        _drop_entity_cols(train).to_csv(train_path, index=False)
        _drop_entity_cols(test).to_csv(test_path, index=False)

        n_tr_pos = int((train[LABEL_COL] == 1).sum())
        n_tr_neg = int((train[LABEL_COL] == 0).sum())
        n_te_pos = int((test[LABEL_COL] == 1).sum())
        n_te_neg = int((test[LABEL_COL] == 0).sum())

        rows.append({
            "design": name,
            "train_total": len(train),
            "train_pos": n_tr_pos,
            "train_neg": n_tr_neg,
            "train_pos_ratio": round(n_tr_pos / max(1, len(train)), 4),
            "test_total": len(test),
            "test_pos": n_te_pos,
            "test_neg": n_te_neg,
            "test_pos_ratio": round(n_te_pos / max(1, len(test)), 4),
            "test_frac": round(len(test) / max(1, len(train) + len(test)), 4),
        })

        print(f"  {name:<22}: train={len(train):>4} (pos={n_tr_pos})"
              f"  test={len(test):>4} (pos={n_te_pos})")

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "split_summary.csv"), index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build all entity-disjoint validation splits for AptaEnsemble.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cosine-threshold", type=float, default=COSINE_THRESHOLD)
    args = parser.parse_args()

    print(f"Data: {args.data}  |  Output: {args.out}  |  "
          f"TestFrac: {args.test_frac}  |  Seed: {args.seed}")

    df_raw = pd.read_csv(args.data)
    print(f"Loaded {len(df_raw):,} pairs | "
          f"pos={int((df_raw[LABEL_COL]==1).sum())} "
          f"neg={int((df_raw[LABEL_COL]==0).sum())}")

    df = assign_entity_ids(df_raw)
    print(f"Unique aptamer entities: {df['aptamer_id'].nunique():,}  |  "
          f"Unique protein entities: {df['protein_id'].nunique():,}")

    splits = {
        "pair_level": build_pair_level(df, args.test_frac, args.seed),
        "protein_disjoint": build_protein_disjoint(df, args.test_frac, args.seed),
        "aptamer_disjoint": build_aptamer_disjoint(df, args.test_frac, args.seed),
        "cluster_disjoint": build_cluster_disjoint(
            df, args.test_frac, args.seed, threshold=args.cosine_threshold),
        "doubly_disjoint": build_doubly_disjoint(df, args.test_frac, args.seed),
    }

    print(f"\nSaving splits to '{args.out}/':")
    summary = save_splits(splits, args.out)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
