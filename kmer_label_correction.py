#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k=4 Feature Label Correction for AptaEnsemble.

The 4-mer column labels in the released feature tables have positions 3 and
4 (0-indexed 2 and 3) transposed relative to the sequence whose frequency
they store, e.g. the column 'aptamer_frequency_GGGC' holds the frequency of
'GGCG'. This affects 192 of the 256 possible 4-mer labels; the remaining 64
are palindromic under the swap and are unaffected. Feature values are
correct; only column names require correction.

Usage
-----
    python kmer_label_correction.py
    python kmer_label_correction.py --fix --data clean.csv M1.csv M2.csv M3.csv
"""

import argparse
import os
from itertools import product
from typing import Tuple

import pandas as pd

NUCLEOTIDES = ["A", "C", "G", "T"]


def stored_to_correct(label: str) -> str:
    """Swap positions 2 and 3 (0-indexed) of a 4-character k-mer label."""
    if len(label) != 4:
        return label
    lst = list(label)
    lst[2], lst[3] = lst[3], lst[2]
    return "".join(lst)


def is_transposed(label: str) -> bool:
    return len(label) == 4 and stored_to_correct(label) != label


def build_correction_map() -> pd.DataFrame:
    rows = []
    for nts in product(NUCLEOTIDES, repeat=4):
        stored = "".join(nts)
        correct = stored_to_correct(stored)
        rows.append({
            "stored_label": stored,
            "correct_label": correct,
            "is_transposed": stored != correct,
        })
    return pd.DataFrame(rows)


def rename_k4_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    rename_map = {}
    for col in df.columns:
        if not col.startswith("aptamer_frequency_"):
            continue
        suffix = col[len("aptamer_frequency_"):]
        if len(suffix) == 4:
            correct_suffix = stored_to_correct(suffix)
            if correct_suffix != suffix:
                rename_map[col] = f"aptamer_frequency_{correct_suffix}"
    return df.rename(columns=rename_map), len(rename_map)


def main():
    parser = argparse.ArgumentParser(
        description="Detect and correct k=4 label transposition in AptaEnsemble feature tables.")
    parser.add_argument("--data", nargs="+", default=[])
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--map-out", default="kmer4_correction_map.csv")
    args = parser.parse_args()

    correction_map = build_correction_map()
    n_transposed = correction_map["is_transposed"].sum()
    correction_map.to_csv(args.map_out, index=False)
    print(f"Correction map written to {args.map_out} "
          f"({n_transposed}/256 labels affected)")

    if not args.fix:
        return

    if not args.data:
        print("--fix requires --data <file(s)>")
        return

    for fpath in args.data:
        if not os.path.exists(fpath):
            print(f"  [skip] {fpath} not found")
            continue

        df = pd.read_csv(fpath)
        k4_cols = [c for c in df.columns
                   if c.startswith("aptamer_frequency_")
                   and len(c[len("aptamer_frequency_"):]) == 4]

        if not k4_cols:
            print(f"  [skip] {fpath} — no k=4 columns found")
            continue

        needs_fix = any(is_transposed(c[len("aptamer_frequency_"):]) for c in k4_cols)
        if not needs_fix:
            print(f"  [ok] {fpath} — labels already correct")
            continue

        df_fixed, n_renamed = rename_k4_columns(df)
        base, ext = os.path.splitext(fpath)
        out_path = f"{base}_kmer_corrected{ext}"
        df_fixed.to_csv(out_path, index=False)
        print(f"  [fixed] {fpath} -> {out_path} ({n_renamed} columns renamed)")


if __name__ == "__main__":
    main()
