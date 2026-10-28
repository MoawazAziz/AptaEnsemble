#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AptaEnsemble: reproducible handling of AptaNet-derived encoded features.

PURPOSE
-------
Reconstruct the 3,003-row, 641-column modeling table from the ORIGINAL
AptaNet Dataset.csv without changing *any* numerical descriptor values.
Produce and check the canonical 4-mer annotation correction (192 changed;
64 unchanged). This script does NOT pretend to regenerate the inherited
300 protein feature values from FASTA sequences.

- The original paper describes PseAAC with omega=0.05 and lag=30; the publicly
  accessible AptaNet extraction script uses a literal 0.15 in some pathways.
  Neither value can be established as the effective value of the supplied
  PRECOMPUTED Dataset.csv without a validated FASTA-to-CSV reconstruction.
- For the same reason, the optional reference_pseaac_block() below demonstrates
  the documented mathematical formula,



DEPENDENCIES: Python standard library only. Does not train models or rewrite
existing input data. Keep the original source file unchanged and commit exact
source provenance, SHA-256 checksums, and the mapping CSV alongside this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict, Counter
from pathlib import Path
from decimal import Decimal, InvalidOperation
from typing import Mapping, Sequence

DNA = 'ACGT'
AA20 = 'ARNDCQEGHILKMFPSTWYV'  # exact protein column order in inherited CSV
GROUPS = 'ABCDEF'
# The 18 scale tables below were extracted from the user-supplied original AptaNet
# Colab feature-extraction source (Groups A–F only; its Group G is not in this
# 300-dimensional benchmark). Verify permission and cite AptaNet when reusing.
APTANET_A_TO_F_PROPERTY_SCALES: dict[str, dict[str, dict[str, float]]] = {'A': {'hydrophobicity': {'A': 0.62,
                          'C': 0.29,
                          'D': -0.9,
                          'E': -0.74,
                          'F': 1.19,
                          'G': 0.48,
                          'H': -0.4,
                          'I': 1.38,
                          'K': -1.5,
                          'L': 1.06,
                          'M': 0.64,
                          'N': -0.78,
                          'P': 0.12,
                          'Q': -0.85,
                          'R': -2.53,
                          'S': -0.18,
                          'T': -0.05,
                          'V': 1.08,
                          'W': 0.81,
                          'Y': 0.26},
       'hydrophilicity': {'A': -0.5,
                          'C': -1.0,
                          'D': 3.0,
                          'E': 3.0,
                          'F': -2.5,
                          'G': 0.0,
                          'H': -0.5,
                          'I': -1.8,
                          'K': 3.0,
                          'L': -1.8,
                          'M': -1.3,
                          'N': 0.2,
                          'P': 0.0,
                          'Q': 0.2,
                          'R': 3.0,
                          'S': 0.3,
                          'T': -0.4,
                          'V': -1.5,
                          'W': -3.4,
                          'Y': -2.3},
       'residue_mass': {'A': 15.0,
                        'C': 47.0,
                        'D': 59.0,
                        'E': 73.0,
                        'F': 91.0,
                        'G': 1.0,
                        'H': 82.0,
                        'I': 57.0,
                        'K': 73.0,
                        'L': 57.0,
                        'M': 75.0,
                        'N': 58.0,
                        'P': 42.0,
                        'Q': 72.0,
                        'R': 101.0,
                        'S': 31.0,
                        'T': 45.0,
                        'V': 43.0,
                        'W': 130.0,
                        'Y': 107.0}},
 'B': {'polarity': {'A': 0.5,
                    'C': 2.5,
                    'D': -1,
                    'E': 2.5,
                    'F': -2.5,
                    'G': 0,
                    'H': -0.5,
                    'I': 1.8,
                    'K': 3,
                    'L': -1.8,
                    'M': -1.3,
                    'N': 0.2,
                    'P': -1.4,
                    'Q': 0.2,
                    'R': 3,
                    'S': 0.3,
                    'T': -0.4,
                    'V': -1.5,
                    'W': -3.4,
                    'Y': -2.3},
       'molecular_weight': {'A': 5.3,
                            'C': 3.6,
                            'D': 1.3,
                            'E': 3.3,
                            'F': 2.3,
                            'G': 4.8,
                            'H': 1.4,
                            'I': 3.1,
                            'K': 4.1,
                            'L': 4.7,
                            'M': 1.1,
                            'N': 3,
                            'P': 2.5,
                            'Q': 2.4,
                            'R': 2.6,
                            'S': 4.5,
                            'T': 3.7,
                            'V': 4.2,
                            'W': 0.8,
                            'Y': 2.3},
       'melting_point': {'A': 0.81,
                         'C': 0.71,
                         'D': 1.17,
                         'E': 0.53,
                         'F': 1.2,
                         'G': 0.88,
                         'H': 0.92,
                         'I': 1.48,
                         'K': 0.77,
                         'L': 1.24,
                         'M': 1.05,
                         'N': 0.62,
                         'P': 0.61,
                         'Q': 0.98,
                         'R': 0.85,
                         'S': 0.92,
                         'T': 1.18,
                         'V': 1.66,
                         'W': 1.18,
                         'Y': 1.23}},
 'C': {'transfer_free_energy': {'A': 58,
                                'C': -97,
                                'D': 116,
                                'E': -131,
                                'F': 92,
                                'G': -11,
                                'H': -73,
                                'I': 107,
                                'K': -24,
                                'L': 95,
                                'M': 78,
                                'N': -93,
                                'P': -79,
                                'Q': -139,
                                'R': -184,
                                'S': -34,
                                'T': -7,
                                'V': 100,
                                'W': 59,
                                'Y': -11},
       'buriability': {'A': 1.37,
                       'C': 8.93,
                       'D': -4.47,
                       'E': 4.04,
                       'F': -7.96,
                       'G': 3.39,
                       'H': -1.65,
                       'I': -7.92,
                       'K': 7.7,
                       'L': -8.68,
                       'M': -7.13,
                       'N': 6.29,
                       'P': 6.25,
                       'Q': 3.88,
                       'R': 1.33,
                       'S': 4.08,
                       'T': 4.02,
                       'V': -6.94,
                       'W': 0.79,
                       'Y': -4.73},
       'bulkiness': {'A': 6.77,
                     'C': 8.57,
                     'D': 0.31,
                     'E': 12.93,
                     'F': 1.92,
                     'G': 7.95,
                     'H': 2.8,
                     'I': 2.72,
                     'K': 10.2,
                     'L': 4.43,
                     'M': 1.87,
                     'N': 5.5,
                     'P': 4.79,
                     'Q': 5.24,
                     'R': 6.87,
                     'S': 5.41,
                     'T': 5.36,
                     'V': 3.57,
                     'W': 0.54,
                     'Y': 2.26}},
 'D': {'solvation_free_energy': {'A': 0.87,
                                 'C': 0.66,
                                 'D': 1.52,
                                 'E': 0.67,
                                 'F': 2.87,
                                 'G': 0.1,
                                 'H': 0.87,
                                 'I': 3.15,
                                 'K': 1.64,
                                 'L': 2.17,
                                 'M': 1.67,
                                 'N': 0.09,
                                 'P': 2.77,
                                 'Q': 0,
                                 'R': 0.85,
                                 'S': 0.07,
                                 'T': 0.07,
                                 'V': 1.87,
                                 'W': 3.77,
                                 'Y': 2.67},
       'relative_mutability': {'A': 1.09,
                               'C': 0.77,
                               'D': 0.5,
                               'E': 0.92,
                               'F': 0.5,
                               'G': 1.25,
                               'H': 0.67,
                               'I': 0.66,
                               'K': 1.25,
                               'L': 0.44,
                               'M': 0.45,
                               'N': 1.14,
                               'P': 2.96,
                               'Q': 0.83,
                               'R': 0.97,
                               'S': 1.21,
                               'T': 1.33,
                               'V': 0.56,
                               'W': 0.62,
                               'Y': 0.94},
       'residue_volume': {'A': 0.91,
                          'C': 1.4,
                          'D': 0.93,
                          'E': 0.97,
                          'F': 0.72,
                          'G': 1.51,
                          'H': 0.9,
                          'I': 0.65,
                          'K': 0.82,
                          'L': 0.59,
                          'M': 0.58,
                          'N': 1.64,
                          'P': 1.66,
                          'Q': 0.94,
                          'R': 1,
                          'S': 1.23,
                          'T': 1.04,
                          'V': 0.6,
                          'W': 0.67,
                          'Y': 0.92}},
 'E': {'volume': {'A': 0.92,
                  'C': 0.48,
                  'D': 1.16,
                  'E': 0.61,
                  'F': 1.25,
                  'G': 0.61,
                  'H': 0.93,
                  'I': 1.81,
                  'K': 0.7,
                  'L': 1.3,
                  'M': 1.19,
                  'N': 0.6,
                  'P': 0.4,
                  'Q': 0.95,
                  'R': 0.93,
                  'S': 0.82,
                  'T': 1.12,
                  'V': 1.81,
                  'W': 1.54,
                  'Y': 1.53},
       'amino_acid_distribution': {'A': 0.96,
                                   'C': 0.9,
                                   'D': 1.13,
                                   'E': 0.33,
                                   'F': 1.37,
                                   'G': 0.9,
                                   'H': 0.87,
                                   'I': 1.54,
                                   'K': 0.81,
                                   'L': 1.26,
                                   'M': 1.29,
                                   'N': 0.72,
                                   'P': 0.75,
                                   'Q': 1.18,
                                   'R': 0.67,
                                   'S': 0.77,
                                   'T': 1.23,
                                   'V': 1.41,
                                   'W': 1.13,
                                   'Y': 1.07},
       'hydration_number': {'A': 0.9,
                            'C': 0.47,
                            'D': 1.24,
                            'E': 0.62,
                            'F': 1.23,
                            'G': 0.56,
                            'H': 1.12,
                            'I': 1.54,
                            'K': 0.74,
                            'L': 1.26,
                            'M': 1.09,
                            'N': 0.62,
                            'P': 0.42,
                            'Q': 1.18,
                            'R': 1.02,
                            'S': 0.87,
                            'T': 1.3,
                            'V': 1.53,
                            'W': 1.75,
                            'Y': 1.68}},
 'F': {'isoelectric_point': {'A': 6,
                             'C': 5.05,
                             'D': 2.77,
                             'E': 5.22,
                             'F': 5.48,
                             'G': 5.97,
                             'H': 7.59,
                             'I': 6.02,
                             'K': 9.74,
                             'L': 5.98,
                             'M': 5.74,
                             'N': 5.41,
                             'P': 6.3,
                             'Q': 5.65,
                             'R': 10.76,
                             'S': 5.68,
                             'T': 5.66,
                             'V': 5.96,
                             'W': 5.89,
                             'Y': 5.66},
       'compressibility': {'A': -25.5,
                           'C': -32.82,
                           'D': -33.12,
                           'E': -36.17,
                           'F': -34.54,
                           'G': -27,
                           'H': -31.84,
                           'I': -31.78,
                           'K': -32.4,
                           'L': -31.78,
                           'M': -31.18,
                           'N': -30.9,
                           'P': -23.25,
                           'Q': -32.6,
                           'R': -26.62,
                           'S': -29.88,
                           'T': -31.23,
                           'V': -30.62,
                           'W': -30.24,
                           'Y': -35.01},
       'chromatographic_index': {'A': 9.9,
                                 'C': 2.8,
                                 'D': 2.8,
                                 'E': 3.2,
                                 'F': 18.8,
                                 'G': 5.6,
                                 'H': 8.2,
                                 'I': 17.1,
                                 'K': 3.5,
                                 'L': 17.6,
                                 'M': 14.7,
                                 'N': 5.4,
                                 'P': 14.8,
                                 'Q': 9,
                                 'R': 4.6,
                                 'S': 6.9,
                                 'T': 9.5,
                                 'V': 14.3,
                                 'W': 17,
                                 'Y': 15}}}

KMER_SIZES = (1, 2, 3, 4)
KMER_COUNTS = {1: 4, 2: 16, 3: 64, 4: 256}
EXPECTED_SOURCE_ROWS = 3404
EXPECTED_FINAL_ROWS = 3003
EXPECTED_COLUMNS = 641


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Read numerical cells as strings, never floating point: no numeric drift."""
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if len(header) != EXPECTED_COLUMNS:
        raise ValueError(f'{path}: expected 641 columns, got {len(header)}')
    for i, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(f'{path}: row {i} has {len(row)} cells, not {len(header)}')
        if row[0] not in {'0', '1'}:
            raise ValueError(f'{path}: invalid class label in row {i}: {row[0]!r}')
    return header, rows


def write_csv(path: str | Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def canonical_kmers(k: int) -> list[str]:
    return [''.join(s) for s in itertools.product(DNA, repeat=k)]


def aptamer_headers(correct_four_mers: bool = True) -> list[str]:
    cols: list[str] = []
    for k in KMER_SIZES:
        for seq in canonical_kmers(k):
            # Legacy AptaNet header has positions three/four transposed,
            # while the numerical frequency column occupies canonical order.
            name = seq if (k != 4 or correct_four_mers) else seq[:2] + seq[3] + seq[2]
            cols.append(f'aptamer_frequency_{name}')
    assert len(cols) == 340 and len(set(cols)) == 340
    return cols


def protein_headers() -> list[str]:
    result: list[str] = []
    for group in GROUPS:
        result += [f'protein_{group}_aac_{aa}' for aa in AA20]
        result += [f'protein_{group}_pseaac_lambda_{lag}' for lag in range(1, 31)]
    assert len(result) == 300
    return result


def full_headers(correct_four_mers: bool) -> list[str]:
    header = ['Class'] + aptamer_headers(correct_four_mers) + protein_headers()
    assert len(header) == EXPECTED_COLUMNS
    return header


def label_mapping_rows() -> list[list[str]]:
    """One row per canonical 4-mer column, with zero-based CSV index and status."""
    mapping: list[list[str]] = []
    for four_index, canonical in enumerate(canonical_kmers(4)):
        original = canonical[:2] + canonical[3] + canonical[2]
        mapping.append([
            str(85 + four_index),   # CSV column index (Class=0)
            str(four_index),
            f'aptamer_frequency_{original}',
            f'aptamer_frequency_{canonical}',
            'changed' if original != canonical else 'unchanged'
        ])
    assert sum(x[-1] == 'changed' for x in mapping) == 192
    assert sum(x[-1] == 'unchanged' for x in mapping) == 64
    return mapping


def write_mapping(path: str | Path) -> None:
    write_csv(path,
              ['csv_column_index_0based', 'four_mer_index_0based',
               'original_header', 'corrected_header', 'status'],
              label_mapping_rows())


def clean_aptanet(source_rows: Sequence[Sequence[str]]) -> tuple[list[list[str]], dict]:
    """Remove exact duplicate labeled rows, then BOTH labels for conflicts.

    Each conflicting profile is the entire 640-value descriptor vector
    associated with two labels. Removing both avoids arbitrarily choosing one.
    """
    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in source_rows:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            unique.append(list(row))
    labels_by_descriptor: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in unique:
        labels_by_descriptor[tuple(row[1:])].add(row[0])
    contradictory = {k for k, labels in labels_by_descriptor.items() if len(labels) > 1}
    cleaned = [r for r in unique if tuple(r[1:]) not in contradictory]
    stats = {
        'source_rows': len(source_rows),
        'exact_duplicate_labeled_rows_removed': len(source_rows) - len(unique),
        'distinct_rows_after_deduplication': len(unique),
        'contradictory_descriptor_profiles': len(contradictory),
        'contradictory_labeled_rows_removed': len(unique) - len(cleaned),
        'final_rows': len(cleaned),
        'class_counts': dict(sorted(Counter(r[0] for r in cleaned).items())),
        'feature_columns': 640,
        'aptamer_columns': 340,
        'protein_columns': 300,
        'four_mer_labels_changed': 192,
        'four_mer_labels_unchanged': 64,
        'note': 'All retained numerical values are inherited without recomputation.'
    }
    return cleaned, stats


def _check_original_source_header(src_header: Sequence[str]) -> None:
    """Confirm structural positions, not the known ambiguous original names."""
    if src_header[0] != 'Class':
        raise ValueError('Original AptaNet first column must be Class')
    if not all(h.startswith('aptamer_frequency_') for h in src_header[1:341]):
        raise ValueError('Expected 340 aptamer frequency columns in positions 1:341')
    if not all(h.startswith('Xc') for h in src_header[341:]):
        raise ValueError('Expected original AptaNet Xc protein columns in positions 341:641')


def assert_same_cells(actual_rows: Sequence[Sequence[str]], expected_rows: Sequence[Sequence[str]]) -> None:
    if len(actual_rows) != len(expected_rows):
        raise AssertionError(f'Row counts differ: {len(actual_rows)} != {len(expected_rows)}')
    for i, (a, b) in enumerate(zip(actual_rows, expected_rows), start=2):
        if a != b:
            first = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
            raise AssertionError(f'Cell mismatch at row {i} column {first}: {a[first]!r} != {b[first]!r}')


def verify_reference_clean(generated_rows, reference_path: str | Path, corrected: bool) -> None:
    expected_header, expected_rows = read_csv(reference_path)
    if expected_header != full_headers(corrected):
        for j, (a, b) in enumerate(zip(expected_header, full_headers(corrected))):
            if a != b:
                raise AssertionError(f'{reference_path}: header mismatch at column {j}: {a!r} != {b!r}')
        raise AssertionError(f'{reference_path}: header mismatch')
    if corrected:
        # The repository's corrected CSV was rewritten through pandas, which
        # formats 0 as 0.0, etc. Decimal equality verifies unchanged numbers.
        if len(generated_rows) != len(expected_rows):
            raise AssertionError('Corrected reference row count mismatch')
        for r, (actual, expected) in enumerate(zip(generated_rows, expected_rows), start=2):
            if actual[0] != expected[0]:
                raise AssertionError(f'Corrected label mismatch in row {r}')
            for c, (a, b) in enumerate(zip(actual[1:], expected[1:]), start=1):
                if Decimal(a) != Decimal(b):
                    raise AssertionError(f'Corrected numeric value mismatch: row {r}, col {c}, {a} != {b}')
    else:
        assert_same_cells(generated_rows, expected_rows)
    print(f'PASS: {reference_path} has matching headers and all {len(expected_rows)} label/numerical rows')


def reference_pseaac_block(
    protein_sequence: str,
    property_triplet: Sequence[Mapping[str, float]],
    *,
    max_lag: int = 30,
    omega: float = 0.05,
) -> list[float]:
    """Illustrate manuscript Eqs. 9–14 with user-supplied property scales.

    IMPORTANT: This is a REFERENCE mathematical function. It has NOT been
    validated to regenerate the AptaNet Dataset.csv values; do not use to
    score new candidate proteins against the inherited matrix without a
    demonstrated, end-to-end encoding/normalization match.
    """
    seq = protein_sequence.strip().upper()
    if not seq or len(seq) <= max_lag or set(seq) - set(AA20):
        raise ValueError('Protein sequence must use 20 standard amino acids and exceed max_lag')
    if len(property_triplet) != 3:
        raise ValueError('Exactly three physicochemical property tables required')
    if max_lag < 1 or omega < 0:
        raise ValueError('max_lag must be positive and omega nonnegative')
    standardized: list[dict[str, float]] = []
    for scale in property_triplet:
        missing = set(AA20) - set(scale)
        if missing:
            raise ValueError(f'Property table missing amino acids: {sorted(missing)}')
        vals = [float(scale[aa]) for aa in AA20]
        mean = sum(vals)/20
        sd = (sum((v-mean)**2 for v in vals)/20)**0.5
        if sd == 0:
            raise ValueError('Constant property table has zero standard deviation')
        standardized.append({aa: (float(scale[aa])-mean)/sd for aa in AA20})
    L = len(seq)
    freqs = [seq.count(aa)/L for aa in AA20]
    theta = []
    for lag in range(1, max_lag+1):
        corr = 0.0
        for j in range(L-lag):
            left, right = seq[j], seq[j+lag]
            corr += sum((scale[left]-scale[right])**2 for scale in standardized)/3
        theta.append(corr/(L-lag))
    denom = 1+omega*sum(theta)
    return [f/denom for f in freqs]+[omega*t/denom for t in theta]



def reference_kmer_frequencies(aptamer_sequence: str) -> list[float]:
    """Compute 340 A/C/G/T k-mer frequencies in canonical order, k=1..4.

    NOTE: This produces newly computed values, whereas the modeling benchmark
    uses precomputed inherited values. The two have not been proven equivalent.
    """
    seq = aptamer_sequence.strip().upper()
    if len(seq) < 4 or set(seq) - set(DNA):
        raise ValueError('Aptamer must have length >=4 and contain only A,C,G,T')
    values: list[float] = []
    for k in KMER_SIZES:
        denominator = len(seq) - k + 1
        counts = Counter(seq[i:i + k] for i in range(denominator))
        block = [counts[word] / denominator for word in canonical_kmers(k)]
        if not math.isclose(sum(block), 1.0, abs_tol=1e-10):
            raise AssertionError('k-mer frequencies must sum to 1 for each k')
        values.extend(block)
    assert len(values) == 340
    return values


def reference_protein_features(
    protein_sequence: str, *, omega: float = 0.05, max_lag: int = 30
) -> list[float]:
    """Compute a *reference* six-group PseAAC representation (300 values).

    It uses the ORIGINAL property scale tables supplied in the user's legacy
    AptaNet Colab code but corrects the conventional mathematics:
      * amino-acid frequency = count / protein length (not count / 20)
      * lag ell uses L-ell residue pairs (not L-30 for every ell)
      * each group has a separate independent 50-value block
      * the 30 order terms and 20 composition terms share the group denominator
      * no arbitrary three-decimal rounding at intermediate steps

    CAUTION: The existing AptaNet Dataset.csv numerical values, including
    centered negative values, are NOT reproduced from a FASTA sequence by this
    function. Never use its outputs with an existing classifier unless feature
    compatibility has been independently validated on known proteins.
    """
    if max_lag != 30:
        raise ValueError('This model representation requires max_lag=30')
    result: list[float] = []
    for group in GROUPS:
        scales = list(APTANET_A_TO_F_PROPERTY_SCALES[group].values())
        block = reference_pseaac_block(
            protein_sequence, scales, omega=omega, max_lag=max_lag
        )
        if len(block) != 50:
            if max_lag != 30:
                raise ValueError('The 300-dimensional model expects max_lag=30')
            raise AssertionError(f'Expected 50 features per group, got {len(block)}')
        result.extend(block)
    assert len(result) == 300
    return result


def write_reference_pair(
    output: str | Path,
    aptamer_sequence: str,
    protein_sequence: str,
    *, omega: float = 0.05,
) -> None:
    """Export a demonstration descriptor vector; NOT TRAINING-COMPATIBLE."""
    values = reference_kmer_frequencies(aptamer_sequence)
    values.extend(reference_protein_features(protein_sequence, omega=omega))
    assert len(values) == 640
    # Use corrected 4-mer annotations; source numeric columns are left alone.
    header = aptamer_headers(correct_four_mers=True) + protein_headers()
    write_csv(output, header, [[format(x, '.17g') for x in values]])
    print('WARNING: This reference encoder is NOT validated against the AptaNet',
          'precomputed feature space; do not use for candidate-ranking claims.')
    print(f'Wrote 640 reference descriptors to {output}')


def verify_reference_formulas() -> None:
    """Fast mathematical unit tests, not a source-dataset reconstruction."""
    dna = 'AACCGGTTACGTACGTACGT'
    protein = 'ACDEFGHIKLMNPQRSTVWY' * 3
    av = reference_kmer_frequencies(dna)
    pv = reference_protein_features(protein, omega=0.05)
    assert len(av) == 340 and len(pv) == 300
    for offset, width in ((0, 4), (4, 16), (20, 64), (84, 256)):
        assert math.isclose(sum(av[offset:offset + width]), 1.0, abs_tol=1e-10)
    for i in range(6):
        block = pv[i * 50 : (i + 1) * 50]
        assert len(block) == 50
        assert all(math.isfinite(v) and v >= 0 for v in block)
        assert math.isclose(sum(block), 1.0, abs_tol=1e-10)
    print('PASS: 340 k-mers, 300 PseAAC descriptors, correct block normalization')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    rebuild = sub.add_parser('rebuild', help='Reconstruct cleaned benchmark, corrected copy and mapping')
    rebuild.add_argument('--source', required=True, help='Original AptaNet Dataset.csv')
    rebuild.add_argument('--legacy-output', default='clean_reconstructed.csv')
    rebuild.add_argument('--corrected-output', default='clean_kmer_corrected_reconstructed.csv')
    rebuild.add_argument('--mapping-output', default='kmer4_correction_map.csv')
    rebuild.add_argument('--verify-against', help='Reference legacy AptaEnsemble clean.csv')
    rebuild.add_argument('--verify-corrected', help='Reference clean_kmer_corrected.csv')
    rebuild.add_argument('--report', default='descriptor_provenance_report.json')
    mapping = sub.add_parser('mapping', help='Write all 256 original/corrected 4-mer headers')
    mapping.add_argument('--output', default='kmer4_correction_map.csv')
    audit = sub.add_parser('audit', help='Check original Dataset.csv against already cleaned matrices')
    audit.add_argument('--source', required=True)
    audit.add_argument('--legacy-clean', required=True)
    audit.add_argument('--corrected-clean')
    encode = sub.add_parser('encode-reference', help='Demonstration FASTA encoder (NOT compatible with inherited matrix)')
    encode.add_argument('--aptamer', required=True, help='DNA aptamer sequence, alphabet A,C,G,T')
    encode.add_argument('--protein', required=True, help='Protein sequence, 20 standard amino acids, length >30')
    encode.add_argument('--output', required=True, help='Where to save a single-row 640-feature demonstration CSV')
    encode.add_argument('--omega', type=float, default=0.05, help='Reference formula sequence-order weight, not a verified source weight')
    encode.add_argument('--acknowledge-not-training-compatible', action='store_true',
                        help='Required acknowledgment; AptaNet Dataset.csv processing is not fully reconstructed')
    sub.add_parser('selftest', help='Test reference formula dimensions and normalization')
    args = parser.parse_args()
    if args.command == 'selftest':
        verify_reference_formulas()
        return
    if args.command == 'encode-reference':
        if not args.acknowledge_not_training_compatible:
            parser.error('Reference encoder has NOT been validated against the inherited training matrix; '
                         'pass --acknowledge-not-training-compatible only for standalone demonstration')
        write_reference_pair(args.output, args.aptamer, args.protein, omega=args.omega)
        return
    if args.command == 'mapping':
        write_mapping(args.output)
        print(f'PASS: wrote 256 mappings to {args.output} (192 changed, 64 unchanged)')
        return
    src_header, rows = read_csv(args.source)
    _check_original_source_header(src_header)
    cleaned, stats = clean_aptanet(rows)
    print(json.dumps(stats, indent=2))
    if args.command == 'rebuild':
        write_csv(args.legacy_output, full_headers(False), cleaned)
        write_csv(args.corrected_output, full_headers(True), cleaned)
        write_mapping(args.mapping_output)
        if args.verify_against:
            verify_reference_clean(cleaned, args.verify_against, corrected=False)
        if args.verify_corrected:
            verify_reference_clean(cleaned, args.verify_corrected, corrected=True)
        stats['source_sha256'] = sha256_file(args.source)
        if args.verify_against:
            stats['reference_clean_sha256'] = sha256_file(args.verify_against)
        Path(args.report).write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
        print('PASS: wrote benchmark files, 4-mer mapping, and provenance report')
    elif args.command == 'audit':
        verify_reference_clean(cleaned, args.legacy_clean, corrected=False)
        if args.corrected_clean:
            verify_reference_clean(cleaned, args.corrected_clean, corrected=True)
        print('PASS: AptaNet numerical descriptors were preserved unchanged')


if __name__ == '__main__':
    main()
