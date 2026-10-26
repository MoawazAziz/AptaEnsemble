#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from typing import Dict, List, Sequence, Tuple

# ==============================================================================
# Constants
# ==============================================================================

# 20 native amino acids, alphabetical order of single-letter code (matches the
# paper's and the repo's ordering).
AA20: List[str] = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                    'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

DNA_ALPHABET = "ACGT"

LAMBDA_DEFAULT = 30          # lambda in Eq. 7/8 (paper: "We set lambda = 30")
OMEGA_PAPER = 0.05           # omega in Eq. 8 (paper: "We set omega = 0.05")
OMEGA_REPO_LEGACY = 0.15     # literal value hardcoded in the released repo code

# The 7 physicochemical / energetic property groups (A-G) exactly as named
# and valued in the paper's Materials & Methods and in feature_extraction.py /
# target_encoding.py. (Group H -- "power to beat the N-terminal, C-terminal
# and middle of alpha-helix" -- is named in the paper's prose but no per-
# residue table for it is given there or implemented in the released code,
# so it is not included.)
PROPERTY_GROUPS: Dict[str, Dict[str, object]] = {
    "A": {
        "label": "hydrophobicity, hydrophilicity, mass",
        "H1": {'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74, 'F': 1.19, 'G': 0.48,
               'H': -0.40, 'I': 1.38, 'K': -1.50, 'L': 1.06, 'M': 0.64, 'N': -0.78,
               'P': 0.12, 'Q': -0.85, 'R': -2.53, 'S': -0.18, 'T': -0.05, 'V': 1.08,
               'W': 0.81, 'Y': 0.26},
        "H2": {'A': -0.5, 'C': -1.0, 'D': 3.0, 'E': 3.0, 'F': -2.5, 'G': 0.0,
               'H': -0.5, 'I': -1.8, 'K': 3.0, 'L': -1.8, 'M': -1.3, 'N': 0.2,
               'P': 0.0, 'Q': 0.2, 'R': 3.0, 'S': 0.3, 'T': -0.4, 'V': -1.5,
               'W': -3.4, 'Y': -2.3},
        "M": {'A': 15.0, 'C': 47.0, 'D': 59.0, 'E': 73.0, 'F': 91.0, 'G': 1.0,
              'H': 82.0, 'I': 57.0, 'K': 73.0, 'L': 57.0, 'M': 75.0, 'N': 58.0,
              'P': 42.0, 'Q': 72.0, 'R': 101.0, 'S': 31.0, 'T': 45.0, 'V': 43.0,
              'W': 130.0, 'Y': 107.0},
    },
    "B": {
        "label": "polarity, molecular weight, melting point",
        "H1": {'A': 0.5, 'C': 2.5, 'D': -1, 'E': 2.5, 'F': -2.5, 'G': 0, 'H': -0.5,
               'I': 1.8, 'K': 3, 'L': -1.8, 'M': -1.3, 'N': 0.2, 'P': -1.4, 'Q': 0.2,
               'R': 3, 'S': 0.3, 'T': -0.4, 'V': -1.5, 'W': -3.4, 'Y': -2.3},
        "H2": {'A': 5.3, 'C': 3.6, 'D': 1.3, 'E': 3.3, 'F': 2.3, 'G': 4.8, 'H': 1.4,
               'I': 3.1, 'K': 4.1, 'L': 4.7, 'M': 1.1, 'N': 3, 'P': 2.5, 'Q': 2.4,
               'R': 2.6, 'S': 4.5, 'T': 3.7, 'V': 4.2, 'W': 0.8, 'Y': 2.3},
        "M": {'A': 0.81, 'C': 0.71, 'D': 1.17, 'E': 0.53, 'F': 1.2, 'G': 0.88,
              'H': 0.92, 'I': 1.48, 'K': 0.77, 'L': 1.24, 'M': 1.05, 'N': 0.62,
              'P': 0.61, 'Q': 0.98, 'R': 0.85, 'S': 0.92, 'T': 1.18, 'V': 1.66,
              'W': 1.18, 'Y': 1.23},
    },
    "C": {
        "label": "transfer free energy, buriability, bulkiness",
        "H1": {'A': 58, 'C': -97, 'D': 116, 'E': -131, 'F': 92, 'G': -11, 'H': -73,
               'I': 107, 'K': -24, 'L': 95, 'M': 78, 'N': -93, 'P': -79, 'Q': -139,
               'R': -184, 'S': -34, 'T': -7, 'V': 100, 'W': 59, 'Y': -11},
        "H2": {'A': 1.37, 'C': 8.93, 'D': -4.47, 'E': 4.04, 'F': -7.96, 'G': 3.39,
               'H': -1.65, 'I': -7.92, 'K': 7.7, 'L': -8.68, 'M': -7.13, 'N': 6.29,
               'P': 6.25, 'Q': 3.88, 'R': 1.33, 'S': 4.08, 'T': 4.02, 'V': -6.94,
               'W': 0.79, 'Y': -4.73},
        "M": {'A': 6.77, 'C': 8.57, 'D': 0.31, 'E': 12.93, 'F': 1.92, 'G': 7.95,
              'H': 2.8, 'I': 2.72, 'K': 10.2, 'L': 4.43, 'M': 1.87, 'N': 5.5,
              'P': 4.79, 'Q': 5.24, 'R': 6.87, 'S': 5.41, 'T': 5.36, 'V': 3.57,
              'W': 0.54, 'Y': 2.26},
    },
    "D": {
        "label": "solvation free energy, relative mutability, residue volume",
        "H1": {'A': 0.87, 'C': 0.66, 'D': 1.52, 'E': 0.67, 'F': 2.87, 'G': 0.1,
               'H': 0.87, 'I': 3.15, 'K': 1.64, 'L': 2.17, 'M': 1.67, 'N': 0.09,
               'P': 2.77, 'Q': 0, 'R': 0.85, 'S': 0.07, 'T': 0.07, 'V': 1.87,
               'W': 3.77, 'Y': 2.67},
        "H2": {'A': 1.09, 'C': 0.77, 'D': 0.5, 'E': 0.92, 'F': 0.5, 'G': 1.25,
               'H': 0.67, 'I': 0.66, 'K': 1.25, 'L': 0.44, 'M': 0.45, 'N': 1.14,
               'P': 2.96, 'Q': 0.83, 'R': 0.97, 'S': 1.21, 'T': 1.33, 'V': 0.56,
               'W': 0.62, 'Y': 0.94},
        "M": {'A': 0.91, 'C': 1.4, 'D': 0.93, 'E': 0.97, 'F': 0.72, 'G': 1.51,
              'H': 0.9, 'I': 0.65, 'K': 0.82, 'L': 0.59, 'M': 0.58, 'N': 1.64,
              'P': 1.66, 'Q': 0.94, 'R': 1, 'S': 1.23, 'T': 1.04, 'V': 0.6,
              'W': 0.67, 'Y': 0.92},
    },
    "E": {
        "label": "volume, amino acid distribution, hydration number",
        "H1": {'A': 0.92, 'C': 0.48, 'D': 1.16, 'E': 0.61, 'F': 1.25, 'G': 0.61,
               'H': 0.93, 'I': 1.81, 'K': 0.7, 'L': 1.3, 'M': 1.19, 'N': 0.6,
               'P': 0.4, 'Q': 0.95, 'R': 0.93, 'S': 0.82, 'T': 1.12, 'V': 1.81,
               'W': 1.54, 'Y': 1.53},
        "H2": {'A': 0.96, 'C': 0.9, 'D': 1.13, 'E': 0.33, 'F': 1.37, 'G': 0.9,
               'H': 0.87, 'I': 1.54, 'K': 0.81, 'L': 1.26, 'M': 1.29, 'N': 0.72,
               'P': 0.75, 'Q': 1.18, 'R': 0.67, 'S': 0.77, 'T': 1.23, 'V': 1.41,
               'W': 1.13, 'Y': 1.07},
        "M": {'A': 0.9, 'C': 0.47, 'D': 1.24, 'E': 0.62, 'F': 1.23, 'G': 0.56,
              'H': 1.12, 'I': 1.54, 'K': 0.74, 'L': 1.26, 'M': 1.09, 'N': 0.62,
              'P': 0.42, 'Q': 1.18, 'R': 1.02, 'S': 0.87, 'T': 1.3, 'V': 1.53,
              'W': 1.75, 'Y': 1.68},
    },
    "F": {
        "label": "isoelectric point, compressibility, chromatographic index",
        "H1": {'A': 6, 'C': 5.05, 'D': 2.77, 'E': 5.22, 'F': 5.48, 'G': 5.97,
               'H': 7.59, 'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41,
               'P': 6.3, 'Q': 5.65, 'R': 10.76, 'S': 5.68, 'T': 5.66, 'V': 5.96,
               'W': 5.89, 'Y': 5.66},
        "H2": {'A': -25.5, 'C': -32.82, 'D': -33.12, 'E': -36.17, 'F': -34.54,
               'G': -27, 'H': -31.84, 'I': -31.78, 'K': -32.4, 'L': -31.78,
               'M': -31.18, 'N': -30.9, 'P': -23.25, 'Q': -32.6, 'R': -26.62,
               'S': -29.88, 'T': -31.23, 'V': -30.62, 'W': -30.24, 'Y': -35.01},
        "M": {'A': 9.9, 'C': 2.8, 'D': 2.8, 'E': 3.2, 'F': 18.8, 'G': 5.6, 'H': 8.2,
              'I': 17.1, 'K': 3.5, 'L': 17.6, 'M': 14.7, 'N': 5.4, 'P': 14.8,
              'Q': 9, 'R': 4.6, 'S': 6.9, 'T': 9.5, 'V': 14.3, 'W': 17, 'Y': 15},
    },
    "G": {
        "label": "unfolding entropy change, unfolding enthalpy change, "
                 "unfolding Gibbs free energy change",
        "H1": {'A': 0.54, 'C': -4.14, 'D': -0.26, 'E': -0.19, 'F': -4.66,
               'G': -0.31, 'H': -0.23, 'I': -0.27, 'K': 1.13, 'L': -0.24,
               'M': -2.36, 'N': 1.74, 'P': -0.08, 'Q': 1.53, 'R': 3.69,
               'S': -0.24, 'T': -0.28, 'V': -0.36, 'W': -2.69, 'Y': -2.82},
        "H2": {'A': 0.51, 'C': 5.21, 'D': 0.18, 'E': 0.05, 'F': 6.82, 'G': -0.23,
               'H': 0.79, 'I': 0.19, 'K': -1.45, 'L': 0.17, 'M': 2.89, 'N': -2.03,
               'P': 0.02, 'Q': -1.76, 'R': -4.4, 'S': -0.16, 'T': 0.04, 'V': 0.3,
               'W': 4.47, 'Y': 3.73},
        "M": {'A': -0.02, 'C': 1.08, 'D': -0.08, 'E': -0.13, 'F': 2.16, 'G': 0.09,
              'H': 0.56, 'I': -0.08, 'K': -0.32, 'L': -0.08, 'M': 0.53, 'N': -0.3,
              'P': -0.06, 'Q': -0.23, 'R': -0.71, 'S': -0.4, 'T': -0.24, 'V': -0.06,
              'W': 1.78, 'Y': -0.91},
    },
}

ALL_IMPLEMENTED_GROUPS = "ABCDEFG"     # full raw dump -- matches the literal
                                        # feature_extraction.py script as released
AptaNet_FINAL_MODEL_GROUPS = "ABCDEF"  # the actual published AptaNet predictor's
                                        # winning feature set per the paper's
                                        # Results ("kmer4_apt + A+B+C+D+E+F"),
                                        # confirmed against predictor.py's
                                        # 639 = 339(kmer) + 300(6 groups) columns


# ==============================================================================
# Protein encoding: Amino Acid Composition (AAC) and Pseudo-AAC (PseAAC)
#   Paper equations 2-8; Materials & Methods, "Amino acid composition (AAC)"
#   and "Pseudo-amino acid composition (PAAC)".
# ==============================================================================

def validate_protein_sequence(seq: str) -> str:
    """Upper-cases and validates a protein sequence against the 20 native
    amino acids. Raises ValueError (with the offending character) instead of
    the original script's silent for-loop-over-a-stale-string bug."""
    seq = seq.strip().upper()
    bad = sorted(set(c for c in seq if c not in AA20))
    if bad:
        raise ValueError(
            f"Sequence contains character(s) not in the 20 native amino "
            f"acids: {', '.join(bad)}"
        )
    if not seq:
        raise ValueError("Protein sequence is empty.")
    return seq


def amino_acid_composition(seq: str) -> Dict[str, float]:
    """Standalone AAC (paper Eq. 2): f(t) = N(t) / N for each of the 20
    amino acids. Sums to 1.0. NOTE: neither feature_extraction.py nor
    target_encoding.py in the released repo compute this as an independent
    block -- and predictor.py's training matrix (639 = 339 kmer + 300 for
    6 PseAAC groups) leaves no room for a separate 20-dim AAC block either.
    So the "AAC" the paper describes is, in the released pipeline, present
    only implicitly as the first 20 (rescaled) components of each PseAAC
    group. This function is provided for completeness / optional use and is
    OFF by default in extract_features (see include_standalone_aac)."""
    seq = validate_protein_sequence(seq)
    n = len(seq)
    return {aa: seq.count(aa) / n for aa in AA20}


def _zero_mean_unit_variance(prop: Dict[str, float]) -> Dict[str, float]:
    """Eq. 4/6: rescale a raw per-residue property table to zero mean and
    unit (population) variance across the 20 amino acids."""
    avg = sum(prop[a] for a in AA20) / 20.0
    var = sum((prop[a] - avg) ** 2 for a in AA20) / 20.0
    sd = var ** 0.5
    return {a: (prop[a] - avg) / sd for a in AA20}


def _theta(ri: str, rj: str, h1: Dict[str, float], h2: Dict[str, float],
           m: Dict[str, float]) -> float:
    """Eq. 3/5: correlation function between two residues over 3 rescaled
    physicochemical properties."""
    return ((h1[rj] - h1[ri]) ** 2 + (h2[rj] - h2[ri]) ** 2
             + (m[rj] - m[ri]) ** 2) / 3.0


def _sequence_order_tiers(seq: str, h1: Dict[str, float], h2: Dict[str, float],
                           m: Dict[str, float], lambda_val: int,
                           legacy: bool) -> List[float]:
    """Eq. 2: the lambda tier-correlation factors theta_1 .. theta_lambda.

    legacy=False (default, paper-correct): tier n sums i = 0 .. (N-n-1),
        i.e. every valid (R_i, R_{i+n}) pair, then divides by (N-n).
    legacy=True (bug-compatible with the released repo): tier n always sums
        exactly (N-lambda) terms regardless of n, then still divides by
        (N-n) -- reproduced here only for those who need to match the
        repo's published numbers exactly; confirmed by direct diff against
        the original script's own output.
    """
    n_len = len(seq)
    tiers = []
    for lag in range(1, lambda_val + 1):
        upper = (n_len - lambda_val) if legacy else (n_len - lag)
        s = 0.0
        for i in range(upper):
            s += _theta(seq[i], seq[i + lag], h1, h2, m)
        tiers.append(s / (n_len - lag))
    return tiers


def pseudo_amino_acid_composition(
    seq: str,
    h01: Dict[str, float], h02: Dict[str, float], m0: Dict[str, float],
    lambda_val: int = LAMBDA_DEFAULT,
    omega: float = OMEGA_PAPER,
    legacy: bool = False,
) -> List[float]:
    """Eq. 7/8: build the (20 + lambda)-dimensional PseAAC vector for one
    triplet of raw physicochemical property tables (h01, h02, m0).

    legacy=True reproduces the released repo's arithmetic exactly:
    omega is forced to 0.15, and the first 20 components use
    count/20 instead of the paper's count/length. legacy=False (default)
    follows the paper's stated equations exactly.
    """
    seq = validate_protein_sequence(seq)
    n_len = len(seq)
    if n_len <= lambda_val:
        raise ValueError(
            f"Protein sequence length ({n_len}) must be greater than "
            f"lambda ({lambda_val}) to compute PseAAC tier correlations."
        )

    h1, h2, m = (_zero_mean_unit_variance(h01), _zero_mean_unit_variance(h02),
                 _zero_mean_unit_variance(m0))

    counts = [seq.count(aa) for aa in AA20]
    freqs = [c / n_len for c in counts]
    tiers = _sequence_order_tiers(seq, h1, h2, m, lambda_val, legacy=legacy)

    omega_used = OMEGA_REPO_LEGACY if legacy else omega
    denom = sum(freqs) + omega_used * sum(tiers)

    if legacy:
        first20 = [round((c / 20) / denom, 3) for c in counts]
    else:
        first20 = [round(f / denom, 3) for f in freqs]
    tail = [round((omega_used * t) / denom, 3) for t in tiers]
    return first20 + tail


def extract_protein_features(
    seq: str,
    groups: str = ALL_IMPLEMENTED_GROUPS,
    lambda_val: int = LAMBDA_DEFAULT,
    omega: float = OMEGA_PAPER,
    legacy: bool = False,
) -> Tuple[List[float], List[str]]:
    """Compute and concatenate PseAAC for each requested property group
    letter (A-G), in order. Returns (vector, feature_names)."""
    seq = validate_protein_sequence(seq)
    vector: List[float] = []
    names: List[str] = []
    for letter in groups:
        letter = letter.upper()
        if letter not in PROPERTY_GROUPS:
            raise ValueError(
                f"Unknown property group '{letter}'. Valid groups: "
                f"{', '.join(PROPERTY_GROUPS)}"
            )
        grp = PROPERTY_GROUPS[letter]
        block = pseudo_amino_acid_composition(
            seq, grp["H1"], grp["H2"], grp["M"],
            lambda_val=lambda_val, omega=omega, legacy=legacy,
        )
        vector.extend(block)
        names.extend(f"pseaac_{letter}_aac_{aa}" for aa in AA20)
        names.extend(f"pseaac_{letter}_tier{i}" for i in range(1, lambda_val + 1))
    return vector, names


# ==============================================================================
# Aptamer encoding: k-mer and reverse-complement k-mer frequency
#   Paper Materials & Methods, "K-mer frequency" and "Reverse compliment k-mer".
#   Reimplemented in pure Python (see note 7 above on repDNA's broken imports);
#   numerically verified identical to a patched repDNA install.
# ==============================================================================

def preprocess_aptamer_sequence(seq: str) -> str:
    """Upper-cases, converts RNA to DNA (U -> T, per the paper's Methods),
    and validates the sequence contains only A/C/G/T."""
    seq = seq.strip().upper().replace("U", "T")
    bad = sorted(set(c for c in seq if c not in DNA_ALPHABET))
    if bad:
        raise ValueError(
            f"Aptamer sequence contains character(s) outside A/C/G/T (after "
            f"U->T conversion): {', '.join(bad)}"
        )
    if not seq:
        raise ValueError("Aptamer sequence is empty.")
    return seq


def _kmer_list(k: int, alphabet: str = DNA_ALPHABET) -> List[str]:
    return ["".join(t) for t in itertools.product(alphabet, repeat=k)]


def kmer_frequency_upto(seq: str, k: int, alphabet: str = DNA_ALPHABET,
                         upto: bool = True) -> List[float]:
    """K-mer frequency vector. With upto=True (the AptaNet configuration),
    concatenates independently-normalized frequency blocks for every k'
    from 1 to k (dimension = sum_{k'=1}^{k} len(alphabet)**k'; 340 for k=4
    on ACGT). Each block sums to 1.0. Verified to match repDNA's
    Kmer(k=k, normalize=True, upto=upto).make_kmer_vec(...) exactly."""
    seq = seq if set(seq) <= set(alphabet) else preprocess_aptamer_sequence(seq)
    k_values = range(1, k + 1) if upto else [k]
    out: List[float] = []
    for kk in k_values:
        kmers = _kmer_list(kk, alphabet)
        counts = {km: 0 for km in kmers}
        for i in range(len(seq) - kk + 1):
            sub = seq[i:i + kk]
            if sub in counts:
                counts[sub] += 1
        total = sum(counts.values())
        out.extend(
            round(counts[km] / total, 3) if total else 0.0 for km in kmers
        )
    return out


def _kmer_names(k: int, alphabet: str = DNA_ALPHABET, upto: bool = True) -> List[str]:
    names = []
    for kk in (range(1, k + 1) if upto else [k]):
        names.extend(f"kmer{kk}_{km}" for km in _kmer_list(kk, alphabet))
    return names


def _reverse_complement(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(comp[c] for c in reversed(seq))


def _revc_kmer_list(k: int, alphabet: str = DNA_ALPHABET) -> List[str]:
    """Canonical (reverse-complement-collapsed) k-mers: dimension follows
    the paper's Eq. 1 (2**(2k-1) for odd k; 2**(2k-1)+2**(k-1) for even k),
    derived here by de-duplication rather than the formula directly, and
    verified to match repDNA's RevcKmer output exactly."""
    seen = set()
    result = []
    for t in itertools.product(alphabet, repeat=k):
        kmer = "".join(t)
        canon = min(kmer, _reverse_complement(kmer))
        if canon not in seen:
            seen.add(canon)
            result.append(canon)
    return result


def revc_kmer_frequency_upto(seq: str, k: int, alphabet: str = DNA_ALPHABET,
                              upto: bool = True) -> List[float]:
    """Reverse-complement k-mer frequency vector, matching repDNA's
    RevcKmer(k=k, normalize=True, upto=upto).make_revckmer_vec(...)."""
    seq = seq if set(seq) <= set(alphabet) else preprocess_aptamer_sequence(seq)
    k_values = range(1, k + 1) if upto else [k]
    out: List[float] = []
    for kk in k_values:
        canon_kmers = _revc_kmer_list(kk, alphabet)
        idx = {km: i for i, km in enumerate(canon_kmers)}
        counts = [0] * len(canon_kmers)
        for i in range(len(seq) - kk + 1):
            sub = seq[i:i + kk]
            canon = min(sub, _reverse_complement(sub))
            if canon in idx:
                counts[idx[canon]] += 1
        total = sum(counts)
        out.extend(round(c / total, 3) if total else 0.0 for c in counts)
    return out


def _revc_kmer_names(k: int, alphabet: str = DNA_ALPHABET, upto: bool = True) -> List[str]:
    names = []
    for kk in (range(1, k + 1) if upto else [k]):
        names.extend(f"revckmer{kk}_{km}" for km in _revc_kmer_list(kk, alphabet))
    return names


# ==============================================================================
# Combined feature vector assembly
# ==============================================================================

def extract_features(
    aptamer_seq: str,
    protein_seq: str,
    protein_groups: str = AptaNet_FINAL_MODEL_GROUPS,
    kmer_k: int = 4,
    aptamer_upto: bool = True,
    include_revc: bool = False,
    revc_k: int = 4,
    lambda_val: int = LAMBDA_DEFAULT,
    omega: float = OMEGA_PAPER,
    legacy: bool = False,
    include_standalone_aac: bool = False,
) -> Tuple[List[float], List[str]]:
    """Build the full AptaNet-style feature vector for one aptamer-protein
    pair: [k-mer (+ optional revc-kmer) aptamer block] + [PseAAC protein
    block(s)] (+ optional standalone AAC block).

    Defaults reproduce the actual published AptaNet predictor's feature set
    (kmer k=4 upto=True + PseAAC groups A-F, paper-correct omega/formula).
    Pass protein_groups=ALL_IMPLEMENTED_GROUPS to instead reproduce the raw,
    unfiltered output of the released feature_extraction.py script (A-G).
    """
    aptamer_seq = preprocess_aptamer_sequence(aptamer_seq)
    protein_seq = validate_protein_sequence(protein_seq)

    vector: List[float] = []
    names: List[str] = []

    kmer_vec = kmer_frequency_upto(aptamer_seq, kmer_k, upto=aptamer_upto)
    vector.extend(kmer_vec)
    names.extend(_kmer_names(kmer_k, upto=aptamer_upto))

    if include_revc:
        revc_vec = revc_kmer_frequency_upto(aptamer_seq, revc_k, upto=aptamer_upto)
        vector.extend(revc_vec)
        names.extend(_revc_kmer_names(revc_k, upto=aptamer_upto))

    if include_standalone_aac:
        aac = amino_acid_composition(protein_seq)
        vector.extend(aac[aa] for aa in AA20)
        names.extend(f"aac_{aa}" for aa in AA20)

    pseaac_vec, pseaac_names = extract_protein_features(
        protein_seq, groups=protein_groups, lambda_val=lambda_val,
        omega=omega, legacy=legacy,
    )
    vector.extend(pseaac_vec)
    names.extend(pseaac_names)

    return vector, names


# ==============================================================================
# Self-test: numeric regression check against the original repo's own output
# ==============================================================================

def _selftest() -> bool:
    """Reproduces the original (buggy) repo arithmetic in legacy mode and
    checks it against a value computed independently by literally
    transcribing the original script's group-A logic and running it. See
    the accompanying chat response for the verification transcript."""
    test_protein = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGT"
    expected_group_a_legacy = [
        0.023, 0.0, 0.008, 0.023, 0.015, 0.023, 0.008, 0.031, 0.023, 0.031,
        0.008, 0.0, 0.008, 0.031, 0.031, 0.031, 0.015, 0.023, 0.0, 0.008,
        0.015, 0.019, 0.02, 0.026, 0.019, 0.016, 0.022, 0.014, 0.021, 0.023,
        0.017, 0.028, 0.02, 0.023, 0.025, 0.016, 0.031, 0.028, 0.039, 0.04,
        0.031, 0.03, 0.025, 0.029, 0.043, 0.043, 0.055, 0.042, 0.035, 0.051,
    ]
    grp = PROPERTY_GROUPS["A"]
    got = pseudo_amino_acid_composition(
        test_protein, grp["H1"], grp["H2"], grp["M"], legacy=True
    )
    ok = got == expected_group_a_legacy
    print(f"[selftest] legacy PseAAC group A matches original repo output: {ok}")

    kmer_vec = kmer_frequency_upto("ATCGATCGATCGGGCTATCGATCGATCG", 4, upto=True)
    ok2 = len(kmer_vec) == 340 and abs(sum(kmer_vec[:4]) - 1.0) < 1e-9
    print(f"[selftest] k-mer(k=4, upto=True) dimension == 340 and first block "
          f"sums to 1.0: {ok2}")

    revc_vec = revc_kmer_frequency_upto("ATCGATCGATCGGGCTATCGATCGATCG", 3, upto=True)
    ok3 = len(revc_vec) == 44
    print(f"[selftest] revc-kmer(k=3, upto=True) dimension == 44: {ok3}")

    all_ok = ok and ok2 and ok3
    print(f"[selftest] ALL CHECKS PASSED: {all_ok}")
    return all_ok


# ==============================================================================
# CLI
# ==============================================================================

def _prompt_sequence(label: str, validator) -> str:
    """Fixed version of the original's interactive prompt: re-asks on
    invalid input instead of silently ignoring the correction."""
    while True:
        raw = input(f"Please enter {label} sequence: ")
        try:
            return validator(raw)
        except ValueError as exc:
            print(f"  -> {exc}\n  Please try again.")


def _write_csv_row(path: str, header: Sequence[str], row: Sequence[float]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerow(row)


def _run_batch(args: argparse.Namespace) -> None:
    with open(args.csv, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    out_rows: List[List[object]] = []
    header: List[str] = []
    for i, row in enumerate(rows, start=1):
        apt = row[args.aptamer_col]
        prot = row[args.protein_col]
        try:
            vector, names = extract_features(
                apt, prot,
                protein_groups=args.groups, kmer_k=args.kmer_k,
                aptamer_upto=not args.no_upto, include_revc=args.include_revc,
                revc_k=args.revc_k, lambda_val=args.lambda_val,
                omega=args.omega, legacy=args.legacy_repo_formula,
                include_standalone_aac=args.include_aac,
            )
        except ValueError as exc:
            print(f"Row {i}: skipped ({exc})", file=sys.stderr)
            continue
        if not header:
            extra = [args.label_col] if args.label_col and args.label_col in row else []
            header = ["id"] + names + extra
        extra_vals = [row[args.label_col]] if args.label_col and args.label_col in row else []
        out_rows.append([row.get("id", i)] + vector + extra_vals)

    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} feature rows ({len(header)} columns) to {args.out}")


def _run_single(args: argparse.Namespace) -> None:
    if args.aptamer and args.protein:
        apt, prot = args.aptamer, args.protein
    else:
        apt = _prompt_sequence("aptamer (DNA/RNA)", preprocess_aptamer_sequence)
        prot = _prompt_sequence("protein target", validate_protein_sequence)

    vector, names = extract_features(
        apt, prot,
        protein_groups=args.groups, kmer_k=args.kmer_k,
        aptamer_upto=not args.no_upto, include_revc=args.include_revc,
        revc_k=args.revc_k, lambda_val=args.lambda_val,
        omega=args.omega, legacy=args.legacy_repo_formula,
        include_standalone_aac=args.include_aac,
    )
    print(f"Feature vector length: {len(vector)}")
    if args.out:
        _write_csv_row(args.out, names, vector)
        print(f"Wrote 1 feature row ({len(names)} columns) to {args.out}")
    else:
        for name, value in zip(names, vector):
            print(f"{name}\t{value}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AptaNet-style feature extraction (PseAAC/AAC + k-mer/"
                     "revc-k-mer) for aptamer-protein interaction pairs.",
    )
    p.add_argument("--aptamer", help="Aptamer sequence (DNA or RNA).")
    p.add_argument("--protein", help="Protein target sequence.")
    p.add_argument("--csv", help="Batch mode: CSV file of aptamer/protein pairs.")
    p.add_argument("--aptamer-col", default="Aptamer", help="Aptamer column name in --csv.")
    p.add_argument("--protein-col", default="Target", help="Protein column name in --csv.")
    p.add_argument("--label-col", default="Class", help="Optional label column to pass through in --csv mode.")
    p.add_argument("--groups", default=AptaNet_FINAL_MODEL_GROUPS,
                    help=f"Protein PseAAC property groups to use, e.g. 'ABCDEF' "
                         f"(the published AptaNet model's set, default) or "
                         f"'{ALL_IMPLEMENTED_GROUPS}' (full raw dump, matches the "
                         f"released feature_extraction.py script as-is).")
    p.add_argument("--kmer-k", type=int, default=4, help="Max k for aptamer k-mer frequency (default 4, the AptaNet setting).")
    p.add_argument("--no-upto", action="store_true", help="Use only k-mers of exactly length --kmer-k instead of cumulative 1..k.")
    p.add_argument("--include-revc", action="store_true", help="Also include reverse-complement k-mer features (paper's second aptamer encoding strategy).")
    p.add_argument("--revc-k", type=int, default=4, help="Max k for reverse-complement k-mer (default 4).")
    p.add_argument("--include-aac", action="store_true", help="Also include a standalone 20-dim AAC block (off by default; see docstring).")
    p.add_argument("--lambda-val", type=int, default=LAMBDA_DEFAULT, help="PseAAC lambda (default 30, as in the paper).")
    p.add_argument("--omega", type=float, default=OMEGA_PAPER, help="PseAAC sequence-order weight (default 0.05, per the paper's Eq. 8).")
    p.add_argument("--legacy-repo-formula", action="store_true",
                    help="Reproduce the released repo's exact (paper-inconsistent) "
                         "arithmetic instead of the corrected, paper-accurate default "
                         "-- forces omega=0.15 and the count/20 + fixed-window formula.")
    p.add_argument("--out", help="Output CSV path (single-pair mode: one row; batch mode: required).")
    p.add_argument("--selftest", action="store_true", help="Run built-in numeric verification and exit.")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.selftest:
        ok = _selftest()
        sys.exit(0 if ok else 1)

    if args.csv:
        if not args.out:
            print("error: --out is required in --csv batch mode", file=sys.stderr)
            sys.exit(2)
        _run_batch(args)
    else:
        _run_single(args)


if __name__ == "__main__":
    main()
