#!/usr/bin/env python3
"""
Pretrained embedding all-model improvement runner for aptamer-protein interaction prediction.

Purpose
-------
This script uses an already-created pretrained pair embedding matrix, e.g.
DNABERT6 aptamer embeddings concatenated with ProtBERT protein embeddings, and evaluates
all requested model families under a leakage-controlled development/M1/M2/M3 protocol:

ML:
  ExtraTrees, RandomForest, LightGBM, XGBoost
ML ensembles:
  SoftVote ML, WeightedVote ML, StackingLR ML
Deep learning:
  MLP_Small, MLP_Deep, ResidualMLP, GatedMLP, CNN1D
ML-DL hybrids:
  ET90/80/70 + each DL model at 10/20/30% DL contribution

Why this version is different
-----------------------------
The first pretrained run was weak mainly because recall on M3 was low. This script tries to
improve that without leaking M3 by:
  1. evaluating multiple threshold objectives from development OOF predictions only
     (mcc, f1, f2, balanced_accuracy, youden);
  2. using stronger class imbalance handling;
  3. using XGBoost, LightGBM, tree ensembles, DL, ML ensembles, and ML-DL hybrids;
  4. standardizing embeddings only inside training folds for models that need scaling;
  5. reporting M1/M2/M3 metrics and confusion counts for every model/objective.

Recommended Colab command
-------------------------
!pip install -q xgboost lightgbm tensorflow scikit-learn pandas numpy joblib

!python /content/run_pretrained_embeddings_all_models_v11.py \
  --embedding_npy /content/pretrained_transformer_results_final/embeddings/DNABERT6__ProtBERT_all_pair_embeddings.npy \
  --metadata_csv /content/pretrained_transformer_results_final/cleaned_sequence_pairs_with_splits.csv \
  --outdir /content/pretrained_all_models_v11 \
  --embedding_combo DNABERT6+ProtBERT \
  --threshold_objectives mcc f1 f2 balanced_accuracy youden \
  --primary_objective mcc

If your split CSV has a different path, use:
  --metadata_csv /content/cleaned_raw_sequence_pairs.csv
The script will regenerate deterministic splits only if no split/cohort column is found.

Outputs
-------
- tables/all_datasets_all_models_all_thresholds.csv
- tables/ranked_M3_all_models_all_thresholds.csv
- tables/ranked_M3_primary_objective.csv
- predictions/all_datasets_all_models_predictions_primary_objective.csv
- models/*.joblib and models/*.keras
"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise RuntimeError("Install xgboost: pip install xgboost") from e

try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise RuntimeError("Install lightgbm: pip install lightgbm") from e

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models, optimizers, regularizers
    from tensorflow.keras import backend as K
except Exception:
    tf = None

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
if tf is not None:
    tf.random.set_seed(SEED)

# -------------------------- utilities --------------------------

def normalize_colname(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def find_label_col(df: pd.DataFrame) -> str:
    candidates = ["class", "label", "y", "target", "interaction", "interacts", "bind", "binding"]
    lower = {normalize_colname(c): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    # fallback: binary numeric column
    for c in df.columns:
        vals = pd.Series(df[c]).dropna().unique()
        if len(vals) <= 3:
            try:
                vv = set(pd.Series(vals).astype(int).tolist())
                if vv.issubset({0, 1}):
                    return c
            except Exception:
                pass
    raise ValueError("Could not find binary label column. Expected Class/label/y/target.")


def find_split_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["split", "cohort", "dataset", "partition", "set"]
    lower = {normalize_colname(c): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def canonical_split_value(x) -> str:
    s = str(x).strip().lower()
    if s in {"train", "training", "development", "dev"}:
        return "development"
    if s in {"m1", "seen", "seen_diagnostic"}:
        return "M1"
    if s in {"m2", "semi", "semi_unseen", "semi-unseen"}:
        return "M2"
    if s in {"m3", "external", "external_unseen", "external-unseen", "test", "c3"}:
        return "M3"
    return s


def prepare_splits(df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    split_col = find_split_col(df)
    if split_col is not None:
        split = df[split_col].map(canonical_split_value).astype(str).values
        if np.isin(split, ["development", "M1", "M2", "M3"]).sum() >= int(0.8 * len(df)):
            df["_split_final"] = split
            # If no explicit M1, M1 is the development partition.
            if "M1" not in set(df["_split_final"]):
                df.loc[df["_split_final"] == "development", "_split_final_m1"] = "M1"
            return df

    # deterministic fallback used only when no split column is present
    idx = np.arange(len(df))
    dev_idx, m3_idx = train_test_split(idx, test_size=0.20, stratify=y, random_state=SEED)
    df["_split_final"] = "unused"
    df.loc[dev_idx, "_split_final"] = "development"
    df.loc[m3_idx, "_split_final"] = "M3"

    # M2 diagnostic: deterministic 600-sample stratified sample from full data when possible
    m2_n = min(600, len(df))
    _, m2_idx = train_test_split(idx, test_size=m2_n, stratify=y, random_state=SEED + 1)
    df["_is_m2"] = False
    df.loc[m2_idx, "_is_m2"] = True
    return df


def get_indices(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    split = df["_split_final"].astype(str).values
    out = {
        "development": np.where(split == "development")[0],
        "M1": np.where(split == "development")[0],
        "M3": np.where(split == "M3")[0],
    }
    if "_is_m2" in df.columns:
        out["M2"] = np.where(df["_is_m2"].astype(bool).values)[0]
    else:
        out["M2"] = np.where(split == "M2")[0]
        if len(out["M2"]) == 0:
            out["M2"] = out["M3"]
    return out


def compute_metrics(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=labels).ravel()
    d = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(thr),
    }
    try:
        d["roc_auc"] = roc_auc_score(y_true, p)
    except Exception:
        d["roc_auc"] = np.nan
    try:
        d["pr_auc"] = average_precision_score(y_true, p)
    except Exception:
        d["pr_auc"] = np.nan
    return d


def threshold_score(y_true: np.ndarray, p: np.ndarray, thr: float, objective: str) -> float:
    pred = (p >= thr).astype(int)
    if objective == "mcc":
        return matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else -1.0
    if objective == "f1":
        return f1_score(y_true, pred, zero_division=0)
    if objective == "f2":
        return fbeta_score(y_true, pred, beta=2, zero_division=0)
    if objective == "balanced_accuracy":
        return balanced_accuracy_score(y_true, pred)
    if objective == "youden":
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        return sens + spec - 1.0
    raise ValueError(f"Unknown threshold objective: {objective}")


def best_threshold(y_true: np.ndarray, p: np.ndarray, objective: str) -> Tuple[float, float]:
    thresholds = np.round(np.arange(0.05, 0.951, 0.005), 3)
    best_t, best_s = 0.5, -1e9
    for t in thresholds:
        s = threshold_score(y_true, p, t, objective)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def class_weight_dict(y: np.ndarray) -> Dict[int, float]:
    n = len(y)
    c0 = max(1, int(np.sum(y == 0)))
    c1 = max(1, int(np.sum(y == 1)))
    return {0: n / (2 * c0), 1: n / (2 * c1)}


def focal_loss(alpha=0.65, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    return loss

# -------------------------- ML models --------------------------

def make_ml_models(y_train: np.ndarray) -> Dict[str, object]:
    c0 = max(1, int(np.sum(y_train == 0)))
    c1 = max(1, int(np.sum(y_train == 1)))
    scale_pos_weight = c0 / c1
    return {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=1200,
            criterion="entropy",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=False,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=1000,
            criterion="entropy",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=False,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=800,
            learning_rate=0.015,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.05,
            reg_lambda=4.0,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=800,
            learning_rate=0.015,
            max_depth=4,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.75,
            gamma=0.05,
            reg_alpha=0.05,
            reg_lambda=4.0,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

# -------------------------- DL models --------------------------

def require_tf():
    if tf is None:
        raise RuntimeError("TensorFlow is required for deep-learning models. Install tensorflow or run --skip_dl.")


def compile_model(model, lr=3e-4, use_focal=True):
    loss = focal_loss(alpha=0.70, gamma=2.0) if use_focal else "binary_crossentropy"
    model.compile(optimizer=optimizers.Adam(lr), loss=loss)
    return model


def build_mlp_small(input_dim: int):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return compile_model(models.Model(inp, out), lr=4e-4)


def build_mlp_deep(input_dim: int):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return compile_model(models.Model(inp, out), lr=3e-4)


def build_residual_mlp(input_dim: int):
    inp = layers.Input(shape=(input_dim,))
    x0 = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(inp)
    x0 = layers.BatchNormalization()(x0)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x0)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(256, activation=None, kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Add()([x0, x])
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return compile_model(models.Model(inp, out), lr=3e-4)


def build_gated_mlp(input_dim: int):
    inp = layers.Input(shape=(input_dim,))
    h = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(inp)
    g = layers.Dense(256, activation="sigmoid")(inp)
    x = layers.Multiply()([h, g])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return compile_model(models.Model(inp, out), lr=3e-4)


def build_cnn1d(input_dim: int):
    inp = layers.Input(shape=(input_dim, 1))
    x = layers.Conv1D(64, kernel_size=7, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(96, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return compile_model(models.Model(inp, out), lr=3e-4)


DL_BUILDERS = {
    "MLP_Small": build_mlp_small,
    "MLP_Deep": build_mlp_deep,
    "ResidualMLP": build_residual_mlp,
    "GatedMLP": build_gated_mlp,
    "CNN1D": build_cnn1d,
}


def dl_predict(name: str, model, X_scaled: np.ndarray) -> np.ndarray:
    if name == "CNN1D":
        return model.predict(X_scaled[..., np.newaxis], verbose=0).ravel()
    return model.predict(X_scaled, verbose=0).ravel()

# -------------------------- core evaluation --------------------------

def add_eval_rows(model_name: str, family: str, probas: Dict[str, np.ndarray], y_by_ds: Dict[str, np.ndarray],
                  thresholds: Dict[str, float], objectives: Iterable[str], rows: List[Dict],
                  embedding_combo: str, feature_source: str, embedding_dim: int):
    for obj in objectives:
        thr = thresholds[obj]
        for ds_name, p in probas.items():
            m = compute_metrics(y_by_ds[ds_name], p, thr)
            m.update({
                "embedding_combo": embedding_combo,
                "feature_source": feature_source,
                "embedding_dim": int(embedding_dim),
                "model": model_name,
                "family": family,
                "dataset": ds_name,
                "threshold_objective": obj,
                "N": int(len(y_by_ds[ds_name])),
            })
            rows.append(m)


def run(args):
    outdir = Path(args.outdir)
    table_dir = outdir / "tables"
    pred_dir = outdir / "predictions"
    model_dir = outdir / "models"
    for d in [table_dir, pred_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    X = np.load(args.embedding_npy)
    df = pd.read_csv(args.metadata_csv)
    if len(df) != X.shape[0]:
        raise ValueError(f"metadata rows ({len(df)}) != embedding rows ({X.shape[0]}). Use the same CSV order used for embedding.")

    label_col = find_label_col(df)
    y = df[label_col].astype(int).values
    df = prepare_splits(df, y)
    idxs = get_indices(df)

    dev_idx = idxs["development"]
    X_train = X[dev_idx]
    y_train = y[dev_idx]

    datasets = {}
    y_by_ds = {}
    for ds in ["M1", "M2", "M3"]:
        ii = idxs[ds]
        if len(ii) == 0:
            continue
        datasets[ds] = X[ii]
        y_by_ds[ds] = y[ii]

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)
    objectives = list(args.threshold_objectives)
    if args.primary_objective not in objectives:
        objectives.append(args.primary_objective)

    all_rows = []
    pred_rows = []
    base_oof: Dict[str, np.ndarray] = {}
    base_probas: Dict[str, Dict[str, np.ndarray]] = {}
    thresholds_all: Dict[str, Dict[str, float]] = {}

    # ML models
    print("\nRunning ML models")
    ml_models = make_ml_models(y_train)
    for name, model_def in ml_models.items():
        print(name)
        if name not in args.models:
            continue

        # LR-like models need scaling; tree models are raw.
        model = clone(model_def)
        oof = np.zeros(len(y_train), dtype=float)
        for tr_idx, va_idx in cv.split(X_train, y_train):
            fold_model = clone(model_def)
            fold_model.fit(X_train[tr_idx], y_train[tr_idx])
            oof[va_idx] = fold_model.predict_proba(X_train[va_idx])[:, 1]

        base_oof[name] = oof
        thresholds_all[name] = {obj: best_threshold(y_train, oof, obj)[0] for obj in objectives}

        final_model = clone(model_def)
        final_model.fit(X_train, y_train)
        dump(final_model, model_dir / f"{name}.joblib")

        probas = {ds: final_model.predict_proba(Xds)[:, 1] for ds, Xds in datasets.items()}
        base_probas[name] = probas
        add_eval_rows(name, "ML", probas, y_by_ds, thresholds_all[name], objectives, all_rows,
                      args.embedding_combo, "pretrained_pair_embeddings", X.shape[1])

    # DL models
    if not args.skip_dl:
        require_tf()
        print("\nRunning DL models")
        input_dim = X_train.shape[1]
        for name, builder in DL_BUILDERS.items():
            if name not in args.models:
                continue
            print(name)
            oof = np.zeros(len(y_train), dtype=float)
            for tr_idx, va_idx in cv.split(X_train, y_train):
                K.clear_session()
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train[tr_idx])
                X_va = scaler.transform(X_train[va_idx])
                y_tr = y_train[tr_idx]

                model = builder(input_dim)
                es = callbacks.EarlyStopping(monitor="val_loss", patience=args.dl_patience,
                                             restore_best_weights=True, min_delta=1e-4)
                rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, args.dl_patience // 2),
                                                 min_lr=1e-5, verbose=0)
                X_fit = X_tr[..., np.newaxis] if name == "CNN1D" else X_tr
                X_val_fit = X_va[..., np.newaxis] if name == "CNN1D" else X_va
                model.fit(
                    X_fit, y_tr,
                    validation_data=(X_val_fit, y_train[va_idx]),
                    epochs=args.dl_epochs,
                    batch_size=args.dl_batch_size,
                    class_weight=class_weight_dict(y_tr),
                    callbacks=[es, rl],
                    verbose=0,
                )
                oof[va_idx] = dl_predict(name, model, X_va)

            base_oof[name] = oof
            thresholds_all[name] = {obj: best_threshold(y_train, oof, obj)[0] for obj in objectives}

            K.clear_session()
            scaler_final = StandardScaler()
            X_full = scaler_final.fit_transform(X_train)
            final_model = builder(input_dim)
            es_final = callbacks.EarlyStopping(monitor="loss", patience=args.dl_patience,
                                               restore_best_weights=True, min_delta=1e-4)
            X_full_fit = X_full[..., np.newaxis] if name == "CNN1D" else X_full
            final_model.fit(
                X_full_fit, y_train,
                epochs=args.dl_epochs,
                batch_size=args.dl_batch_size,
                class_weight=class_weight_dict(y_train),
                callbacks=[es_final],
                verbose=0,
            )
            final_model.save(model_dir / f"{name}.keras")
            dump(scaler_final, model_dir / f"{name}_scaler.joblib")

            probas = {}
            for ds, Xds in datasets.items():
                X_scaled = scaler_final.transform(Xds)
                probas[ds] = dl_predict(name, final_model, X_scaled)
            base_probas[name] = probas
            add_eval_rows(name, "Deep learning", probas, y_by_ds, thresholds_all[name], objectives, all_rows,
                          args.embedding_combo, "pretrained_pair_embeddings", X.shape[1])

    # ML ensembles
    print("\nRunning ML ensembles")
    ml_names = [n for n in ["ExtraTrees", "RandomForest", "LightGBM", "XGBoost"] if n in base_oof]
    if len(ml_names) >= 2:
        # soft vote
        soft_oof = np.mean([base_oof[n] for n in ml_names], axis=0)
        soft_thr = {obj: best_threshold(y_train, soft_oof, obj)[0] for obj in objectives}
        soft_probas = {ds: np.mean([base_probas[n][ds] for n in ml_names], axis=0) for ds in datasets}
        base_oof["SoftVote ML"] = soft_oof
        base_probas["SoftVote ML"] = soft_probas
        thresholds_all["SoftVote ML"] = soft_thr
        add_eval_rows("SoftVote ML", "ML ensemble", soft_probas, y_by_ds, soft_thr, objectives, all_rows,
                      args.embedding_combo, "pretrained_pair_embeddings", X.shape[1])

        # weighted vote based on OOF MCC, clipped to avoid negative weights
        mcc_weights = {}
        for n in ml_names:
            t = best_threshold(y_train, base_oof[n], "mcc")[0]
            mcc = max(0.05, compute_metrics(y_train, base_oof[n], t)["mcc"])
            mcc_weights[n] = mcc
        denom = sum(mcc_weights.values())
        weighted_oof = sum(mcc_weights[n] * base_oof[n] for n in ml_names) / denom
        weighted_thr = {obj: best_threshold(y_train, weighted_oof, obj)[0] for obj in objectives}
        weighted_probas = {ds: sum(mcc_weights[n] * base_probas[n][ds] for n in ml_names) / denom for ds in datasets}
        base_oof["WeightedVote ML"] = weighted_oof
        base_probas["WeightedVote ML"] = weighted_probas
        thresholds_all["WeightedVote ML"] = weighted_thr
        add_eval_rows("WeightedVote ML", "ML ensemble", weighted_probas, y_by_ds, weighted_thr, objectives, all_rows,
                      args.embedding_combo, "pretrained_pair_embeddings", X.shape[1])

        # stacking LR from OOF predictions only
        stack_train_X = np.column_stack([base_oof[n] for n in ml_names])
        stack_meta = LogisticRegression(class_weight="balanced", max_iter=5000, C=0.5, random_state=SEED)
        stack_oof = cross_val_predict(stack_meta, stack_train_X, y_train, cv=cv, method="predict_proba")[:, 1]
        stack_thr = {obj: best_threshold(y_train, stack_oof, obj)[0] for obj in objectives}
        stack_meta.fit(stack_train_X, y_train)
        dump(stack_meta, model_dir / "StackingLR_ML_meta.joblib")
        stack_probas = {}
        for ds in datasets:
            stack_X = np.column_stack([base_probas[n][ds] for n in ml_names])
            stack_probas[ds] = stack_meta.predict_proba(stack_X)[:, 1]
        base_oof["StackingLR ML"] = stack_oof
        base_probas["StackingLR ML"] = stack_probas
        thresholds_all["StackingLR ML"] = stack_thr
        add_eval_rows("StackingLR ML", "ML ensemble", stack_probas, y_by_ds, stack_thr, objectives, all_rows,
                      args.embedding_combo, "pretrained_pair_embeddings", X.shape[1])

    # ML-DL hybrids
    if not args.skip_dl and "ExtraTrees" in base_oof:
        print("\nRunning ML-DL hybrids")
        hybrid_specs = []
        for dl in ["CNN1D", "GatedMLP", "MLP_Small", "MLP_Deep", "ResidualMLP"]:
            label = dl.replace("_", " ")
            if dl == "MLP_Small": label = "MLP Small"
            if dl == "MLP_Deep": label = "MLP Deep"
            for et_w, dl_pct in [(0.90, 10), (0.80, 20), (0.70, 30)]:
                hybrid_specs.append((f"ET{int(et_w*100)} + {label}{dl_pct}", dl, et_w))

        for hybrid_name, dl_name, et_w in hybrid_specs:
            if dl_name not in base_oof:
                continue
            dl_w = 1.0 - et_w
            oof = et_w * base_oof["ExtraTrees"] + dl_w * base_oof[dl_name]
            thr = {obj: best_threshold(y_train, oof, obj)[0] for obj in objectives}
            probas = {ds: et_w * base_probas["ExtraTrees"][ds] + dl_w * base_probas[dl_name][ds] for ds in datasets}
            add_eval_rows(hybrid_name, "ML-DL hybrid", probas, y_by_ds, thr, objectives, all_rows,
                          args.embedding_combo, "pretrained_pair_embeddings", X.shape[1])

    results = pd.DataFrame(all_rows)
    # column order
    front = ["embedding_combo", "feature_source", "embedding_dim", "model", "family", "dataset", "threshold_objective", "N"]
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "f2", "mcc", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp", "threshold"]
    results = results[front + metric_cols]
    results.to_csv(table_dir / "all_datasets_all_models_all_thresholds.csv", index=False)

    ranked = results[results["dataset"] == "M3"].sort_values(["mcc", "f2", "balanced_accuracy", "accuracy"], ascending=False)
    ranked.to_csv(table_dir / "ranked_M3_all_models_all_thresholds.csv", index=False)

    primary = results[(results["dataset"] == "M3") & (results["threshold_objective"] == args.primary_objective)].copy()
    primary = primary.sort_values(["mcc", "f2", "balanced_accuracy", "accuracy"], ascending=False)
    primary.to_csv(table_dir / "ranked_M3_primary_objective.csv", index=False)

    # save primary-objective predictions for reproducibility
    for model_name, probas in base_probas.items():
        if model_name not in thresholds_all:
            continue
        if args.primary_objective not in thresholds_all[model_name]:
            continue
        thr = thresholds_all[model_name][args.primary_objective]
        for ds, p in probas.items():
            yy = y_by_ds[ds]
            pred = (p >= thr).astype(int)
            for i in range(len(yy)):
                pred_rows.append({
                    "embedding_combo": args.embedding_combo,
                    "model": model_name,
                    "dataset": ds,
                    "threshold_objective": args.primary_objective,
                    "threshold": thr,
                    "row_index_in_embedding_file": int(idxs[ds][i]) if ds in idxs and i < len(idxs[ds]) else i,
                    "true_label": int(yy[i]),
                    "probability": float(p[i]),
                    "predicted_label": int(pred[i]),
                })
    pd.DataFrame(pred_rows).to_csv(pred_dir / "all_datasets_all_models_predictions_primary_objective.csv", index=False)

    status = {
        "embedding_npy": str(args.embedding_npy),
        "metadata_csv": str(args.metadata_csv),
        "embedding_combo": args.embedding_combo,
        "embedding_shape": list(X.shape),
        "label_col": label_col,
        "class_counts_all": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()},
        "development_rows": int(len(dev_idx)),
        "M1_rows": int(len(idxs.get("M1", []))),
        "M2_rows": int(len(idxs.get("M2", []))),
        "M3_rows": int(len(idxs.get("M3", []))),
        "threshold_objectives": objectives,
        "primary_objective": args.primary_objective,
        "top_M3_primary_objective": primary.head(10).to_dict(orient="records"),
    }
    with open(outdir / "run_status.json", "w") as f:
        json.dump(status, f, indent=2)

    print("\nDONE")
    print("All results:", table_dir / "all_datasets_all_models_all_thresholds.csv")
    print("Ranked M3:", table_dir / "ranked_M3_all_models_all_thresholds.csv")
    print("Primary ranked M3:", table_dir / "ranked_M3_primary_objective.csv")
    print("Top primary M3:")
    print(primary.head(15).to_string(index=False))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding_npy", required=True, help="Pair embedding matrix .npy; rows must match metadata_csv.")
    ap.add_argument("--metadata_csv", required=True, help="CSV used to create embeddings; must include Class/label and optional split/cohort.")
    ap.add_argument("--outdir", default="pretrained_all_models_v11")
    ap.add_argument("--embedding_combo", default="DNABERT6+ProtBERT")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=[
        "ExtraTrees", "RandomForest", "LightGBM", "XGBoost",
        "MLP_Small", "MLP_Deep", "ResidualMLP", "GatedMLP", "CNN1D",
    ])
    ap.add_argument("--threshold_objectives", nargs="+", default=["mcc", "f1", "f2", "balanced_accuracy", "youden"])
    ap.add_argument("--primary_objective", default="mcc")
    ap.add_argument("--skip_dl", action="store_true")
    ap.add_argument("--dl_epochs", type=int, default=80)
    ap.add_argument("--dl_patience", type=int, default=10)
    ap.add_argument("--dl_batch_size", type=int, default=64)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
