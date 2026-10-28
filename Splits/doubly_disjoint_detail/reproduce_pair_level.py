#!/usr/bin/env python3
"""Reproduce the current manuscript's pair-level 94.34% result.

This script uses the exact pair-level development/M3 files and the fixed
threshold 0.4300 reported in the current manuscript.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"data"/"source"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--threshold",type=float,default=0.4300)
    args=ap.parse_args()

    tr=pd.read_csv(SRC/"pair_level_development_2402.csv")
    te=pd.read_csv(SRC/"pair_level_test_M3_601.csv")
    X=tr.iloc[:,1:].to_numpy(float); y=tr.iloc[:,0].to_numpy(int)
    Xt=te.iloc[:,1:].to_numpy(float); yt=te.iloc[:,0].to_numpy(int)

    model=ExtraTreesClassifier(
        n_estimators=500, criterion="entropy", max_depth=None,
        min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
        bootstrap=False, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X,y)
    p=model.predict_proba(Xt)[:,1]
    pred=(p>=args.threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(yt,pred).ravel()

    result={
        "threshold":args.threshold,
        "accuracy":accuracy_score(yt,pred),
        "balanced_accuracy":balanced_accuracy_score(yt,pred),
        "precision":precision_score(yt,pred),
        "recall":recall_score(yt,pred),
        "f1":f1_score(yt,pred),
        "mcc":matthews_corrcoef(yt,pred),
        "roc_auc":roc_auc_score(yt,p),
        "pr_auc":average_precision_score(yt,p),
        "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)
    }
    out=ROOT/"results"/"pair_level_reproduction.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
