#!/usr/bin/env python3
"""Validate the frozen final doubly-disjoint assignment against the manuscript-reported endpoint.

This script does NOT create a final partition. It only validates one supplied by the authors.
"""
import argparse, hashlib, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

EXPECTED = {
    "all_rows": 3003,
    "test_rows": 770,
    "test_positive": 204,
    "test_negative": 566,
    "development_rows": 2233,
    "development_positive": 605,
    "development_negative": 1628,
}

def hid(arr,prefix):
    s=",".join(f"{float(v):.17g}" for v in arr)
    return prefix+hashlib.sha256(s.encode()).hexdigest()[:16]

def ids(df):
    x=df.iloc[:,1:].to_numpy(float)
    A=[hid(a,"A_") for a in x[:,:340]]
    P=[hid(p,"P_") for p in x[:,340:]]
    y=df.iloc[:,0].astype(int).tolist()
    pair=[hashlib.sha256((a+"|"+p+"|"+str(l)).encode()).hexdigest()[:20] for a,p,l in zip(A,P,y)]
    return pair,A,P,y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source", default="data/source/clean_3003.csv")
    ap.add_argument("--assignments", default="data/final_doubly_disjoint/doubly_disjoint_assignments.csv")
    ap.add_argument("--predictions", default=None)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    df=pd.read_csv(root/args.source)
    ass=pd.read_csv(root/args.assignments)
    required={"pair_id","aptamer_id","protein_id","label","doubly_disjoint_partition"}
    missing=required-set(ass.columns)
    if missing: raise SystemExit(f"Missing columns: {sorted(missing)}")
    pair,A,P,y=ids(df)
    if set(pair)!=set(ass.pair_id): raise SystemExit("Assignment file does not contain exactly the source pair IDs.")
    if ass.doubly_disjoint_partition.isna().any() or (~ass.doubly_disjoint_partition.isin(["development","test"])).any():
        raise SystemExit("Partition column must contain only development/test.")
    test=ass.doubly_disjoint_partition.eq("test")
    dev=~test
    if int(test.sum())!=EXPECTED["test_rows"]: raise SystemExit(f"Expected {EXPECTED['test_rows']} test rows; got {int(test.sum())}.")
    if int((ass.loc[test,'label']==1).sum())!=EXPECTED["test_positive"]: raise SystemExit("Test positive count mismatch.")
    if int((ass.loc[test,'label']==0).sum())!=EXPECTED["test_negative"]: raise SystemExit("Test negative count mismatch.")
    if int(dev.sum())!=EXPECTED["development_rows"]: raise SystemExit("Development row count mismatch.")
    if int((ass.loc[dev,'label']==1).sum())!=EXPECTED["development_positive"]: raise SystemExit("Development positive count mismatch.")
    if int((ass.loc[dev,'label']==0).sum())!=EXPECTED["development_negative"]: raise SystemExit("Development negative count mismatch.")
    ta=set(ass.loc[test,'aptamer_id']); da=set(ass.loc[dev,'aptamer_id'])
    tp=set(ass.loc[test,'protein_id']); dp=set(ass.loc[dev,'protein_id'])
    if ta & da: raise SystemExit(f"Aptamer overlap detected: {len(ta&da)}")
    if tp & dp: raise SystemExit(f"Protein overlap detected: {len(tp&dp)}")
    print(json.dumps({
      "status":"PASS",
      "test_rows":int(test.sum()),"test_positive":int((ass.loc[test,'label']==1).sum()),
      "test_negative":int((ass.loc[test,'label']==0).sum()),
      "development_rows":int(dev.sum()),"development_positive":int((ass.loc[dev,'label']==1).sum()),
      "development_negative":int((ass.loc[dev,'label']==0).sum()),
      "aptamer_overlap":len(ta&da),"protein_overlap":len(tp&dp)
    },indent=2))
    if args.predictions:
        pr=pd.read_csv(root/args.predictions)
        if not {'pair_id','y_true','y_pred'}.issubset(pr.columns):
            raise SystemExit("Prediction file must have pair_id,y_true,y_pred.")
        yt=pr.y_true.astype(int); yp=pr.y_pred.astype(int)
        print(json.dumps({
          "prediction_accuracy":accuracy_score(yt,yp),
          "prediction_precision":precision_score(yt,yp,zero_division=0),
          "prediction_recall":recall_score(yt,yp,zero_division=0),
          "prediction_f1":f1_score(yt,yp,zero_division=0),
          "prediction_mcc":matthews_corrcoef(yt,yp)
        },indent=2))
if __name__=="__main__": main()
