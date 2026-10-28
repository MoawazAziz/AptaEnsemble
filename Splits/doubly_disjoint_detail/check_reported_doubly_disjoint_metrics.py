#!/usr/bin/env python3
"""Check whether a prediction file reproduces the rounded Table 8 doubly-disjoint metrics."""
import argparse, json, pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,roc_auc_score,average_precision_score,confusion_matrix
from pathlib import Path
TARGET={'accuracy':0.7506,'precision':0.8,'recall':0.0784,'f1':0.1429,'mcc':0.1980,'roc_auc':0.5589}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--file',required=True);args=ap.parse_args()
    d=pd.read_csv(args.file); y=d.y_true.astype(int); yp=d.y_pred.astype(int)
    out={'n':len(d),'accuracy':accuracy_score(y,yp),'precision':precision_score(y,yp,zero_division=0),'recall':recall_score(y,yp,zero_division=0),'f1':f1_score(y,yp,zero_division=0),'mcc':matthews_corrcoef(y,yp)}
    if 'probability' in d: out['roc_auc']=roc_auc_score(y,d.probability);out['pr_auc']=average_precision_score(y,d.probability)
    out['confusion_matrix']=confusion_matrix(y,yp,labels=[0,1]).tolist()
    print(json.dumps(out,indent=2))
    print('Target values are from the manuscript Table 8. Exact reproduction requires the recovered final partition and final fitted configuration.')
if __name__=='__main__':main()
