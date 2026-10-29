#!/usr/bin/env python3
"""Pair-bootstrap uncertainty for reproducibility diagnostics only.

These confidence intervals describe uncertainty within each fixed, already-generated
held-out cohort. They do NOT account for split selection or model refitting, and
must not be presented as replication of historical manuscript confidence intervals.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, recall_score

DESIGNS = [
    'pair-level', 'protein-disjoint', 'aptamer-disjoint',
    'cluster-disjoint-FIXED', 'doubly-disjoint-released'
]

def bootstrap(p, n_boot, seed):
    y = p['y_true'].to_numpy(int)
    probability = p['probability'].to_numpy(float)
    pred = p['y_pred'].to_numpy(int)
    rng = np.random.default_rng(seed)
    vals = {'mcc': [], 'roc_auc': [], 'recall': []}
    for _ in range(n_boot):
        inds = rng.integers(0, len(y), size=len(y))
        yt, pr, ph = y[inds], probability[inds], pred[inds]
        if len(np.unique(yt)) < 2:
            continue
        vals['mcc'].append(matthews_corrcoef(yt, ph))
        vals['roc_auc'].append(roc_auc_score(yt, pr))
        vals['recall'].append(recall_score(yt, ph, zero_division=0))
    return {k+'_ci_'+bound: float(np.percentile(v, q)) for k, v in vals.items()
            for bound, q in [('lo', 2.5), ('hi', 97.5)]}

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--results', type=Path, required=True)
    a.add_argument('--bootstrap', type=int, default=600)
    a.add_argument('--seed', type=int, default=42)
    opt=a.parse_args()
    rows=[]
    for index, name in enumerate(DESIGNS):
        p=pd.read_csv(opt.results/(name+'_predictions.csv'))
        y=p.y_true.to_numpy(int)
        d={'design':name,'n':len(p),'positive':int(y.sum()),
           'mcc':float(matthews_corrcoef(y,p.y_pred)),
           'roc_auc':float(roc_auc_score(y,p.probability)),
           'recall':float(recall_score(y,p.y_pred))}
        d.update(bootstrap(p,opt.bootstrap,opt.seed+index))
        rows.append(d)
        print(f"{name:27s}: MCC {d['mcc']:.4f} [{d['mcc_ci_lo']:.3f}, {d['mcc_ci_hi']:.3f}] AUC {d['roc_auc']:.4f} [{d['roc_auc_ci_lo']:.3f}, {d['roc_auc_ci_hi']:.3f}]",flush=True)
    pd.DataFrame(rows).to_csv(opt.results/'five_designs_diagnostic_bootstrap_ci.csv',index=False)

if __name__=='__main__':main()
