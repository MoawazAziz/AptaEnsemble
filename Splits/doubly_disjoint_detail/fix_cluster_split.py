#!/usr/bin/env python3
"""Create an exact 3-mer/cosine>=0.90 single-linkage cluster-disjoint split.
Outputs are NEW experimental splits and DO NOT reproduce the earlier manuscript Table 8.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold


def split(df,seed=42,fraction=.2,threshold=.9):
    apt=[c for c in df if c.startswith('aptamer_frequency_')]
    km3=[c for c in apt if len(c.removeprefix('aptamer_frequency_'))==3]
    assert len(km3)==64, f'Expected 64 3-mer features, got {len(km3)}'
    mat=df[apt].to_numpy(dtype=float)
    _, indices, aid=np.unique(np.round(mat,6),axis=0,return_index=True,return_inverse=True)
    u3=df.iloc[indices][km3].to_numpy(float)
    sims=cosine_similarity(u3);adj=csr_matrix(sims>=threshold)
    nclusters,cl=connected_components(adj,directed=False)
    row_cluster=cl[aid]; n=len(df)
    # StratifiedGroupKFold prevents all cluster overlap and approximately preserves class balance.
    # Select the fold whose size and positive prevalence are closest to the predeclared targets.
    folds=list(StratifiedGroupKFold(n_splits=int(round(1/fraction)),shuffle=True,random_state=seed).split(df,df.Class,row_cluster))
    base_prevalence=float(df.Class.mean())
    fold_scores=[(abs(len(te)/n-fraction)+abs(float(df.iloc[te].Class.mean())-base_prevalence),i) for i,(_,te) in enumerate(folds)]
    selected_fold=min(fold_scores)[1]
    train_idx,test_idx=folds[selected_fold]
    is_test=np.zeros(n,dtype=bool);is_test[test_idx]=True
    picked=set(row_cluster[is_test])
    assert set(row_cluster[is_test]).isdisjoint(set(row_cluster[~is_test]))
    result={'source_rows':n,'clusters':int(nclusters),'unique_aptamer_profiles':int(len(indices)),
            'clusters_in_test':int(len(picked)),'selected_stratified_group_fold':int(selected_fold),'train_rows':int((~is_test).sum()),
            'test_rows':int(is_test.sum()),'train_positive':int(df.loc[~is_test,'Class'].sum()),
            'test_positive':int(df.loc[is_test,'Class'].sum()),'cross_partition_cluster_overlap':0,
            'threshold':threshold,'seed':seed,'test_fraction_requested':fraction}
    return df.loc[~is_test].copy(),df.loc[is_test].copy(),result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    df=pd.read_csv(a.data)
    tr,te,meta=split(df,a.seed)
    a.out.mkdir(parents=True,exist_ok=True)
    tr.to_csv(a.out/'cluster_disjoint_FIXED_train.csv',index=False)
    te.to_csv(a.out/'cluster_disjoint_FIXED_test.csv',index=False)
    (a.out/'cluster_split_manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
