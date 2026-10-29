#!/usr/bin/env python3
"""Audit uploaded AptaEnsemble snapshot; never rewrites experimental data."""
import argparse, json, ast, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def read(repo):
    return pd.read_csv(repo/'clean.csv')

def unique_vec(mat):
    _, idx, inverse = np.unique(np.round(mat.astype('float64'), 6), axis=0, return_index=True, return_inverse=True)
    return inverse, idx

def audit(repo, out):
    dat=read(repo)
    apt=[c for c in dat if c.startswith('aptamer_frequency_')]
    prot=[c for c in dat if c.startswith('protein_')]
    km3=[c for c in apt if len(c.split('aptamer_frequency_')[-1])==3]
    apt_ids, idx=unique_vec(dat[apt].to_numpy()); pro_ids,_=unique_vec(dat[prot].to_numpy())
    napt=int(apt_ids.max()+1); npro=int(pro_ids.max()+1)
    s=np.arange(len(dat)); a=apt_ids; p=pro_ids+napt
    graph=csr_matrix((np.ones(len(s)*4), (np.r_[a,p,a,p],np.r_[p,a,a,p])),shape=(napt+npro,napt+npro))
    nc, labs=connected_components(graph,directed=False)
    comp_counts=np.bincount(labs[a]); largest=int(comp_counts.max())
    apt_samples=dat.iloc[idx][km3].to_numpy(dtype=float)
    cos=cosine_similarity(apt_samples)
    np.fill_diagonal(cos, 0)
    adj=cos>=.9
    ncl, cl=connected_components(csr_matrix(adj), directed=False)
    sizes=np.bincount(cl)
    print('APT',napt,'PRO',npro,'ENTITIES_GRAPH_COMPONENTS',nc,'LARGEST_GRAPH_EDGES',largest)
    print('3MER CLUSTERS',ncl,'LARGEST_SIZE',sizes.max(),'SIMILARITY_EDGES',np.count_nonzero(np.triu(adj,1)))
    split_aud={}
    allcols=list(dat.columns)
    for name in ['pair_level','protein_disjoint','aptamer_disjoint','cluster_disjoint','doubly_disjoint']:
      d=pd.read_csv(repo/'Splits'/f'{name}_train.csv'); t=pd.read_csv(repo/'Splits'/f'{name}_test.csv')
      assert list(d.columns)==allcols and list(t.columns)==allcols
      a0=set(map(tuple,np.round(d[apt].to_numpy(),6))); a1=set(map(tuple,np.round(t[apt].to_numpy(),6)))
      p0=set(map(tuple,np.round(d[prot].to_numpy(),6))); p1=set(map(tuple,np.round(t[prot].to_numpy(),6)))
      aid_of={tuple(np.round(dat.iloc[i][apt].to_numpy(dtype=float),6)): int(apt_ids[i]) for i in range(len(dat))}
      d_ids=[aid_of[tuple(np.round(row,6))] for row in d[apt].to_numpy()]
      t_ids=[aid_of[tuple(np.round(row,6))] for row in t[apt].to_numpy()]
      dev_cl={int(cl[x]) for x in d_ids}; test_cl={int(cl[x]) for x in t_ids}
      split_aud[name]={'train_n':len(d),'test_n':len(t),'train_pos':int(d.Class.sum()),'test_pos':int(t.Class.sum()),'aptamer_overlap':len(a0&a1),'protein_overlap':len(p0&p1),'true_3mer_single_linkage_cluster_overlap':len(dev_cl&test_cl)}
      print(name,split_aud[name])
    summary={'n':len(dat),'positive':int(dat.Class.sum()),'negative':int((dat.Class==0).sum()),'n_aptamer_profiles':napt,'n_protein_profiles':npro,'bipartite_connected_components':int(nc),'largest_component_pairs':largest,'3mer_cosine_threshold':.9,'3mer_true_single_linkage_clusters':int(ncl),'largest_3mer_cluster_size':int(sizes.max()),'splits':split_aud}
    out.mkdir(parents=True,exist_ok=True)
    (out/'split_audit.json').write_text(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
  a=argparse.ArgumentParser();a.add_argument('--repo',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args(); audit(args.repo,args.out)
