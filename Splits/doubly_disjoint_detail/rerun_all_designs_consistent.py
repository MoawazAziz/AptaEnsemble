#!/usr/bin/env python3
"""New, transparent supplementary rerun of all five designs on saved/corrected splits.
This analysis is NOT claimed to reproduce manuscript Table 8. It uses an a-priori
model family and descriptor block per design, not a model selection search.
"""
from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold,StratifiedKFold
from rerun_extra_trees import features,labels,model,threshold_mcc,blocked_oof,report


def group_ids(df,design):
    apt=[c for c in df if c.startswith('aptamer_frequency_')]
    protein=[c for c in df if c.startswith('protein_')]
    if design=='protein':return labels(df,protein)
    if design=='aptamer':return labels(df,apt)
    if design=='cluster':
        k3=[c for c in apt if len(c.removeprefix('aptamer_frequency_'))==3]
        _,idx,inv=np.unique(np.round(df[apt].to_numpy(float),6),axis=0,return_index=True,return_inverse=True)
        _,comp=connected_components(csr_matrix(cosine_similarity(df.iloc[idx][k3].to_numpy(float))>=.90),directed=False)
        return comp[inv]
    raise ValueError(design)

def oof_grouped(df,cols,design):
    Y=df.Class.to_numpy(int);X=df[cols].to_numpy(float)
    if design=='pair':
        folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(X,Y)
    else:
        groups=group_ids(df,design)
        folds=GroupKFold(n_splits=5).split(X,Y,groups)
    probs=np.full(len(df),np.nan)
    logs=[]
    for i,(tr,te) in enumerate(folds):
        mdl=model();mdl.fit(X[tr],Y[tr]);probs[te]=mdl.predict_proba(X[te])[:,1]
        logs.append({'fold':i,'n_train':len(tr),'n_validation':len(te),'n_validation_positive':int(Y[te].sum())})
    mask=np.isfinite(probs)
    return Y[mask],probs[mask],logs

def main():
    a=argparse.ArgumentParser();a.add_argument('--repo',type=Path,required=True);a.add_argument('--fixed-dir',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    names=[
        ('pair-level','full640','pair',args.repo/'M1.csv',args.repo/'M3.csv'),
        ('protein-disjoint','aptamer340_AAC120','protein',args.repo/'Splits/protein_disjoint_train.csv',args.repo/'Splits/protein_disjoint_test.csv'),
        ('aptamer-disjoint','compact184','aptamer',args.repo/'Splits/aptamer_disjoint_train.csv',args.repo/'Splits/aptamer_disjoint_test.csv'),
        ('cluster-disjoint-FIXED','compact184','cluster',args.fixed_dir/'cluster_disjoint_FIXED_train.csv',args.fixed_dir/'cluster_disjoint_FIXED_test.csv'),
        ('doubly-disjoint-released','compact184','double',args.repo/'Splits/doubly_disjoint_train.csv',args.repo/'Splits/doubly_disjoint_test.csv')]
    output={};rowout=[]
    for name,mod,folds,trp,tep in names:
        start=time.time();print('RUN',name,flush=True)
        tr=pd.read_csv(trp);te=pd.read_csv(tep);cols=features(tr,'compact184' if mod=='compact184' else 'full640') if mod!='aptamer340_AAC120' else [c for c in tr if c.startswith('aptamer_frequency_') or (c.startswith('protein_') and '_aac_' in c)]
        if mod=='aptamer340_AAC120':assert len(cols)==460
        if folds=='double':
            yoof,poof,logs=blocked_oof(tr,cols,'double')
        else:
            yoof,poof,logs=oof_grouped(tr,cols,folds)
        thr,oofmcc=threshold_mcc(yoof,poof)
        clf=model();clf.fit(tr[cols].to_numpy(float),tr.Class.to_numpy(int))
        prob=clf.predict_proba(te[cols].to_numpy(float))[:,1];scores=report(te.Class.to_numpy(int),prob,thr)
        pd.DataFrame({'y_true':te.Class,'probability':prob,'y_pred':(prob>=thr).astype(int),'threshold':thr}).to_csv(args.out/(name+'_predictions.csv'),index=False)
        payload={'source_dev':str(trp),'source_test':str(tep),'model':'ExtraTrees 500, entropy, sqrt, balanced, seed42','feature_set':mod,'n_features':len(cols),
                 'threshold_selection':f'5-fold development {folds}-exclusion CV', 'oof_n':len(yoof),'oof_mcc':float(oofmcc),'threshold':thr,'folds':logs,
                 'training_n':len(tr),'test_n':len(te),'test_metrics':scores,'elapsed_seconds':round(time.time()-start,2)}
        output[name]=payload;rowout.append({'design':name,'development_n':len(tr),'test_n':len(te),'n_features':len(cols),'threshold':thr,'oof_n':len(yoof),**scores})
        print('DONE',name,'MCC',round(scores['mcc'],4),'AUC',round(scores['roc_auc'],4),'THR',thr,flush=True)
        (args.out/'all_five_metrics.json').write_text(json.dumps(output,indent=2))
    pd.DataFrame(rowout).to_csv(args.out/'all_five_diagnostic_results.csv',index=False)
    print(pd.DataFrame(rowout)[['design','test_n','n_features','threshold','mcc','roc_auc','recall','accuracy']].to_string(index=False))
if __name__=='__main__':main()
