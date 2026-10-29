#!/usr/bin/env python3
"""Re-run 500-tree ExtraTrees for existing and corrected splits.
Never mislabels NEW split metrics as reproduction of manuscript Table 8.
Cold-start threshold: five-fold *two-axis blocked* development CV; validation cells are A_f x P_f,
training excludes all A_f and P_f entities, mixed edges omitted per fold.
Cluster threshold: group CV with true 3-mer similarity components.
"""
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,precision_score,recall_score,matthews_corrcoef,f1_score,confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

SEED=42

def features(df,mode):
    apt=[c for c in df if c.startswith('aptamer_frequency_')]
    protein=[c for c in df if c.startswith('protein_')]
    k3=[c for c in apt if len(c.removeprefix('aptamer_frequency_'))==3]
    aac=[c for c in protein if '_aac_' in c]
    if mode=='compact184':
        assert len(k3)==64 and len(aac)==120, (len(k3),len(aac))
        return k3+aac
    assert len(apt)==340 and len(protein)==300
    return apt+protein

def model():
    return ExtraTreesClassifier(n_estimators=500,criterion='entropy',max_features='sqrt',class_weight='balanced',bootstrap=False,random_state=42,n_jobs=-1)

def labels(df,cols):
    _,ids=np.unique(np.round(df[cols].to_numpy(float),6),axis=0,return_inverse=True)
    return ids

def threshold_mcc(yt,pr):
    # Equal threshold tie => lower threshold selected, with grid declared.
    grid=np.round(np.arange(.05,.951,.005),3)
    vals=[matthews_corrcoef(yt,pr>=t) for t in grid]
    j=int(np.argmax(vals));return float(grid[j]),float(vals[j])

def blocked_oof(df,cols,kind,seed=42):
    y=df.Class.astype(int).to_numpy()
    apt=[c for c in df if c.startswith('aptamer_frequency_')]
    prot=[c for c in df if c.startswith('protein_')]
    apt_ids=labels(df,apt)
    if kind=='double':
        prot_ids=labels(df,prot)
        a_u=np.unique(apt_ids);p_u=np.unique(prot_ids)
        rng=np.random.default_rng(seed);rng.shuffle(a_u);rng.shuffle(p_u)
        am={int(k):i%5 for i,k in enumerate(a_u)};pm={int(k):i%5 for i,k in enumerate(p_u)}
        af=np.array([am[int(k)] for k in apt_ids]);pf=np.array([pm[int(k)] for k in prot_ids])
        folds=[(np.flatnonzero((af!=f)&(pf!=f)),np.flatnonzero((af==f)&(pf==f))) for f in range(5)]
    elif kind=='cluster':
        k3=[c for c in apt if len(c.removeprefix('aptamer_frequency_'))==3]
        _,idx,_=np.unique(np.round(df[apt].to_numpy(float),6),axis=0,return_index=True,return_inverse=True)
        _,_,inverse=np.unique(np.round(df[apt].to_numpy(float),6),axis=0,return_index=True,return_inverse=True)
        ncomp,components=connected_components(csr_matrix(cosine_similarity(df.iloc[idx][k3].to_numpy(float))>=.90),directed=False)
        groups=components[inverse]
        folds=list(GroupKFold(n_splits=5).split(df,y,groups=groups))
    else: raise ValueError(kind)
    oof=np.full(len(df),np.nan);fold_out=[]
    X=df[cols].to_numpy(float)
    for f,(it,iv) in enumerate(folds):
        if len(iv)==0 or len(np.unique(y[it]))<2:
            fold_out.append({'fold':f,'development':len(it),'validation':len(iv),'SKIPPED':True});continue
        m=model();m.fit(X[it],y[it]);oof[iv]=m.predict_proba(X[iv])[:,1]
        fold_out.append({'fold':f,'development':len(it),'validation':len(iv),
                         'positive_validation':int(y[iv].sum()),'positive_development':int(y[it].sum())})
        print('Fold',f,'train',len(it),'validate',len(iv),flush=True)
    ok=np.isfinite(oof)
    if ok.sum()==0: raise ValueError('No OOF validation records')
    return y[ok],oof[ok],fold_out

def report(yt,pr,thr):
    yhat=pr>=thr;tn,fp,fn,tp=confusion_matrix(yt,yhat,labels=[0,1]).ravel()
    return dict(n=len(yt),positives=int(yt.sum()),negatives=int((yt==0).sum()),
       threshold=thr,accuracy=float(accuracy_score(yt,yhat)),precision=float(precision_score(yt,yhat,zero_division=0)),
       recall=float(recall_score(yt,yhat,zero_division=0)),mcc=float(matthews_corrcoef(yt,yhat)),
       f1=float(f1_score(yt,yhat,zero_division=0)),roc_auc=float(roc_auc_score(yt,pr)),
       pr_auc=float(average_precision_score(yt,pr)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--corrected',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--skip-pair',action='store_true');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    experiments=[]
    if not args.skip_pair: experiments.append(('pair_level','full640',args.repo/'M1.csv',args.repo/'M3.csv','fixed'))
    experiments.extend([
       ('released_doubly_disjoint','compact184',args.repo/'Splits/doubly_disjoint_train.csv',args.repo/'Splits/doubly_disjoint_test.csv','double'),
       ('corrected_cluster_disjoint','compact184',args.corrected/'cluster_disjoint_FIXED_train.csv',args.corrected/'cluster_disjoint_FIXED_test.csv','cluster'),
    ])
    outcomes={}
    for name,mode,train_path,test_path,threshold_policy in experiments:
        print('\nEXPERIMENT',name,mode,flush=True);t0=time.time()
        dev=pd.read_csv(train_path);test=pd.read_csv(test_path)
        columns=features(dev,mode)
        if list(test[columns].columns)!=columns:raise ValueError('Columns mismatch')
        ydev=dev.Class.to_numpy(dtype=int);ytest=test.Class.to_numpy(dtype=int)
        if threshold_policy=='fixed':
            threshold=.455;cross_validation=None;criterion='reported pair-level fixed threshold'
        else:
            yoof,poof,folds=blocked_oof(dev,columns,threshold_policy)
            threshold,oof_mcc=threshold_mcc(yoof,poof)
            cross_validation={'oof_n':len(yoof),'oof_mcc':oof_mcc,'folds':folds};criterion='blocked development-only CV'
        clf=model();clf.fit(dev[columns].to_numpy(float),ydev)
        pr=clf.predict_proba(test[columns].to_numpy(float))[:,1]
        scores=report(ytest,pr,threshold)
        probs=pd.DataFrame({'y_true':ytest,'probability':pr,'y_pred':(pr>=threshold).astype(int),'threshold':threshold})
        probs.to_csv(args.out/f'{name}_predictions.csv',index=False)
        outcomes[name]={'status':'new diagnostic result' if name!='pair_level' else 'independent reproduced endpoint',
                        'description':f'{mode} model 500 entropy sqrt balanced, {criterion}',
                        'features':len(columns),'train_rows':len(dev),'test_rows':len(test),
                        'threshold_optimization':cross_validation,'test_metrics':scores,'elapsed_seconds':round(time.time()-t0,2)}
        print(json.dumps(scores,indent=2),flush=True)
    (args.out/'rerun_metrics.json').write_text(json.dumps(outcomes,indent=2))
    print('\nWROTE',args.out/'rerun_metrics.json',flush=True)
if __name__=='__main__':main()
