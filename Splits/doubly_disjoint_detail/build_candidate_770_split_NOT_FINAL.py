#!/usr/bin/env python3
"""Build a deterministic candidate 770-row doubly-disjoint split.

WARNING: this is NOT the recovered final manuscript partition. It is supplied only
as a reconstruction aid. Do not publish it as the source of the reported 75.06/0.1980/0.5589 result.
"""
import csv, json, collections, hashlib, numpy as np
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"data/source/clean_3003.csv"
OUT=ROOT/"data/final_doubly_disjoint"
SEED=42; DEV_NEG_SEED=1001; TEST_NEG_SEED=2001; TEST_POS_TARGET=204; TEST_NEG=566; DEV_NEG=1628

def h(arr,prefix):
    return prefix+hashlib.sha256(",".join(f"{float(v):.17g}" for v in arr).encode()).hexdigest()[:16]

rows=list(csv.reader(open(SRC,newline="",encoding="utf-8-sig"))); header=rows[0]; rows=rows[1:]
y=np.array([int(r[0]) for r in rows]); Akeys=[tuple(r[1:341]) for r in rows]; Pkeys=[tuple(r[341:641]) for r in rows]
Aid={};Pid={};Ai=[];Pi=[];avec=[];pvec=[]
for a,p in zip(Akeys,Pkeys):
    if a not in Aid: Aid[a]=len(Aid); avec.append(a)
    if p not in Pid: Pid[p]=len(Pid); pvec.append(p)
    Ai.append(Aid[a]);Pi.append(Pid[p])
pos=np.flatnonzero(y==1); pospairs={(int(Ai[i]),int(Pi[i])) for i in pos}
parent=list(range(len(avec)+len(pvec))); size=[1]*len(parent)
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(a,b):
    a,b=find(a),find(b)
    if a==b:return
    if size[a]<size[b]:a,b=b,a
    parent[b]=a;size[a]+=size[b]
for i in pos: union(Ai[i],len(avec)+Pi[i])
cc=collections.defaultdict(list)
for i in pos: cc[find(Ai[i])].append(int(i))
comps=[v for v in cc.values()]
rng=np.random.default_rng(SEED); order=np.arange(len(comps));rng.shuffle(order)
dp={0:[]}
for j in order:
    s=len(comps[int(j)])
    for total,lst in sorted(list(dp.items()),reverse=True):
        nt=total+s
        if nt<=TEST_POS_TARGET and nt not in dp: dp[nt]=lst+[int(j)]
    if TEST_POS_TARGET in dp: break
if TEST_POS_TARGET not in dp: raise RuntimeError("Target positive count not reachable.")
T=set(dp[TEST_POS_TARGET])
tepos=[i for j,c in enumerate(comps) if j in T for i in c]
dvpos=[i for j,c in enumerate(comps) if j not in T for i in c]
tA={Ai[i] for i in tepos}; tP={Pi[i] for i in tepos}; dA={Ai[i] for i in dvpos}; dP={Pi[i] for i in dvpos}
def negsample(Aset,Pset,n,seed):
    cand=[(a,p) for a in sorted(Aset) for p in sorted(Pset) if (a,p) not in pospairs]
    ix=np.random.default_rng(seed).choice(len(cand),n,replace=False)
    return [(int(cand[int(i)][0]),int(cand[int(i)][1])) for i in ix]
dn=negsample(dA,dP,DEV_NEG,DEV_NEG_SEED); tn=negsample(tA,tP,TEST_NEG,TEST_NEG_SEED)
D=[[1]+list(rows[i][1:]) for i in dvpos]+[[0]+list(avec[a])+list(pvec[p]) for a,p in dn]
TST=[[1]+list(rows[i][1:]) for i in tepos]+[[0]+list(avec[a])+list(pvec[p]) for a,p in tn]
rg=np.random.default_rng(SEED);rg.shuffle(D);rg.shuffle(TST)
for fn,data in [("candidate_doubly_disjoint_development_2233.csv",D),("candidate_doubly_disjoint_test_770.csv",TST)]:
    with open(OUT/fn,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(header);w.writerows(data)
print(json.dumps({"warning":"NOT_FINAL","test_rows":len(TST),"test_positive":sum(r[0]=='1' for r in TST),"test_negative":sum(r[0]=='0' for r in TST),"development_rows":len(D)},indent=2))
