#!/usr/bin/env python3
import pandas as pd, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "source"
LEG = ROOT / "data" / "legacy"

def entity_ids(df):
    X = df.iloc[:,1:].to_numpy(float)
    def h(a,p):
        s = ",".join(f"{float(v):.17g}" for v in a)
        return p + hashlib.sha256(s.encode()).hexdigest()[:16]
    A = [h(a,"A_") for a in X[:,:340]]
    P = [h(p,"P_") for p in X[:,340:]]
    return A,P

def pair_ids(df):
    A,P = entity_ids(df)
    y = df.iloc[:,0].astype(int).tolist()
    return [hashlib.sha256((a+"|"+p+"|"+str(l)).encode()).hexdigest()[:20]
            for a,p,l in zip(A,P,y)]

clean = pd.read_csv(SRC/"clean_3003.csv")
train = pd.read_csv(SRC/"pair_level_development_2402.csv")
m3 = pd.read_csv(SRC/"pair_level_test_M3_601.csv")
legacy_dev = pd.read_csv(LEG/"doubly_disjoint_development_legacy_2402.csv")
legacy_test = pd.read_csv(LEG/"doubly_disjoint_test_legacy_601.csv")

assert len(clean)==3003
assert len(train)==2402 and len(m3)==601
assert len(set(pair_ids(train)) & set(pair_ids(m3)))==0
assert set(pair_ids(train)) | set(pair_ids(m3)) == set(pair_ids(clean))

Adev,Pdev=entity_ids(legacy_dev)
Atest,Ptest=entity_ids(legacy_test)
assert not (set(Adev)&set(Atest))
assert not (set(Pdev)&set(Ptest))

report = {
    "clean_rows": len(clean),
    "pair_level_development_rows": len(train),
    "pair_level_test_rows": len(m3),
    "legacy_doubly_disjoint_development_rows": len(legacy_dev),
    "legacy_doubly_disjoint_test_rows": len(legacy_test),
    "legacy_doubly_disjoint_aptamer_overlap": len(set(Adev)&set(Atest)),
    "legacy_doubly_disjoint_protein_overlap": len(set(Pdev)&set(Ptest)),
    "status": "PASS"
}
print(json.dumps(report, indent=2))
( ROOT/"results"/"legacy_audit_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
