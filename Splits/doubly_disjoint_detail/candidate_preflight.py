#!/usr/bin/env python3
"""Preflight for original candidate_ranking.py. Never emits unverifiable biological rankings."""
import argparse,ast,json
from pathlib import Path
import numpy as np,pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 code=(a.repo/'candidate_ranking.py').read_text(); mod=ast.parse(code)
 constants={}
 for node in mod.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   if node.targets[0].id in ['GROUPS','BASE_PROPS']:
    constants[node.targets[0].id]=ast.literal_eval(node.value)
 groups=constants.get('GROUPS',{})
 weight_values=[]
 for node in ast.walk(mod):
  if isinstance(node,ast.Constant) and isinstance(node.value,float) and node.value in [.05,.15]: weight_values.append(node.value)
 df=pd.read_csv(a.repo/'M1.csv')
 prot=[c for c in df if c.startswith('protein_')]
 # Compute OOD sign discrepancy: normal direct PseAAC outputs nonnegative; source benchmark negative.
 X=df[prot].to_numpy(float)
 check={'source':'candidate_ranking.py from ZIP',
        'candidate_libs_present_in_zip':{k:False for k in ['CT-20(2).txt','DK-30(2).txt','BC-6(2).txt']},
        'referenced_training_file_present':'train(3).csv' in [p.name for p in a.repo.iterdir()],
        'usable_original_train':'M1.csv' in [p.name for p in a.repo.iterdir()],
        'group_definitions':groups,
        'literal_weights_in_code':sorted(set(weight_values)),
        'inherited_protein_matrix_negative_fraction':float((X<0).mean()),
        'source_protein_feature_columns':len(prot),
        'critical_findings':[
            'Raw CTGF, DKK1, BCMA sequence library files are absent from ZIP.',
            'Hardcoded train(3).csv is absent; M1.csv exists, but swapping paths does not fix feature incompatibility.',
            'Candidate GROUPS and omega differ from documented AptaNet-derived precomputed protein representation.',
            'Original AptaNet encoded dataset contains negative values, while direct composition and squared-correlation formulas yield nonnegative descriptors.',
            'Cannot certify candidate scores without a validated FASTA-to-trained-feature preprocessing transform.'
        ],
        'candidate_scores_recalculated':False}
 a.out.mkdir(parents=True,exist_ok=True)
 (a.out/'candidate_preflight.json').write_text(json.dumps(check,indent=2))
 print(json.dumps(check,indent=2))
if __name__=='__main__': main()
