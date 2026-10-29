# AptaEnsemble reproducibility repair — verified rerun package
**Audit date:** 29 October 2026  
**Source:** user-uploaded `AptaEnsemble-main (1).zip` (snapshot, not verified live GitHub HEAD).  
**Status:** reproducibility improvements and new diagnostic evaluations; **NOT** a validated reproduction of the original manuscript Table 8 or its candidate-ranking table.

## Important results

| Design | Development | Test | Descriptor count | OOF threshold | MCC | ROC-AUC | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pair level (M1→M3) | 2402 | 601 | 640 | 0.455 | 0.8717 | 0.9495 | 0.8395 |
| Protein-disjoint (released) | 2386 | 617 | 460 | 0.495 | 0.8737 | 0.9573 | 0.8258 |
| Aptamer-disjoint (released) | 2365 | 638 | 184 | 0.725 | 0.5184 | 0.7125 | 0.3958 |
| **Cluster-disjoint (new FIXED)** | **2403** | **600** | **184** | **0.755** | **0.3932** | **0.7558** | **0.2099** |
| Doubly-disjoint (released alternative) | 1324 | 345 | 184 | 0.590 | 0.1366 | 0.6020 | 0.3394 |

The five reruns use ExtraTrees with `n_estimators=500`, `criterion='entropy'`, `max_features='sqrt'`, `class_weight='balanced'`, `bootstrap=False`, `random_state=42`. Protein-disjoint uses the 340 aptamer and 120 protein AAC columns, while aptamer-, cluster-, doubly-disjoint use the 64 aptamer 3-mer and 120 AAC-labeled protein columns. Pair-level uses all 640. For each design, its threshold was selected using five development folds only (pair-level stratified; protein/aptamer entity-grouped; cluster grouped using the true 3-mer cosine graph; doubly-disjoint using both-axis blocked folds). The blocked doubly-disjoint development folds exclude mixed-entity rows and consequently produce only **262** out-of-fold predictions from 1,324 development rows. The held-out test set was not used for selecting thresholds or hyperparameters. These descriptor assignments and model hyperparameters were fixed before this diagnostic, not selected using a fresh search. Numerical results are reproducible from the supplied split files and code, but have not been accompanied by the full manuscript's model-selection, confidence-interval, or permutation-control protocols.

## Four problems: finding → actionable repair

### 1. Doubly disjoint / manuscript Table 8 — NOT resolved
- Current `Splits/doubly_disjoint_train.csv` 1,324 and `Splits/doubly_disjoint_test.csv` 345 pairs (positive test 109), with **no common aptamer/protein profiles**, but 1,334 additional mixed-entity rows are excluded.
- Manuscript Table 8 uses MCC 0.1980, ROC-AUC 0.5589, recall 7.84%, apparently derived from a different 770-pair evaluation. Source assignments, metadata, and predictions matching that figure are **not present**.
- The original 3,003-pair dataset forms a bipartite graph whose largest connected component contains **2,879 edges/pairs**. It is mathematically impossible to divide *all* 3,003 original edges into development 2,233 and test 770 with complete disjointness of both node types and no exclusions. If the historical 770-pair endpoint is real, document whether pairs were generated, excluded, or entities were defined differently.
- **Options:** (A) recover the historical source partitions, pair-generation and evaluation code; prove entity separation and recalculate all claims; OR (B) explicitly replace the original design with a newly preregistered, fixed genuinely doubly-disjoint protocol, excluding mixed edges, then rerun all affected experiments, CIs, figures, abstract, and conclusion. **Do not claim this 345-pair rerun reproduces Table 8.**

### 2. Candidate ranking — NOT resolved; numerical incompatibility VERIFIED
- `candidate_ranking.py` uses other property groups (`hydro`, `volume`, `charge`, `helix`, `sheet`, `flex`), omega **0.15**, training file `/mnt/data/train(3).csv` (absent), and CTGF/DKK1/BCMA libraries `CT-20(2).txt`, `DK-30(2).txt`, `BC-6(2).txt` (absent).
- When the embedded `protein_vec()` function is run on target FASTA sequences, each 300-D output is nonnegative, ranging about 0.00037 to 0.03333, while about **55.32%** of the benchmark's 300 protein values in M1 are negative; overall benchmark protein range −0.896 to 0.989. **Column count and label agreement alone are not sufficient for feature compatibility.**
- **Options:** recover AptaNet's exact FASTA→encoded-feature pipeline *including the archived preprocessing transforms* and verify it quantitatively against the original 300-D AptaNet records with known source FASTA; OR obtain/precompute new aptamer and protein descriptors for training and candidates with a single identical, verified encoder, retrain from scratch, revalidate, then recalculate all candidate probabilities and priority scores. Do not change the sequence-order weight and retain existing scores without re-execution. The top 5000 library filtering and 0.70/0.20/0.10 composite formula should be reported only after the scoring code and sources are verified.

### 3. Similarity cluster split — SPLIT repaired; new model result obtained
- Independently computed **718 single-linkage clusters** across 803 unique aptamer profiles from 64 3-mer features and cosine ≥0.90, matching manuscript's cluster count.
- The released `Splits/cluster_disjoint_*` partition has **9** such clusters shared between development and test. So it is not strictly cluster disjoint despite having no *identical* aptamer profiles in both.
- `fix_cluster_split.py` creates `cluster_disjoint_FIXED_train.csv` and `_test.csv` with **0** leaked 3-mer clusters. StratifiedGroupKFold creates a balanced 2403/600 partition with 162 positives in test, threshold selected by within-development GroupKFold gives MCC **0.3932**, ROC-AUC **0.7558**. These are NEW results; manuscript Table 8/Figures 2–4 must not claim they were always the original evaluation.

### 4. GitHub execution/reproducibility — minimally repaired; full pipeline NOT certified
- `main_pipeline.py` contained an unclosed triple-quote. `main_pipeline_FIXED.py` closes it, maps missing `train.csv`→`M1.csv`, `test.csv`→`M3.csv`, handles absent TensorFlow for an ML-only mode, repairs `build_mlp_small`, and corrects M3 labeling to pair-level held-out.
- **Verified:** `py_compile` passes, and `APTA_SMOKE_TEST=1` loads and audits M1/M2/M3 successfully. A full core ML run progressed through ExtraTrees and RandomForest training and into LightGBM, but exceeded execution limits; it was **not** a successful full-run verification. TensorFlow is not installed in the audit runtime, so DL and hybrids were not executed.
- The standalone `rerun_all_designs_consistent.py` **did** finish and write reproducible test predictions and metrics for all five ExtraTrees designs. These are the outputs to use for checking this package.
- Update README paths/casing, clarify M3 is a **pair-level held-out set** rather than externally unseen molecular entities, separate the distinct 580-pair embedding protocol, and remove claims about scripts/files absent from the snapshot.

## Run this patch against the ZIP-extracted original repository
```bash
pip install -r requirements_minimal.txt
python audit_four_issues.py --repo /path/to/AptaEnsemble-main --out ./results
python fix_cluster_split.py --data /path/to/AptaEnsemble-main/clean.csv --out ./results
python rerun_all_designs_consistent.py --repo /path/to/AptaEnsemble-main --fixed-dir ./results --out ./results
python candidate_preflight.py --repo /path/to/AptaEnsemble-main --out ./results
# Optional main-pipeline syntax/load smoke (does NOT fit models):
APTA_DATA_DIR=/path/to/AptaEnsemble-main APTA_SMOKE_TEST=1 python main_pipeline_FIXED.py
```
Do not run the old `kmer_label_correction.py --fix` on an already-corrected matrix: its transposition is not idempotent.

## Distribution notes
- Original CSVs should stay as historical inputs for traceability; corrected splits/predictions must have new filenames.
- Publish generated code, split assignments, threshold-selection methods, feature-column manifests, and raw predictions before citing specific rerun results in a manuscript. Check source dataset licenses and acknowledgment requirements.
- `all_five_metrics.json`, `all_five_diagnostic_results.csv`, and `*_predictions.csv` are new reproducible artifacts, **not** recovered historical outputs.
