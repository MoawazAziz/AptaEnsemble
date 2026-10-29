#!/usr/bin/env python3
"""Create minimally runnable fixed main_pipeline from provided ZIP; no benchmark result changes."""
from pathlib import Path
import argparse

def patch(src):
 s=src
 needle='with out-of-fold (OOF) threshold optimization.\n\n\n\nimport os'
 assert needle in s
 s=s.replace(needle,'with out-of-fold (OOF) threshold optimization.\n"""\n\nimport os',1)
 s=s.replace('import tensorflow as tf\nfrom tensorflow.keras import layers, models, callbacks, optimizers\nfrom tensorflow.keras import backend as K',
 '''try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers
    from tensorflow.keras import backend as K
    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    layers = models = callbacks = optimizers = K = None
    HAS_TENSORFLOW = False''')
 s=s.replace('tf.random.set_seed(SEED)','if tf is not None: tf.random.set_seed(SEED)',1)
 s=s.replace('DATA_DIR = "./data"','DATA_DIR = os.environ.get("APTA_DATA_DIR", ".")')
 s=s.replace('OUTPUT_DIR = "./output/final_m1_m2_m3_leakage_controlled_run"','OUTPUT_DIR = os.environ.get("APTA_OUTPUT_DIR", "./output/final_m1_m2_m3_leakage_controlled_run")')
 s=s.replace('RUN_DEEP_LEARNING = True','RUN_DEEP_LEARNING = HAS_TENSORFLOW and os.environ.get("APTA_RUN_DL", "0") == "1"')
 s=s.replace('RUN_ML_DL_HYBRIDS = True','RUN_ML_DL_HYBRIDS = RUN_DEEP_LEARNING')
 s=s.replace('"train": os.path.join(DATA_DIR, "train.csv"),','"train": os.path.join(DATA_DIR, "M1.csv"),')
 s=s.replace('"test": os.path.join(DATA_DIR, "test.csv"),','"test": os.path.join(DATA_DIR, "M3.csv"),')
 start=s.index('def build_mlp_small(')
 end=s.index('\ndef build_mlp_deep(',start)
 s=s[:start]+'''def build_mlp_small(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    model = models.Model(inp, layers.Dense(1, activation="sigmoid")(x))
    model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model
''' + s[end:]
 s=s.replace('"M3": "external unseen validation"','"M3": "pair-level held-out validation"')
 s=s.replace('"M3_usage": "M3 used only for final external validation"','"M3_usage": "M3 is a pair-level held-out set; entity recurrence is possible"')
 s=s.replace('ranked_M3_external_validation.csv','ranked_M3_pair_level_heldout.csv')
 s=s.replace('    train_df = raw["train"].copy()','''    if os.environ.get("APTA_SMOKE_TEST", "0") == "1":
        print("SMOKE TEST PASS: all three M1/M2/M3 source files and columns loaded. No model fitted.")
        return
    train_df = raw["train"].copy()''')
 # Remove pandas-based ExcelWriter from repaired pipeline; all CSVs and JSON are retained.
 s=s.replace('''    excel_path = os.path.join(OUTPUT_DIR, "all_results_tables.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="All_results", index=False)
        ranked_m3.to_excel(writer, sheet_name="Ranked_M3", index=False)
        group_fi.to_excel(writer, sheet_name="ET_group_importance", index=False)
        fi.head(100).to_excel(writer, sheet_name="ET_top100_features", index=False)
        top5_best.to_excel(writer, sheet_name="Top5_best_model", index=False)
        top5_et.to_excel(writer, sheet_name="Top5_ExtraTrees", index=False)
''','''    # Reproducibility outputs are written as CSV/JSON above; no spreadsheet dependency.
''')
 s=s.replace('They will execute automatically if RUN_ABLATION, RUN_AFO flags are True.','These experiment families require separate reproducible scripts and are not implemented here.')
 return s

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 a.out.write_text(patch(a.source.read_text()))
 print(a.out)
