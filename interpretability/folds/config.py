import os 
 
FOLD       = 0
TIME_COL   = "time"                # <-- set to your CSV's time column
EVENT_COL  = "event"               # <-- set to your CSV's event column (1 = event)
  
OUT_FEATURES = 9                   # = number of pseudo-eval times your model outputs
SEED        = 42

# survshap cost knobs (kernel SHAP is slow; raise for the final run)
N_BACKGROUND = 60
N_EXPLAIN    = 150
B            = 50

FEATURE_RENAME = {} 

DEVICE     = "cuda:1" 
   
DATA_DIR   = "/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/processed_data/federated_data/seer/centralized"
os.environ["DEVICE"] = DEVICE