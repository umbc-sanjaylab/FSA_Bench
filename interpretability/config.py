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
os.environ["DEVICE"] = DEVICE

MODEL_DIRECTORIES = {
    "FedPDNN": "FedPseudo_Flare",
    "FedPLSTM": "FedPseudo_Flare",
    "FedPAttn": "FedPseudo_Flare",
    "AFT_Weibull": "AFT_Weibull",
    "DeepHit": "DeepHit",
    "DeepPseudo": "DeepPseudo",
    "Fed_CoxPH": "Fed_CoxPH",
    "LinearPH": "LinearPH_NNPH_NNnPH",
    "NNPH": "LinearPH_NNPH_NNnPH",
    "NNnPH": "LinearPH_NNPH_NNnPH",
}

DATASET_FOLDS = {
    "seer": 5,
    "Pbc3": 3,
    "LeukSurv": 3,
    "cgd": 3,
    "scania": 5,
    "uis": 5,
    "d.oropha.rec": 3,
    "Dialysis": 3,
    "Aids2": 5
}

DATASET_CLIENTS = {
    "seer": [1, 2, 3],
    "Pbc3":  ["Barcelona", "Copenhagen", "Hvidovre", "London", "Lyon", "Munich"],
    "LeukSurv": list(range(1, 25, 1)),
    "cgd":  ["Europe:Amsterdam", "Europe:other", "US:NIH", "US:other"],
    "scania": list(range(1, 6, 1)),
    "uis": ['A', 'B'],
    "d.oropha.rec":  list(range(1, 7, 1)),
    "Dialysis": [ 120, 128, 184, 217, 344, 561, 562, 641, 741, 743, 917, 918, 936, 954, 983, 984, 1029, 1039, 1048, 1051, 1053, 1061, 1064, 1070, 1071, 1072, 1093, 1100, 1159, 1372, 1437, 1654, 1692, 1699, 2026, 2210, 2211, 2242, 2320, 2351, 2611, 2632, 2811, 2844, 2987, 3064, 3596, 3675, 4122, 4157, 4163, 4779, 4825, 5089, 5135, 5419, 5423, 5432, 5481, 5495, 5513, 5514, 5543, 5653, 5681, 5692, 5768 ],
    "Aids2": ["NSW", "QLD", "VIC", "Other"]
}


DATASET_CHUNKS = {
    "seer": 1500,
    "Pbc3": 100,
    "LeukSurv": 100,
    "cgd": 100,
    "scania": 100,
    "uis": 100,
    "d.oropha.rec": 100, 
    "Dialysis": 100,
    "Aids2": 100 
}

MODELS = [
    "FedPAttn",
    "FedPDNN",
    "FedPLSTM",
    "AFT_Weibull",
    "Fed_CoxPH",
    "DeepPseudo",
    "LinearPH",
    "NNPH",
    "NNnPH",
    "DeepHit"
]