from pathlib import Path

#  Dataset Directories (Raw & Processed data) 
DATASET_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = DATASET_ROOT / "datasets/raw_data"
PROCESSED_DATA_DIR = DATASET_ROOT / "datasets/processed_data"  

#  Configurations & Properties of Survival Datasets 
DATASET_CONFIG = {
    "TCGA": {
        "region_column": "Region",
        "n_folds": 5,
        "n_clients": 7,
        "batch_size": 128
    },

    "Aids2": {
        "region_column": "fac_state",
        "n_folds": 5,
        "n_clients": 4,
        "batch_size": 64
    },

    "LeukSurv": {
        "region_column": "fac_district",
        "n_folds": 3,
        "n_clients": 24,
        "batch_size": 32
    },

    "Pbc3": {
        "region_column": "fac_unit",
        "n_folds": 3,
        "n_clients": 6,
        "batch_size": 32
    },

    "Dialysis": {
        "region_column": "fac_center",
        "n_folds": 3,
        "n_clients": 67,
        "batch_size": 32
    },

    "d.oropha.rec": {
        "region_column": "fac_inst",
        "n_folds": 3,
        "n_clients": 6,
        "batch_size": 32
    },

    "cgd": {
        "region_column": "fac_hos_cat",
        "n_folds": 3,
        "n_clients": 4,
        "batch_size": 32
    },

    "uis": {
        "region_column": "fac_site",
        "n_folds": 5,
        "n_clients": 2,
        "batch_size": 64
    },

    "scania": {
        "region_column": "fac_parish",
        "n_folds": 5,
        "n_clients": 5,
        "batch_size": 32
    },

    "seer": {
        "region_column": "Area_3",
        "n_folds": 5,
        "n_clients": 3,
        "batch_size": 128
    }
}
