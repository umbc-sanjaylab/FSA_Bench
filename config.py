# bash scripts/run_fl.sh NNnPH Pbc3 0
# bash scripts/run_cl.sh NNnPH Pbc3 0 
# model → dataset → fold/run

from pathlib import Path 

DATASET_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = DATASET_ROOT / "raw_data"
PROCESSED_DATA_DIR = DATASET_ROOT / "processed_data"  

DATASET_CONFIG = {
    "TCGA": {
        "region_column": "Region",
        "n_folds": 5,
        "n_clients": 7
    },
    
    "Aids2": {
        "region_column": "fac_state",
        "n_folds": 5,
        "n_clients": 4
    },
    
    "LeukSurv": {
        "region_column": "fac_district",
        "n_folds": 3,
        "n_clients": 24
    },
    
    "Pbc3": {
        "region_column": "fac_unit",
        "n_folds": 3,
        "n_clients": 6
    },
    
    "Dialysis": {
        "region_column": "fac_center",
        "n_folds": 3,
        "n_clients": 67
    },
    
    "d.oropha.rec": {
        "region_column": "fac_inst",
        "n_folds": 3,
        "n_clients": 6
    },
    "cgd": {
        "region_column": "fac_hos_cat",
        "n_folds": 3,
        "n_clients": 4
    },
    
    "uis": {
        "region_column": "fac_site",
        "n_folds": 5,
        "n_clients": 2
    },
    
    "scania": {
        "region_column": "fac_parish",
        "n_folds": 5,
        "n_clients": 5
    },
    
    "seer": {
        "region_column": "Area_3",
        "n_folds": 5,
        "n_clients": 3
    }
}