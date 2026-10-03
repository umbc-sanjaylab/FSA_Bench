from pathlib import Path
import pandas as pd

DATASET_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = DATASET_ROOT / "raw_data"
PROCESSED_DATA_DIR = DATASET_ROOT / "processed_data" 

DATASET_CONFIG = {
    "TCGA": {
        "region_column": "Region",
        "n_folds": 5,
        "n_clients": 7,
    },
    
    "Aids2": {
        "region_column": "fac_state",
        "n_folds": 5,
        "n_clients": 4,
    },
    
    "LeukSurv": {
        "region_column": "fac_district",
        "n_folds": 3,
        "n_clients": 24,
    },
    
    "Pbc3": {
        "region_column": "fac_unit",
        "n_folds": 3,
        "n_clients": 6,
    },
    
    "Dialysis": {
        "region_column": "fac_center",
        "n_folds": 3,
        "n_clients": 67,
    },
    
    "d.oropha.rec": {
        "region_column": "fac_inst",
        "n_folds": 3,
        "n_clients": 6,
    },
    "cgd": {
        "region_column": "fac_hos_cat",
        "n_folds": 3,
        "n_clients": 4,
    },
    
    "uis": {
        "region_column": "fac_site",
        "n_folds": 5,
        "n_clients": 2,
    },
    
    "scania": {
        "region_column": "fac_parish",
        "n_folds": 5,
        "n_clients": 5,
    },
    
    "seer": {
        "region_column": "Area_3",
        "n_folds": 5,
        "n_clients": 3,
    }
}


def get_dataset_config(dataset_name):
    """Return metadata associated with a federated dataset."""

    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DATASET_CONFIG[dataset_name]

######################################### Read dataset for a specific client ################################

def load_raw_dataset(dataset_name):
    """Load a raw survival dataset."""

    file_path = RAW_DATA_DIR / f"{dataset_name}.csv"
    if not file_path.exists():
        raise FileNotFoundError( f"Dataset not found: {file_path}" )

    return pd.read_csv(file_path)

######################################### Read dataset for a specific client ################################
 
def get_local_dataset( dataset_name, partition, client_id, run_id):
    """
    Load train/validation/test data for one client.

    Returns
    -------
    train_df : pd.DataFrame
    val_df : pd.DataFrame
    test_df : pd.DataFrame
    client_name : str
    feature_list : list[str]
    """ 
    base_dir =  PROCESSED_DATA_DIR /  dataset_name / partition / str(client_id) / str(run_id) 

    train_path = base_dir / "train.csv"
    val_path = base_dir / "val.csv"
    test_path = base_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    feature_list = [ col for col in train_df.columns if col not in ("time", "event")]
    return train_df, val_df, test_df, str(client_id), feature_list  