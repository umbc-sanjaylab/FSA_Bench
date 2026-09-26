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
    base_dir =  PROCESSED_DATA_DIR / "federated_data" / dataset_name / partition / str(client_id) / str(run_id) 

    train_path = base_dir / "train.csv"
    val_path = base_dir / "val.csv"
    test_path = base_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    feature_list = [ col for col in train_df.columns if col not in ("time", "event")]
    return train_df, val_df, test_df, str(client_id), feature_list 
  

######################################### Region from Federated Dataset by Client ID ####################################

def get_client_name_for_federated_datasets(dataset, client_id):
    dataset_with_region = {
        "TCGA": ["South", "West", "Midwest", "Northeast", "Europe", "Canada", "Other"],
        "Aids2": ['NSW', 'Other', 'QLD', 'VIC'],
        "LeukSurv": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        "Pbc3": ['London', 'Copenhagen', 'Hvidovre', 'Barcelona', 'Munich', 'Lyon'],
        "Dialysis": [120, 128, 184, 217, 344, 561, 562, 641, 741, 743, 917, 918, 936, 954, 983, 984, 1029, 1039,
                     1048, 1051, 1053, 1061, 1064, 1070, 1071, 1072, 1093, 1100, 1159, 1372, 1437, 1654, 1692,
                     1699, 2026, 2210, 2211, 2242, 2320, 2351, 2611, 2632, 2811, 2844, 2987, 3064, 3596, 3675,
                     4122, 4157, 4163, 4779, 4825, 5089, 5135, 5419, 5423, 5432, 5481, 5495, 5513, 5514, 5543,
                     5653, 5681, 5692, 5768],
        "d.oropha.rec": [2, 5, 4, 6, 3, 1],
        "cgd": ['Europe:Amsterdam', 'Europe:other', 'US:NIH', 'US:other'],
        "uis": ["A", "B"],
        "scania":  [1, 2, 3, 4, 5],
        "seer": [1, 2, 3]
    }

    region_list = dataset_with_region[dataset]
    return region_list[client_id]

################################################## Get Local Dataset ######################################################


def get_local_dataset(dataset, partition, client_id, run_id, args):
    if partition == 'centralized':
        client_id = 0

    client_name = None
    if partition == "federated":
        client_name = get_client_name_for_federated_datasets(dataset, client_id)
        dataset_directory = f"{args.datadir}/{dataset}/{partition}/{client_name}/{run_id}/"
    else:
        client_name = client_id
        dataset_directory = f"{args.datadir}/{dataset}/{partition}/{client_id}/{run_id}/"
        print(f"client is is {client_id} and dataset directory is {dataset_directory}") 

    train_df = pd.read_csv(dataset_directory + "train.csv")
    val_df = pd.read_csv(dataset_directory + "val.csv")
    test_df = pd.read_csv(dataset_directory + "test.csv")

    cols_to_remove = ['time', 'event']
    new_col_name = train_df.columns.tolist()
    feature_list = [x for x in new_col_name if x not in cols_to_remove]

    return train_df, val_df, test_df, client_name, feature_list
     