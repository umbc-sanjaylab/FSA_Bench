# bash scripts/run_fl.sh NNnPH Pbc3 0
# bash scripts/run_cl.sh NNnPH Pbc3 0 

# bash scripts/run_fl.sh   
# model → dataset → fold/run

import argparse
from pathlib import Path

#################################### Dataset Directories ############################################################################ 

DATASET_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = DATASET_ROOT / "datasets/raw_data"
PROCESSED_DATA_DIR = DATASET_ROOT / "datasets/processed_data"  

#################################### Configurations & Properties of Survival Datasets ################################################ 
 
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


#################################### Regions name for Federated Datasets ############################################################ 

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


#################################### Arguments for Survival Model Experiments ############################################################ 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CoxPH', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='Aids2', help='dataset used for training')
    parser.add_argument( '--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy')
    parser.add_argument('--Pseudo_value', type=str, default='non-DP', help='the pseudo value calculation strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)') 
    parser.add_argument("--lr", type=float, default=1e-3, help='learning rate (default: 0.01)') 
    parser.add_argument("--epochs", type=int, default=250, help='number of local epochs') 
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg', help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication round')
    parser.add_argument('--patience', type=int, default=25, help='number of patience')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=5, help="Random seed")
    parser.add_argument('--run_id', type=int, default=0, help="fold number of 5-fold cross validation") 
    parser.add_argument("--dropout", type=float,  required=False, default=0.05, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./Fed_CoxPH/data/", help="Data directory") 
    parser.add_argument("--reg", type=float, default=5e-5, help="L2 regularization strength") 
    parser.add_argument('--logdir', type=str, required=False, default="./Fed_CoxPH/logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./Fed_CoxPH/models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1.0, help='Sample ratio for each communication round')
    parser.add_argument('--pseudo_perc_interval', type=int, default=10,  help='Interval for pseudo values calculation')
    parser.add_argument('--epsilon', type=float, default=5.0, help='epsilon')

    args = parser.parse_args()
    return args 