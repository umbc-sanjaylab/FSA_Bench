import numpy as np
import pandas as pd
import torch
from pycox.evaluation import EvalSurv
import argparse
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import numpy as np
from datetime import datetime
import os
import json
import numpy as np
import torch
import logging
import builtins
from SurvivalEVAL.Evaluator import SurvivalEvaluator
torch.cuda.set_per_process_memory_fraction(0.15, 0)

################################################## Argument Extraction ###################################################### 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,  default='FedPDNN', help='neural network used in training') 
    parser.add_argument('--run_id', type=int, default=0, help="ID for 5-fold cross validation") 
    parser.add_argument('--dataset', type=str,  default='metabric',  help='dataset used for training')
    parser.add_argument('--partition',  type=str,  default='iid',  help='the data partitioning strategy')
    parser.add_argument('--batch_size',  type=int,  default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float,  default=0.001, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int,  default=20, help='number of local epochs')
    parser.add_argument('--n_parties',  type=int, default=5, help='number of clients in the federated setting')
    parser.add_argument('--alg',  type=str,  default='fedavg', help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int,  default=20,  help='number of maximum communication round')
    parser.add_argument('--patience',
                        type=int,
                        default=20,
                        help='number of patience')
    parser.add_argument('--eval_every',
                        type=int,
                        default=5,
                        help='compute C-index every N epochs')
    parser.add_argument('--reg',
                        type=float,
                        default=1e-5,
                        help="L2 regularization strength")
    parser.add_argument('--is_same_initial',
                        type=int,
                        default=1,
                        help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed',
                        type=int,
                        default=0,
                        help="Random seed")

    parser.add_argument('--log_file',
                        type=str,
                        default=None,
                        help='The log file name')
    parser.add_argument('--log_file_name',
                        type=str,
                        default=None,
                        help='The log file name')
    parser.add_argument('--dropout_p',
                        type=float,
                        required=False,
                        default=0.1,
                        help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir',
                        type=str,
                        required=False,
                        default="./Data/", help="Data directory")
    parser.add_argument('--logdir',
                        type=str,
                        required=False,
                        default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir',
                        type=str,
                        required=False,
                        default="./models/", help='Model directory path')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='The device to run the program')
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help='the optimizer')
    parser.add_argument('--sample',
                        type=float,
                        default=1,
                        help='Sample ratio for each communication round')
    parser.add_argument('--pseudo_interval',
                        type=int,
                        default=10,
                        help='Interval for pseudo values calculation')
    parser.add_argument('--time_horizon_interval', type=float,  default=1.0,  help='Interval for survival function calculation')
    parser.add_argument('--sensitivity',  type=float,  default=2.0,  help='sensitivity parameter')
    parser.add_argument('--epsilon', type=float, default=5.5,  help='privacy budget parameter')
    parser.add_argument('--is_DP', default=False,  action="store_true",  help='Enforcing DP in pseudo values')
    
    # experiment_id
    parser.add_argument( "--experiment_id", type=str, default=None)
    
    args = parser.parse_args()
    return args
  
def save_results_to_csv(final_val_cidx, final_test_cidx, final_test_brier, final_test_nbll, final_test_p_value, final_test_MAE_PO, total_time_counter, memory, final_round, seed, run_id, dataset, partition, logdir, model, lr, batch_size, epochs, fileName="results_final.csv"):
            
            csv_path = os.path.join(logdir, fileName)
            print(csv_path)
            
            row = {
                "comm_round": final_round,
                "seed": int(seed),
                "fold_id": int(run_id),
                "dataset": dataset,
                "partition": partition,
                "model": model,
                "val_c_idx": builtins.round(float(final_val_cidx), 4) if final_val_cidx is not None else None,
                "c_idx": builtins.round(float(final_test_cidx), 4) if final_test_cidx is not None else None,
                "brier": builtins.round(float(final_test_brier), 4) if final_test_brier is not None else None,
                "nbll": builtins.round(float(final_test_nbll), 4) if final_test_nbll is not None else None,
                "p_value": builtins.round(float(final_test_p_value), 4) if final_test_p_value is not None else None,
                "MAE_PO": builtins.round(float(final_test_MAE_PO), 4) if final_test_MAE_PO is not None else None,
                "lr": builtins.round(float(lr), 5),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "time_counter": builtins.round(float(total_time_counter), 4),
                "memory": memory,
            }

            df = pd.DataFrame([row])
            file_exists = os.path.exists(csv_path)
            df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
            
            import numpy as np
import torch

#################################### Finds whether value is finite ############################################################

def is_valid_number(value):
    if value is None:
        return False

    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False

#################################### Aggregate Model with finite parameters =#################################################

def aggregate_model_with_finite_parameters(results, logger):
    valid_results = []

    for result in results:
        valid, invalid_key = parameters_are_finite(result.params)

        if not valid:
            logger.warning(
                "Skipping client update because parameters are invalid. "
                f"Invalid parameter: {invalid_key}"
            )
            continue

        valid_results.append(result)

    if not valid_results:
        logger.error( "No finite client updates were received. Keeping the previous global model.")
        return None 
    
    return valid_results  
     

#################################### Check whether value is None ############################################################

def parameters_are_finite(params):
    """
    Returns:
        (is_valid, invalid_key)
    """
    if params is None:
        return False, "params_is_none"

    if isinstance(params, dict):
        items = params.items()
    else:
        # Adapt this part if NVFlare stores params differently.
        try:
            items = enumerate(params)
        except TypeError:
            return False, "unsupported_parameter_container"

    for key, value in items:
        if value is None:
            return False, str(key)

        try:
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError):
            return False, str(key)

        if tensor.numel() == 0:
            return False, str(key)

        if not torch.isfinite(tensor).all().item():
            return False, str(key)

    return True, None

#################################### Early Stopping Criteria ############################################################


class EarlyStopping:
    """Stop training if validation C-index does not improve for `patience` rounds."""

    def __init__(self, patience=25, verbose=False, delta=0.0, directory=None, checkpoint_pth="checkpoint.pt"):
        self.patience = int(patience)
        self.verbose = verbose
        self.directory = directory
        self.delta = float(delta)
        self.logger = logging.getLogger("fedPseudo.server")
        self.checkpoint_pth = checkpoint_pth

        self.counter = 0
        self.best_val_cindex = float("-inf")
        self.early_stop = False

        # new: keep metadata for the selected best round
        self.best_round = None
        self.best_test_cindex = None
        self.best_test_brier = None
        self.best_test_nbll = None
        self.best_test_p_value = None
        self.best_test_MAE_PO = None

    def __call__(self, val_c_index, param, round_idx=None, test_metrics=None):
        if val_c_index is None or not np.isfinite(val_c_index):
            if self.verbose:
                self.logger.info(
                    f"[EarlyStopping] Skipping invalid val_c_index = {val_c_index}")
            return False

        improved = val_c_index > self.best_val_cindex + self.delta

        if improved:
            if self.verbose:
                self.logger.info(
                    f"Validation C-index improved "
                    f"({self.best_val_cindex:.6f} -> {val_c_index:.6f}). Saving checkpoint..."
                )

            self.best_val_cindex = val_c_index
            self.counter = 0
            self.best_round = round_idx

            if test_metrics is not None:
                self.best_test_cindex = test_metrics.get("test_c_index")
                self.best_test_brier = test_metrics.get("test_brier")
                self.best_test_nbll = test_metrics.get("test_nbll")
                self.best_test_p_value = test_metrics.get("test_p_value")
                self.best_test_MAE_PO = test_metrics.get("test_MAE_PO")

            self.save_checkpoint(param)
            return True

        self.counter += 1
        if self.verbose:
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True

        return False

    def save_checkpoint(self, param):
        if param is not None:
            os.makedirs(self.directory, exist_ok=True)
            torch.save(param, os.path.join(
                self.directory, self.checkpoint_pth))

####################################### Store Local Models parameters and results ##################################################


def store_local_model_information(client_name, logdir, modeldir, net,  test_cindex, test_brier, test_nbll, local_test_p_value, local_test_weighted_MAE_PO, seed, run_id, dataset, partition, model_name, lr, batch_size, epochs):
    # create directory path
    checkpoint_dir = os.path.join(modeldir, dataset)

    # create directory if it does not exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    csv_path = os.path.join(logdir, "results_both_early_stopping_local.csv")
    print(csv_path)

    row = {
        "seed": int(seed),
        "client_name": client_name,
        "dataset": dataset,
        "partition": partition,
        "model": model_name,
        "c_idx":  round(test_cindex, 4),
        "brier":  round(test_brier, 4),
        "nbll": round(test_nbll, 4),
        "p_value": round(local_test_p_value, 4),
        "weighted_MAE_PO": round(local_test_weighted_MAE_PO, 4),
        "lr":  round(float(lr), 5),
        "batch_size": int(batch_size),
        "epochs": int(epochs)
    }

    df = pd.DataFrame([row])
    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)


####################################### Logger configuration of Nvidia Flare ########################################################

def write_logger_configuration1122(args):
    os.makedirs(args.logdir, exist_ok=True)
    log_file_path = os.path.join(args.logdir, f'experiment_{args.run_id}_{args.dataset}_{args.model}_{datetime.now().strftime("%Y-%m-%d-%H:%M-%S")}.log')
    log_config_path = os.path.join(args.logdir, f'experiment_arguments-{datetime.now().strftime("%Y-%m-%d-%H:%M-%S")}.json')

    # (C) Logging DICT config (this must be valid JSON!)
    log_cfg = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
        },

        "handlers": {
            # ---- Your app handlers (INFO+)
            "console_app": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
                "stream": "ext://sys.stdout",
            },
            "file_app": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": log_file_path,
                "level": "INFO",
                "encoding": "utf-8",
            },

            # ---- NVFlare handlers (WARNING+)
            "console_flare": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "WARNING",
                "stream": "ext://sys.stderr",
            },
            "file_flare": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": log_file_path,     # or a separate flare log file
                "level": "WARNING",
                "encoding": "utf-8",
            },
        },

        # Root can be quiet (prevents random spam)
        "root": {
            "handlers": ["console_flare", "file_flare"],
            "level": "WARNING",
        },

        "loggers": {
            # ✅ Your logger (use SAME NAME in getLogger)
            "fedPseudo": {                       # <-- match this exactly
                "handlers": ["console_app", "file_app"],
                "level": "INFO",
                "propagate": False,
            },

            # ✅ NVFlare logger(s)
            "nvflare": {
                "handlers": ["console_flare", "file_flare"],
                "level": "WARNING",
                "propagate": False,
            },
        },
    }

    with open(log_config_path, "w", encoding="utf-8") as f:
        json.dump(log_cfg, f, indent=2)

    return log_config_path, log_cfg
  
import json
import logging
import os
import re
from typing import Optional


def write_logger_configuration(
    args,
    role: str,
    experiment_id: str,
    save_config: bool = False,
):  

    safe_role = re.sub( r"[^A-Za-z0-9_.-]+", "_", str(role), )
    run_name = ( f"{args.dataset}_{args.model}_" f"fold{args.run_id}_{experiment_id}" )
    
    run_log_dir = os.path.abspath( os.path.join(args.logdir, run_name) )
    os.makedirs( run_log_dir, exist_ok=True, )

    # Deterministic filename:
    log_file_path = os.path.join( run_log_dir, f"{safe_role}.log")

    log_cfg = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "default": {
                "format": (
                    "%(asctime)s - %(process)d - "
                    "%(name)s - %(levelname)s - "
                    "%(message)s"
                )
            }
        },

        "handlers": {
            "console_info": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
                "stream": "ext://sys.stdout",
            },

            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": log_file_path,
                "level": "INFO",
                "encoding": "utf-8",

                # A restarted client continues in the same file.
                "mode": "a",
            },
        },

        "root": {
            "handlers": ["console_info", "file"],
            "level": "WARNING",
        },

        "loggers": {
            # Matches:
            # fedPseudo.server
            # fedPseudo.client0
            # fedPseudo.client1
            "fedPseudo": {
                "handlers": [
                    "console_info",
                    "file",
                ],
                "level": "INFO",
                "propagate": False,
            },

            # NVFlare messages produced inside this process
            # are saved in that process's log file.
            "nvflare": {
                "handlers": [
                    "console_info",
                    "file",
                ],
                "level": "WARNING",
                "propagate": False,
            },
        },
    }

    config_path: Optional[str] = None

    if save_config:
        config_path = os.path.join( run_log_dir, "server_logging_config.json", )

        with open( config_path, "w", encoding="utf-8", ) as file:
            json.dump(log_cfg, file, indent=2)

    return config_path, log_cfg, log_file_path

################################################## Data Builder ######################################################


class DataBuilder(Dataset):
    def __init__(self, X_covariates, pseudo_values, col_name, features_for_standardization, scaler=None):
        self.pseudo_values = torch.from_numpy(pseudo_values)
        self.scaler = None

        self.col_name = col_name
        self.features_for_standardization = features_for_standardization
        logger = logging.getLogger("fedPseudo")
        logger.info(f"stanadrdized columns are: {features_for_standardization}")

        self.X_covariates = None
        X_covariates_df = pd.DataFrame(X_covariates, columns=col_name) 
        
        logger.info(f"Length of standardized columns: {len(features_for_standardization)}") 
        self.X_covariates = torch.from_numpy(X_covariates_df.to_numpy(dtype=np.float32))

    def __len__(self):
        return len(self.X_covariates)

    def __getitem__(self, idx):
        return self.X_covariates[idx].float(), self.pseudo_values[idx].float()
 
def get_model_spec(args):

  # "FedPLSTM" "FedPAttn")

    print(f"model name is {args.model}") 
    if args.model == "FedPLSTM":
        hidden_size, num_layers = 128, 2
    elif args.model == "FedPAttn":
        hidden_size, num_layers = 128, 2
    elif args.model == "FedPDNN":
        hidden_size, num_layers = 0, 0
    else:
        raise ValueError(f"Unknown model name: {args.model}")

    input_size = get_input_feature(args)
    model_spec = {
        "path": f"model.{args.model}",
        "args": {
            "dataset": args.dataset,
            "in_features": int(input_size),
            "out_features": 9,
            "dropout": float(getattr(args, "dropout_p", 0.1)),
            "hidden_size": hidden_size,
            "num_layers": num_layers
        }
    }
    return model_spec

################################################## Get Input Feature ######################################################


def get_input_feature(args):
    if args.partition == "federated":
        client_name = get_client_name_for_federated_datasets(args.dataset, client_id=0)
        dataset_directory = f"{args.datadir}/{args.dataset}/{args.partition}/{client_name}/{args.run_id}/"
    else:
        dataset_directory = f"{args.datadir}/{args.dataset}/{args.partition}/0/{args.run_id}/"

    train_df = pd.read_csv(dataset_directory + "train.csv")
    X_train = np.array(train_df.drop( columns=["time", "event"]), dtype='float32')
    return X_train.shape[1]

######################################### Region from Federated Dataset by Client ID ########################################


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
 
def get_local_dataset(logger,  dataset, partition, client_id, run_id, args):
    if partition == 'centralized':
        client_id = 0

    client_name = None
    if partition == "federated":
        logger.info(f"client ID is {client_id}")
        client_name = get_client_name_for_federated_datasets(dataset, client_id)
        dataset_directory = f"{args.datadir}/{dataset}/{partition}/{client_name}/{run_id}/"
    else:
        dataset_directory = f"{args.datadir}/{dataset}/{partition}/{client_id}/{run_id}/"
        client_name = client_id

    train_df = pd.read_csv(dataset_directory + "train.csv")
    val_df = pd.read_csv(dataset_directory + "val.csv")
    test_df = pd.read_csv(dataset_directory + "test.csv") 
    logger.info(f"train_df shape: {train_df.shape}, val_df shape: {val_df.shape}, test_df shape: {test_df.shape}")

    cols_to_remove = ['time', 'event']
    new_col_name = train_df.columns.tolist()
    feature_list = [x for x in new_col_name if x not in cols_to_remove]

    return train_df, val_df, test_df, client_name, feature_list


######################################### Convert Torch Tensor to NumPy Array ####################################################

def to_1d_numpy(x):
    # torch tensor -> numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # pandas Series/DataFrame column -> numpy
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()

    # list/others -> numpy
    x = np.asarray(x)
    return x.reshape(-1)


################################################## Evaluation metrics ######################################################

def Evaluation(model, train_time, train_status, X_test, test_time, test_status, evaltime, logger, device="cpu"):
    """
    Attributes:
    ----------
        model: Trained Model
        train_time: Event time array for train set
        train_status: Event status array for train set

        X_test: Covariates in test set
        test_time: Observed times array for test set
        test_status: Event status array for test set

        evaltime: Prespecified evaluation times
        device: Device to run the model on

    Returns
    -------
        cindex, brier, nbll, p_value, weighted_MAE_PO
    """
    logger = logging.getLogger("fedPseudo.client")

    model.eval()
    model.to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)

    # --- Build survival DataFrames ---
    try:
        sp_test = model(X_test_tensor)
        surv_array = sp_test.cpu().detach().numpy()  # (n_samples, n_eval_times)

        # EvalSurv (pycox): expects (n_eval_times, n_samples) with evaltime as row index
        surv_for_evsurv = pd.DataFrame(
            np.transpose(surv_array),            # → (n_eval_times, n_samples)
            index=evaltime
        )

        # SurvivalEvaluator (survivaleval): expects (n_samples, n_eval_times) with evaltime as columns
        surv_for_eval = pd.DataFrame(
            surv_array,                          # → (n_samples, n_eval_times)
            columns=np.asarray(evaltime, dtype=float)
        )

        ev = EvalSurv(surv_for_evsurv, test_time,
                      test_status, censor_surv='km')

    except Exception as e:
        logger.error(
            f"[Evaluation] Failed to build survival frames: {e}", exc_info=True)
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # --- C-index ---
    try:
        cindex = float(ev.concordance_td())
        cindex = cindex if np.isfinite(cindex) else np.nan
    except Exception as e:
        logger.warning(f"[C-index WARNING] {e}")
        cindex = np.nan

    # --- Integrated Brier Score ---
    try:
        brier = float(ev.integrated_brier_score(evaltime))
        brier = brier if np.isfinite(brier) else np.nan
    except Exception as e:
        logger.warning(f"[Brier WARNING] {e}")
        brier = np.nan

    # --- Integrated NBLL ---
    try:
        nbll = float(ev.integrated_nbll(evaltime))
        nbll = nbll if np.isfinite(nbll) else np.nan
    except Exception as e:
        logger.warning(f"[NBLL WARNING] {e}")
        nbll = np.nan

    # --- SurvivalEvaluator (D-Calibration & Weighted MAE) ---
    test_time_arr = np.asarray(test_time,    dtype=float).reshape(-1)
    test_status_arr = np.asarray(test_status,  dtype=int).reshape(-1)
    train_time_arr = np.asarray(train_time,   dtype=float).reshape(-1)
    train_status_arr = np.asarray(train_status, dtype=int).reshape(-1)

    logger.info(
        f"[SurvivalEvaluator shapes] "
        # expect (n_test_samples, n_eval_times) e.g. (649, 9)
        f"surv={surv_for_eval.shape}, "
        f"evaltime={np.asarray(evaltime).shape}, "  # expect (9,)
        f"test_time={test_time_arr.shape}, "  # expect (649,)
        f"train_time={train_time_arr.shape}"  # expect (1944,)
    )

    p_value = np.nan
    weighted_MAE_PO = np.nan

    try:
        evaluator = SurvivalEvaluator(
            surv_for_eval,                        # (n_samples, n_eval_times)
            np.asarray(evaltime, dtype=float),    # (n_eval_times,)
            test_time_arr,                        # (n_samples,)
            test_status_arr,                      # (n_samples,)
            train_time_arr,                       # (n_train_samples,)
            train_status_arr,                     # (n_train_samples,)
        )

        try:
            p_val_result, _ = evaluator.d_calibration()
            p_value = float(p_val_result) if np.isfinite(
                p_val_result) else np.nan
        except Exception as e:
            logger.warning(f"[D-Calibration WARNING] {e}")

        try:
            mae_result = evaluator.mae(method='Pseudo_obs', weighted=True)
            weighted_MAE_PO = float(mae_result) if np.isfinite(
                mae_result) else np.nan
        except Exception as e:
            logger.warning(f"[Weighted MAE WARNING] {e}")

    except Exception as e:
        logger.error(
            f"[SurvivalEvaluator] Failed to initialize: {e}", exc_info=True)

    logger.info(
        f"[Evaluation] cindex={cindex}, brier={brier}, nbll={nbll}, "
        f"p_value={p_value}, weighted_MAE_PO={weighted_MAE_PO}"
    )

    return cindex, brier, nbll, p_value, weighted_MAE_PO

################################################## Concodance Index metrics ######################################################


def Concordance(model, x, durations, events, evaltime, logger, device="cpu"):
    """
    Attributes:
    ----------    
        model: Model 
        x: Covariates
        durations: Event status array 
        events: Observed times array 
        evaltime: Prespecified evaluation times

    Returns
    -------
        Time-dependent C-index
    """
    model = model.to(device)
    model.eval()
    x = x.to(device)

    surv = model(x)
    y_pred = pd.DataFrame(np.transpose(surv.cpu().detach().numpy()))
    y_pred = y_pred.set_index([evaltime])

    test_time_np = to_1d_numpy(durations).astype(np.float64)
    test_status_np = to_1d_numpy(events).astype(np.int64)
    ev = EvalSurv(y_pred, test_time_np, test_status_np, censor_surv='km') 
    
    if len(durations) < 2:
        logger.warning("C-index is not estimable: fewer than two samples.")
        return np.nan

    if np.sum(events == 1) == 0:
        logger.warning("C-index is not estimable: no observed events.")
        return np.nan


    cindex = np.nan
    try:
        cindex = float(ev.concordance_td())
        if not np.isfinite(cindex):
            logger.info("[C-index WARNING] NaN/Inf returned.")
            cindex = np.nan
    except ZeroDivisionError:
        print("[C-index WARNING] Zero comparable pairs in validation set.")
        cindex = np.nan

    return cindex


################################################## Pseudo Value based Loss Function ##############################################
def pseudo_loss(output, target):
    loss = torch.mean(target*(1-2*output)+(output**2))
    return loss


################################################## Model Save and Load ###########################################################

def save_model(model, model_index, args):
    logger = logging.getLogger("fedPseudo.client")
    logger.info(f"saving local model-{model_index}")
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model
