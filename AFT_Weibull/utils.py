import numpy as np
import pandas as pd
import torch
import random
from model import *
from pycox.evaluation import EvalSurv
import torch.nn as nn
import torch
import numpy as np
import traceback

import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from SurvivalEVAL.Evaluator import SurvivalEvaluator
torch.cuda.set_per_process_memory_fraction(0.15, 0)


############################################# Store local training metrics #########################################################################


def store_results_in_csv(args, metrics_dict):
    csv_path = os.path.join(args.logdir, "results_local_training.csv")
    print(csv_path)

    row = {
        "seed": args.init_seed,
        "fold_id": int(args.run_id),
        "dataset": args.dataset,
        "partition": args.partition,
        "model": args.model,
        "c_idx": round(metrics_dict["c_idx"], 4),
        "brier": round(metrics_dict["brier"], 4),
        "nbll": round(metrics_dict["nbll"], 4),
        "p_value": round(metrics_dict["p_value"], 4),
        "weighted_MAE_PO": round(metrics_dict["weighted_MAE_PO"], 4),
        "lr": round(float(args.lr), 5),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "dropout": float(args.dropout_p)
    }

    df = pd.DataFrame([row])
    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)

#################################### Early Stopping Criteria ############################################################


class EarlyStopping:
    """Stop training if validation C-index does not improve for `patience` rounds."""

    def __init__(self, patience=25, verbose=False, delta=0.0, directory=None, checkpoint_pth="checkpoint.pt"):
        self.patience = int(patience)
        self.verbose = verbose
        self.directory = directory

        self.delta = float(delta)
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
                print(
                    f"[EarlyStopping] Skipping invalid val_c_index = {val_c_index}")
            return False

        improved = val_c_index > self.best_val_cindex + self.delta

        if improved:
            if self.verbose:
                print(
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
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True

        return False

    def save_checkpoint(self, param):
        if param is not None:
            os.makedirs(self.directory, exist_ok=True)
            torch.save(param, os.path.join(
                self.directory, self.checkpoint_pth))


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

#################################### Check whether there is any comparable paris ###############################################


def has_comparable_pairs(durations, events):
    """
    Attributes:
    ----------    
        durations: Observed times array 
        events: Event status array 

    Returns
    -------
        Boolean flag whether there is comparable paris 
    """

    durations = np.asarray(durations).reshape(-1)
    events = np.asarray(events).reshape(-1)

    n = len(durations)
    comparable = 0

    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            if durations[i] < durations[j]:
                comparable += 1

    return comparable > 0

#################################### Compute Concordance Score from validation data ###############################################


def Concordance(model, x, durations, events, evaltime, device="cpu"):
    """
    Attributes:
    ----------
        model: Model
        x: Covariates
        durations: Observed times array
        events: Event status array
        evaltime: Prespecified evaluation times
    Returns
    -------
        Time-dependent C-index
    """

    model = model.to(device)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)

    durations = np.asarray(durations).reshape(-1)
    events = np.asarray(events).reshape(-1)
    evaltime = np.asarray(evaltime).reshape(-1)

    if len(durations) < 2:
        return np.nan

    if np.sum(events == 1) == 0:
        return np.nan

    if not has_comparable_pairs(durations, events):
        return np.nan

    model.eval()
    with torch.no_grad():
        log_params = model(x).detach().cpu().numpy()

    log_scale = log_params[:, 0]
    log_shape = log_params[:, 1]

    scale = np.exp(log_scale)
    shape = np.exp(log_shape)

    eps = 1e-12
    scale = np.clip(scale, eps, 1e12)
    shape = np.clip(shape, eps, 1e12)

    surv_np = np.exp(
        - (evaltime[np.newaxis, :] /
           scale[:, np.newaxis]) ** shape[:, np.newaxis]
    )
    surv_np = np.clip(surv_np, eps, 1.0)

    if surv_np.shape[0] != len(durations):
        raise ValueError(
            f"Sample mismatch: surv_np has {surv_np.shape[0]} rows, "
            f"but durations has length {len(durations)}"
        )

    if surv_np.shape[1] != len(evaltime):
        raise ValueError(
            f"Time mismatch: surv_np has {surv_np.shape[1]} columns, "
            f"but evaltime has length {len(evaltime)}"
        )

    y_pred = pd.DataFrame(surv_np.T, index=evaltime)
    ev = EvalSurv(y_pred, durations, events, censor_surv="km")

    try:
        cindex = float(ev.concordance_td())
        if not np.isfinite(cindex):
            print("[C-index WARNING] NaN/Inf returned.")
            return np.nan
        return cindex
    except ZeroDivisionError:
        print("[C-index WARNING] Zero comparable pairs in validation set.")
        return np.nan

################################################## Data Builder ######################################################


class DataBuilder(Dataset):
    def __init__(self, covariates, time, status, feature_list):
        self.feature_list = feature_list
        X_covariates_df = pd.DataFrame(covariates, columns=feature_list)
        self.X_covariates = torch.from_numpy(
            np.array(X_covariates_df, dtype=np.float32))

        self.time = time
        self.status = status

    def __len__(self):
        return len(self.X_covariates)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.X_covariates[idx], dtype=torch.float32),
            torch.as_tensor(self.time[idx], dtype=torch.float32),
            torch.as_tensor(self.status[idx], dtype=torch.float32))

################################################## Evaluation metrics ######################################################


def Evaluation(model, train_time, train_status, X_test, test_time, test_status, evaltime, device="cpu"):
    """
    Attributes:
    ----------    
        model: Trained Model 
        train_time: Observed times array for training set
        train_status: Event status array for training set

        X_test: Covariates in test set
        test_status: Event status array for test set
        test_time: Observed times array for test set

        evaltime: Prespecified evaluation times

    Returns
    -------
        Time-dependent C-index and Integrated Brier Score 
    """
    ################ Performance Measure on entire test data #####################
    model.to(device)
    model.eval()

    X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)

    durations = np.asarray(test_time).reshape(-1)
    events = np.asarray(test_status).reshape(-1)
    evaltime = np.asarray(evaltime).reshape(-1)

    # if len(durations) < 2:
    #     return np.nan, np.nan, np.nan

    # if np.sum(events == 1) == 0:
    #     return np.nan, np.nan, np.nan

    # if not has_comparable_pairs(durations, events):
    #     return np.nan, np.nan, np.nan

    with torch.no_grad():
        log_params_test = model(X_test).detach().cpu().numpy()

    log_scale = log_params_test[:, 0]
    log_shape = log_params_test[:, 1]

    scale = np.exp(log_scale)
    shape = np.exp(log_shape)

    eps = 1e-12
    scale = np.clip(scale, eps, 1e12)
    shape = np.clip(shape, eps, 1e12)

    surv_test_np = np.exp(
        - (evaltime[np.newaxis, :] /
           scale[:, np.newaxis]) ** shape[:, np.newaxis]
    )

    surv_test_np = np.clip(surv_test_np, eps, 1.0)

    if surv_test_np.shape[0] != len(durations):
        raise ValueError(
            f"Sample mismatch: surv has {surv_test_np.shape[0]} rows, "
            f"but durations has length {len(durations)}"
        )

    if surv_test_np.shape[1] != len(evaltime):
        raise ValueError(
            f"Time mismatch: surv has {surv_test_np.shape[1]} columns, "
            f"but evaltime has length {len(evaltime)}"
        )

    y_pred = pd.DataFrame(surv_test_np.T, index=evaltime)
    surv_for_eval = pd.DataFrame(
        surv_test_np,                          # → (n_samples, n_eval_times)
        columns=np.asarray(evaltime, dtype=float)
    )

    ev = EvalSurv(y_pred, durations, events, censor_surv="km")

    cindex = np.nan
    try:
        cindex = float(ev.concordance_td())
        if not np.isfinite(cindex):
            print("[C-index WARNING] NaN/Inf returned.")
            cindex = np.nan
    except ZeroDivisionError:
        print("[C-index WARNING] Zero comparable pairs.")
        cindex = np.nan

    brier = np.nan
    try:
        brier = float(ev.integrated_brier_score(evaltime))
        if not np.isfinite(brier):
            print("[Brier Score WARNING] NaN/Inf returned.")
            brier = np.nan
    except ZeroDivisionError:
        print("[Brier WARNING] Could not compute integrated Brier score.")
        brier = np.nan

    nbll = np.nan
    try:
        nbll = float(ev.integrated_nbll(evaltime))
        if not np.isfinite(nbll):
            print("[Negative Log-Likelihood WARNING] NaN/Inf returned.")
            nbll = np.nan
    except ZeroDivisionError:
        print("[Negative Log-Likelihood WARNING] Could not compute integrated NBLL.")
        nbll = np.nan

    # surv_for_eval
    # --- SurvivalEvaluator (D-Calibration & Weighted MAE) ---
    test_time_arr = np.asarray(test_time,    dtype=float).reshape(-1)
    test_status_arr = np.asarray(test_status,  dtype=int).reshape(-1)
    train_time_arr = np.asarray(train_time,   dtype=float).reshape(-1)
    train_status_arr = np.asarray(train_status, dtype=int).reshape(-1)

    print(
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
            print(f"[D-Calibration WARNING] {e}")

        try:
            mae_result = evaluator.mae(method='Pseudo_obs', weighted=True)
            weighted_MAE_PO = float(mae_result) if np.isfinite(
                mae_result) else np.nan
        except Exception as e:
            print(f"[Weighted MAE WARNING] {e}")

    except Exception as e:
        print(f"[SurvivalEvaluator] Failed to initialize: {e}")

    print(f"[Evaluation] cindex={cindex}, brier={brier}, nbll={nbll}, "
          f"p_value={p_value}, weighted_MAE_PO={weighted_MAE_PO}"
          )

    return cindex, brier, nbll, p_value, weighted_MAE_PO

########################################## Parameters from trained model ################################################################


def get_trainable_parameters(net):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # logger.info("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    N = 0
    for params in paramlist:
        N += params.numel()
    X = torch.empty(N, dtype=torch.float64)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset +
                numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset += numel
    return X

########################################## Parameters to PyTorch Model ################################################################


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset += numel

########################################## Store the PyTorch Model in storage ############################################################


def save_model(model, model_index, args):
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

########################################## Retrieve the PyTorch Model from storage ########################################################


def load_model(model, model_index, device="cpu"):
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if (type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif (type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    return

########################################## Pseudo Value based Loss Function ##############################################################


def pseudo_loss(output, target):
    loss = torch.mean((target-output)**2)
    return loss
