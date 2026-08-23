import os
import logging
import numpy as np
import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split
from pycox.evaluation import EvalSurv
import torch.nn as nn
import random
from pycox.models import DeepHit
import torchtuples as tt
from collections import defaultdict
import h5py
import copy
import numpy as np
from pycox.evaluation import EvalSurv
import torchtuples as tt
from SurvivalEVAL.Evaluator import SurvivalEvaluator
import traceback 

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
                print(f"[EarlyStopping] Skipping invalid val_c_index = {val_c_index}")
            return False

        improved = val_c_index > self.best_val_cindex + self.delta
        if improved:
            if self.verbose:
                print(
                    f"Validation C-index improved "
                    f"({self.best_val_cindex:.6f} -> {val_c_index:.6f}). Saving checkpoint...")

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
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

        if self.counter >= self.patience:
              self.early_stop = True

        return False

    def save_checkpoint(self, param):
        if param is not None:
            os.makedirs(self.directory, exist_ok=True)
            torch.save(param, os.path.join(self.directory, self.checkpoint_pth))

#################################### Early Stopping Criteria ############################################################


class EarlyStopCIndex(tt.callbacks.Callback):
    """
    Early stop on validation time-dependent C-index (maximize).
    Works for DeepHit via model.predict_surv_df(X_val).
    """
    # [ EarlyStopCIndex( X_val=X_val, val_time=val_time, val_status=val_status, patience=args.patience, check_every=1, verbose=True)]

    def __init__(self, X_val, val_time, val_status, patience=25, check_every=1, verbose=True):
        super().__init__()
        self.X_val = X_val
        self.val_time = np.asarray(val_time).reshape(-1)
        self.val_status = np.asarray(val_status).reshape(-1)

        self.patience = int(patience)
        self.check_every = int(check_every)
        self.verbose = verbose

        self.best_cindex = -np.inf
        self.best_state = None
        self.num_bad_checks = 0

    def on_epoch_end(self, **kwargs):
        epoch = int(kwargs.get("epoch", -1))
        if epoch < 0:
            return

        # only check every N epochs
        if (epoch + 1) % self.check_every != 0:
            return

        # survival df: (n_times, n_samples)
        surv_df = self.model.predict_surv_df(self.X_val).clip(1e-8, 1 - 1e-8)

        # consider "event 1" as event-of-interest for c-index
        events = (self.val_status == 1).astype(int)

        ev = EvalSurv(surv_df, self.val_time, events, censor_surv='km')
        cindex = float(ev.concordance_td())

        improved = cindex > self.best_cindex
        if improved:
            self.best_cindex = cindex
            self.best_state = copy.deepcopy(self.model.net.state_dict())
            self.num_bad_checks = 0
        else:
            self.num_bad_checks += 1

        if self.verbose:
            print(f"[EarlyStopCIndex] epoch={epoch+1} val_cindex={cindex:.4f} "
                  f"best={self.best_cindex:.4f} bad={self.num_bad_checks}/{self.patience}")

        # STOP condition
        if self.num_bad_checks > self.patience:
            if self.verbose:
                print(
                    f"[EarlyStopCIndex] Early stopping triggered at epoch={epoch+1}")

            # torchtuples uses this flag
            self.model.stop_training = True

    def on_fit_end(self, **kwargs):
        if self.best_state is not None:
            self.model.net.load_state_dict(self.best_state)
            if self.verbose:
                print(
                    f"[EarlyStopCIndex] Restored best model with val_cindex={self.best_cindex:.4f}")


#################################### Creating new directory ############################################################

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


########################################## Evaluation using Test Data ################################################################


def Evaluation(model, X_test, test_time, test_status, evaltime, device="cpu"):
    """
    Returns:
        cindex, brier, nbll
    If any metric fails, return np.nan for that metric.
    """

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        sp_test = model(X_test_tensor)

    surv_np = sp_test.detach().cpu().numpy()

    if surv_np.ndim == 3 and surv_np.shape[1] == 1:
        surv_np = surv_np.squeeze(axis=1)

    if surv_np.shape[1] != len(evaltime):
        raise ValueError(
            f"Time mismatch: surv_np has {surv_np.shape[1]} columns, "
            f"but evaltime has length {len(evaltime)}"
        )

    surv = pd.DataFrame(surv_np.T, index=evaltime)

    ev = EvalSurv(
        surv,
        np.asarray(test_time).reshape(-1),
        np.asarray(test_status).reshape(-1),
        censor_surv="km",
    )

    try:
        cindex = float(ev.concordance_td())
        if not np.isfinite(cindex):
            print("[C-index ERROR] concordance_td returned NaN/Inf")
            cindex = np.nan
    except Exception as e:
        print(f"[C-index ERROR] {type(e).__name__}: {e}")
        cindex = np.nan

    try:
        brier = float(ev.integrated_brier_score(evaltime))
        if not np.isfinite(brier):
            print("[Brier ERROR] integrated_brier_score returned NaN/Inf")
            brier = np.nan
    except Exception as e:
        print(f"[Brier ERROR] {type(e).__name__}: {e}")
        brier = np.nan

    try:
        nbll = float(ev.integrated_nbll(evaltime))
        if not np.isfinite(nbll):
            print("[NBLL ERROR] integrated_nbll returned NaN/Inf")
            nbll = np.nan
    except Exception as e:
        print(f"[NBLL ERROR] {type(e).__name__}: {e}")
        nbll = np.nan

    return cindex, brier, nbll

#################################### Evaluation of Trained Model using Test Data ####################################### 


def _safe(fn, *fn_args, label='', **fn_kwargs):
    try:
        v = float(fn(*fn_args, **fn_kwargs))
        return v if np.isfinite(v) else np.nan
    except Exception as e:
        print(f"[{label}] {type(e).__name__}: {e}")
        return np.nan 

def Evaluation(model, train_time, train_status, X_test, test_time, test_status, evaltime, labtrans, device):

    cindex, brier, nbll, p_value, weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan
    cif = model.predict_cif(X_test)  
    
    eval_times_arr = np.asarray(evaltime, dtype=float)
    duration_index = np.asarray(model.duration_index, dtype=float)

    # Snap evaltime to nearest cut points available in duration_index
    valid_eval_times = eval_times_arr[eval_times_arr >= duration_index[0]]
    pos = np.searchsorted(duration_index, valid_eval_times, side="right") - 1 
    pos = np.clip(pos, 0, len(duration_index) - 1)

    # Remove duplicate snapped positions
    pos = np.unique(pos) 
    snapped_eval_times = duration_index[pos]
     
    cif_event_one = pd.DataFrame(cif[0], index=model.duration_index)     # (n_timepoints, n_samples)    
    cif_for_eval_times = cif_event_one.iloc[pos]
    
    cif_for_eval_times.index = snapped_eval_times
    surv_for_eval_times = 1 - cif_for_eval_times

    ######## EvalSurv metrics ################################################## 
    ev = EvalSurv(surv_for_eval_times, test_time, test_status, censor_surv='km')
    try:
        cindex = float(ev.concordance_td())
        cindex = cindex if np.isfinite(cindex) else np.nan
    except Exception as e:
        logger.warning(f"[C-index WARNING] {e}")
        cindex = np.nan 
    
    ######### Integrated Brier Score ################################
    try:
        brier = float(ev.integrated_brier_score(snapped_eval_times))
        brier = brier if np.isfinite(brier) else np.nan
    except Exception as e:
        logger.warning(f"[Brier WARNING] {e}")
        brier = np.nan 
        
    try:
        nbll = float(ev.integrated_nbll(snapped_eval_times))
        nbll = nbll if np.isfinite(nbll) else np.nan
    except Exception as e:
        logger.warning(f"[NBLL WARNING] {e}")
        nbll = np.nan 
    
    test_time_arr = np.asarray(test_time,    dtype=float).reshape(-1)
    test_status_arr = np.asarray(test_status,  dtype=int).reshape(-1)
    train_time_arr = np.asarray(train_time,   dtype=float).reshape(-1)
    train_status_arr = np.asarray(train_status, dtype=int).reshape(-1)
      
    surv_for_eval = pd.DataFrame( surv_for_eval_times.T.to_numpy(), columns=snapped_eval_times) 
    try:
        evaluator = SurvivalEvaluator(surv_for_eval, snapped_eval_times, test_time_arr, test_status_arr, train_time_arr, train_status_arr)
    except Exception as e:
        print(f"[SurvivalEvaluator Initialization WARNING] {e}")
        return cindex, brier, nbll, p_value, weighted_MAE_PO
       
    try:
        p_value, _ = evaluator.d_calibration()
        p_value = float(p_value) if np.isfinite(p_value) else np.nan
    except Exception as e:
        print(f"[D-Calibration WARNING] {e}")
        p_value = np.nan

    try:
        mae_result = evaluator.mae(method="Pseudo_obs", weighted=True)
        weighted_MAE_PO = float(mae_result) if np.isfinite(mae_result) else np.nan
    except Exception as e:
        print(f"[Weighted MAE WARNING] {e}")
        weighted_MAE_PO = np.nan
 

    print(f"[Evaluation] cindex={cindex:.4f}, brier={brier:.4f}, nbll={nbll:.4f}, p_value={p_value:.4f}, weighted_MAE_PO={weighted_MAE_PO:.4f}") 
    return cindex, brier, nbll, p_value, weighted_MAE_PO

# def Evaluation(model, train_time, train_status, X_test, test_time, test_status, evaltime, labtrans, device):
#     """
#     Attributes:
#     ----------    
#         model: Trained DeepPseudo Model 
#         data_event_train: Event status array for train set
#         data_time_train: Observed times array for train set
#         data_X_test: Covariates in test set
#         data_event_test: Event status array for test set
#         data_time_test: Observed times array for test set
#         evaltime: Prespecified evaluation times

#     Returns
#     -------
#         Time-dependent C-index, Integrated Brier Score and Integrated nbll score 
#     """

#     ################ Performance Measure on entire test data #####################

#     # ── 1. Predict & normalise CIF ──────────────────────────────────────────
#     cif = np.asarray(model.predict_cif(X_test))
#     print("raw cif shape:", cif.shape)
    
#     cif1 = cif[0] if cif.ndim == 3 else cif
#     n_times = len(model.duration_index)
#     if cif1.shape[0] != n_times:
#         cif1 = cif1.T
        
#     print("cif1 shape after fix:", cif1.shape)
#     print("min CIF:", cif1.min(), "max CIF:", cif1.max())
#     print("std at last timepoint:", cif1[-1, :].std())
   
#     # → (n_timepoints, n_samples)
#     cif1 = pd.DataFrame(cif1, index=model.duration_index)
#     # Debug AFTER conversion
#     print("CIF first patient (should be increasing):", cif1.iloc[:, 0].values)
#     print("CIF last patient:", cif1.iloc[:, -1].values)

#     # ── 2. pycox metrics (C-index, Brier, NBLL) ────────────────────────────
#     ev = EvalSurv(1 - cif1, test_time, test_status == 1, censor_surv='km')
#     eval_times_arr = np.asarray(evaltime, dtype=float)

#     def _safe(fn, label):
#         try:
#             v = float(fn())
#             return v if np.isfinite(v) else (print(f"[{label}] returned NaN/Inf") or np.nan)
#         except Exception as e:
#             print(f"[{label}] {type(e).__name__}: {e}")
#             return np.nan

#     cindex = _safe(ev.concordance_td,                              "C-index")
#     brier  = _safe(lambda: ev.integrated_brier_score(eval_times_arr), "Brier")
#     nbll   = _safe(lambda: ev.integrated_nbll(eval_times_arr),        "NBLL")

#     # ── 3. Build survival matrix for SurvivalEvaluator ─────────────────────
#     try:
#         duration_index = np.asarray(model.duration_index, dtype=float)
#         surv_at_eval = np.stack([
#             np.interp(eval_times_arr, duration_index, (1 - cif1).iloc[:, i].values,
#                       left=1.0, right=0.0)
#             for i in range(cif1.shape[1])
#         ])                                               # (n_samples, n_eval_times)
#         surv_at_eval = np.clip(np.nan_to_num(surv_at_eval, nan=0.0), 0.0, 1.0)
#         surv_for_eval = pd.DataFrame(surv_at_eval, columns=eval_times_arr)
#     except Exception as e:
#         print(f"[surv_for_eval] {e}")
#         return cindex, brier, nbll, np.nan, np.nan

#     # ── 4. SurvivalEvaluator metrics (D-cal, Weighted MAE) ─────────────────
#     p_value, weighted_MAE_PO = np.nan, np.nan
#     try:
#         evaluator = SurvivalEvaluator(surv_for_eval, eval_times_arr,
#             np.asarray(test_time,    dtype=float).ravel(),
#             np.asarray(test_status,  dtype=int).ravel(),
#             np.asarray(train_time,   dtype=float).ravel(),
#             np.asarray(train_status, dtype=int).ravel(),
#         )
#         p_value  = _safe(lambda: evaluator.d_calibration()[0], "D-Calibration")
#         weighted_MAE_PO = _safe(lambda: evaluator.mae(method='Pseudo_obs', weighted=True), "Weighted MAE")
#     except Exception as e:
#         print(f"[SurvivalEvaluator] {e}")

#     print(f"[Evaluation] cindex={cindex:.4f}, brier={brier:.4f}, nbll={nbll:.4f}, "
#           f"p_value={p_value:.4f}, weighted_MAE_PO={weighted_MAE_PO:.4f}")

#     return cindex, brier, nbll, p_value, weighted_MAE_PO


#################################### Evaluation of Global Model using Test Data #######################################

def Evaluation_final(net, optim,  alpha, sigma, X_test, test_time, test_status, evaltime, labtrans, device):
    """
    Attributes:
    ----------    
    model: Trained DeepPseudo Model 
    data_event_train: Event status array for train set
    data_time_train: Observed times array for train set
    data_X_test: Covariates in test set
    data_event_test: Event status array for test set
    data_time_test: Observed times array for test set
    evaltime: Prespecified evaluation times

    Returns
    -------
    Time-dependent C-index, Integrated Brier Score and Integrated nbll score 
    """
    ################ Performance Measure on entire test data #####################

    model = DeepHit(net.to(device), optim, alpha=alpha, sigma=sigma,
                    duration_index=labtrans.cuts)
    cif = model.predict_cif(X_test)
    cif = np.asarray(cif)
    cif1 = cif[0] if cif.ndim == 3 else cif

    n_dur = len(model.duration_index)
    if cif1.shape[0] != n_dur and cif1.shape[1] == n_dur:
        cif1 = cif1.T

    cif1 = pd.DataFrame(cif1, model.duration_index)
    ev1 = EvalSurv(1-cif1, test_time, test_status == 1, censor_surv='km')

    # cindex=ev1.concordance_td()
    # time_grid = evaltime ## Grid of times for calculating brier score.
    # brier=ev1.integrated_brier_score(time_grid)

    try:
        cindex = float(ev1.concordance_td())
    except Exception as e:
        print(f"[C-index ERROR] {type(e).__name__}: {e}")
        cindex = 0.0

    # -------- Brier score (independent) --------
    try:
        brier = float(ev1.integrated_brier_score(evaltime))
    except Exception as e:
        print(f"[Brier ERROR] {type(e).__name__}: {e}")
        brier = 0.0

    return cindex, brier


#################################### Load Dataset from Dictionary #######################################

def load_datasets(dataset_file):
    datasets = defaultdict(dict)

    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets


#################################### One-hot Encoding ##################################################

def one_hot_encoder(data, encode):
    """
    Attributes:
    ----------    
    data: A dataframe 
    encode: A list of columns in the dataframe to be one-hot-encoded

    Returns
    -------
    An one-hot-encoded dataframe
    """
    data_encoded = data.copy()
    encoded = pd.get_dummies(data_encoded, prefix=encode,
                             columns=encode, drop_first=False)
    return encoded

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
        dataset_directory = f"{args.datadir}/{dataset}/{partition}/{client_id}/{run_id}/"
        client_name = client_id

    train_df = pd.read_csv(dataset_directory + "train.csv")
    val_df = pd.read_csv(dataset_directory + "val.csv")
    test_df = pd.read_csv(dataset_directory + "test.csv")

    cols_to_remove = ['time', 'event']
    new_col_name = train_df.columns.tolist()
    feature_list = [x for x in new_col_name if x not in cols_to_remove]

    return train_df, val_df, test_df, client_name, feature_list


#################################### Get Local Dataset from Directory ####################################

def get_trainable_parameters(net):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # logger.info("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    N = 0
    for params in paramlist:
        N += params.numel()
        # logger.info("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset +
                numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset += numel
    # logger.info("get trainable x:", X)
    return X


#################################### Get Local Dataset from Directory ####################################

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


#################################### Get Local Dataset from Directory ####################################

def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


#################################### Get Local Dataset from Directory ####################################

def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model


#################################### Get Local Dataset from Directory ####################################

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


###########################  Calculate the maximum survival times of the local clients  ###################################
def local_max_time(client_time):
    max_time = np.max(client_time)
    return max_time


#################################### Get Local Dataset from Directory #####################################################
def global_max_time(num_clients, net_dataidx_map, time):
    """
    Arguments:
        num_clients: Total number of clients
        net_dataidx_map: index mapping dictionary
        time: entire training time        
   """

    max_times = np.zeros((num_clients))
    for i in range(num_clients):
        max_times[i] = local_max_time(time[net_dataidx_map[i]])
    global_max_time = np.max(max_times)
    return global_max_time


#################################### Get Local Dataset from Directory #####################################################

def Pseudo_Evaltime1(global_max_time, interval):
    """
    Arguments:
        global_max_time: maximum survival time
        pseudo_df_train: train data with time and status
        time_horizon_interval: Equal distant interval for dividing the time horizon
        pseudo_perc_interval:interval for percentile of time at which pseudo values will be calculated 
    Returns:
        A vector of the evaluation time points at which pseudo values will be calculated.
    """
    PseudoEvalTime = np.percentile(
        np.arange(0.0, int(global_max_time), 1), np.arange(interval, 99, interval))
    return PseudoEvalTime
