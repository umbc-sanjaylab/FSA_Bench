import os
import logging
import numpy as np
import pandas as pd
import torch
from pycox.evaluation import EvalSurv
import torch.nn as nn
import warnings
from pycox.models import CoxPH
import torchtuples as tt
from torch import Tensor
import numpy as np
from lifelines import KaplanMeierFitter
import torch.nn.functional as F
from SurvivalEVAL.Evaluator import SurvivalEvaluator
import traceback
import builtins

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
                self.logger.info(f"Validation C-index improved  ({self.best_val_cindex:.6f} -> {val_c_index:.6f}). Saving checkpoint...")

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
        "c_idx": builtins.round(metrics_dict["c_idx"], 4),
        "brier": builtins.round(metrics_dict["brier"], 4),
        "nbll": builtins.round(metrics_dict["nbll"], 4),
        "p_value": builtins.round(metrics_dict["p_value"], 4),
        "weighted_MAE_PO": builtins.round(metrics_dict["weighted_MAE_PO"], 4),
        "lr": builtins.round(float(args.lr), 5),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "dropout": float(args.dropout_p)
    }

    df = pd.DataFrame([row])
    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)

######################################### Create a new directory ####################################################


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


############################### Scale the loss to make it data size invariant #######################################

def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    raise ValueError(
        f"`reduction` = {reduction} is not valid. Use 'none', or 'mean'.")


######################################### Negative Log Likelihood ####################################################


def negative_llh(phi: Tensor, idx_durations: Tensor, events: Tensor,  reduction: str = 'mean') -> Tensor:
    """
    Arguments:
        phi -- N x T. Estimates in (-inf, inf) where hazard = sigmoid(phi). 
        idx_durations -- N. Event times for each individual represented as indices based on discretisation.
        events -- N. Indicator of event (1.) or censoring (0.) for each individual. Floats.
    Returns:
        Negative LLH as a 1dim Tensor
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`." +
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`," +
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    # stack events and idx_durations to get Nx1 Tensor
    # .view(-1, 1) -- infer first dimension to get second dimension as size=1
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    # create NxT zero Tensor and put in each row at the idx_duration whether the event occurred
    # censored ppl have row of zeros, uncensored ppl have row with a single one
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events.float())
    # with logits to pass phi through a sigmoid first -- recall it's unconstrained
    bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')
    # sum the loss until failure time:
    # first sum until each time
    # then gather the summed loss at relevant failure time
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)


################################## Evaluation metrics ################################################################

def _get_target(df):
    """Return (time_array, status_array) from a DataFrame whose first two
    columns are time and event."""
    return (df.iloc[:, 0].values, df.iloc[:, 1].values)


# ── Evaluation ───────────────────────────────────────────────────────────────

def Evaluation( model, train_time, train_status,  X_train, y_train,     X_eval,  eval_time,  eval_status,   evaltime,  device):
    """
    Compute survival metrics for one set (val or test).

    Parameters
    ----------
    model        : pycox CoxPH wrapper (already has global weights loaded)
    train_time   : 1-D array, training observed times
    train_status : 1-D array, training event indicators
    
    X_train      : 2-D array, training covariates  ← used ONLY for baseline hazard
    y_train      : tuple (train_time, train_status) ← pycox format
    
    X_eval       : 2-D array, covariates for the set being evaluated
    eval_time    : 1-D array, observed times for the eval set
    eval_status  : 1-D array, event indicators for the eval set
    evaltime     : 1-D array-like, fixed evaluation time grid
    device       : str, kept for interface consistency

    Returns
    -------
    cindex, brier, nbll, p_value, weighted_MAE_PO  (all float or np.nan)
    """
    # ── baseline hazard (on training data, as required by Cox) ──────────────
    _ = model.compute_baseline_hazards(X_train, y_train)
    
    # ── DIAGNOSTIC: remove once nan is resolved ──────────────────────────
    baseline = model.baseline_hazards_
    print(f"[DEBUG] baseline_hazards_ shape: {baseline.shape}, "
          f"any_nan: {baseline.isna().any().any()}, "
          f"any_inf: {np.isinf(baseline.values).any()}")

    risk_scores = model.predict(X_eval)
    print(f"[DEBUG] risk scores — min: {risk_scores.min():.4f}, "
          f"max: {risk_scores.max():.4f}, "
          f"nan_count: {np.isnan(risk_scores).sum()}")

    # ── survival predictions on the eval set ────────────────────────────────
    survival = model.predict_surv_df(X_eval)
    index_surv = np.array(survival.index)

    # FIX 1: early-exit now returns 5 values (was 3)
    if index_surv.size == 0:
        print("[Evaluation WARNING] survival.index is empty — returning nan metrics.")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Map each requested eval time to the closest available index
    new_idx = []
    for t in evaltime:
        pos = int(np.clip(  np.searchsorted(index_surv, t, side="right") - 1,  0, len(index_surv) - 1, ))
        new_idx.append(index_surv[pos])
    new_idx = np.array(new_idx)
    surv = survival.loc[new_idx]
    
    survival = model.predict_surv_df(X_eval)

    # ── pycox metrics via EvalSurv ───────────────────────────────────────────
    ev = EvalSurv(surv, eval_time, eval_status, censor_surv="km")

    try:
        cindex = float(ev.concordance_td())
        if not np.isfinite(cindex):
            print("[C-index WARNING] non-finite value returned.")
            cindex = np.nan
    except Exception as e:
        print(f"[C-index ERROR] {type(e).__name__}: {e}")
        cindex = np.nan

    try:
        brier = float(ev.integrated_brier_score(evaltime))
        if not np.isfinite(brier):
            print("[Brier WARNING] non-finite value returned.")
            brier = np.nan
    except Exception as e:
        print(f"[Brier ERROR] {type(e).__name__}: {e}")
        brier = np.nan

    try:
        # FIX 2: cast nbll to float consistently with cindex / brier
        nbll = float(ev.integrated_nbll(evaltime))
        if not np.isfinite(nbll):
            print("[NBLL WARNING] non-finite value returned.")
            nbll = np.nan
    except Exception as e:
        print(f"[NBLL ERROR] {type(e).__name__}: {e}")
        nbll = np.nan

    # ── SurvivalEvaluator (d-calibration + MAE) ─────────────────────────────
    p_value = np.nan
    weighted_MAE_PO = np.nan

    try:
        # lazy import
        from SurvivalEVAL.Evaluator import SurvivalEvaluator
        eval_time_arr   = np.asarray(eval_time,   dtype=float)
        eval_status_arr = np.asarray(eval_status, dtype=float)
        
        train_time_arr  = np.asarray(train_time,  dtype=float)
        train_status_arr= np.asarray(train_status,dtype=float)
        evaltime_arr    = np.asarray(evaltime,     dtype=float)

        # Interpolate survival to the fixed eval-time grid
        surv_interp = pd.DataFrame(index=evaltime_arr, columns=surv.columns, dtype=float)
        for t in evaltime_arr:
            idx = max(0, min(surv.index.searchsorted(t, side="right") - 1,
                             len(surv.index) - 1))
            surv_interp.loc[t] = surv.iloc[idx].values

        # SurvivalEvaluator expects (n_samples, n_eval_times)
        surv_for_eval = surv_interp.values.T.astype(float)

        evaluator = SurvivalEvaluator(
            surv_for_eval,
            evaltime_arr,
            eval_time_arr,
            eval_status_arr,
            train_time_arr,
            train_status_arr,
        )

        try:
            p_val_result, _ = evaluator.d_calibration()
            p_value = float(p_val_result) if np.isfinite(p_val_result) else np.nan
        except Exception as e:
            print(f"[D-Calibration WARNING] {e}")

        try:
            mae_result = evaluator.mae(method="Pseudo_obs", weighted=True)
            weighted_MAE_PO = float(mae_result) if np.isfinite(mae_result) else np.nan
        except Exception as e:
            print(f"[Weighted MAE WARNING] {e}")

    except Exception as e:
        print(f"[SurvivalEvaluator] Failed to initialize: {e}")
        traceback.print_exc()

    print(f"[Evaluation] cindex={cindex:.4f}, brier={brier:.4f}, nbll={nbll:.4f}, p_value={p_value}, weighted_MAE_PO={weighted_MAE_PO}")
    return cindex, brier, nbll, p_value, weighted_MAE_PO

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
        client_id = client_name 
        
    dataset_directory = f"{args.datadir}/{dataset}/{partition}/{client_id}/{run_id}/" 
    train_df = pd.read_csv(dataset_directory + "train.csv") 
    
    val_df = pd.read_csv(dataset_directory + "val.csv")
    test_df = pd.read_csv(dataset_directory + "test.csv")

    cols_to_remove = ['time', 'event']
    new_col_name = train_df.columns.tolist()
    feature_list = [x for x in new_col_name if x not in cols_to_remove]

    return train_df, val_df, test_df, client_name, feature_list

############################### Convert Multi-dimensional Torch Tensor to 1D NumPy Array ########################################


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


######################################### Parameter from the Trained Model ######################################################

def get_trainable_parameters(net):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    N = 0
    for params in paramlist:
        N += params.numel()
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset +
                numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset += numel
    # print("get trainable x:", X)
    return X


######################################### Parameter to the PyTorch Model ######################################################

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


######################################### Store PyTorch Model in Storage ######################################################

def save_model(model, model_index, args):
    print("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


######################################### Load PyTorch Model from Storage ######################################################

def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model


######################################### Initialise PyTorch Model using default Values #########################################

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
