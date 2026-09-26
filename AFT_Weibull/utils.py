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
 
