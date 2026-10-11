import torch
import numpy as np 
from torch.utils.data import DataLoader
import pandas as pd 
from pycox.evaluation import EvalSurv
from SurvivalEVAL.Evaluator import SurvivalEvaluator 
from ..utils import DataBuilder

##################################### Optimizer from Arguments ############################################################### 

def build_optimizer(model, args): 
    parameters = filter(lambda p: p.requires_grad, model.parameters()) 
    if args.optimizer == "adam":
        return torch.optim.Adam( parameters, lr=args.lr, weight_decay=args.reg) 
    elif args.optimizer == "amsgrad":
        return torch.optim.Adam( parameters, lr=args.lr, weight_decay=args.reg, amsgrad=True) 
    elif args.optimizer == "sgd":
        return torch.optim.SGD( parameters, lr=args.lr, momentum=args.rho, weight_decay=args.reg)  
    else: 
      raise ValueError( f"Unsupported optimizer: {args.optimizer}")   
    

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

    surv_np = np.exp( - (evaltime[np.newaxis, :] /scale[:, np.newaxis]) ** shape[:, np.newaxis])
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
        f"test_status={test_status_arr.shape}, "  # expect (649,)
        f"train_time={train_time_arr.shape}, "  # expect (1944,)
        f"train_status={train_status_arr.shape}"  # expect (1944,)
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
            p_value = float(p_val_result) if np.isfinite(p_val_result) else np.nan
        except Exception as e:
            print(f"[D-Calibration WARNING] {e}")

        try:
            mae_result = evaluator.mae(method='Pseudo_obs', weighted=True)
            weighted_MAE_PO = float(mae_result) if np.isfinite( mae_result) else np.nan
        except Exception as e:
            print(f"[Weighted MAE WARNING] {e}")

    except Exception as e:
        print(f"[SurvivalEvaluator] Failed to initialize: {e}")

    print(f"[Evaluation] cindex={cindex}, brier={brier}, nbll={nbll}, "
          f"p_value={p_value}, weighted_MAE_PO={weighted_MAE_PO}"
          )

    return cindex, brier, nbll, p_value, weighted_MAE_PO