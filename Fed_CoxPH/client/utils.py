import numpy as np, traceback
import torchtuples as tt  
import pandas as pd  
from pycox.evaluation import EvalSurv
from SurvivalEVAL.Evaluator import SurvivalEvaluator

import logging
logger = logging.getLogger("experiment")

##################################### Optimizer from Arguments ############################################################### 

def build_optimizer(args):  
    """
    Build an optimizer for the CoxPH model.
    Parameters
    ----------
        args : argparse.Namespace
            Arguments containing optimizer configuration.
    Returns
    -------
        torch.optim.Optimizer
            Configured optimizer for the CoxPH model.
    """ 
    if args.optimizer == "adam":
        return tt.optim.Adam( lr=args.lr, weight_decay=args.reg) 
    elif args.optimizer == "sgd":
        return tt.optim.SGD( lr=args.lr, weight_decay=args.reg)
    elif args.optimizer == "amsgrad":
        raise ValueError( "AMSGrad is not configured for the torchtuples backend.")
    else:
        raise ValueError( f"Unsupported optimizer backend: {args.optimizer}") 


################################## Method for checking whether metric is finite ##################################################

def is_finite_metric(value):
    """
    Check if a metric value is finite and valid.
    """ 
    return (
        value is not None
        and np.isscalar(value)
        and np.isfinite(float(value))
    )
    
    
def accumulate(test_acc, metric, value, weight):
    if is_finite_metric(value):
        test_acc[metric]["weighted_sum"] += float(value) * weight 
        test_acc[metric]["n"] += weight 
         

def weighted_mean(test_acc, metric):
    acc = test_acc[metric]

    if acc["n"] == 0:
        return np.nan

    return acc["weighted_sum"] / acc["n"]
  

def safe_metric(name, metric_fn):
    try:
        value = float(metric_fn())
        return value if np.isfinite(value) else np.nan
    except Exception as exc:
        logger.warning("[%s WARNING] %s", name, exc)
        return np.nan 
 
######################################### Evaluation Method ####################################################     

def Evaluation(model, train_time, train_status, X_train, X_eval, eval_time, eval_status, eval_times):
    """
    Evaluate a CoxPH model on validation or test data.

    Parameters
    ----------
    model : pycox.models.CoxPH
        Fitted CoxPH model.
    train_time : np.ndarray
        Observed survival times from the training set.
    train_status : np.ndarray
        Event indicators from the training set.
    X_train : np.ndarray
        Training covariates used to estimate the baseline hazard.
    X_eval : np.ndarray
        Covariates for the evaluation set.
    eval_time : np.ndarray
        Observed survival times for the evaluation set.
    eval_status : np.ndarray
        Event indicators for the evaluation set.
    eval_times : np.ndarray
        Common time grid used for survival metric evaluation.

    Returns
    -------
    tuple
        C-index, integrated Brier score, integrated NBLL,
        D-calibration p-value, and weighted MAE-PO.
    """

    # Validate training outcomes
    if len(train_time) == 0 or not np.any(train_status == 1):
        logger.warning("Evaluation skipped because the training split contains no events.")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Estimate baseline hazards and predict survival functions
    y_train = (train_time, train_status)
    model.compute_baseline_hazards(X_train, y_train)
    
    survival = model.predict_surv_df(X_eval) 
    if survival.empty:
        logger.warning("Evaluation skipped because survival predictions are empty.")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Align survival predictions with the common evaluation grid
    survival_times = np.asarray(survival.index, dtype=float)
    positions = np.searchsorted(survival_times, eval_times, side="right") - 1
    positions = np.clip(positions, 0, len(survival_times) - 1)

    surv = survival.iloc[positions].copy()
    surv.index = eval_times

    # Initialize metric evaluators
    ev = EvalSurv(surv, eval_time, eval_status, censor_surv="km")
    surv_for_eval = surv.to_numpy(dtype=float).T
    evaluator = SurvivalEvaluator(surv_for_eval, eval_times, eval_time, eval_status, train_time, train_status)

    # Compute evaluation metrics
    cindex = safe_metric("C-index", ev.concordance_td)
    brier = safe_metric("Brier", lambda: ev.integrated_brier_score(eval_times))
    nbll = safe_metric("NBLL", lambda: ev.integrated_nbll(eval_times))
    p_value = safe_metric("D-Calibration", lambda: evaluator.d_calibration()[0])
    weighted_mae_po = safe_metric("Weighted MAE", lambda: evaluator.mae(method="Pseudo_obs", weighted=True))

    logger.info("[Evaluation] C-index=%s | Brier=%s | NBLL=%s | D-calibration=%s | MAE-PO=%s",
                cindex, brier, nbll, p_value, weighted_mae_po)

    return cindex, brier, nbll, p_value, weighted_mae_po