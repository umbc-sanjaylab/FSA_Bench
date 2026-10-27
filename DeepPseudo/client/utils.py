import torch 
import pandas as pd
import numpy as np
from pycox.evaluation import EvalSurv
from SurvivalEVAL.Evaluator import SurvivalEvaluator 

import logging
logger = logging.getLogger("experiment")
 
########################################## Pseudo Value based Loss Function ############################################################## 

def pseudo_loss(output, target):
    loss = torch.mean((target-output)**2)
    return loss
  
  
#################################### Compute Concordance Score from validation data ###############################################

def Concordance(model, X_covariates, durations, events, evaltime, device="cpu"):
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
        Time-dependent C-index (or np.nan if not computable)
    """

    model.to(device)
    model.eval()
    X = torch.from_numpy(np.array(X_covariates, dtype=np.float32)) .to(device)

    durations = np.asarray(durations).reshape(-1)
    events = np.asarray(events).reshape(-1)
    evaltime = np.asarray(evaltime)

    # --- guards: not enough data / no events / no comparable pairs ---
    if len(durations) < 2:
        return np.nan

    n_events = np.sum(events == 1)
    if n_events == 0:
        return np.nan

    if not has_comparable_pairs(durations, events):
        return np.nan

    # --- model prediction ---
    with torch.no_grad():
        out = model(X)
    out_np = out.detach().cpu().numpy()                 # raw model output

    # Remove singleton event dim: (n_samples, 1, n_evaltime) -> (n_samples, n_evaltime)
    if out_np.ndim == 3 and out_np.shape[1] == 1:
        out_np = out_np.squeeze(axis=1)

    if out_np.shape[0] != len(durations):
        raise ValueError(
            f"Sample mismatch: out_np has {out_np.shape[0]} rows, "
            f"but durations has length {len(durations)}"
        )
    if out_np.shape[1] != len(evaltime):
        raise ValueError(
            f"Time mismatch: out_np has {out_np.shape[1]} columns, "
            f"but evaltime has length {len(evaltime)}"
        )

    # survival probability from F(t)   
    surv_np = 1.0 - out_np 
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

def Evaluation(model, train_time, train_status, X_eval, test_time, test_status, eval_times, device="cpu"):
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
    test_time : np.ndarray
        Observed survival times for the evaluation set.
    test_status : np.ndarray
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
    model.to(device)
    model.eval() 
    
    # Check input covariates
    if not np.isfinite(X_eval).all():
        logger.error( "Invalid X_eval: shape=%s | NaN=%d | Inf=%d",
            X_eval.shape,  np.isnan(X_eval).sum(),  np.isinf(X_eval).sum()
        )
        return np.nan, np.nan, np.nan, np.nan, np.nan
      
    X_covariates = torch.from_numpy(np.array(X_eval, dtype=np.float32)) .to(device) 
    with torch.no_grad(): 
        sp_test = model(X_covariates)
    
    # obtain cumulative incidence fucntion on test set and validate it 
    cif = sp_test.cpu().detach().numpy()   
    if cif.size == 0:
        logger.warning("Evaluation skipped because survival predictions are empty.")
        return np.nan, np.nan, np.nan, np.nan, np.nan 
      
    if cif.ndim == 3 and cif.shape[1] == 1:
          cif = cif.squeeze(axis=1)
    
    # Check network outputs
    if not np.isfinite(cif).all():
        logger.error("Invalid DeepPseudo predictions:  shape=%s | NaN=%d | Inf=%d", cif.shape,  np.isnan(cif).sum(), np.isinf(cif).sum())
        return np.nan, np.nan, np.nan, np.nan, np.nan
          
    # Align survival predictions with the common evaluation grid
    survival = 1.0 - cif  

    # Initialize metric evaluators from PyCox & SurvivalEVAL packages
    surv_for_evsurv = pd.DataFrame(survival.T, index=eval_times)
    ev = EvalSurv(surv_for_evsurv, test_time, test_status, censor_surv="km")
    
    surv_for_eval = pd.DataFrame(survival, columns=eval_times)
    evaluator = SurvivalEvaluator(surv_for_eval, eval_times,  test_time, test_status, train_time, train_status)
  
    # Compute evaluation metrics
    cindex = safe_metric("C-index", ev.concordance_td)
    brier = safe_metric("Brier", lambda: ev.integrated_brier_score(eval_times))
    nbll = safe_metric("NBLL", lambda: ev.integrated_nbll(eval_times))
    p_value = safe_metric("D-Calibration", lambda: evaluator.d_calibration()[0])
    weighted_mae_po = safe_metric("Weighted MAE", lambda: evaluator.mae(method="Pseudo_obs", weighted=True))

    logger.info("[Evaluation] C-index=%s | Brier=%s | NBLL=%s | D-calibration=%s | MAE-PO=%s",
                cindex, brier, nbll, p_value, weighted_mae_po)

    return cindex, brier, nbll, p_value, weighted_mae_po