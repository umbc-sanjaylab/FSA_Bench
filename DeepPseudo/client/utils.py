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

def Evaluation(model, train_time, train_status, X_train, X_eval, test_time, test_status, eval_times, device="cpu"):
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
    with torch.no_grad():
        sp_test = model(X_eval)
    
    # obtain cumulative incidence fucntion on test set and validate it 
    cif = sp_test.cpu().detach().numpy()   
    if cif.size == 0:
        logger.warning("Evaluation skipped because survival predictions are empty.")
        return np.nan, np.nan, np.nan, np.nan, np.nan 
      
    if cif.ndim == 3 and cif.shape[1] == 1:
          cif = cif.squeeze(axis=1)  
          
    # Align survival predictions with the common evaluation grid
    survival = 1.0 - cif 
    survival_times = np.asarray(survival.index, dtype=float)
    positions = np.searchsorted(survival_times, eval_times, side="right") - 1
    positions = np.clip(positions, 0, len(survival_times) - 1)

    surv = survival.iloc[positions].copy()
    surv.index = eval_times

    # Initialize metric evaluators from PyCox & SurvivalEVAL packages
    surv_for_evsurv = pd.DataFrame(surv.T, index=eval_times)
    ev = EvalSurv(surv_for_evsurv, test_time, test_status, censor_surv="km")
    
    surv_for_eval = pd.DataFrame(surv, columns=np.asarray(eval_times, dtype=float))
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