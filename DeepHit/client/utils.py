import numpy as np, traceback
import torchtuples as tt  
import pandas as pd  
from pycox.evaluation import EvalSurv
from SurvivalEVAL.Evaluator import SurvivalEvaluator 
from pycox.preprocessing.label_transforms import LabTransDiscreteTime 

import logging
logger = logging.getLogger("experiment") 

##################################### Label Transformer ############################################################### 
class LabTransform(LabTransDiscreteTime):
  def __init__(self, num_durations):
      super().__init__(num_durations)
      self.num_durations = num_durations

  def transform(self, durations, events):
    durations = np.asarray(durations, dtype=np.float64).reshape(-1)
    events = np.asarray(events).copy().reshape(-1)

    if len(durations) != len(events):
        raise ValueError("durations and events must have the same length.")

    # Handle a genuinely empty split.
    if len(durations) == 0:
        return ( np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

    if not np.isfinite(durations).all():
        raise ValueError("durations contain NaN or infinite values.")

    event_mask = events > 0

    # Pycox's transform fails when event_mask contains no True values.
    # Manually discretize an all-censored split using the shared cuts.
    if not np.any(event_mask):
        cuts = np.asarray(self.cuts)

        # Pycox uses right censoring for durations beyond the last cut.
        clipped_durations = np.minimum(durations, cuts[-1])

        # For censoring, use the cut at or immediately before the censoring time.
        duration_idx = (np.searchsorted( cuts, clipped_durations, side="right", ) - 1)
        duration_idx = np.clip( duration_idx, 0, len(cuts) - 1)
        transformed_events = np.zeros(len(events), dtype=np.int64)

        return ( duration_idx.astype(np.int64), transformed_events)

##################################### Optimizer from Arguments ############################################################### 

def build_optimizer(args):  
    """
    Build an optimizer for the DeepHit model.
    Parameters
    ----------
        args : argparse.Namespace
            Arguments containing optimizer configuration.
    Returns
    -------
        torch.optim.Optimizer
            Configured optimizer for the DeepHit model.
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

#################################### Evaluation of Trained Model using Test Data #######################################  

def Evaluation(model, train_time, train_status, X_test, test_time, test_status, evaltime, labtrans, device):

    """
    Evaluate a DeepHit model on validation or test data.

    Parameters
    ----------
    model : pycox.models.DeepHit
        Fitted DeepHit model.
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
        
    # cumulative incidence function inference  
    cif = model.predict_cif(X_test)   
    duration_index = np.asarray(model.duration_index, dtype=float)

    # Snap evaltime to nearest cut points available in duration_index
    valid_eval_times = evaltime[evaltime >= duration_index[0]]
    pos = np.searchsorted(duration_index, valid_eval_times, side="right") - 1 
    pos = np.clip(pos, 0, len(duration_index) - 1)

    # Remove duplicate snapped positions
    pos = np.unique(pos) 
    snapped_eval_times = duration_index[pos]
     
    cif_event_one = pd.DataFrame(cif[0], index=model.duration_index)     # (n_timepoints, n_samples)    
    cif_for_eval_times = cif_event_one.iloc[pos]
    
    cif_for_eval_times.index = snapped_eval_times
    surv_for_eval_times = 1 - cif_for_eval_times
    
    # Initialize metric evaluators
    ev = EvalSurv(surv_for_eval_times, test_time, test_status, censor_surv="km")
    surv_for_eval = pd.DataFrame( surv_for_eval_times.T.to_numpy(), columns=snapped_eval_times)  
    evaluator = SurvivalEvaluator(surv_for_eval, snapped_eval_times, test_time, test_status, train_time, train_status)

    # Compute evaluation metrics
    cindex = safe_metric("C-index", ev.concordance_td)
    brier = safe_metric("Brier", lambda: ev.integrated_brier_score(snapped_eval_times))
    nbll = safe_metric("NBLL", lambda: ev.integrated_nbll(snapped_eval_times))
    p_value = safe_metric("D-Calibration", lambda: evaluator.d_calibration()[0])
    weighted_MAE_PO = safe_metric("Weighted MAE", lambda: evaluator.mae(method="Pseudo_obs", weighted=True))  

    print(f"[Evaluation] cindex={cindex:.4f}, brier={brier:.4f}, nbll={nbll:.4f}, p_value={p_value:.4f}, weighted_MAE_PO={weighted_MAE_PO:.4f}") 
    return cindex, brier, nbll, p_value, weighted_MAE_PO  