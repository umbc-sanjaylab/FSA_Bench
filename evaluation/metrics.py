import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv

def compute_concordance( survival, eval_times, durations, events):
    """
    Compute time-dependent concordance from predicted
    survival probabilities.
    """

    survival = np.asarray( survival, dtype=float)
    eval_times = np.asarray( eval_times, dtype=float).reshape(-1)

    durations = np.asarray( durations, dtype=float).reshape(-1)
    events = np.asarray( events, dtype=int).reshape(-1)

    # Basic validity checks
    if len(durations) < 2:
        return np.nan

    if np.sum(events == 1) == 0:
        return np.nan

    if survival.shape[0] != len(durations):
        raise ValueError(f"Sample mismatch: survival has {survival.shape[0]} rows, but durations has {len(durations)} samples.")

    if survival.shape[1] != len(eval_times):
        raise ValueError(f"Time mismatch: survival has  {survival.shape[1]} columns, but eval_times has {len(eval_times)} values.")

    # pytorch expects (times, samples) as (row, col)  
    survival_df = pd.DataFrame(survival.T, index=eval_times)
    evaluator = EvalSurv(survival_df, durations,  events, censor_surv="km")

    try:
        cindex = float( evaluator.concordance_td())
        return  cindex if np.isfinite(cindex) else np.nan 
    except ZeroDivisionError: 
        return np.nan