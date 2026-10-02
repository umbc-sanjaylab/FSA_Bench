import logging
import numpy as np
import pandas as pd
import torch  
from pycox.evaluation import EvalSurv  
import numpy as np
from pycox.evaluation import EvalSurv 
from SurvivalEVAL.Evaluator import SurvivalEvaluator 

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO) 
 

#################################### Evaluation of Trained Model using Test Data #######################################  

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