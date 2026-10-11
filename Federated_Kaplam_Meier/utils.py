from pathlib import Path
import logging
import numpy as np
import pandas as pd 

#################################### Experiment Logger ############################################################ 

def create_experiment_logger(logdir, setting, model, dataset, run_id):
    logger_name = f"{setting}_{model}_{dataset}_fold_{run_id}"
    log_path = Path(logdir) / f"{model}_{dataset}_fold_{run_id}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]: 
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%m-%d %H:%M"
    )

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
  
  
################################  Kaplan-Meier Evaluation at eval times  ############################


def evaluate_km_on_grid(km_times, km_surv, eval_times):
    """
    Arguments: 
        km_times: event times (sorted)
        km_surv: survival at those times
        eval_times: quantile grid (cuts)

    Returns:
        1D numpy array of survival probs evaluated on target_times
    """
    surv_at_grid = [] 
    for eval_time in eval_times:

        # find last KM time <= t
        idx = np.searchsorted(km_times, eval_time, side='right') - 1 
        interp_time_index = np.argmin(np.abs(eval_time - km_times)) 

        if idx >= 0:
            surv_at_grid.append(km_surv[interp_time_index])
        else:
            surv_at_grid.append(1.0)  # before first event

    return np.array(surv_at_grid)
  

#################################### Evaluation of Global Model using Test Data #######################################

def Evaluation(surv_test_np,  train_time, train_status,  test_time, test_status, evaltime, logger):
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
    
    test_cindex, test_brier, test_nbll, test_p_value,  test_weighted_MAE_PO= np.nan, np.nan, np.nan, np.nan, np.nan
    
    surv_for_eval_surv = pd.DataFrame( np.transpose(surv_test_np), index=evaltime) 
    eval_pycox = EvalSurv(surv_for_eval_surv, test_time, test_status, censor_surv='km')  
    
    surv_for_eval = pd.DataFrame(surv_test_np, columns=np.asarray(evaltime, dtype=float))
    evaluator = SurvivalEvaluator( surv_for_eval, np.asarray(evaltime, dtype=float), test_time, test_status, train_time, train_status)
 
    try:
        test_cindex = eval_pycox.concordance_td()
        if not np.isfinite(test_cindex):
            logger.error("[C-index ERROR] concordance_td returned NaN/Inf")
            test_cindex = np.nan
    except Exception as e:
        logger.error(f"[C-index ERROR] {type(e).__name__}: {e}")
        test_cindex = np.nan

    try:
        test_brier = eval_pycox.integrated_brier_score( evaltime)  # integrated brier score
        if not np.isfinite(test_brier):
            logger.error( "[Brier Score ERROR] integrated_brier_score returned NaN/Inf")
            test_brier = np.nan
    except Exception as e:
        logger.error(f"[Brier Score ERROR] {type(e).__name__}: {e}")
        test_brier = np.nan 

    try:
        test_nbll = eval_pycox.integrated_nbll(evaltime)
        if not np.isfinite(test_nbll):
            logger.error("[NBLL ERROR] integrated_nbll returned NaN/Inf")
            test_nbll = np.nan
    except Exception as e:
        logger.error(f"[NBLL ERROR] {type(e).__name__}: {e}")
        test_nbll = np.nan 

    try:
        p_val_result, _ = evaluator.d_calibration()
        test_p_value = float(p_val_result) if np.isfinite(p_val_result) else np.nan
    except Exception as e:
        logger.warning(f"[D-Calibration WARNING] {e}")
        test_p_value = np.nan

    try:
        mae_result = evaluator.mae(method='Pseudo_obs', weighted=True)
        test_weighted_MAE_PO = float(mae_result) if np.isfinite(mae_result) else np.nan
    except Exception as e:
        logger.warning(f"[Weighted MAE WARNING] {e}")
        test_weighted_MAE_PO = np.nan
 
    logger.info( f"[Evaluation] cindex={test_cindex}, brier={test_brier}, nbll={test_nbll}, p_value={test_p_value}, weighted_MAE_PO={test_weighted_MAE_PO}")
    return  test_cindex, test_brier, test_nbll, test_p_value,  test_weighted_MAE_PO 