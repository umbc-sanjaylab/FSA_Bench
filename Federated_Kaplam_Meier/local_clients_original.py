import numpy as np
from pycox.models import DeepHit
from math import *
import torchtuples as tt
from utils import *
from model import *
import numpy as np
import torchtuples as tt
import traceback
from lifelines import KaplanMeierFitter
from utils import local_partial_matrix
from pycox.evaluation import EvalSurv
from SurvivalEVAL.Evaluator import SurvivalEvaluator
from scipy.stats import combine_pvalues

################  Training DeepPseudo Model with "Centralized" framework in Local Client  #################################


def train_net_all(train_time_all, train_status_all, val_time_all, val_status_all,  test_time_all, test_status_all, evaltime, args, device):

    client_time = np.concatenate((train_time_all, val_time_all))
    client_status = np.concatenate((train_status_all, val_status_all))

    local_dataframe, columns = local_partial_matrix(client_time, client_status, init_seed=1234)
    global_partial_matrix = local_dataframe.groupby('time', as_index=False).sum()

    global_partial_matrix.loc[0, 'nrisk'] = np.sum(global_partial_matrix.loc[:, 'nrisk'])
    global_partial_matrix.loc[1:, 'nrisk'] = 0

    global_full_matrix, global_surv = global_survival_probability(global_partial_matrix)
    global_full_matrix = pd.DataFrame(global_full_matrix, columns=columns)

    km_times = global_full_matrix['time'].values
    km_surv = global_surv

    time_grid_KM_np = evaltime
    surv_at_grid = evaluate_km_on_grid(km_times, km_surv, time_grid_KM_np)

    surv_test_np = np.tile(surv_at_grid, (test_time_all.shape[0], 1))
    time_grid_test_np = np.unique(test_time_all)

    surv_test_df = pd.DataFrame(surv_test_np.T, index=time_grid_KM_np)
    eval_pycox = EvalSurv(surv_test_df, test_time_all,
                          test_status_all, censor_surv='km')

    cindex, brier, nbll = np.nan, np.nan, np.nan

    try:
        cindex = eval_pycox.concordance_td()
    except Exception as e:
        logger.error(f"[C-index ERROR] {type(e).__name__}: {e}")
        cindex = np.nan

    try:
        brier = eval_pycox.integrated_brier_score(
            evaltime)  # integrated brier score
    except Exception as e:
        logger.error(f"[Brier Score ERROR] {type(e).__name__}: {e}")
        brier = np.nan

     # integrated brier score computation.
    try:
        nbll = eval_pycox.integrated_nbll(evaltime)
    except Exception as e:
        print(f"[Brier ERROR] {type(e).__name__}: {e}")
        nbll = np.nan

    # SurvivalEvaluator (survivaleval): expects (n_samples, n_eval_times) with evaltime as columns
    surv_for_eval = pd.DataFrame( surv_test_np,  columns=np.asarray(time_grid_KM_np, dtype=float))

    # --- SurvivalEvaluator (D-Calibration & Weighted MAE) ---
    test_time_arr = np.asarray(test_time_all,    dtype=float).reshape(-1)
    test_status_arr = np.asarray(test_status_all,  dtype=int).reshape(-1)
    train_time_arr = np.asarray(train_time_all,   dtype=float).reshape(-1)
    train_status_arr = np.asarray(train_status_all, dtype=int).reshape(-1)

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
            logger.warning(f"[D-Calibration WARNING] {e}")

        try:
            mae_result = evaluator.mae(method='Pseudo_obs', weighted=True)
            weighted_MAE_PO = float(mae_result) if np.isfinite(mae_result) else np.nan
        except Exception as e:
            logger.warning(f"[Weighted MAE WARNING] {e}")

    except Exception as e:
        logger.error(
            f"[SurvivalEvaluator] Failed to initialize: {e}", exc_info=True)

    logger.info(
        f"[Evaluation] cindex={cindex}, brier={brier}, nbll={nbll}, "
        f"p_value={p_value}, weighted_MAE_PO={weighted_MAE_PO}"
    )

    print(f"cindex: {cindex}, brier score: {brier}")
    return cindex, brier, nbll, p_value, weighted_MAE_PO

###############################  Training DeepPseudo Model with "FedAvg" framework in Each Client  ####################################################

# def train_net(km_times, km_surv, train_time, train_status, test_time, test_status, evaltime, args, device):
  
def train_net(km_times, km_surv, train_time, train_status, test_time, test_status, evaltime, args, device, ):
    km_times = np.asarray(km_times, dtype=np.float64, ).reshape(-1)
    km_surv = np.asarray(km_surv, dtype=np.float64, ).reshape(-1)
    
    train_time = np.asarray(train_time, dtype=np.float64).reshape(-1)
    train_status = np.asarray( train_status, dtype=np.int64).reshape(-1)
    
    test_time = np.asarray(test_time, dtype=np.float64).reshape(-1)
    test_status = np.asarray( test_status, dtype=np.int64).reshape(-1)

    # Sort and retain unique KM time points.
    order = np.argsort(km_times)
    km_times = km_times[order]
    km_surv = km_surv[order]

    km_times, unique_indices = np.unique(km_times, return_index=True)
    km_surv = km_surv[unique_indices]

    # Use the complete unique KM timeline.
    prediction_times = km_times
    prediction_survival = km_surv
    surv_test_np = np.tile( prediction_survival, (len(test_time), 1))

    surv_test_df = pd.DataFrame(surv_test_np.T, index=prediction_times)
    eval_pycox = EvalSurv(surv_test_df, test_time, test_status, censor_surv="km")
    test_cindex = eval_pycox.concordance_td()

    # Use unique test times within prediction support.
    metric_times = np.unique(test_time)
    metric_times = metric_times[ (metric_times > prediction_times[0]) & (metric_times < prediction_times[-1]) ]
    
    test_brier = eval_pycox.integrated_brier_score( metric_times )
    test_nbll = eval_pycox.integrated_nbll( metric_times )
    surv_for_eval = pd.DataFrame( surv_test_np, columns=prediction_times, )
    
    evaluator = SurvivalEvaluator( surv_for_eval, prediction_times, test_time, test_status, train_time, train_status, )
    test_p_value, _ = evaluator.d_calibration()
    test_weighted_MAE_PO = evaluator.mae(method="Pseudo_obs", weighted=True, )

    return ( test_cindex, test_brier, test_nbll, float(test_p_value), float(test_weighted_MAE_PO), )

def train_net(km_times, km_surv, train_time, train_status, test_time, test_status, evaltime, args, device):

    time_grid_KM_np = evaltime 
    test_cindex, test_brier, test_nbll, p_value,  test_weighted_MAE_PO= np.nan, np.nan, np.nan, np.nan, np.nan 
    
    
    survival_curve_test_np = np.tile(km_surv, (test_time.shape[0], 1)) 
    surv_at_grid = evaluate_km_on_grid(survival_curve_test_np, km_surv, time_grid_KM_np) 

    surv_test_df = pd.DataFrame(surv_test_np.T, index=time_grid_KM_np)
    eval_pycox = EvalSurv(surv_test_df, test_time, test_status, censor_surv='km') 
   
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

    # → (n_samples, n_eval_times)
    surv_for_eval = pd.DataFrame( surv_test_np, columns=np.asarray(time_grid_KM_np, dtype=float) )
    
    # --- SurvivalEvaluator (D-Calibration & Weighted MAE) ---
    test_time_arr = np.asarray(test_time,    dtype=float).reshape(-1)
    test_status_arr = np.asarray(test_status,  dtype=int).reshape(-1)

    train_time_arr = np.asarray(train_time,   dtype=float).reshape(-1)
    train_status_arr = np.asarray(train_status, dtype=int).reshape(-1) 

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

    except Exception as e:
        logger.error(f"[SurvivalEvaluator] Failed to initialize: {e}", exc_info=True)

    logger.info(
        f"[Evaluation] cindex={test_cindex}, brier={test_brier}, nbll={test_nbll}, "
        f"p_value={p_value}, weighted_MAE_PO={test_weighted_MAE_PO}"
    )

    return test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO


###############################  Training DeepPseudo Model with "FedAvg" framework in Local Client  ####################################################

def local_train_net(train_time_local, train_status_local, val_time_local, val_status_local, test_time_local, test_status_local, evaltime, args, device):

    avg_cindex = 0.0
    avg_brier = 0.0

    avg_nbll = 0.0
    avg_p_value =  0.0 
    client_p_values = [] 

    avg_weighted_MAE_PO = 0.0 
    total_point_for_cindex, total_point_for_IBS, total_point_for_NLL,  total_point_for_MAE_PO = 0.0, 0.0, 0.0, 0.0 

    train_time_all = np.concatenate(train_time_local)
    train_status_all = np.concatenate(train_status_local)

    val_time_all = np.concatenate(val_time_local)
    val_status_all = np.concatenate(val_status_local)

    train_val_time = np.concatenate([train_time_all, val_time_all])
    train_val_status = np.concatenate([train_status_all, val_status_all])

    client_KM_components, columns = local_partial_matrix(train_val_time, train_val_status, init_seed=1234)
    global_partial_matrix = client_KM_components.groupby('time', as_index=False).sum()

    global_partial_matrix.loc[0, 'nrisk'] = np.sum(global_partial_matrix.loc[:, 'nrisk'])
    global_partial_matrix.loc[1:, 'nrisk'] = 0

    global_full_matrix, global_surv = global_survival_probability(global_partial_matrix)
    global_full_matrix = pd.DataFrame(global_full_matrix, columns=columns)

    km_surv = global_surv
    km_times = global_full_matrix['time'].values

    for index, test_time in enumerate(test_time_local):
        test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = train_net(km_times, km_surv, train_time_local[index], train_status_local[index], test_time_local[index], test_status_local[index], evaltime, args, device)
        
        # total_point_for_cindex, total_point_for_IBS, total_point_for_NLL, total_point_for_P_value, total_point_for_MAE_PO

        if test_cindex is not None and np.isfinite(test_cindex):
            avg_cindex += test_cindex * len(test_time)
            total_point_for_cindex += len(test_time)

        if test_brier is not None and np.isfinite(test_brier):
            avg_brier += test_brier * len(test_time)
            total_point_for_IBS += len(test_time)

        if test_nbll is not None and np.isfinite(test_nbll):
            avg_nbll += test_nbll * len(test_time)
            total_point_for_NLL += len(test_time)
            
        if test_p_value is not None and np.isfinite(test_p_value):
            client_p_values.append(float(test_p_value))  

        if test_weighted_MAE_PO is not None and np.isfinite(test_weighted_MAE_PO):
            avg_weighted_MAE_PO += test_weighted_MAE_PO * len(test_time)
            total_point_for_MAE_PO += len(test_time)

    avg_cindex /= total_point_for_cindex
    avg_brier /= total_point_for_IBS
    avg_nbll /= total_point_for_NLL

    if len(client_p_values) > 0:
        avg_p_value = combine_pvalues(client_p_values, method="fisher")[1]
    else:
        avg_p_value = np.nan
        
    avg_weighted_MAE_PO /= total_point_for_MAE_PO
    return avg_cindex, avg_brier, avg_nbll, avg_p_value, avg_weighted_MAE_PO

############################################################################################################################################
