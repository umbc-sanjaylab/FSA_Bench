import numpy as np 
from math import * 
from utils import *
from model import *
import numpy as np  
from utils import local_partial_matrix 
from scipy.stats import combine_pvalues
from utils import evaluate_km_on_grid

################  Training DeepPseudo Model with "Centralized" framework in Local Client  #################################
 
def train_net(train_time, train_status, val_time, val_status, test_time, test_status, evaltime):  
     
    train_val_time = np.concatenate([train_time, val_time]) 
    train_val_status = np.concatenate([train_status, val_status])
    client_KM_components, columns = local_partial_matrix(train_val_time, train_val_status, init_seed=1234)

    global_full_matrix, global_surv = global_survival_probability(client_KM_components)
    global_full_matrix = pd.DataFrame(global_full_matrix, columns=columns)

    km_times = global_full_matrix['time'].values
    km_surv = global_surv 
    
    km_surv_at_eval_time = evaluate_km_on_grid(km_times, km_surv, evaltime)  
    surv_test_np = np.tile(km_surv_at_eval_time, (test_time.shape[0], 1)) 
    test_cindex, test_brier, test_nbll, test_p_value,  test_weighted_MAE_PO = Evaluation(surv_test_np,  train_time, train_status,  test_time, test_status, evaltime, logger) 

    return test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO



###############################  Training DeepPseudo Model with "FedAvg" framework in Local Client  ####################################################

def local_train_net(train_time_local, train_status_local, val_time_local, val_status_local, test_time_local, test_status_local, evaltime):

    avg_cindex = 0.0
    avg_brier = 0.0

    avg_nbll = 0.0
    avg_p_value =  0.0 
    client_p_values = [] 

    avg_weighted_MAE_PO = 0.0 
    total_point_for_cindex, total_point_for_IBS, total_point_for_NLL,  total_point_for_MAE_PO = 0.0, 0.0, 0.0, 0.0  
  

    for index, _ in enumerate(test_time_local):
        train_time = np.asarray(train_time_local[index], dtype=np.float64).reshape(-1) 
        train_status = np.asarray(train_status_local[index], dtype=np.float64).reshape(-1) 
        
        val_time = np.asarray(val_time_local[index], dtype=np.float64).reshape(-1) 
        val_status = np.asarray(val_status_local[index], dtype=np.float64).reshape(-1)  
        
        test_time =  np.asarray(test_time_local[index], dtype=np.float64).reshape(-1)  
        test_status = np.asarray(test_status_local[index], dtype=np.float64).reshape(-1)  
        test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = train_net(train_time, train_status, val_time, val_status, test_time, test_status, evaltime)
        
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
