import os
import logging
import numpy as np
import pandas as pd
import torch
import random
from pycox.evaluation import EvalSurv
import random
from pycox.models import DeepHit
from collections import defaultdict
import h5py
import numpy as np
from pycox.evaluation import EvalSurv
from SurvivalEVAL.Evaluator import SurvivalEvaluator

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#################################### Creating new directory ############################################################


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

#################################### Compute subjects at risk ##########################################################


def compute_d_and_n(data, t):
    """
    Arguments:
        data: dataframe with time and event status
        t : unique time points of dataframe   

    Return:
        event and censored status at the given time.
    """

    temp = data[data["time"] == t].groupby("status").count()
    try:
        d = temp.loc[1.0, "time"]
    except KeyError:
        d = 0
    try:
        c = temp.loc[0.0, "time"]
    except KeyError:
        c = 0
    return d, c

#################################### Compute elememts olf local matrix #####################################################


def compute_d_and_n_matrix(data):
    """
    Arguments:
        data: dataframe with time and event status

    Return:
        A new dataframe with number of subjects at risk and censored observations.
    """

    timeline = data["time"].sort_values().unique()
    di = np.full(len(timeline) + 1, np.nan)
    ci = np.full(len(timeline) + 1, np.nan)

    ni = np.full(len(timeline) + 1, 0)
    ni[0] = data.shape[0]

    for i in range(len(timeline)):
        d, c = compute_d_and_n(data, timeline[i])
        di[i] = d
        ci[i] = c

    m = pd.DataFrame(index=timeline)
    m["ti"] = timeline

    m["ni"] = ni[:-1]
    m["di"] = di[:-1]
    m["ci"] = ci[:-1]

    return m

###############################  Client Kaplan-Meier Summary Statistics ###########################################


def local_partial_matrix(client_time, client_status, init_seed):
    """
    Arguments:
        client_time: Client's event time
        client_status: Client's event status 

    Return:
        A partial table with KM estimator components.
    """

    np.random.seed(init_seed)
    df = pd.concat([pd.DataFrame(client_time.copy()), pd.DataFrame(client_status.copy())], axis=1)
    df.columns = ['time', 'status']

    local_partial_matrix = compute_d_and_n_matrix(df)
    local_partial_matrix.reset_index(inplace=True)

    local_partial_matrix = local_partial_matrix.rename(columns={'index': 'time'})
    local_partial_matrix.loc[1:, 'ni'] = 0
    local_dataframe = pd.DataFrame(local_partial_matrix)

    local_dataframe.columns = ['time', 'time', 'nrisk', 'di', 'ci']
    local_dataframe = local_dataframe.iloc[:, 1:]

    return local_dataframe, list(local_dataframe.columns)
 

#####################################  Global Survival Curve Probabilities  ########################################


def Kaplan_Meier_from_components(global_km_components):

    df = global_km_components.sort_values("ti").copy()

    surv_probs = []
    s = 1.0

    for _, row in df.iterrows():
        ni = row["ni"]
        di = row["di"]

        if ni > 0:
            s *= (1.0 - di / ni)

        surv_probs.append(s)

    out = pd.DataFrame({
        "time": df["ti"].to_numpy(dtype=float),
        "survival_prob": np.asarray(surv_probs, dtype=float)
    })
    return out

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

#####################################  Global Survival Curve Probabilities  ########################################

# global survival function calculation


def global_survival_probability(global_partial_matrix):
    """
    Arguments:
        global_partial_matrix: aggregated Kaplan-Meier Components for all clients

    Return:
        A partial table with KM estimator components.
    """

    global_full_matrix = np.array(global_partial_matrix.copy())
    r_last = global_full_matrix[0, 1]
    d_last = global_full_matrix[0, 2]
    c_last = global_full_matrix[0, 3]

    for j in range(1, global_full_matrix.shape[0]):
        r = r_last - d_last - c_last
        d = global_full_matrix[j, 2]
        c = global_full_matrix[j, 3]

        if r < 0:
            global_full_matrix[j, 1] = 0
        else:
            global_full_matrix[j, 1] = r
        if d < 0:
            global_full_matrix[j, 2] = 0
        if c < 0:
            global_full_matrix[j, 3] = 0

        r_last = r
        d_last = d
        c_last = c

    # Calculate global survival probabilities using product rule formula
    global_surv = np.cumprod(
        1-(global_full_matrix[:, 2]/(global_full_matrix[:, 1]+0.000000001)))
    return global_full_matrix, global_surv

 
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

#################################### Load Dataset from Dictionary #######################################

def load_datasets(dataset_file):
    datasets = defaultdict(dict)

    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets


#################################### One-hot Encoding ##################################################

def one_hot_encoder(data, encode):
    """
    Attributes:
    ----------    
    data: A dataframe 
    encode: A list of columns in the dataframe to be one-hot-encoded

    Returns
    -------
    An one-hot-encoded dataframe
    """
    data_encoded = data.copy()
    encoded = pd.get_dummies(data_encoded, prefix=encode,
                             columns=encode, drop_first=False)
    return encoded


#################################### Get Local Dataset from Directory ####################################

######################################### Region from Federated Dataset by Client ID ########################################

def get_client_name_for_federated_datasets(dataset, client_id):
    dataset_with_region = {
        "TCGA": ["South", "West", "Midwest", "Northeast", "Europe", "Canada", "Other"],
        "Aids2": ['NSW', 'Other', 'QLD', 'VIC'],
        "LeukSurv": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        "Pbc3": ['London', 'Copenhagen', 'Hvidovre', 'Barcelona', 'Munich', 'Lyon'],
        "Dialysis": [120, 128, 184, 217, 344, 561, 562, 641, 741, 743, 917, 918, 936, 954, 983, 984, 1029, 1039,
                     1048, 1051, 1053, 1061, 1064, 1070, 1071, 1072, 1093, 1100, 1159, 1372, 1437, 1654, 1692,
                     1699, 2026, 2210, 2211, 2242, 2320, 2351, 2611, 2632, 2811, 2844, 2987, 3064, 3596, 3675,
                     4122, 4157, 4163, 4779, 4825, 5089, 5135, 5419, 5423, 5432, 5481, 5495, 5513, 5514, 5543,
                     5653, 5681, 5692, 5768],
        "d.oropha.rec": [2, 5, 4, 6, 3, 1],
        "cgd": ['Europe:Amsterdam', 'Europe:other', 'US:NIH', 'US:other'],
        "uis": ["A", "B"],
        "scania":  [1, 2, 3, 4, 5],
        "seer": [1, 2, 3]
    }

    region_list = dataset_with_region[dataset]
    return region_list[client_id]

################################################## Get Local Dataset ######################################################


def get_local_dataset(dataset, partition, client_id, run_id, args):
    if partition == 'centralized':
        client_id = 0

    client_name = None
    if partition == "federated":
        client_name = get_client_name_for_federated_datasets(dataset, client_id)
        dataset_directory = f"{args.datadir}/{dataset}/{partition}/{client_name}/{run_id}/"
    else: 
        dataset_directory = f"{args.datadir}/{dataset}/{partition}/{client_id}/{run_id}/"
        print(f"client is is {client_id} and dataset directory is {dataset_directory}") 

    train_df = pd.read_csv(dataset_directory + "train.csv")
    val_df = pd.read_csv(dataset_directory + "val.csv")
    test_df = pd.read_csv(dataset_directory + "test.csv")

    cols_to_remove = ['time', 'event']
    new_col_name = train_df.columns.tolist()
    feature_list = [x for x in new_col_name if x not in cols_to_remove]

    return train_df, val_df, test_df, client_name, feature_list 

#################################### Get Local Dataset from Directory ####################################


def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


#################################### Get Local Dataset from Directory ####################################

def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model


###########################  Calculate the maximum survival times of the local clients  ###################################
def local_max_time(client_time):
    max_time = np.max(client_time)
    return max_time


#################################### Get Local Dataset from Directory #####################################################
def global_max_time(num_clients, net_dataidx_map, time):
    """
    Arguments:
        num_clients: Total number of clients
        net_dataidx_map: index mapping dictionary
        time: entire training time        
   """

    max_times = np.zeros((num_clients))
    for i in range(num_clients):
        max_times[i] = local_max_time(time[net_dataidx_map[i]])
    global_max_time = np.max(max_times)
    return global_max_time


#################################### Get Local Dataset from Directory #####################################################

def Pseudo_Evaltime1(global_max_time, interval):
    """
    Arguments:
        global_max_time: maximum survival time
        pseudo_df_train: train data with time and status
        time_horizon_interval: Equal distant interval for dividing the time horizon
        pseudo_perc_interval:interval for percentile of time at which pseudo values will be calculated 
    Returns:
        A vector of the evaluation time points at which pseudo values will be calculated.
    """
    PseudoEvalTime = np.percentile(
        np.arange(0.0, int(global_max_time), 1), np.arange(interval, 99, interval))
    return PseudoEvalTime
