import numpy as np
import pandas as pd
import torch, logging  
from pycox.evaluation import EvalSurv
from SurvivalEVAL.Evaluator import SurvivalEvaluator 

######################################### Total Events and Censoring Computation ####################################################

def compute_d_and_n(data, t):
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

######################################### Kaplan Meier components computation  ####################################################

def compute_d_and_n_matrix(data):
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

######################################### Pseudo Value computation from client time and event ####################################################

def local_partial_matrix(client_time, client_status, sensitivity, epsilon, is_DP, init_seed):
    """
    Arguments:
        client_time: Client's event time
        client_status: Client's event status

        sensitivity: sensitivity parameter
        epsilon: Privacy budget parameter. if epsilon is None it doesn't add Laplace noise, i.e., no differential privacy is enforced. If epsilon is provided, it adds Laplace noise, i.e.,
        differential privacy is enforced.

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

    if is_DP:
        scale_parameter = sensitivity / epsilon
        Z = np.zeros( (local_partial_matrix.shape[0], (local_partial_matrix.shape[1] - 1)))
        for i in range(local_partial_matrix.shape[0]):
            for j in range(local_partial_matrix.shape[1] - 1):
                Z[i, j] = np.random.laplace(loc=0, scale=scale_parameter)

        local_partial_matrix.iloc[:, 1:] = local_partial_matrix.iloc[:, 1:] + Z
        local_partial_matrix[local_partial_matrix < 0] = 0

    local_dataframe = pd.DataFrame(local_partial_matrix)
    return local_dataframe.to_numpy(), list(local_dataframe.columns)

######################################### Convert Torch Tensor to NumPy Array ####################################################

def to_1d_numpy(x):
    # torch tensor -> numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # pandas Series/DataFrame column -> numpy
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()

    # list/others -> numpy
    x = np.asarray(x)
    return x.reshape(-1)


################################################## Concodance Index metrics ######################################################


def Concordance(model, x, durations, events, evaltime, logger, device="cpu"):
    """
    Attributes:
    ----------    
        model: Model 
        x: Covariates
        durations: Event status array 
        events: Observed times array 
        evaltime: Prespecified evaluation times

    Returns
    -------
        Time-dependent C-index
    """
    model = model.to(device)
    model.eval()
    x = x.to(device)

    surv = model(x)
    y_pred = pd.DataFrame(np.transpose(surv.cpu().detach().numpy()))
    y_pred = y_pred.set_index([evaltime])

    test_time_np = to_1d_numpy(durations).astype(np.float64)
    test_status_np = to_1d_numpy(events).astype(np.int64)
    ev = EvalSurv(y_pred, test_time_np, test_status_np, censor_surv='km') 
    
    if len(durations) < 2:
        logger.warning("C-index is not estimable: fewer than two samples.")
        return np.nan

    if np.sum(events == 1) == 0:
        logger.warning("C-index is not estimable: no observed events.")
        return np.nan


    cindex = np.nan
    try:
        cindex = float(ev.concordance_td())
        if not np.isfinite(cindex):
            logger.info("[C-index WARNING] NaN/Inf returned.")
            cindex = np.nan
    except ZeroDivisionError:
        print("[C-index WARNING] Zero comparable pairs in validation set.")
        cindex = np.nan

    return cindex


################################################## Evaluation metrics ######################################################

def Evaluation(model, train_time, train_status, X_test, test_time, test_status, evaltime, logger, device="cpu"):
    """
    Attributes:
    ----------
        model: Trained Model
        train_time: Event time array for train set
        train_status: Event status array for train set

        X_test: Covariates in test set
        test_time: Observed times array for test set
        test_status: Event status array for test set

        evaltime: Prespecified evaluation times
        device: Device to run the model on

    Returns
    -------
        cindex, brier, nbll, p_value, weighted_MAE_PO
    """
    logger = logging.getLogger("fedPseudo.client")

    model.eval()
    model.to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)

    # --- Build survival DataFrames ---
    try:
        sp_test = model(X_test_tensor)
        surv_array = sp_test.cpu().detach().numpy()  # (n_samples, n_eval_times)

        # EvalSurv (pycox): expects (n_eval_times, n_samples) with evaltime as row index
        surv_for_evsurv = pd.DataFrame(
            np.transpose(surv_array),            # → (n_eval_times, n_samples)
            index=evaltime
        )

        # SurvivalEvaluator (survivaleval): expects (n_samples, n_eval_times) with evaltime as columns
        surv_for_eval = pd.DataFrame(
            surv_array,                          # → (n_samples, n_eval_times)
            columns=np.asarray(evaltime, dtype=float)
        )

        ev = EvalSurv(surv_for_evsurv, test_time,
                      test_status, censor_surv='km')

    except Exception as e:
        logger.error(
            f"[Evaluation] Failed to build survival frames: {e}", exc_info=True)
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # --- C-index ---
    try:
        cindex = float(ev.concordance_td())
        cindex = cindex if np.isfinite(cindex) else np.nan
    except Exception as e:
        logger.warning(f"[C-index WARNING] {e}")
        cindex = np.nan

    # --- Integrated Brier Score ---
    try:
        brier = float(ev.integrated_brier_score(evaltime))
        brier = brier if np.isfinite(brier) else np.nan
    except Exception as e:
        logger.warning(f"[Brier WARNING] {e}")
        brier = np.nan

    # --- Integrated NBLL ---
    try:
        nbll = float(ev.integrated_nbll(evaltime))
        nbll = nbll if np.isfinite(nbll) else np.nan
    except Exception as e:
        logger.warning(f"[NBLL WARNING] {e}")
        nbll = np.nan

    # --- SurvivalEvaluator (D-Calibration & Weighted MAE) ---
    test_time_arr = np.asarray(test_time,    dtype=float).reshape(-1)
    test_status_arr = np.asarray(test_status,  dtype=int).reshape(-1)
    train_time_arr = np.asarray(train_time,   dtype=float).reshape(-1)
    train_status_arr = np.asarray(train_status, dtype=int).reshape(-1)

    logger.info(
        f"[SurvivalEvaluator shapes] "
        # expect (n_test_samples, n_eval_times) e.g. (649, 9)
        f"surv={surv_for_eval.shape}, "
        f"evaltime={np.asarray(evaltime).shape}, "  # expect (9,)
        f"test_time={test_time_arr.shape}, "  # expect (649,)
        f"train_time={train_time_arr.shape}"  # expect (1944,)
    )

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
            p_value = float(p_val_result) if np.isfinite(
                p_val_result) else np.nan
        except Exception as e:
            logger.warning(f"[D-Calibration WARNING] {e}")

        try:
            mae_result = evaluator.mae(method='Pseudo_obs', weighted=True)
            weighted_MAE_PO = float(mae_result) if np.isfinite(
                mae_result) else np.nan
        except Exception as e:
            logger.warning(f"[Weighted MAE WARNING] {e}")

    except Exception as e:
        logger.error(
            f"[SurvivalEvaluator] Failed to initialize: {e}", exc_info=True)

    logger.info(
        f"[Evaluation] cindex={cindex}, brier={brier}, nbll={nbll}, "
        f"p_value={p_value}, weighted_MAE_PO={weighted_MAE_PO}"
    )

    return cindex, brier, nbll, p_value, weighted_MAE_PO


################################################## Evaluate Global Model on Client ###########################################################################

def evaluate_global_model_on_client(net, train_time, train_status,  X_valid_array,  valid_time,  valid_status,  X_test_array,  test_time,  test_status, pseudo_evaltime, logger, device): 
    X_val_tensor = torch.as_tensor(X_valid_array, dtype=torch.float32, device=device) 
    validation_c_index = Concordance(net, X_val_tensor, np.asarray(valid_time),  np.asarray(valid_status), pseudo_evaltime,  logger, device) 
    test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation(net,  train_time,  train_status,  X_test_array, test_time,  test_status, pseudo_evaltime, logger, device)  
    return validation_c_index, test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO
  
  
################################################## Model Save and Load ###########################################################

def save_model(model, model_index, args):
    logger = logging.getLogger("fedPseudo.client")
    logger.info(f"saving local model-{model_index}")
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model
