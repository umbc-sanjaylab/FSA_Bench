import numpy as np  
from scipy.stats import combine_pvalues 
from .model import init_net, local_partial_matrix, aggregate_local_matrices 
from .utils import Evaluation, evaluate_km_on_grid


def run_federated(clients, eval_times, args):
    local_matrices = []

    # Compute local KM summary statistics
    for client in clients:
        train_val_time = np.concatenate([client.train.time, client.val.time]) 
        train_val_event = np.concatenate([ client.train.event, client.val.event])

        local_matrix = local_partial_matrix(train_val_time, train_val_event)
        local_matrices.append(local_matrix)

    # Aggregate local statistics at server
    global_partial_matrix = aggregate_local_matrices( local_matrices )

    # Construct global Kaplan-Meier model
    model = init_net(args)
    model.fit_from_partial_matrix(global_partial_matrix)

    km_times, km_surv = model.predict_survival()

    # Evaluate global curve at common evaluation times
    km_surv_at_eval_time = evaluate_km_on_grid(km_times, km_surv, eval_times)

    # Aggregate client-level evaluation metrics
    weighted_val_cindex = 0.0
    weighted_cindex = 0.0
    weighted_brier = 0.0
    weighted_nbll = 0.0
    weighted_mae_po = 0.0

    total_val_cindex = 0
    total_cindex = 0
    total_brier = 0
    total_nbll = 0
    total_mae_po = 0

    client_p_values = []

    for client in clients:
        # Same global KM curve is used for every subject
        surv_val = np.tile( km_surv_at_eval_time, (len(client.val.time), 1))
        surv_test = np.tile(km_surv_at_eval_time, (len(client.test.time), 1))

        # Validation C-index
        val_cindex, _, _, _, _ = Evaluation(surv_val, client.train.time, client.train.event, client.val.time, client.val.event, eval_times)

        # Test metrics
        test_cindex, test_brier, test_nbll, test_p_value, test_mae_po = Evaluation( surv_test, client.train.time, client.train.event, client.test.time, client.test.event, eval_times) 
        n_val, n_test = len(client.val.time), len(client.test.time)

        if np.isfinite(val_cindex):
            weighted_val_cindex += val_cindex * n_val
            total_val_cindex += n_val

        if np.isfinite(test_cindex):
            weighted_cindex += test_cindex * n_test
            total_cindex += n_test

        if np.isfinite(test_brier):
            weighted_brier += test_brier * n_test
            total_brier += n_test

        if np.isfinite(test_nbll):
            weighted_nbll += test_nbll * n_test
            total_nbll += n_test

        if np.isfinite(test_p_value):
            client_p_values.append(float(test_p_value))

        if np.isfinite(test_mae_po):
            weighted_mae_po += test_mae_po * n_test
            total_mae_po += n_test

    val_cindex =   weighted_val_cindex / total_val_cindex if total_val_cindex > 0 else np.nan  
    test_cindex =  weighted_cindex / total_cindex if total_cindex > 0 else np.nan 
    test_brier =  weighted_brier / total_brier if total_brier > 0 else np.nan  
    test_nbll =  weighted_nbll / total_nbll if total_nbll > 0 else np.nan  
    test_mae_po =  weighted_mae_po / total_mae_po if total_mae_po > 0 else np.nan 

    if  client_p_values: 
        test_p_value = combine_pvalues(client_p_values, method="fisher")[1]
    else:
        test_p_value = np.nan 

    return {
        "best_round": 0,
        "val_cindex": val_cindex,
        "cindex": test_cindex,
        "brier": test_brier,
        "nbll": test_nbll,
        "d_calibration": test_p_value,
        "mae_po": test_mae_po
    }