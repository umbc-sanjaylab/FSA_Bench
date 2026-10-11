import numpy as np 
from .model import init_net
from .utils import Evaluation, evaluate_km_on_grid
  

##########################################  Kaplan-Meier Model in Centralized Settings ################################################# 

def train_centralized(data, eval_times, args):
    """
    Fit centralized Kaplan-Meier estimator and evaluate on the test split.
    Arguments: 
        data: an object containing train, val, and test splits with time and event attributes.
        eval_times: array-like, time points at which to evaluate the survival curve.
        args: arguments for initializing the Kaplan-Meier model.
    
    Returns: 
        dictionary: evaluation metrics for the test set. 
    """

    # Combine training and validation outcomes for KM estimation
    train_val_time = np.concatenate([ data.train.time, data.val.time]) 
    train_val_event = np.concatenate([ data.train.event, data.val.event])

    # Initialize and fit Kaplan-Meier estimator
    model = init_net(args)
    model.fit(train_val_time, train_val_event)

    # Obtain KM survival curve
    km_times, km_surv = model.predict_survival()

    # Evaluate KM curve at common evaluation times
    km_surv_at_eval_time = evaluate_km_on_grid( km_times, km_surv, eval_times )

    # Same KM survival curve is assigned to every test subject
    surv_test = np.tile(km_surv_at_eval_time, (len(data.test.time), 1))

    # evaluate model performance on test set
    test_cindex, test_brier, test_nbll, test_p_value, test_mae_po = Evaluation(surv_test, data.train.time, data.train.event, data.test.time, data.test.event, eval_times)
    return {
        "best_round": 0,
        "val_cindex": np.nan,
        "cindex": test_cindex,
        "brier": test_brier,
        "nbll": test_nbll,
        "d_calibration": test_p_value,
        "mae_po": test_mae_po
    }

