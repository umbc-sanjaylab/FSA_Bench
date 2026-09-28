import torch
from torch import optim
import pandas as pd
import numpy as np
from utils import pseudo_loss, Concordance, Evaluation
import logging
from copy import deepcopy
import os
from scipy.stats import combine_pvalues
torch.cuda.set_per_process_memory_fraction(0.15, 0) 

def Pseudo_Evaltime(global_full_table):
    """
    Arguments:
        global_max_time: global unique times
        time_horizon_interval: Equal distant interval for dividing the time horizon
        pseudo_interval:interval for percentile of time at which pseudo values will be calculated

    Return:
        A vector of the evaluation time points at which pseudo values will be calculated.
    """
    pseudo_interval = 10
    global_unique_times = np.unique(global_full_table[:, 0])
    PseudoEvalTime = np.percentile(
        global_unique_times, np.arange(10, 99, pseudo_interval))
    return PseudoEvalTime


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
    df = pd.concat([pd.DataFrame(client_time.copy()),
                   pd.DataFrame(client_status.copy())], axis=1)
    df.columns = ['time', 'status']
    local_partial_matrix = compute_d_and_n_matrix(df)
    local_partial_matrix.reset_index(inplace=True)
    local_partial_matrix = local_partial_matrix.rename(
        columns={'index': 'time'})
    local_partial_matrix.loc[1:, 'ni'] = 0

    if is_DP:
        scale_parameter = sensitivity / epsilon
        Z = np.zeros(
            (local_partial_matrix.shape[0], (local_partial_matrix.shape[1] - 1)))
        for i in range(local_partial_matrix.shape[0]):
            for j in range(local_partial_matrix.shape[1] - 1):
                Z[i, j] = np.random.laplace(loc=0, scale=scale_parameter)

        local_partial_matrix.iloc[:, 1:] = local_partial_matrix.iloc[:, 1:] + Z
        local_partial_matrix[local_partial_matrix < 0] = 0

    local_dataframe = pd.DataFrame(local_partial_matrix)
    return local_dataframe.to_numpy(), list(local_dataframe.columns)


def local_loo_survival_probability(global_partial_matrix, client_time, client_status):
    global_partial_df = pd.DataFrame(global_partial_matrix, columns=[
                                     'time', 'nrisk', 'di', 'ci'])
    Surv_Eval_Time = np.array(global_partial_df.loc[:, 'time'])

    logger = logging.getLogger("fedPseudo")
    logger.info(f"length of client time is {len(client_status)}")
    logger.info(f"length of survival eval time is {len(Surv_Eval_Time)}")

    local_loo_surv_prob = np.zeros((len(client_time), len(Surv_Eval_Time)))
    local_loo_matrix = np.zeros((len(client_time), len(Surv_Eval_Time), 3))

    for i in range(len(client_time)):
        local_loo_matrix[i, 0, 0] = -1
        idx = np.where(client_time[i] == Surv_Eval_Time)
        if client_status[i] == 1:
            local_loo_matrix[i, idx, 1] = -1
        elif client_status[i] == 0:
            local_loo_matrix[i, idx, 2] = -1
        local_loo_matrix[i, :, :] = np.array(global_partial_df)[
            :, 1:]+local_loo_matrix[i, :, :]
        for p in range(len(Surv_Eval_Time)-1):
            local_loo_matrix[i, (p+1), 0] = local_loo_matrix[i, (p), 0] - \
                (local_loo_matrix[i, (p), 1]+local_loo_matrix[i, (p), 2])

    for j in range(len(client_time)):
        local_loo_surv_prob[j, :] = np.cumprod(
            1-(local_loo_matrix[j, :, 1]/(local_loo_matrix[j, :, 0]+0.000000001)))

    return local_loo_surv_prob


def federated_pseudo_values(global_full_matrix, global_survival_probability, local_loo_survival_probability, Pseudo_Eval_Time, args):

    is_DP = args.is_DP

    local_loo_surv_probability = local_loo_survival_probability.copy()
    if is_DP:
        idx = np.where(global_survival_probability == 0)
        local_loo_surv_probability[:, idx] = 0

    pseudo_values_all = global_full_matrix[0, 1] * global_survival_probability - (
        (global_full_matrix[0, 1] - 1) * local_loo_surv_probability)
    index = np.searchsorted(np.array(global_full_matrix)[
                            :, 0], Pseudo_Eval_Time, side='right') - 1
    pseudo_values = np.array(pd.DataFrame(pseudo_values_all).iloc[:, index])

    return pseudo_values

# Function for training the model  
def train(dataloader, model, loss_fn, optimizer, device):
    # Total number of observation divided by batch size
    num_batches = len(dataloader)
    model.train()
    model.to(device)

    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # x is covariates and y is the pseudo values in the batch
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    total_loss /= num_batches
    return total_loss

# Function for evaluating the model


def evaluate(dataloader, model, loss_fn, device):

    num_batches = len(dataloader)
    model.eval()

    # device="cuda:1"
    model.to(device)

    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            # x is covarites and y is the pseudo values in the batch
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += float(loss_fn(pred, y).item())
    test_loss /= num_batches

    return test_loss


#####################################  Global Survival Curve Probabilities  ########################################

# global survival function calculation
def global_survival_probability(global_partial_matrix, is_DP):
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

    if is_DP:
        indices = np.where(
            global_full_matrix[:, 1] <= global_full_matrix[:, 2])[0]
        if indices.size == 0:
            global_full_matrix = global_full_matrix
        else:
            idx = indices[0]
            global_full_matrix[idx:, 1] = global_full_matrix[idx-1, 1]
            global_full_matrix[idx:, 2] = 0

        global_full_matrix[global_full_matrix < 0] = 0

    # Calculate global survival probabilities using product rule formula
    global_surv = np.cumprod(
        1-(global_full_matrix[:, 2]/(global_full_matrix[:, 1]+0.000000001)))
    return global_full_matrix, global_surv


def fix_cindex_direction(cindex):
    if np.isfinite(cindex) and cindex < 0.5:
        return 1.0 - cindex
    return cindex 
        
def train_net(net_id, client_name, net, modeldir, dataset, train_dataloader, val_dataloader, train_time, train_status, X_val, val_time, val_status, evaltime, args, logger, device="cpu"):
  
    # logger initialization  
    logger.info(f"Client: {net_id} training is started")

    # validation data initialization
    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    val_time_array = np.asarray(val_time)
    val_status_array = np.asarray(val_status)

    # optimizer for model parameters udpate
    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters( )), lr=args.lr, weight_decay=args.reg)  # Default optimizer
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters( )), lr=args.lr, weight_decay=args.reg, amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter( lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)

    # learning scheduler
    if args.partition == 'centralized':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'min', patience=args.patience)  # Learning rate scheduler
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'min', patience=args.patience)  # Learning rate scheduler

    # epochs and patience
    patience = args.patience 

    # training and validation loss
    train_losses = []
    val_losses = []

    # concordance index
    cindex = []
    best_val_cindex = float("-inf")
    best_model_state = deepcopy(net.state_dict())

    # Pseudo value based loss function
    criterion = pseudo_loss

    es = 0
    for epoch in range(args.epochs):
        # Calculate the train loss
        train_loss = train(train_dataloader, net, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Calculate the validation loss
        val_loss = evaluate(val_dataloader, net, criterion, device)
        scheduler.step(val_loss)
        val_losses.append(val_loss)

        # Calculate the validation Cindex
        metrics = Concordance(net, X_val_tensor, val_time_array, val_status_array, evaltime, logger, device)
        cindex.append(metrics)

        # handle invalid concordance
        if np.sum(val_status_array == 1) == 0:
            continue  
        if metrics is None or not np.isfinite(metrics):
            logger.warning(
                f"Client: {net_id} invalid validation C-index at epoch {epoch}: {metrics}. "
                f"Skipping early-stopping update for this epoch."
            )
            es += 1
            if es >= patience:
                break
            continue  # Early Stopping Criteria

        elif metrics > best_val_cindex:
            best_val_cindex = metrics
            es = 0
            best_model_state = deepcopy(net.state_dict())
        else:
            es += 1
            if es >= patience:
                break

    # create directory if it does not exist
    checkpoint_dir = os.path.join(modeldir, dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # store and load the model parameters
    torch.save(best_model_state, f'{checkpoint_dir}/{args.model}_{args.run_id}_{client_name}.pt')
    net.load_state_dict(best_model_state)
    net.to(device) 

    return net 

################################################################################################################################

def evaluate_global_model_on_client(net, train_time, train_status,  X_valid_array,  valid_time,  valid_status,  X_test_array,  test_time,  test_status, pseudo_evaltime, logger, device):
    
    X_val_tensor = torch.as_tensor(X_valid_array, dtype=torch.float32, device=device) 
    validation_c_index = Concordance(net, X_val_tensor, np.asarray(valid_time),  np.asarray(valid_status), pseudo_evaltime,  logger, device)

    test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation(net,  train_time,  train_status,  X_test_array, test_time,  test_status, pseudo_evaltime, logger, device) 

    return validation_c_index, test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO

 