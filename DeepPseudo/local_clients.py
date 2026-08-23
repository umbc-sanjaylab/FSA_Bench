import numpy as np
import torch
import torch.optim as optim
import argparse
import logging
from copy import deepcopy
from math import *
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model import *
# from experiments import    *
from utils import *
import os
import traceback

########################## Function for training the model  #####################################


def train(dataloader, model, loss_fn, optimizer, device="cpu"):
    # Total number of observation divided by batch size
    num_batches = len(dataloader)
    total_loss = 0

    model.train()
    model.to(device)

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

########################## Function for evaluating the model ##############################################################


def evaluate(dataloader, model, loss_fn, device="cpu"):
    model.eval()
    model.to(device)

    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            # x is covarites and y is the pseudo values in the batch
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += float(loss_fn(pred, y).item())

    test_loss /= num_batches

    return test_loss
  

################  Fix C-Index if reverse direction  #################################
  
def fix_cindex_direction(cindex):
    if np.isfinite(cindex) and cindex < 0.5:
        return 1.0 - cindex
    return cindex

################  Training DeepPseudo Model with "Centralized" framework in Local Client  #################################


def train_net_all( net_id, net,  train_dataloader, val_dataloader, epochs,  lr, args_optimizer, train_time, train_status, X_val, val_time, val_status, X_test, test_time,  test_status,  evaltime, args, logger, device="cpu"):
    
    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    val_time_array = np.asarray(val_time)
    val_status_array = np.asarray(val_status)

    logger.info(f"Training network {net_id}")
    net = net.to(device)

    if args_optimizer == "adam":
        optimizer = optim.Adam( filter(lambda p: p.requires_grad, net.parameters()), lr=lr,   weight_decay=args.reg,  )
    elif args_optimizer == "adamw":
        optimizer = AdamW(net.parameters(), lr=lr, correct_bias=False)
    else:
        raise ValueError(f"Unsupported optimizer: {args_optimizer}")

    criterion = pseudo_loss

    if args_optimizer == "adam":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience)
    elif args_optimizer == "adamw":
        num_training_steps = epochs * len(train_dataloader)
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    patience = args.patience
    best_val_cindex = -float("inf")
    best_epoch = -1
    best_state = deepcopy(net.state_dict())

    train_losses = []
    val_losses = [] 
    es = 0

    for epoch in range(epochs):
        train_loss = train(train_dataloader, net, criterion, optimizer, device)
        train_losses.append(train_loss)

        val_loss = evaluate(val_dataloader, net, criterion, device)
        val_losses.append(val_loss)

        if args_optimizer == "adam":
            scheduler.step(val_loss)
        elif args_optimizer == "adamw":
            scheduler.step()

        logger.info(f"epoch={epoch}, train_loss={train_loss}, val_loss={val_loss}") 
        cindex_metric = Concordance( net, X_val_tensor, val_time_array, val_status_array, evaltime, device)  
        logger.info(f"epoch={epoch}, validation C-index={cindex_metric}")

        # invalid validation metric -> count as no improvement
        if cindex_metric is None or not np.isfinite(cindex_metric):
            logger.info(f"[Validation WARNING] Invalid validation C-index at epoch {epoch}: {cindex_metric}")
            es += 1
            if es >= patience:
                logger.info(f"Early stopping at epoch {epoch} due to repeated invalid/non-improving validation.")
                break
            continue

        if cindex_metric > best_val_cindex:
            best_val_cindex = cindex_metric
            best_epoch = epoch
            es = 0
            best_state = deepcopy(net.state_dict())
        else:
            es += 1
            if es >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info("Done!")

    os.makedirs(args.modeldir, exist_ok=True)
    model_path = os.path.join(args.modeldir, f"{args.model}_{args.run_id}_{net_id}.pt")
    torch.save(best_state, model_path)

    net.load_state_dict(best_state)
    net = net.to(device)
    net.eval()

    test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation( net, train_time, train_status, X_test, test_time, test_status, evaltime, logger, device) 
    test_cindex = fix_cindex_direction(test_cindex)

    return test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO

################  Training DeepPseudo Model with "FedAvg" framework in Local Client  #################################
def model_is_finite(model):
    return all(
        torch.isfinite(p).all().item()
        for p in model.parameters()
    )

def train_net(net_id, client_name, net, train_dataloader, val_dataloader,  train_time, train_status,  X_val, val_time, val_status, X_test, test_time, test_status, evaltime, args, comm_round, logger, device="cpu"):

    modeldir, epochs, lr, optimizer, patience = args.modeldir, args.epochs, args.lr, args.optimizer, args.patience
    logger.info(f'Training network {str(net_id)}')

    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32)
    val_time_array = np.asarray(val_time)
    val_status_array = np.asarray(val_status) 

    if optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters( )), lr=lr, weight_decay=args.reg)  # Default optimizer
    elif optimizer == 'adamw':
        optimizer = AdamW(net.parameters(), lr=lr, correct_bias=False)
 
    # learning scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'min', patience=args.patience)  # Learning rate scheduler
    criterion = pseudo_loss  # Pseudo value based loss function

    # concordance index
    best_val_cindex = float("-inf")
    best_model_state = deepcopy(net.state_dict())  

    es = 0
    for epoch in range(epochs):
        # Calculate the train loss using train function
        train_loss = train(train_dataloader, net, criterion, optimizer, device)  
        val_loss = evaluate(val_dataloader, net, criterion, device) 
        
        logger.info(f'client {net_id} train loss: {train_loss}, validation loss: {val_loss}') 
        scheduler.step(val_loss) 
      
        # Calculate the validation Cindex
        metrics = Concordance(net, X_val_tensor, val_time_array, val_status_array, evaltime, device) 
        logger.info(f'client {net_id} evaluation metric: {metrics}')  
        
        # Early Stopping Criteria
        if np.sum(val_status_array == 1) == 0:
            continue  
        elif metrics is None or not np.isfinite(metrics):
            es += 1
            if es >= patience:
                break
            continue
        elif metrics > best_val_cindex:
            best_val_cindex = metrics
            es = 0
            best_model_state = deepcopy(net.state_dict())
        else:
            es += 1
            if es >= patience:
                break
    
    best_model_state_cpu = { k: v.detach().cpu().clone() for k, v in best_model_state.items()}
    
    # store and load the model parameters
    torch.save(best_model_state_cpu, f'{modeldir}/{args.model}_{args.run_id}_{net_id}.pt') 
    net.load_state_dict(best_model_state_cpu)
    
    net.to(device)  
    return  net 


def local_train_net(client_name_array, nets, selected, args, X_train_all, train_pseudo_all, train_time_all, train_status_all, X_val_all, val_pseudo_all, val_time_all, val_status_all, X_test_all, test_time_all, test_status_all, pseudo_evaltime, tr_data_len, comm_round, logger, device="cpu"): 
 
    for net_id, net in nets.items():

        client_name = client_name_array[net_id]
        if net_id not in selected:
            continue

        # move the model to cuda device:
        net.to(device)

        X_train = X_train_all[net_id]
        train_pseudo = train_pseudo_all[net_id]
        train_time = train_time_all[net_id]
        train_status = train_status_all[net_id]

        X_val = X_val_all[net_id]
        val_pseudo = val_pseudo_all[net_id]
        val_time = val_time_all[net_id]
        val_status = val_status_all[net_id]

        X_test = X_test_all[net_id]
        test_time = test_time_all[net_id]
        test_status = test_status_all[net_id]

        train_dataloader = FastTensorDataLoader(torch.from_numpy(X_train), torch.from_numpy(  train_pseudo), batch_size=args.batch_size, shuffle=True)
        val_dataloader = FastTensorDataLoader(torch.from_numpy(X_val),  torch.from_numpy(  val_pseudo), batch_size=args.batch_size, shuffle=False) 
        
        nets[net_id] = train_net( net_id, client_name, net, train_dataloader, val_dataloader, train_time, train_status, X_val, val_time, val_status, X_test, test_time, test_status, pseudo_evaltime, args, comm_round, logger, device=device) 
    
    return nets 

############################################### Evaluate Global Model ##################################################################### 


def evaluate_global_model_on_clients(global_model,  selected, X_val_local_std, val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local, train_time_local, train_status_local, pseudo_evaltime, logger, device="cpu"):

    global_model.to(device)
    global_model.eval()

    # --- validation accumulators (sample-weighted) ---
    total_val_samples = 0
    weighted_val_cindex = 0.0

    # --- test accumulators (sample-weighted, per metric) ---
    # each metric tracks its own weight sum, since a client can be NaN on
    # one metric but finite on another
    test_acc = {
        "cindex": {"weighted_sum": 0.0, "n": 0},
        "brier":  {"weighted_sum": 0.0, "n": 0},
        "nbll":   {"weighted_sum": 0.0, "n": 0},
        "MAE_PO": {"weighted_sum": 0.0, "n": 0},
    }
    # p-values are NOT averaged numerically; collect per-client for reporting
    test_p_values = [] 

    # --- bookkeeping for auditability (IID vs non-IID drop comparison) ---
    val_dropped = []
    test_dropped = {k: [] for k in test_acc}

    def _accumulate(metric_key, value, weight, client_id):
        """Add a per-client metric into its weighted accumulator if finite."""
        if value is not None and np.isfinite(value):
            test_acc[metric_key]["weighted_sum"] += float(value) * weight
            test_acc[metric_key]["n"] += weight
        else:
            test_dropped[metric_key].append(client_id)
    
    
    for net_id in selected: 
            
        X_val = X_val_local_std[net_id]
        val_time = np.asarray(val_time_local[net_id])
        val_status = np.asarray(val_status_local[net_id])

        X_test = X_test_local_std[net_id]
        test_time = np.asarray(test_time_local[net_id])
        test_status = np.asarray(test_status_local[net_id])

        train_time = np.asarray(train_time_local[net_id])
        train_status = np.asarray(train_status_local[net_id])

        n_val = len(val_time)
        n_test = len(test_time)

        X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=device)

        try:
            val_cindex = Concordance(global_model, X_val_tensor, val_time, val_status, pseudo_evaltime, device)
        except Exception as e:
            logger.info(f"[Client {net_id}] Validation failed: {type(e).__name__}: {e}")
            val_cindex = np.nan

        try:
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation(global_model, train_time, train_status, X_test,  test_time, test_status,  pseudo_evaltime,  logger, device)
        except Exception as e:
            logger.info(f"[Client {net_id}] Test evaluation failed: {type(e).__name__}: {e}")
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = [np.nan] * 5

        logger.info(f"[Client {net_id}]  val_n={n_val}, val_events={int(np.sum(val_status))},  val_cindex={val_cindex}, test_n={n_test}, test_events={int(np.sum(test_status))},  test_cindex={test_cindex}") 
    
        # --- validation: sample-weighted ---
        if val_cindex is not None and np.isfinite(val_cindex):
            weighted_val_cindex += float(val_cindex) * n_val
            total_val_samples += n_val 
        else:
            val_dropped.append(net_id) 
        
        # --- test: sample-weighted, per metric, weight = n_test ---
        _accumulate("cindex", test_cindex, n_test, net_id)
        _accumulate("brier",  test_brier, n_test, net_id)
        _accumulate("nbll",   test_nbll, n_test, net_id)
        _accumulate("MAE_PO", test_weighted_MAE_PO, n_test, net_id)

        # p-value: collected, not numerically averaged
        if test_p_value is not None and np.isfinite(test_p_value):
            test_p_values.append(test_p_value)

    # --- finalize validation ---
    # if total_val_samples > 0:  
    global_val_cindex =  weighted_val_cindex / total_val_samples 
    
    # --- finalize test (sample-weighted means) ---
    def _weighted_mean(metric_key):
        acc = test_acc[metric_key]
        return acc["weighted_sum"] / acc["n"] if acc["n"] > 0 else np.nan
 
    avg_test_cindex = _weighted_mean("cindex")  
    avg_test_brier = _weighted_mean("brier")
    avg_test_nbll = _weighted_mean("nbll")
    avg_test_MAE_PO = _weighted_mean("MAE_PO")

    # p-value: median is a placeholder summary; report the list separately if needed
    median_test_p_value = np.nanmedian(test_p_values) if len(test_p_values) > 0 else np.nan

    # --- auditability: log how many / which clients were dropped ---
    n_selected = len(list(selected))
    if val_dropped:
        logger.info(f"[Aggregation] Val: dropped {len(val_dropped)}/{n_selected} clients (NaN): {val_dropped}")
        
    for k, dropped in test_dropped.items():
        if dropped:
            logger.info(f"[Aggregation] Test {k}: dropped {len(dropped)}/{n_selected} clients (NaN): {dropped}")

    return global_val_cindex, avg_test_cindex, avg_test_brier, avg_test_nbll, median_test_p_value, avg_test_MAE_PO 