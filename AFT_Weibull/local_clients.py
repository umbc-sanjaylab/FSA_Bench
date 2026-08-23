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
from scipy.stats import combine_pvalues
import traceback
from torchsurv.loss.weibull import neg_log_likelihood_weibull
torch.cuda.set_per_process_memory_fraction(0.15, 0)

########################## Function for training the model  #####################################


def train(train_loader, weibull_model, optimizer, device="cpu"):
    weibull_model.train()
    weibull_model.to(device)

    total_loss = 0.0
    num_points = 0

    for X_batch, Y_batch, D_batch in train_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        D_batch = D_batch.to(device).bool()

        # skip fully censored batch
        if torch.sum(D_batch).item() == 0:
            continue

        log_hazards_batch = weibull_model(X_batch) 
        loss_batch = neg_log_likelihood_weibull(
            log_hazards_batch, D_batch, Y_batch)

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        batch_loss_sum = neg_log_likelihood_weibull(
            log_hazards_batch, D_batch, Y_batch, reduction='sum'
        ).item()

        total_loss += batch_loss_sum
        num_points += X_batch.size(0)

    if num_points == 0:
        return np.nan

    train_loss = total_loss / num_points
    return train_loss

########################## Function for evaluating the model ##############################################################


def evaluate(val_loader, weibull_model, device="cpu"):

    weibull_model.eval()
    weibull_model.to(device)

    with torch.no_grad():
        val_loss = torch.tensor(0.0, dtype=torch.float, device=device)
        num_points = 0

        for X_batch, Y_batch, D_batch in val_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            D_batch = D_batch.to(device).bool()

            # skip fully censored batches
            if torch.sum(D_batch).item() == 0:
                continue

            log_hazards_batch = weibull_model(X_batch)
            val_loss += neg_log_likelihood_weibull(
                log_hazards_batch, D_batch, Y_batch, reduction='sum')
            num_points += X_batch.size(0)

        if num_points == 0:
            return np.nan

        val_loss = float(val_loss / num_points)

    return val_loss


################  Training DeepPseudo Model with "Centralized" framework in Local Client  #################################
def train_net_all(net_id, net, X_train, train_time, train_status, X_val, val_time, val_status, X_test, test_time, test_status, evaltime, feature_list, args, device="cpu"):

    print('Training network %s' % str(net_id))

    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)  # Default optimizer
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg, amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.rho, weight_decay=args.reg)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience)  # Learning rate scheduler
    Epochs = args.epochs  # Number of local epochs

    best_val_cindex = 0.0
    best_model_state = deepcopy(net.state_dict())

    cindex = []
    es = 0

    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    val_time_array = np.asarray(val_time)
    val_status_array = np.asarray(val_status)

    # train data loader from covariates, time and event status information
    train_x = DataBuilder(X_train, train_time, train_status, feature_list)
    train_dataloader = DataLoader(dataset=train_x, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # validation data loader from covariates, time and event status information
    val_x = DataBuilder(X_val, val_time, val_status, feature_list)
    val_dataloader = DataLoader(dataset=val_x, batch_size=args.batch_size, shuffle=False, num_workers=1)

    for epoch in range(Epochs):
        # Calculate the train loss using train function

        train_loss = train(train_dataloader, net, optimizer, device)
        val_loss = evaluate(val_dataloader, net, device)
        scheduler.step(val_loss)

        # Calculate the validation Cindex
        metrics = Concordance(net, X_val_tensor, val_time_array, val_status_array, evaltime, device)
        cindex.append(metrics)

        # Early Stopping Criteria
        if np.sum(val_status_array == 1) == 0:
              continue  
                    
        elif metrics is None or not np.isfinite(metrics):
            es += 1
            if es >= args.patience:
                break
            continue
        
        elif metrics > best_val_cindex:
            best_val_cindex = metrics
            es = 0
            best_model_state = deepcopy(net.state_dict())
        
        else:
            es += 1
            if es >= args.patience:
                break

    print("Done!")

    # Load the trained model and evaluate the performances
    torch.save(best_model_state, f"{args.modeldir}/{args.model}_{args.run_id}_{net_id}.pt")
    print(f"model is saved at {args.modeldir}/{args.model}_{args.run_id}_{net_id}.pt")

    net.load_state_dict(best_model_state)
    net.to(device)
    test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation(net, train_time, train_status, X_test, test_time, test_status, evaltime, device)

    print('>> Test Cindex: {}'.format(test_cindex))
    print('>> Test Brier: {}'.format(test_brier))
    print('>> Test NBLL: {}'.format(test_nbll))

    print(' ** Training complete **')
    return test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO


################  Training Weibull AFT Model with "FedAvg" framework in Local Client  #################################
def train_net(net_id,  net, train_dataloader, val_dataloader,  X_val, val_time, val_status,  evaltime, args,  device="cpu"):

    modeldir = args.modeldir
    epochs = args.epochs
    
    lr = args.lr
    opt_name = args.optimizer
    
    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    val_time_array = np.asarray(val_time)
    val_status_array = np.asarray(val_status)
    
    valid_val_metric_seen = False
    has_val_events =  len(val_status_array) > 0 and np.any(val_status_array == 1) 

    print(f"Training network {net_id}")  
    if opt_name == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif opt_name == "adamw":
        optimizer = AdamW(net.parameters(), lr=lr, correct_bias=False)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience)

    best_val_cindex = float("-inf")
    best_model_state = deepcopy(net.state_dict())
    es = 0

    for epoch in range(epochs):
        train_loss = train(train_dataloader, net, optimizer, device)
        val_loss = evaluate(val_dataloader, net, device)
        
        if val_loss is not None and np.isfinite(float(val_loss)):
            scheduler.step(float(val_loss))
        else:
            print(f"Client {net_id}: invalid validation loss at epoch {epoch}: {val_loss}")
        
        scheduler.step(val_loss)
        print(f"client: {net_id} epoch: {epoch}, train loss: {train_loss} and val loss: {val_loss}")
        
        metrics = None 
        if has_val_events:
            try:
                metrics = Concordance(net, X_val_tensor, val_time_array, val_status_array, evaltime, device)
            except Exception as exc:
                print( f"Client {net_id}: validation C-index unavailable at epoch {epoch}: {exc}") 

        # Calculate the validation Cindex
        print(f"client: {net_id}, epoch: {epoch}, and  validation metric: {metrics}")  
        
        # Early Stopping Criteria
        if metrics is None or not np.isfinite(metrics):
              continue

        valid_val_metric_seen = True
        if metrics > best_val_cindex:
            best_val_cindex = metrics
            es = 0
            best_model_state = deepcopy(net.state_dict())
        
        else:
            es += 1
            if es >= args.patience:
                break
    
    # Retain the final locally trained state.
    if not valid_val_metric_seen:
        best_model_state = deepcopy(net.state_dict())
    
    torch.save(best_model_state,  f"{modeldir}/{args.model}_{args.run_id}_{net_id}.pt")
    net.load_state_dict(best_model_state)
    net.to(device)

    return net
 

def local_train_net(nets, selected, args, X_train_all, train_time_all, train_status_all, X_val_all, val_time_all, val_status_all, pseudo_evaltime, feature_list, device="cpu"): 

    for net_id, net in nets.items():  
        if net_id not in selected:
            continue

        # move the model to cuda device:
        net.to(device)

        X_train = X_train_all[net_id]
        train_time = train_time_all[net_id]
        train_status = train_status_all[net_id]
        
        # skip a client if there are no events in the training data
        X_val = X_val_all[net_id]
        val_time = val_time_all[net_id]
        val_status = val_status_all[net_id] 

        # train data loader from covariates, time and event status information
        train_x = DataBuilder(X_train, train_time, train_status, feature_list)
        train_dataloader = DataLoader(dataset=train_x, batch_size=args.batch_size, shuffle=True, num_workers=1)

        # validation data loader from covariates, time and event status information
        val_x = DataBuilder(X_val, val_time, val_status, feature_list)
        val_dataloader = DataLoader(dataset=val_x, batch_size=args.batch_size, shuffle=False, num_workers=1)  
        nets[net_id] = train_net(net_id,  net, train_dataloader, val_dataloader,  X_val, val_time, val_status,  pseudo_evaltime, args,  device=device) 
    return nets  
      
############################################################################################################################################

def is_finite_metric(value):
    return (
        value is not None
        and np.isscalar(value)
        and np.isfinite(float(value))
    )
      
################################## Evaluate Global Model on Clients ################################################################### 


def evaluate_global_model_on_clients(global_model,  selected,  X_val_local_std,  val_time_local,  val_status_local, X_test_local_std, test_time_local, test_status_local,  train_time_local,  train_status_local,  pseudo_evaltime,  device="cpu"):
    global_model.to(device)
    global_model.eval()
    
    total_samples_cindex, total_samples_brier= 0, 0
    total_samples_nbll, total_samples_mae = 0, 0

    total_val_samples = 0
    total_test_samples = 0
    weighted_val_cindex = 0.0
    client_p_values = []  

    avg_test_cindex = 0.0
    avg_test_brier = 0.0
    avg_test_nbll = 0.0
    avg_test_p_value = 0.0
    avg_test_MAE_PO = 0.0 

    for net_id in selected:
        X_val = X_val_local_std[net_id]
        val_time = np.asarray(val_time_local[net_id])
        val_status = np.asarray(val_status_local[net_id])

        X_test = X_test_local_std[net_id]
        test_time = np.asarray(test_time_local[net_id])
        test_status = np.asarray(test_status_local[net_id])

        train_time = np.asarray(train_time_local[net_id])
        train_status = np.asarray(train_status_local[net_id])

        X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=device)
         
        val_cindex = np.nan
        if len(val_status) > 0 and np.any(val_status == 1):
            try:
                val_cindex = Concordance( global_model, X_val_tensor, val_time, val_status, pseudo_evaltime, device)
            except Exception as exc:
                print(f"Client {net_id}: validation C-index could not be calculated: {exc}")
        
        test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation(global_model, train_time,  train_status, X_test, test_time, test_status, pseudo_evaltime, device)
         
        
        if val_cindex is not None and np.isfinite(val_cindex):
            weighted_val_cindex += float(val_cindex) * len(val_time)
            total_val_samples += len(val_time) 
  
        if is_finite_metric(test_cindex):
            avg_test_cindex +=  test_cindex  * len(test_time)
            total_samples_cindex += len(test_time)
            
        if is_finite_metric(test_brier):
            avg_test_brier += test_brier  * len(test_time)
            total_samples_brier += len(test_time)
        
        if is_finite_metric(test_nbll):
            avg_test_nbll += test_nbll  * len(test_time)
            total_samples_nbll += len(test_time)
        
        if is_finite_metric(test_weighted_MAE_PO): 
            avg_test_MAE_PO += test_weighted_MAE_PO  * len(test_time)
            total_samples_mae += len(test_time)
        
        if is_finite_metric(test_p_value): 
            client_p_values.append(float(test_p_value))
                
      # global_val_cindex = fix_cindex_direction(weighted_val_cindex / total_val_samples) 
    global_val_cindex = weighted_val_cindex / total_val_samples
    
    if len(client_p_values) > 0:
        avg_test_p_value = combine_pvalues(client_p_values, method="fisher")[1]
    else:
        avg_test_p_value = np.nan
         
    global_val_cindex = weighted_val_cindex / total_val_samples if total_val_samples > 0 else np.nan    
    avg_test_cindex =  avg_test_cindex / total_samples_cindex if total_samples_cindex > 0 else np.nan    
    avg_test_brier =  avg_test_brier / total_samples_brier if total_samples_brier > 0 else np.nan  
    avg_test_nbll =  avg_test_nbll / total_samples_nbll if total_samples_nbll > 0 else np.nan  
    avg_test_MAE_PO =  avg_test_MAE_PO / total_samples_mae if total_samples_mae > 0 else np.nan  

    return global_val_cindex, avg_test_cindex, avg_test_brier, avg_test_nbll, avg_test_p_value, avg_test_MAE_PO  
    
############################################################################################################################################