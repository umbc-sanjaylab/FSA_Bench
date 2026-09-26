import os

import numpy as np
import torch
import torch.optim as optim 
from copy import deepcopy
from math import *
from torch.optim import AdamW   
from datasets.data_processing import SurvivalDataset
from torch.utils.data import DataLoader
from models.aft_weibull import train_epoch, evaluate

from models.aft_weibull import predict_survival
from evaluation.metrics import evaluate_survival_predictions, compute_concordance  


def create_optimizer(model, args):
    """
    Create the optimizer used for local client training.
    """ 
    trainable_parameters = filter( lambda parameter: parameter.requires_grad, model.parameters()) 
    if args.optimizer == "adam": 
        return optim.Adam( trainable_parameters, lr=args.lr, weight_decay=args.reg)

    elif args.optimizer == "adamw": 
        return optim.AdamW( trainable_parameters, lr=args.lr, weight_decay=args.reg)

    else: 
        raise ValueError( f"Unsupported optimizer: {args.optimizer}" )

################  Training DeepPseudo Model with "Centralized" framework in Local Client  #################################
def train_net_all( net_id, net, X_train, train_time, train_status,  X_val, val_time, val_status, X_test, test_time,  test_status, evaltime, feature_list, args, device="cpu"):
    """
    Train and evaluate the AFT-Weibull model under centralized learning.

    Model selection is performed using validation time-dependent C-index.
    Final test metrics are computed using the model corresponding to the
    best validation C-index.
    """

    print(f"Training network {net_id}")
 
    # 1. Optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,)

    elif args.optimizer == "amsgrad":
        optimizer = optim.Adam( filter( lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg, amsgrad=True)

    elif args.optimizer == "sgd": 
        optimizer = optim.SGD(filter( lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.rho, weight_decay=args.reg)

    else:
        raise ValueError( f"Unsupported optimizer: {args.optimizer}")

    # 2. Learning-rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode="min", patience=args.patience)
 
    # 3. Prepare datasets
    train_dataset = SurvivalDataset(X_train, train_time, train_status, feature_list)
    train_dataloader = DataLoader( dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    
    val_dataset = SurvivalDataset( X_val, val_time, val_status, feature_list)
    val_dataloader = DataLoader( dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
 
    # 4. Validation arrays 
    val_time_array = np.asarray(val_time, dtype=float).reshape(-1)
    val_status_array = np.asarray(val_status,  dtype=int).reshape(-1) 
    has_val_events = (  len(val_status_array) > 0 and np.any(val_status_array == 1) ) 
    
    # 5. Early-stopping initialization  
    best_val_cindex = float("-inf")
    best_model_state = deepcopy(net.state_dict())

    early_stop_counter = 0
    valid_val_metric_seen = False
 
    # 6. Training loop  
    for epoch in range(args.epochs):

        # AFT-Weibull training loss
        train_loss = train_epoch(train_dataloader, net,  optimizer,  device)

        # AFT-Weibull validation NLL
        val_loss = evaluate(val_dataloader, net, device)

        # Update scheduler only when validation loss is valid
        if  val_loss is not None and np.isfinite(float(val_loss)): 
            scheduler.step(float(val_loss))
        else:
            print(f"Epoch {epoch}: invalid validation loss = {val_loss}")
 
        # 6.1 Validation survival prediction
        val_cindex = np.nan
        if has_val_events:
            try:
                val_survival = predict_survival(model=net, X=X_val, eval_times=evaltime, device=device) 
                val_cindex = compute_concordance(survival=val_survival, eval_times=evaltime, durations=val_time_array, events=val_status_array)   # validation C-index 
            except Exception as exc: 
                print(f"Epoch {epoch}: validation C-index unavailable: {exc}")
                val_cindex = np.nan

       
        # 6.2 Logging 
        print(f"Epoch {epoch}: train_loss={train_loss},  val_loss={val_loss},  val_cindex={val_cindex}")
 
        # 6.3 Early stopping
        if not has_val_events:  
            continue    # No validation events 

        if  val_cindex is None or not np.isfinite(val_cindex): 
            early_stop_counter += 1

            if early_stop_counter >= args.patience:
                print( "Early stopping: validation C-index remained undefined.")
                break

            continue

        valid_val_metric_seen = True 
        if val_cindex > best_val_cindex: 
            best_val_cindex = val_cindex 
            
            early_stop_counter = 0 
            best_model_state = deepcopy(net.state_dict())

        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print( f"Early stopping at epoch {epoch}")
                break

    print("Training completed.")
    
    # 7. retain final model if validation C-index was never available,   
    if not valid_val_metric_seen:
        best_model_state = deepcopy(  net.state_dict() )
 
    # 8. Save best model 
    os.makedirs(args.modeldir,  exist_ok=True)
    model_path = os.path.join(args.modeldir, f"{args.model}_{args.run_id}_{net_id}.pt")
    torch.save(best_model_state, model_path)
    print(f"Model saved at {model_path}") 
     
    # 9. Restore best validation model 
    net.load_state_dict( best_model_state)
    net.to(device)
    net.eval()

    # 10. test survival probabilities
    test_survival = predict_survival(model=net, X=X_test, eval_times=evaltime, device=device)

    # 11. Compute model-independent survival metrics
    test_cindex, test_brier, test_nbll,  test_p_value,  test_weighted_MAE_PO = evaluate_survival_predictions(
        survival=test_survival, eval_times=evaltime,
        train_time=train_time, train_event=train_status,
        test_time=test_time, test_event=test_status,
    )

     
    # 12. Report test results 
    print( f">> Test C-index: {test_cindex}" ) 
    print( f">> Test Brier: {test_brier}" ) 
    print( f">> Test NBLL: {test_nbll}" ) 
    print( f">> Test D-calibration p-value: {test_p_value}" ) 
    print( f">> Test weighted MAE-PO: {test_weighted_MAE_PO}" )

    print("** Training complete **") 
    return test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO
     

################  Training Weibull AFT Model with "FedAvg" framework in Local Client  #################################
def train_client(net_id,  net, train_dataloader, val_dataloader,  X_val, val_time, val_status,  evaltime, args,  device="cpu"): 
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
        train_loss = train_epoch(train_dataloader, net, optimizer, device)
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
 