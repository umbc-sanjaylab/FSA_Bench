import numpy as np, torch
from copy import deepcopy 
from pathlib import Path

from ..model import train, validate
from scipy.stats import combine_pvalues
from .utils import prepare_client_dataloaders, negative_llh 
from .utils import build_optimizer, Concordance, Evaluation 

################  Logger Initialization  ################################# 

import logging
logger = logging.getLogger("experiment")

################  Finite Metric Check  ################################# 

def is_finite_metric(value):
    return value is not None and np.isscalar(value) and np.isfinite(float(value))   

################  Training NNnPH Model with "FedAvg" framework in Local Client  ################################# 

def train_local_client( global_model, client, discretiser, args):
    device = args.device

    # Each selected client starts from the current global model.
    local_model = deepcopy(global_model).to(device) 
    train_loader, val_loader, discretiser = prepare_client_dataloaders(client, discretiser, args)

    optimizer = build_optimizer(local_model, args) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience) 
    best_model_state = deepcopy(local_model.state_dict())

    best_val_cindex = -np.inf
    early_stop_counter = 0
    
    valid_val_metric_seen = False 
    has_val_events = np.any(client.val.event == 1) 
    X_val = torch.as_tensor( client.val.X, dtype=torch.float32, device=device)
    loss_fn =  negative_llh   
    
    for epoch in range(args.epochs): 
        train_loss = train(train_loader, local_model, loss_fn, optimizer, device)  
        val_loss = validate(val_loader, local_model, loss_fn, device) 

        if  val_loss is not None and np.isfinite(val_loss):
            scheduler.step(val_loss)

        val_cindex = np.nan 
        if has_val_events: 
            try:
                val_cindex = Concordance( local_model, X_val, client.val.time, client.val.event, discretiser, device) 
            except Exception as exc:
                logger.exception(f"[Client {client.client_id}] validation C-index failed: {exc}")

        logger.info( f"Client={client.client_id} | epoch={epoch + 1} | train_loss={train_loss} | val_loss={val_loss} |  val_cindex={val_cindex}")

        # check validity of C-index before updating early stopping   
        if  val_cindex is None or not np.isfinite(val_cindex):
            continue

        valid_val_metric_seen = True 
        if val_cindex > best_val_cindex: 
            best_val_cindex = val_cindex
            early_stop_counter = 0 
            best_model_state = deepcopy( local_model.state_dict()) 
        else: 
            early_stop_counter += 1 
            if early_stop_counter >= args.patience:
                break

    # If C-index was never computable, use the final locally trained model.
    if not valid_val_metric_seen:
        best_model_state = deepcopy(local_model.state_dict()) 
        
    model_file_name = Path(args.modeldir) / f"{args.model}_client_{client.client_id}_{args.run_id}.pt"
    torch.save({"model": best_model_state, "discretizer": discretiser}, model_file_name)

    local_model.load_state_dict(best_model_state) 
    return local_model

################  Evaluate global model on clients using their local data ################################# 

def evaluate_global_model_on_clients( global_model, clients, eval_times, discretiser, args):
    device = args.device 
    global_model.to(device)
    global_model.eval()

    val_sum = 0.0
    val_n = 0

    test_acc = {
        "cindex": [0.0, 0],
        "brier": [0.0, 0],
        
        "nbll": [0.0, 0],
        "mae_po": [0.0, 0]
    }

    p_values = [] 
    for client in clients: 
        
        if np.any(client.val.event == 1): 
            X_val = torch.as_tensor(client.val.X, dtype=torch.float32, device=device)

            try:
                val_cindex = Concordance( global_model, X_val, client.val.time, client.val.event, discretiser, device ) 
            except Exception as exc:
                val_cindex = np.nan 
                logger.exception(f"[Client {client.client_id}] validation evaluation failed: {exc}")

        try: 
            test_cindex, test_brier, test_nbll, test_p_value, test_mae_po = Evaluation(global_model,
                client.train.time, client.train.event,
                client.test.X, client.test.time, client.test.event,
                eval_times, discretiser, device) 
        except Exception as exc: 
            logger.exception(f"[Client {client.client_id}] test evaluation failed: {exc}")
            test_cindex, test_brier, test_nbll = np.nan, np.nan, np.nan
            test_p_value, test_mae_po = np.nan, np.nan

        if is_finite_metric(val_cindex): 
            n_val = len(client.val.time) 
            val_sum +=  float(val_cindex) * n_val 
            val_n += n_val

        n_test = len(client.test.time) 
        metrics = {
            "cindex": test_cindex,
            "brier": test_brier,
            "nbll": test_nbll,
            "mae_po": test_mae_po
        }

        for key, value in metrics.items(): 
            if is_finite_metric(value): 
                test_acc[key][0] +=  float(value) * n_test  
                test_acc[key][1] += n_test

        if is_finite_metric(test_p_value):
            p_values.append(float(test_p_value))

    global_val_cindex = val_sum / val_n if val_n > 0 else np.nan 
    
    def weighted_mean(key): 
        weighted_sum, n = test_acc[key] 
        return weighted_sum / n if n > 0 else np.nan  

    test_cindex = weighted_mean("cindex") 
    test_brier = weighted_mean("brier") 
    test_nbll = weighted_mean("nbll") 
    test_mae_po = weighted_mean("mae_po")

    if p_values: 
        test_p_value = combine_pvalues( p_values, method="fisher" )[1] 
    else:
        test_p_value = np.nan 
    return global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_mae_po
     