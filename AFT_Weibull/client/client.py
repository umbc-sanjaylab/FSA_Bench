import numpy as np, torch 
from copy import deepcopy
from .utils import Concordance, train, evaluate, build_optimizer, Evaluation
from ..utils import is_finite_metric 
from scipy.stats import combine_pvalues
from ..utils import DataBuilder
from torch.utils.data import DataLoader

import logging 
logger = logging.getLogger("experiment")

################  Training Weibull AFT Model with "FedAvg" framework in Local Client  #################################

def train_local_client( global_model, client, args):  
    # Each selected client starts from the current global model.
    local_model = deepcopy(global_model).to(args.device)  
    
    feature_list = client.train.X.columns.tolist()
    train_x = DataBuilder(client.train.X, client.train.time, client.train.event, feature_list)
    train_dataloader = DataLoader(dataset=train_x, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # validation data loader from covariates, time and event status information
    feature_list = client.val.X.columns.tolist()
    val_x = DataBuilder(client.val.X, client.val.time, client.val.event, feature_list)
    val_dataloader = DataLoader(dataset=val_x, batch_size=args.batch_size, shuffle=False, num_workers=1) 
    
    valid_val_metric_seen = False 
    has_val_events =  len(client.val.event.values) > 0 and np.any(client.val.event.values == 1)

    logger.info(f"Training client {client.client_id}")  
    optimizer = build_optimizer(local_model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience)

    best_val_cindex = float("-inf")
    best_model_state = deepcopy(local_model.state_dict())
    es = 0

    for epoch in range(args.epochs):
        train_loss = train(train_dataloader, local_model, optimizer, args.device)
        val_loss = evaluate(val_dataloader, local_model, args.device)
        
        if val_loss is not None and np.isfinite(float(val_loss)):
            scheduler.step(float(val_loss))
        else:
            logger.warning(f"Client {client.client_id}: invalid validation loss at epoch {epoch}: {val_loss}")
        
         
        logger.info(f"client: {client.client_id} epoch: {epoch}, train loss: {train_loss} and val loss: {val_loss}")
        
        metrics = None 
        if has_val_events:
            try:
                metrics = Concordance(local_model, client.val.X.values, client.val.time.values, client.val.event.values, args.evaltime, args.device)
            except Exception as exc:
                logger.exception( f"Client {client.client_id}: validation C-index unavailable at epoch {epoch}: {exc}") 

        # Calculate the validation Cindex
        logger.info(f"client: {client.client_id}, epoch: {epoch}, and  validation metric: {metrics}")  
        
        
        if not is_finite_metric(metrics):
              continue
        
        # Early Stopping Criteria 
        valid_val_metric_seen = True
        if metrics > best_val_cindex:
            best_val_cindex = metrics
            es = 0
            best_model_state = deepcopy(local_model.state_dict()) 
        else:
            es += 1
            if es >= args.patience:
                break
    
    # Retain the final locally trained state.
    if not valid_val_metric_seen:
        best_model_state = deepcopy(local_model.state_dict())
    
    torch.save(best_model_state,  f"{args.modeldir}/{args.model}_{args.run_id}_{client.client_id}.pt")
    local_model.load_state_dict(best_model_state)
    local_model.to(args.device)

    return local_model

################################## Evaluate Global Model on Clients ###################################################################  

def evaluate_global_model_on_clients(global_model,  selected,  X_val_local_std,  val_time_local,  val_status_local, X_test_local_std, test_time_local, test_status_local,  train_time_local,  train_status_local,  pseudo_evaltime,  device="cpu"):
    global_model.to(device)
    global_model.eval()
    
    total_samples_cindex, total_samples_brier= 0, 0
    total_samples_nbll, total_samples_mae = 0, 0

    total_val_samples = 0 
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