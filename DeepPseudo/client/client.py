from copy import deepcopy
from .utils import Concordance
from .utils import  pseudo_loss
from ..model import train, evaluate
from ..utils import is_finite_metric 
from scipy.stats import combine_pvalues 
import pandas as pd, numpy as np, torch  
from torch.utils.data import DataLoader
from ..utils import DataBuilder, pseudoFunction
from .utils import build_optimizer, Evaluation

import logging 
logger = logging.getLogger("experiment")  
     
################  Training DeepPseudo Model with "FedAvg" framework in Local Client  #################################
def train_local_client( global_model, client, eval_times, args):  
  
    # Create DataFrame with time and status for pseudo values calculation
    central_train_data = pd.DataFrame(pd.concat([pd.DataFrame(client.train.time), pd.DataFrame(client.train.event)], axis=1))
    central_train_data.columns = ['time', 'status']
    
    central_val_data = pd.DataFrame(  pd.concat([pd.DataFrame(client.val.time), pd.DataFrame(client.val.event)], axis=1))
    central_val_data.columns = ['time', 'status']
    
    # Calculate Pseudo values at prespecified evaltime
    central_data = pd.concat( [central_train_data, central_val_data], axis=0)
    pseudo_central = pseudoFunction(central_data, eval_times) 
    logger.info( "Client %s | pseudo shape=%s | NaN=%d | Inf=%d", client.client_id, pseudo_central.shape,  np.isnan(pseudo_central).sum(), np.isinf(pseudo_central).sum())

    if not np.isfinite(pseudo_central).all():
        raise ValueError(f"Client {client.client_id}:  Pseudo-observations contain NaN or Inf.")
      
    train_pseudo=pseudo_central[:len(central_train_data),:]
    val_pseudo=pseudo_central[len(central_train_data):,:]

    modeldir, epochs, optimizer, patience = args.modeldir, args.epochs,  args.optimizer, args.patience
    logger.info(f'Training network') 
    
    # Each selected client starts from the current global model.
    local_model = deepcopy(global_model).to(args.device) 

    optimizer = build_optimizer(local_model, args)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'min', patience=args.patience)  # Learning rate scheduler
    criterion = pseudo_loss  # Pseudo value based loss function

    # concordance index
    best_val_cindex = float("-inf")
    best_val_loss = float("inf") 
    best_model_state = deepcopy(local_model.state_dict())  
    
    # train data loader from covariates, time and event status information
    train_x = DataBuilder(client.train.X,  train_pseudo)
    train_dataloader = DataLoader(dataset=train_x, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # validation data loader from covariates, time and event status information
    val_x = DataBuilder(client.val.X, val_pseudo)
    val_dataloader = DataLoader(dataset=val_x, batch_size=args.batch_size, shuffle=False, num_workers=1) 

    es = 0 
    for _ in range(epochs):
        # Calculate the train loss using train function
        train_loss = train(train_dataloader, local_model, criterion, optimizer, args.device)  
        val_loss = evaluate(val_dataloader, local_model, criterion, args.device) 
        
        logger.info(f'client {client.client_id} train loss: {train_loss}, validation loss: {val_loss}') 
        scheduler.step(val_loss) 
      
        # Calculate the validation Cindex
        metrics = Concordance(local_model, client.val.X, client.val.time, client.val.event, eval_times, args.device)  
        logger.info(f'client {client.client_id} evaluation metric: {metrics}')  
        
        # Early Stopping Criteria
        if np.sum(client.val.event == 1) == 0:
            best_model_state = deepcopy(local_model.state_dict()) 
            continue
        elif metrics is None or not np.isfinite(metrics):
            es += 1
            if es >= patience:
                break 
        elif metrics > best_val_cindex:
            best_val_cindex = metrics
            es = 0
            best_model_state = deepcopy(local_model.state_dict())
        else:
            es += 1
            if es >= patience:
                break
    
    best_model_state_cpu = { k: v.detach().cpu().clone() for k, v in best_model_state.items()}
    
    # store and load the model parameters
    torch.save(best_model_state_cpu, f'{modeldir}/{args.model}_{args.run_id}_{client.client_id}.pt') 
    local_model.load_state_dict(best_model_state_cpu)
    
    local_model.to(args.device)  
    return  local_model   

################################## Evaluate Global Model on Clients ###################################################################  
 
def evaluate_global_model_on_clients(global_model, selected_indices, clients, pseudo_evaltime, args): 
    global_model.to(args.device)
    global_model.eval()
    
    total_samples_cindex, total_samples_brier = 0, 0
    total_samples_nbll, total_samples_mae = 0, 0

    total_val_samples = 0 
    weighted_val_cindex = 0.0
    client_p_values = []  

    avg_test_cindex = 0.0
    avg_test_brier = 0.0
    avg_test_nbll = 0.0
    avg_test_p_value = 0.0
    avg_test_MAE_PO = 0.0  

    for net_id in selected_indices:
        client = clients[net_id]
        X_val = client.val.X 
        val_time = client.val.time
        val_status = client.val.event

        X_test = client.test.X
        test_time = client.test.time
        test_status = client.test.event
         
        val_cindex = np.nan
        if len(val_status) > 0 and np.any(val_status == 1):
            try:
                val_cindex = Concordance( global_model, X_val, val_time, val_status, pseudo_evaltime, args.device) 
            except Exception as exc:
                print(f"Client {net_id}: validation C-index could not be calculated: {exc}")
         
        print("X_test shape:", np.asarray(X_test).shape)
        print("Number of test samples:", len(test_time))
        print("Number of features:", len(client.train.X[0]))
 
        test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation(global_model, client.train.time,  client.train.event, X_test, test_time, test_status, pseudo_evaltime, args.device) 
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
    