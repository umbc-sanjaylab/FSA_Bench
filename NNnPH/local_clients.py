import numpy as np
import torch
import torch.optim as optim
import logging
from math import *
from model import *
from utils import *
from copy import deepcopy
from scipy.stats import combine_pvalues  
from model import train_epoch, validation_loss

################  Training NNnPH Model with "Centralized" framework in Local Client  #################################

def train_centralized(model, data, eval_times, args): 
    device = args.device 
    model = model.to(device) 
    
    # NNnPH-specific target preparation
    train_loader, val_loader, discretiser =   prepare_nnnph_dataloaders(data, args) 

    ################################################ Optimizer ################################################

    if args.optimizer == "adam": 
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg) 
    elif args.optimizer == "amsgrad": 
        optimizer = torch.optim.Adam( filter( lambda p: p.requires_grad, model.parameters() ), lr=args.lr, weight_decay=args.reg, amsgrad=True) 
    elif args.optimizer == "sgd": 
        optimizer = torch.optim.SGD( filter( lambda p: p.requires_grad, model.parameters() ), lr=args.lr, momentum=args.rho, weight_decay=args.reg ) 
    else:
        raise ValueError( f"Unsupported optimizer: {args.optimizer}") 

    ################################################ Scheduler ################################################ 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience)

    ################################################ Early Stopping ###########################################

    best_val_cindex = -np.inf 
    best_model_state = deepcopy( model.state_dict()) 
    early_stop_counter = 0


    ################################################ Validation Data ########################################## 
    X_val_tensor = torch.as_tensor(data.val.X, dtype=torch.float32, device=device)


    ################################################ Training ################################################# 
    for epoch in range(args.epochs): 
        train_loss = train_epoch( train_loader, model, optimizer, device) 
        val_loss = validation_loss(val_loader, model, device)

        if np.isfinite(val_loss):
            scheduler.step(val_loss)

        val_cindex = Concordance( model, X_val_tensor, data.val.time, data.val.event, discretiser, device) 
        print( f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | " f"val_loss={val_loss:.4f} |  val_cindex={val_cindex}") 

        ################################################ Early stopping ######################################

        if val_cindex is None or not np.isfinite(val_cindex):
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                break
            continue 

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_model_state = deepcopy( model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
            if early_stop_counter >= args.patience:
                break


    ################################################ Restore Best Model #######################################

    model.load_state_dict(best_model_state) 
    model.eval() 

    ################################################ Save Model ################################################

    os.makedirs( args.modeldir, exist_ok=True ) 
    model_path = os.path.join( args.modeldir, (f"Centralized_" f"{args.model}_" f"{args.dataset}_" f"{args.run_id}_0.pt" ) )

    torch.save( { "model": model.state_dict(), "discretizer": discretiser }, model_path) 
    print( f"Model saved at {model_path}" )


    ################################################ Test Evaluation #########################################

    try:
        test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation( model,
            data.train.time, data.train.event,
            data.test.X, data.test.time, data.test.event,
            eval_times, discretiser, device)

    except Exception as e: 
        print( f"[Centralized evaluation error] " f"{type(e).__name__}: {e}" ) 
        test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan  


    print( f">> Test C-index: {test_cindex}") 
    print( f">> Test Brier: {test_brier}") 
    print( f">> Test NBLL: {test_nbll}") 
    print( f">> Test D-Calibration: {test_p_value}") 
    print( f">> Test Weighted MAE PO: {test_weighted_MAE_PO}") 

    return {
        "cindex": test_cindex,
        "brier": test_brier,
        "nbll": test_nbll,
        "d_calibration": test_p_value,
        "mae_po": test_weighted_MAE_PO,
        "best_val_cindex": best_val_cindex
    } 

################  Training NNnPH Model with "FedAvg" framework in Local Client  #################################


def train_net(net_id, net, train_dataloader, val_dataloader, epochs, lr, args_optimizer, train_time, train_status, X_val, val_time, val_status, X_test, test_time, test_status, evaltime, discretiser, args, comm_round, device="cpu"):
    print('Training network %s' % str(net_id))
    print(f"train_net method: {device}")

    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32)
    val_time_array = np.asarray(val_time)
    val_status_array = np.asarray(val_status)
    
    has_val_events = np.any(val_status_array == 1)   
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)  # Default optimizer
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)

    criterion = negative_llh  # Negative log-likelihood loss function
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience)  # Learning rate scheduler
    best_model_state = deepcopy(net.state_dict())

    Epochs = epochs
    patience = args.patience  # Number of patience for early stopping
    best_val_cindex = float("-inf")
    
    valid_val_metric_seen = False 
    train_losses = []
    val_losses = []
    cindex = []
    es = 0

    for epoch in range(Epochs):
        # Calculate the train loss using train function
        train_loss = train(train_dataloader, net, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Calculate the validation loss using evaluate function
        val_loss = evaluate(val_dataloader, net, criterion, device) 
        val_losses.append(val_loss)
        
        if val_loss is not None and np.isfinite(float(val_loss)):
            scheduler.step(float(val_loss))
        else:
            logger.warning(f"Client {net_id}: invalid validation loss at epoch {epoch}: {val_loss}")
        
        metrics = None 
        if has_val_events:
            try:
                metrics = Concordance( net, X_val_tensor, val_time_array, val_status_array, discretiser, device)
            except Exception as exc:
                logger.warning( f"Client {net_id}: validation C-index could not be calculated at epoch {epoch}: {exc}") 
        
        if metrics is None or not np.isfinite(metrics):
              logger.warning(f"Client: {net_id} invalid validation C-index at epoch {epoch}: {metrics}. Skipping early-stopping update for this epoch.") 
              continue  

        valid_val_metric_seen = True   
        if metrics > best_val_cindex:
            best_val_cindex = metrics
            es = 0
            best_model_state = deepcopy(net.state_dict())
        else:
            es += 1
            if es >= patience:
                break
    if not valid_val_metric_seen:
        best_model_state = deepcopy(net.state_dict()) 

    # torch.save(best_model_state, f'{args.modeldir}/{args.model}_{args.run_id}_{net_id}.pt')
    torch.save({
                "model": best_model_state, 
                "discretizer": discretiser   
              }, f'{args.modeldir}/{args.model}_{args.run_id}_{net_id}.pt')
    
    net.load_state_dict(best_model_state)  
    return net 
  
########################################  Training NNnPH Models with "FedAvg" framework with all clients  ################################# 

def local_train_net(nets, client_name_array, selected, args, X_train_local_std, train_time_local, train_status_local, X_val_local_std, val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local, pseudo_evaltime, discretiser, tr_data_len, comm_round, device="cpu"):
 
    print(f"local_train_net method: {device}")  
    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        print(f"Training network: {str(net_id)}")

        X_train = X_train_local_std[net_id]
        train_time = train_time_local[net_id]
        train_status = train_status_local[net_id]

        X_val = X_val_local_std[net_id]
        val_time = val_time_local[net_id]
        val_status = val_status_local[net_id]

        X_test = X_test_local_std[net_id]
        test_time = test_time_local[net_id]
        test_status = test_status_local[net_id]

        train_time = np.array(train_time, dtype='float32')
        train_status = np.array(train_status, dtype='int64')
        val_time = np.array(val_time, dtype='float32')
        val_status = np.array(val_status, dtype='int64')

        y_train = (train_time, train_status)
        y_val = (val_time, val_status)

        y_train_trans = data_transform( y_train, discretiser, fit_transform=False)
        train_time_trans = y_train_trans[0]
        train_status_trans = y_train_trans[1]

        y_val_trans = data_transform(y_val, discretiser, fit_transform=False)
        val_time_trans = y_val_trans[0]
        val_status_trans = y_val_trans[1]

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor( train_time_trans), torch.tensor(train_status_trans))
        validate_dataset = TensorDataset(torch.tensor(X_val), torch.tensor( val_time_trans), torch.tensor(val_status_trans))
        
        train_dl = DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True) 
        val_dl = DataLoader( validate_dataset, batch_size=args.batch_size, shuffle=True)
        
        nets[net_id] = train_net( client_name_array[net_id], net, train_dl, val_dl, args.epochs, args.lr, args.optimizer, train_time, train_status, X_val, val_time, val_status, X_test, test_time, test_status, pseudo_evaltime, discretiser, args, comm_round, device=device)
    
    return nets 
  
   
############################################################################################################################################

def is_finite_metric(value):
    return (
        value is not None
        and np.isscalar(value)
        and np.isfinite(float(value))
    )

############################################################################################################################################


def evaluate_global_model_on_clients(global_model, selected,  X_val_local_std,  val_time_local,  val_status_local, X_test_local_std,  test_time_local,  test_status_local,  train_time_local,
    train_status_local,  pseudo_evaltime,  discretiser,  device="cpu"):
    
    global_model.to(device)
    global_model.eval()

    avg_val_cindex = 0.0 
    total_val_samples = 0

    avg_test_cindex = 0.0
    total_samples_cindex = 0 

    avg_test_brier = 0.0
    total_samples_brier = 0 

    avg_test_nbll = 0.0 
    total_samples_nbll = 0

    avg_test_MAE_PO = 0.0
    total_samples_mae = 0 
    client_p_values = [] 

    with torch.no_grad():
        for net_id in selected:
            X_val = X_val_local_std[net_id]
            val_time = np.asarray(val_time_local[net_id])
            val_status = np.asarray(val_status_local[net_id])

            X_test = X_test_local_std[net_id]
            test_time = np.asarray(test_time_local[net_id])
            test_status = np.asarray(test_status_local[net_id])

            train_time = np.asarray(train_time_local[net_id])
            train_status = np.asarray(train_status_local[net_id])
 
            val_cindex = None         
            X_val_tensor = torch.as_tensor( X_val,  dtype=torch.float32,  device=device) 
            
            if len(val_status) > 0 and np.any(val_status == 1):
                try:
                    val_cindex = Concordance( global_model, X_val_tensor, val_time, val_status, discretiser, device)
                except Exception as exc:
                    logger.warning( f"Client {net_id}: validation C-index could not be calculated: {exc}") 
            
            try:
                test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation( global_model, train_time, train_status, X_test, test_time, test_status, pseudo_evaltime, discretiser,  device)
            except Exception as exc:
                  logger.warning( f"Client {net_id}: test evaluation failed: {exc}")
      
                  test_cindex = np.nan
                  test_brier = np.nan
                  test_nbll = np.nan
                  test_p_value = np.nan
                  test_weighted_MAE_PO = np.nan
            
            if val_cindex is not None and np.isfinite(val_cindex):
                  avg_val_cindex += float(val_cindex) * len(val_time)
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

    global_val_cindex = avg_val_cindex / total_val_samples if total_val_samples > 0 else np.nan 
    avg_test_cindex = avg_test_cindex / total_samples_cindex if total_samples_cindex > 0 else np.nan
    avg_test_brier = avg_test_brier / total_samples_brier if total_samples_brier > 0 else np.nan
    avg_test_nbll = avg_test_nbll / total_samples_nbll if total_samples_nbll > 0 else np.nan
    avg_test_MAE_PO = avg_test_MAE_PO / total_samples_mae if total_samples_mae > 0 else np.nan  

    if len(client_p_values) > 0:
        avg_test_p_value = combine_pvalues(client_p_values, method="fisher")[1]
    else:
        avg_test_p_value = np.nan

    return  global_val_cindex, avg_test_cindex, avg_test_brier, avg_test_nbll, avg_test_p_value, avg_test_MAE_PO
     