import os
from copy import deepcopy

import numpy as np
import torch

from .model import train, validate, init_net
from .client.utils import prepare_nnnph_dataloaders, build_optimizer, Concordance, Evaluation 

################  Training NNnPH Model with "Centralized" framework in Local Client  #################################

def train_centralized(data, eval_times, args, logger):
    """Train and evaluate the selected survival model in the centralized setting."""
     
    #  load the train set, val set, model and discretizer    
    train_loader, val_loader, discretiser =   prepare_nnnph_dataloaders(data, args)  
    model = init_net(data.train.X.shape[1], len(discretiser.cuts), args) 
    model.to(args.device)
      
    # Optimizer & Scheduler  
    optimizer = build_optimizer(model, args) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience)

    # Early Stopping   
    best_val_cindex = -np.inf 
    best_model_state = deepcopy(model.state_dict()) 
    early_stop_counter = 0  

    # Training   
    for epoch in range(args.epochs): 
        train_loss = train( train_loader, model, optimizer, args.device) 
        val_loss = validate(val_loader, model, args.device)

        if np.isfinite(val_loss):
            scheduler.step(val_loss)
        
        val_cindex = Concordance( model, data.val.X, data.val.time, data.val.event, discretiser, args.device) 
        logger.info( f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | " f"val_loss={val_loss:.4f} |  val_cindex={val_cindex}") 

        # Early stopping   
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


   # Restore Best Model  
    model.load_state_dict(best_model_state) 
    model.eval()  

    #  Save Model  
    os.makedirs( args.modeldir, exist_ok=True ) 
    model_path = os.path.join( args.modeldir, (f"{args.model}_CL_{args.run_id}.pt")) 
    torch.save( { "model": model.state_dict(), "discretizer": discretiser }, model_path) 
    logger.info( f"Model saved at {model_path}") 
    
   # Test Evaluation  
    try:
        test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation( model,
            data.train.time, data.train.event,
            data.test.X, data.test.time, data.test.event,
            eval_times, discretiser, args.device)

    except Exception as e: 
        logger.error( f"[Centralized evaluation error] " f"{type(e).__name__}: {e}" ) 
        test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan  


    logger.info( f">> Test C-index: {test_cindex}") 
    logger.info( f">> Test Brier: {test_brier}") 
    logger.info( f">> Test NBLL: {test_nbll}") 
    logger.info( f">> Test D-Calibration: {test_p_value}") 
    logger.info( f">> Test Weighted MAE PO: {test_weighted_MAE_PO}") 

    return {
        "cindex": test_cindex,
        "brier": test_brier,
        "nbll": test_nbll,
        "d_calibration": test_p_value,
        "mae_po": test_weighted_MAE_PO,
        "best_val_cindex": best_val_cindex
    }  