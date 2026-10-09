import os
from copy import deepcopy

import numpy as np
import torch

from .model import train, validate, init_net
from .client.utils import prepare_nnnph_dataloaders, build_optimizer, Concordance, Evaluation

################  Training NNnPH Model with "Centralized" framework in Local Client  #################################

def train_centralized(model, data, eval_times, args): 
    device = args.device   
    train_loader, val_loader, discretiser =   prepare_nnnph_dataloaders(data, args) 
     
    model = init_net(data.feature_size, len(discretiser.cuts), args) 
    model.to(device)
    
    # dim_out=len(discretiser.cuts)``

    ######################################  Optimizer & Scheduler ################################################
    optimizer = build_optimizer(model, args) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience)

    ################################################ Early Stopping ########################################### 
    best_val_cindex = -np.inf 
    best_model_state = deepcopy(model.state_dict()) 
    early_stop_counter = 0 

    ################################################ Validation Data ########################################## 
    X_val_tensor = torch.as_tensor(data.val.X, dtype=torch.float32, device=device)


    ################################################ Training ################################################# 
    for epoch in range(args.epochs): 
        train_loss = train( train_loader, model, optimizer, device) 
        val_loss = validate(val_loader, model, device)

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