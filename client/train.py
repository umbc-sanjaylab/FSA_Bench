import os
import numpy as np
from torch import optim

from federated.server.server_controller import deepcopy
import torch

import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from federated.client.eval import compute_concordance
      
################  Training Weibull AFT Model with "FedAvg" framework in Local Client  #################################

def train_client( client_id, model, train_dataloader, val_dataloader, X_val, val_time, val_event, eval_times, model_spec, args, device="cpu"):
    """
    Train one federated client model.

    Model-specific operations such as training loss,
    validation loss, and survival prediction are obtained
    from MODEL_REGISTRY through model_spec.

    Client model selection is based on validation C-index.
    """

    print(f"\nTraining client {client_id}") 
    
    # 1. Get model-specific functions 
    train_epoch_fn = model_spec["train_epoch"]
    validation_loss_fn = model_spec[ "validation_loss"]
    predict_survival_fn = model_spec["predict_survival"]
 
    # 2. Prepare validation information
    val_time_array = np.asarray(val_time, dtype=float).reshape(-1) 
    val_event_array = np.asarray(val_event, dtype=int).reshape(-1)
    has_val_events =  len(val_event_array) > 0 and np.any(val_event_array == 1) 
    
    # 3. Create optimizer 
    trainable_parameters = filter( lambda parameter: parameter.requires_grad, model.parameters()) 
    if args.optimizer == "adam": 
        optimizer = optim.Adam( trainable_parameters, lr=args.lr, weight_decay=args.reg) 
    elif args.optimizer == "adamw": 
        optimizer = optim.AdamW( trainable_parameters, lr=args.lr, weight_decay=args.reg) 
    else: 
        raise ValueError( f"Unsupported optimizer: {args.optimizer}" )
      
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode="min", patience=args.patience)  
    
    # 4. Initialize client-side early stopping 
    best_val_cindex = float("-inf")
    best_model_state = deepcopy(model.state_dict())

    early_stop_counter = 0 
    valid_val_metric_seen = False 
    
    # 5. Local training 
    for epoch in range(args.epochs): 
        train_loss = train_epoch_fn( model, train_dataloader, optimizer, device) # training loss          
        val_loss = validation_loss_fn( model, val_dataloader, device)  # validation loss
 
        # Learning-rate scheduler
        if val_loss is not None and np.isfinite(float(val_loss)):  
            scheduler.step(float(val_loss)) 
        else: 
            print(f"Client {client_id}, epoch {epoch}: invalid validation loss = {val_loss}")
 
        # Generate validation survival probabilities
        val_cindex = np.nan 
        if has_val_events: 
            try:
                val_survival = predict_survival_fn( model=model, X=X_val, eval_times=eval_times, device=device) 
                val_cindex = compute_concordance( survival=val_survival, eval_times=eval_times, durations=val_time_array, events=val_event_array) 
            except Exception as exc: 
                print(f"Client {client_id}, epoch {epoch}:  validation C-index unavailable: {exc}") 
                val_cindex = np.nan

        
        # Training progress 
        print(f"Client {client_id} |  Epoch {epoch + 1}/{args.epochs} |  Train loss: {train_loss} | Val loss: {val_loss} | Val C-index: {val_cindex}")

        # Continue as we could not calculate the Validation C-index. 
        if not has_val_events: 
            continue
 
         # continue if validation metrics are undefined
        if val_cindex is None or not np.isfinite(val_cindex): 
            continue
        
        # 6. Validation-based local early stopping
        valid_val_metric_seen = True 
        if val_cindex > best_val_cindex: 
            best_val_cindex = val_cindex 
            early_stop_counter = 0 
            best_model_state = deepcopy(model.state_dict()) 
        else: 
            early_stop_counter += 1 
            if early_stop_counter >= args.patience: 
                print( f"Client {client_id}: early stopping at epoch {epoch + 1}.") 
                break 
    
    # 7. If C-index was never available, retain final local state 
    if not valid_val_metric_seen: 
        best_model_state = deepcopy(model.state_dict())

    # 8. Save client checkpoint
    os.makedirs(args.modeldir, exist_ok=True) 
    model_path = os.path.join(args.modeldir, f"{args.model}_fold_{args.run_id}_client_{client_id}.pt")  
    torch.save( best_model_state, model_path) 
     
    # 9. Restore selected local model 
    model.load_state_dict(best_model_state) 
    model.to(device) 
    print(f"Client {client_id}: model saved at {model_path}")

    return model