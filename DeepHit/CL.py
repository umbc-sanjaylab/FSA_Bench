import numpy as np, os, torch    
import torchtuples as tt    
from model import init_net
from pycox.models import DeepHit 
from .client.utils import build_optimizer, LabTransform, Evaluation

import logging 
logger = logging.getLogger("experiment")  
    
def  train_centralized(data,  evaltime, alpha, sigma, labtrans, args): 
    logger.info('Training network') 
    ckpt_path = os.path.join(args.modeldir,   f"DeepHit_{args.run_id}_global.pt")  
    
    # DeepHitSingle.label_transform standardize the durations    
    y_train = labtrans.fit_transform(data.train.time, data.train.event)
    y_val = labtrans.transform(data.val.time, data.val.event)

    # Create the validation set
    val_data = (data.val.X, y_val)
    test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan 
    
    
    # Initialize model and Torchtuples optimizer
    model = init_net(data.train.X.shape[1], args)
    optimizer = build_optimizer(args=args)    
    
    verbose = False
    callbacks = [tt.callbacks.EarlyStopping( patience=args.patience, file_path=ckpt_path, load_best=True, rm_file=True )]   
    optimizer = build_optimizer(args=args) 

    # initialize DeepHit model and train the model on train and validation data
    net = DeepHit(model.to(args.device), optimizer, alpha=alpha, sigma=sigma, duration_index=labtrans.cuts)
    log = net.fit(data.train.X, y_train, args.batch_size, args.epochs, callbacks, verbose, val_data=val_data, val_batch_size=args.batch_size)  
    
    # Save the model checkpoint
    os.makedirs(args.modeldir, exist_ok=True) 
    torch.save({"model_state_dict": net.net.state_dict(),
                  "duration_index": labtrans.cuts,
                  "num_durations": args.num_duration,
                  "alpha": alpha, "sigma": sigma,
                  "input_size": data.train.X.shape[1],
                  "dropout_p": args.dropout_p,
                  }, ckpt_path) 
  
    # evaluate the model on test dataset and return the evaluation metrics
    test_cindex, test_brier, test_nbll, test_p_value, test_mae_po  = Evaluation( net, data.train.time, data.train.event, data.train.X, data.test.X, data.test.time, data.test.event, evaltime, args.device) 
    return {
        "best_round": 0,
        "val_cindex": np.nan,
        "cindex": test_cindex,
        "brier": test_brier,
        "nbll": test_nbll,
        "d_calibration": test_p_value,
        "mae_po": test_mae_po
    } 

