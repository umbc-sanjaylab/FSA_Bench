import numpy as np, os, torch    
import torchtuples as tt  
from pycox.models import CoxPH
from .client.utils import build_optimizer, Evaluation
from model import init_net

import logging 
logger = logging.getLogger("experiment")


####################################  Training Centralized CoxPH Model  ##################################################################### 

def  train_centralized(data,  evaltime, args): 
    logger.info('Training network')
    
    # Prepare CoxPH targets for training time and events
    y_train = (data.train.time, data.train.event) 
    
    # Initialize model and Torchtuples optimizer
    model = init_net(data.train.X.shape[1], args)
    optimizer = build_optimizer(args=args)
    
    # callback configuration for torch tuples. 
    model_path = os.path.join( args.modeldir, f"Fed_CoxPH_CL_{args.run_id}.pt") 
    callbacks = [tt.callbacks.EarlyStopping( patience=args.patience, file_path=model_path, load_best=True, rm_file=True )]  
    
    # validation events checking 
    if np.sum(data.val.event == 1) == 0:
         val_data=None  
         val_batch_size=None  
    else: 
        y_val = (data.val.time, data.val.event)
        val_data = (data.val.X, y_val)
        val_batch_size = args.batch_size 

    # PyCox CoxPH wrapper
    net = CoxPH(model, optimizer)
    log = net.fit(data.train.X,  y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=False, val_data=val_data,
        val_batch_size=val_batch_size 
    )

    logger.info("Centralized CoxPH training completed: %s", log)   
    torch.save(net.net.state_dict(), model_path)
    logger.info("Model saved at %s", model_path)

    # Test evaluation
    test_cindex, test_brier, test_nbll, test_p_value, test_mae_po  = Evaluation( net, data.train.time, data.train.event, data.train.X, y_train, data.test.X, data.test.time, data.test.event, evaltime, args.device)

    return {
        "best_round": 0,
        "val_cindex": np.nan,
        "cindex": test_cindex,
        "brier": test_brier,
        "nbll": test_nbll,
        "d_calibration": test_p_value,
        "mae_po": test_mae_po
    } 