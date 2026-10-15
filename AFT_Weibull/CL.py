from copy import deepcopy 
import numpy as np
import torch 
from torch.utils.data import DataLoader
from .utils import DataBuilder

from .model import train,  evaluate 
from .client.utils import  build_optimizer, Concordance, Evaluation 

import logging 
logger = logging.getLogger("experiment")

################  Training AFT Weibull Model with "Centralized" Setting in Local Client  ################################# 
def train_centralized(net, data, feature_list, evaltime, args): 
    print('Training network')

    optimizer = build_optimizer(net, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience)  # Learning rate scheduler
    
    best_val_cindex = 0.0
    best_model_state = deepcopy(net.state_dict()) 
    es = 0 

    # train data loader from covariates, time and event status information
    train_x = DataBuilder(data.train.X,  data.train.time, data.train.event, feature_list)
    train_dataloader = DataLoader(dataset=train_x, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # validation data loader from covariates, time and event status information
    val_x = DataBuilder(data.val.X, data.val.time, data.val.event, feature_list)
    val_dataloader = DataLoader(dataset=val_x, batch_size=args.batch_size, shuffle=False, num_workers=1)

    for epoch in range(args.epochs):
      
        # Calculate the train loss using train function 
        train_loss = train(train_dataloader, net, optimizer, args.device)
        val_loss = evaluate(val_dataloader, net, args.device)
        scheduler.step(val_loss) 

        # Calculate the validation Cindex 
        val_cindex = Concordance(net, data.val.X, data.val.time, data.val.event, evaltime, args.device)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | " f"val_loss={val_loss:.4f} |  val_cindex={val_cindex}") 

        # Early Stopping Criteria
        if np.sum(data.val.event == 1) == 0:
              continue     
        elif val_cindex is None or not np.isfinite(val_cindex):
            es += 1
            if es >= args.patience:
                break
            continue 
        elif val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            es = 0
            best_model_state = deepcopy(net.state_dict()) 
        else:
            es += 1
            if es >= args.patience:
                break

    logger.info("Done!")

    # Load the trained model and evaluate the performances
    torch.save(best_model_state, f"{args.modeldir}/{args.model}_CL_{args.run_id}.pt")
    logger.info(f"model is saved at {args.modeldir}/{args.model}_CL_{args.run_id}.pt")  

    # Load the best model state before evaluation
    net.load_state_dict(best_model_state) 
    test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation(net, data.train.time, data.train.event, data.test.X, data.test.time, data.test.event, evaltime, args.device)

    # Log the test results
    logger.info(f'>> Test Cindex: {test_cindex}, Test Brier: {test_brier} Test NBLL: {test_nbll}')
    logger.info(f'>> Test D-Calibration: {test_p_value}, Test MAE_PO: {test_weighted_MAE_PO}')   
    
    result = {
        "cindex": test_cindex,
        "brier": test_brier,
        "nbll": test_nbll,
        "d_calibration": test_p_value,
        "mae_po": test_weighted_MAE_PO,
        "best_val_cindex": best_val_cindex
    }
    
    return result    