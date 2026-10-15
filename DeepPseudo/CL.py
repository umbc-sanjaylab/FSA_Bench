from copy import deepcopy 
import numpy as np, os, torch  
from torch.utils.data import DataLoader
from .utils import DataBuilder, pseudoFunction

from .model import train,  evaluate 
from .client.utils import Concordance, Evaluation 
from .client.utils import  build_optimizer, pseudo_loss

import logging, pandas as pd  
logger = logging.getLogger("experiment") 

################################ Centralized DeepPseudo Training ################################ 
      
def train_centralized(net, data,  evaltime, args):  
    """
    Train and evaluate DeepPseudo in the centralized setting.

    Parameters
    ----------
    net : torch.nn.Module
        DeepPseudo neural network.
    data : CentralizedSurvivalData
        Centralized training, validation, and test splits.
    evaltime : array-like
        Evaluation time grid used for survival metrics.
    args : argparse.Namespace
        Experiment configuration and training hyperparameters.
     
    Returns
    -------
    dict
        Test metrics and the best validation C-index.
    """     

    # initializes network
    logger.info("Training network")
    net = net.to(args.device)
    
    # Create DataFrame with time and status for pseudo values calculation
    central_train_data = pd.DataFrame(pd.concat([pd.DataFrame(data.train.time), pd.DataFrame(data.train.status)], axis=1))
    central_train_data.columns = ['time', 'status']
    
    central_val_data = pd.DataFrame(  pd.concat([pd.DataFrame(data.val.time), pd.DataFrame(data.val.status)], axis=1))
    central_val_data.columns = ['time', 'status']
    
    # Calculate Pseudo values at prespecified evaltime
    central_data = pd.concat( [central_train_data, central_val_data], axis=0)
    pseudo_central = pseudoFunction(central_data, evaltime)
      
    train_pseudo=pseudo_central[:len(central_train_data),:]
    val_pseudo=pseudo_central[len(central_train_data):,:]
    
    # Optimizer, loss function, and learning rate scheduler
    optimizer = build_optimizer(net, args)   
    criterion = pseudo_loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=args.patience)
    
    patience = args.patience
    best_val_cindex = -float("inf") 
    best_state = deepcopy(net.state_dict())

    train_losses = []
    val_losses = [] 
    es = 0
    
    # train data loader from covariates, time and event status information
    train_x = DataBuilder(data.train.X,  train_pseudo)
    train_dataloader = DataLoader(dataset=train_x, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # validation data loader from covariates, time and event status information
    val_x = DataBuilder(data.val.X, val_pseudo)
    val_dataloader = DataLoader(dataset=val_x, batch_size=args.batch_size, shuffle=False, num_workers=1) 
    
    for epoch in range(args.epochs):
        train_loss = train(train_dataloader, net, criterion, optimizer, args.device)
        train_losses.append(train_loss)

        val_loss = evaluate(val_dataloader, net, criterion, args.device)
        val_losses.append(val_loss)
        scheduler.step(val_loss) 

        logger.info(f"epoch={epoch}, train_loss={train_loss}, val_loss={val_loss}") 
        cindex_metric = Concordance(net, data.val.X, data.val.time, data.val.event, evaltime, args.device)  
        logger.info(f"epoch={epoch}, validation C-index={cindex_metric}")

        # invalid validation metric -> count as no improvement
        if cindex_metric is None or not np.isfinite(cindex_metric):
            logger.info(f"[Validation WARNING] Invalid validation C-index at epoch {epoch}: {cindex_metric}")
            es += 1
            if es >= patience:
                logger.info(f"Early stopping at epoch {epoch} due to repeated invalid/non-improving validation.")
                break
            continue

        if cindex_metric > best_val_cindex:
            best_val_cindex = cindex_metric 
            es = 0
            best_state = deepcopy(net.state_dict())
        else:
            es += 1
            if es >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info("Done!")

    # store the best model state
    os.makedirs(args.modeldir, exist_ok=True)
    model_path = os.path.join(args.modeldir, f"{args.model}_CL_{args.run_id}.pt")
    torch.save(best_state, model_path)  

    # load the best model state for evaluation
    net.load_state_dict(best_state)
    net = net.to(args.device)
    net.eval()

    # evaluate the model and log the results
    test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation(net, data.train.time, data.train.event, data.test.X, data.test.time, data.test.event, evaltime, args.device) 
    logger.info(f'>> Test Cindex: {test_cindex}, Test Brier: {test_brier} Test NBLL: {test_nbll}')
    logger.info(f'>> Test D-Calibration: {test_p_value}, Test MAE_PO: {test_weighted_MAE_PO}')   
    
    # return the evaluation results
    result = {
        "cindex": test_cindex,
        "brier": test_brier,
        "nbll": test_nbll,
        "d_calibration": test_p_value,
        "mae_po": test_weighted_MAE_PO,
        "best_val_cindex": best_val_cindex
    }
    return result
 