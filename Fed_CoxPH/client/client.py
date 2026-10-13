from copy import deepcopy 
from pathlib import Path

import numpy as np 
import torch, logging 
import torchtuples as tt
from pycox.models import CoxPH

from .utils import build_optimizer, Evaluation 
from .utils import accumulate, is_finite_metric, weighted_mean
logger = logging.getLogger("experiment")  

################################ Local CoxPH Training ################################
 
def train_local_client(global_model, client, args):
    # Cox partial likelihood cannot be trained without observed events
    if np.sum(client.train.event == 1) == 0:
        raise ValueError(f"Client {client.client_id} has no training events.")

    # Every client starts from the current global model
    local_model = deepcopy(global_model).to(args.device)

    y_train = (client.train.time, client.train.event) 
    y_val = (client.val.time, client.val.event) 
    val_data = (client.val.X, y_val)

    # PyCox requires a Torchtuples optimizer
    optimizer = build_optimizer(args=args, backend="torchtuples") 
    net = CoxPH(local_model, optimizer)

    # Client-specific temporary checkpoint
    ckpt_path = Path(args.modeldir) / ( f"{args.model}_{args.dataset}_fold_{args.run_id}_client_{client.client_id}") 
    callbacks = [ 
        tt.callbacks.EarlyStopping(
            patience=args.patience,
            file_path=str(ckpt_path),
            load_best=True,
            rm_file=True
        )
    ]
    
    # validation events checking 
    if np.sum(client.val.event == 1) == 0:
          val_data=None  
          val_batch_size=None  
    else: 
        y_val = (client.val.time, client.val.event)
        val_data = (client.val.X, y_val)
        val_batch_size = args.batch_size 

    # PyCox CoxPH wrapper
    net = CoxPH(local_model, optimizer)
    log = net.fit(client.train.X,  y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=False, val_data=val_data,
        val_batch_size=val_batch_size 
    )
    logger.info("Client %s local training completed: %s", client.client_id, log) 
    
    # Preserve your existing local checkpoint behavior 
    model_path = Path(args.modeldir) / ( f"{args.model}_client_{client.client_id}_run_{args.run_id}.pt") 
    torch.save( net.net.state_dict(), model_path ) 
    return net.net


################################## Evaluate Global Model on Clients ###################################################################   
  
def evaluate_global_model_on_clients( global_model, clients, eval_times, args ):
    # global model in evaluator mode 
    global_model.to(args.device)
    global_model.eval()
    
    # Validation
    weighted_val_cindex = 0.0
    total_val_samples = 0
    
    # Test metrics
    test_acc = {
        "cindex": {"weighted_sum": 0.0, "n": 0},
        "brier": {"weighted_sum": 0.0, "n": 0},
        "nbll": {"weighted_sum": 0.0, "n": 0},
        "mae_po": {"weighted_sum": 0.0, "n": 0}
    }
    
    test_p_values = [] 
    for client in clients:
        y_train = (client.train.time, client.train.event)

        # Evaluation computes client-specific baseline hazards.
        optimizer = build_optimizer(args=args, backend="torchtuples") 
        eval_model = deepcopy(global_model).to(args.device) 
        net = CoxPH(eval_model, optimizer)

        ################ Validation ################ 
        val_cindex = np.nan 
        if len(client.val.event) > 0 and np.any(client.val.event == 1):
            try:
                val_cindex, _, _, _, _ = Evaluation(net, client.train.time, client.train.event, client.train.X, y_train, client.val.X, client.val.time, client.val.event, eval_times, args.device) 
            except Exception:
                logger.exception("Client %s validation evaluation failed.", client.client_id)

        if is_finite_metric(val_cindex):
            n_val = len(client.val.time) 
            weighted_val_cindex += float(val_cindex) * n_val 
            total_val_samples += n_val

        ################ Test ################ 
        try:
            test_cindex, test_brier, test_nbll, test_p_value, test_mae_po = Evaluation(net, client.train.time, client.train.event, client.train.X, y_train, client.test.X, client.test.time, client.test.event, eval_times, args.device) 
        except Exception:
            logger.exception("Client %s test evaluation failed.", client.client_id) 
            test_cindex, test_brier, test_nbll, test_p_value, test_mae_po = ( np.nan, np.nan, np.nan, np.nan, np.nan )

        n_test = len(client.test.time) 
        accumulate(test_acc, "cindex", test_cindex, n_test) 
        accumulate(test_acc, "brier", test_brier, n_test)
        accumulate(test_acc, "nbll", test_nbll, n_test)
        accumulate(test_acc, "mae_po", test_mae_po, n_test)

        if is_finite_metric(test_p_value):
            test_p_values.append(float(test_p_value))

    ################ Final Aggregation ################ 
    global_val_cindex =  weighted_val_cindex / total_val_samples if total_val_samples > 0 else np.nan        
    test_cindex = weighted_mean(test_acc, "cindex" ) 
    test_brier = weighted_mean(test_acc, "brier" ) 
    test_nbll = weighted_mean(test_acc, "nbll" ) 
    test_mae_po = weighted_mean(test_acc, "mae_po" ) 
    
    # Preserve your existing CoxPH behavior for now
    test_p_value = np.nanmedian(test_p_values) if test_p_values else np.nan   
    return global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_mae_po   
    
############################################################################################################################################ 