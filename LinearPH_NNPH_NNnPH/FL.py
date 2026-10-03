import numpy as np  
from client.client import train_local_client, evaluate_global_model_on_clients 
from client.utils import Discretiser 
from server.server import aggregate_fedavg  
from server.utils import EarlyStopping 

import logging 
logger = logging.getLogger("experiment")

def is_finite_metric(value):
    return value is not None and np.isscalar(value) and np.isfinite(float(value))  

def run_federated(model, clients, eval_times, args):
    num_durations = 40
    device = args.device
    
    global_model = model.to(device)
    discretiser = Discretiser(num_durations, scheme='km') 
    early_stopping = EarlyStopping(patience=args.patience, directory=args.modeldir,\
                                  verbose=True, discretiser=discretiser,\
                                  checkpoint_pth=f"{args.model}_model_run_{args.run_id}.pt")

    # Initialize outputs safely.
    global_val_cindex, test_cindex, test_brier = np.nan, np.nan, np.nan 
    test_nbll, test_p_value, test_mae_po  = np.nan, np.nan, np.nan 

    last_round = -1 
    for comm_round in range(args.comm_round): 
        logger.info("Communication round %d/%d", comm_round + 1, args.comm_round)
        n_clients = len(clients) 
        n_selected = max(1, int(n_clients * args.sample)) 
        
        selected_indices = np.random.permutation(n_clients)[:n_selected] 
        selected_clients = [clients[idx] for idx in selected_indices]

        ################################################ Local Training ############################

        local_models = {} 
        for client in selected_clients: 
            try: 
                local_model = train_local_client(global_model=global_model, client=client, discretiser=discretiser, args=args) 
                local_models[client.client_id] = local_model 
            except Exception as exc: 
                logger.exception("Client %s local training failed.", client.client_id)

        valid_clients = [client for client in selected_clients if client.client_id in local_models] 
        if not valid_clients: 
            logger.warning("Round %d: no valid client updates. Global model is retained.", comm_round)
            continue

        ################################################ FedAvg ####################################

        aggregated = aggregate_fedavg(global_model, local_models, valid_clients) 
        if not aggregated: 
            logger.warning("Round %d: aggregation skipped. Global model is retained.", comm_round)
            continue

        # Global Evaluation  
        try:
            global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_mae_po = evaluate_global_model_on_clients(global_model, clients, eval_times, discretiser, args)
        except Exception as exc: 
            logger.exception("Round %d: global model evaluation failed.", comm_round)
            global_val_cindex, test_cindex, test_brier = np.nan, np.nan, np.nan
            test_nbll, test_p_value, test_mae_po = np.nan, np.nan, np.nan
            
        # Early Stopping at Server 
        current_global_state = {
            key: value.detach().cpu().clone()
            for key, value in global_model.state_dict().items()
        }

        improved = False 
        if is_finite_metric( global_val_cindex ): 
            improved = early_stopping( global_val_cindex,
                current_global_state, round_idx=comm_round,
                test_metrics={
                    "test_c_index": test_cindex,
                    "test_brier":  test_brier,
                    "test_nbll": test_nbll,
                    "test_p_value":  test_p_value,
                    "test_MAE_PO":  test_mae_po }) 
        else: 
            logger.warning("Round %d: global validation C-index is undefined.", comm_round)

        if improved: 
            logger.info("New best global model | round=%d | val_cindex=%.4f", comm_round, global_val_cindex)

        logger.info( "Round %d | Val C-index: %.4f | Test C-index: %.4f | Brier: %.4f | NBLL: %.4f | D-calibration: %.4f | MAE PO: %.4f",
            comm_round, global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_mae_po)

        last_round = comm_round 
        if early_stopping.early_stop: 
            logger.info("Federated training stopped early.") 
            break

    # Final Result 
    final_round = early_stopping.best_round if early_stopping.best_round is not None else last_round   
    final_val_cindex =  early_stopping.best_val_cindex if np.isfinite( early_stopping.best_val_cindex ) else global_val_cindex  
    
    final_test_cindex =  early_stopping.best_test_cindex if early_stopping.best_test_cindex is not None else test_cindex    
    final_test_brier = early_stopping.best_test_brier if early_stopping.best_test_brier is not None else test_brier  
    
    final_test_nbll = early_stopping.best_test_nbll if early_stopping.best_test_nbll is not None else test_nbll   
    final_test_p_value =  early_stopping.best_test_p_value if early_stopping.best_test_p_value is not None else test_p_value  
    final_test_mae_po = early_stopping.best_test_MAE_PO if early_stopping.best_test_MAE_PO is not None else test_mae_po 

    results =  {
        "best_round": final_round,
        "val_cindex": final_val_cindex,
        "cindex": final_test_cindex,
        "brier": final_test_brier,
        "nbll": final_test_nbll,
        "d_calibration": final_test_p_value,
        "mae_po": final_test_mae_po
    } 
    
    logger.info("Federated training completed | Best round: %s | Results: %s", final_round, results)
    return results 