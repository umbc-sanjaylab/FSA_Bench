import numpy as np  
from model import *  
from client.client import train_local_client, evaluate_global_model_on_clients 
from server.utils import get_args 
from client.utils import Discretiser 
from datasets.data_processing import prepare_survival_data,  compute_eval_times 
from model import init_nets
from server.server import aggregate_fedavg  
from server.utils import EarlyStopping, mkdirs
from CL import train_centralized
from config import DATASET_CONFIG
from pathlib import Path  

def is_finite_metric(value):
    return value is not None and np.isscalar(value) and np.isfinite(float(value))  

def run_federated(model, clients, eval_times, args):
    num_durations = 40 
    device = args.device 
    
    global_model = model.to(device)
    discretiser = Discretiser(num_durations, scheme='km') 
    early_stopping = EarlyStopping(patience=args.patience, directory=args.modeldir, verbose=True, discretiser=discretiser, checkpoint_pth=f"{args.model}_model_run_{args.run_id}.pt")

    # Initialize outputs safely.
    global_val_cindex, test_cindex, test_brier = np.nan, np.nan, np.nan 
    test_nbll, test_p_value, test_mae_po  = np.nan, np.nan, np.nan 

    last_round = -1 
    for comm_round in range(args.comm_round): 
        print( f"\nCommunication round {comm_round + 1}/{args.comm_round}") 
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
                print(f"[Client {client.client_id}] " f"local training failed: {exc}")

        valid_clients = [client for client in selected_clients if client.client_id in local_models] 
        if not valid_clients: 
            print( f"Round {comm_round}: no valid client updates.") 
            continue

        ################################################ FedAvg ####################################

        aggregated = aggregate_fedavg(global_model, local_models, valid_clients) 
        if not aggregated: 
            print(f"Round {comm_round}: aggregation skipped.") 
            continue

        ################################################ Global Evaluation ########################## 

        try:
            global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_mae_po = evaluate_global_model_on_clients(global_model, clients, eval_times, discretiser, args)
        except Exception as exc: 
            print( f"[Global evaluation error] {type(exc).__name__}: {exc}")

            global_val_cindex = np.nan
            test_cindex = np.nan
            test_brier = np.nan
            test_nbll = np.nan
            test_p_value = np.nan
            test_mae_po = np.nan

        ################################################ Server Early Stopping

        current_global_state = {
            key: value.detach().cpu().clone()
            for key, value in global_model.state_dict().items()
        }

        improved = False

        if is_finite_metric( global_val_cindex ): 
            improved = early_stopping(
                global_val_cindex,
                current_global_state,
                round_idx=comm_round,
                test_metrics={
                    "test_c_index": test_cindex,
                    "test_brier":  test_brier,
                    "test_nbll": test_nbll,
                    "test_p_value":  test_p_value,
                    "test_MAE_PO":  test_mae_po
                }
            )

        else: 
            print( f"Round {comm_round}:  global validation C-index  is undefined.")

        if improved: 
            print( f"New best global model |  round={comm_round} | val_cindex={global_val_cindex:.4f}" )

        print(f"Global validation C-index:  {global_val_cindex}" ) 
        print(f"Global test C-index: {test_cindex}") 
        print(f"Global test Brier:  {test_brier}") 
        print( f"Global test NBLL:  {test_nbll}" ) 
        print( f"Global D-calibration: {test_p_value}" ) 
        print(f"Global weighted MAE PO: " f"{test_mae_po}")

        last_round = comm_round 
        if early_stopping.early_stop: 
            print( "Federated training stopped early.") 
            break

    ################################################ Final Result

    final_round = early_stopping.best_round if early_stopping.best_round is not None else last_round   
    final_val_cindex =  early_stopping.best_val_cindex if np.isfinite( early_stopping.best_val_cindex ) else global_val_cindex  
    
    final_test_cindex =  early_stopping.best_test_cindex if early_stopping.best_test_cindex is not None else test_cindex    
    final_test_brier = early_stopping.best_test_brier if early_stopping.best_test_brier is not None else test_brier  
    
    final_test_nbll = early_stopping.best_test_nbll if early_stopping.best_test_nbll is not None else test_nbll   
    final_test_p_value =  early_stopping.best_test_p_value if early_stopping.best_test_p_value is not None else test_p_value  
    final_test_mae_po = early_stopping.best_test_MAE_PO if early_stopping.best_test_MAE_PO is not None else test_mae_po 

    return {
        "best_round": final_round,
        "val_cindex": final_val_cindex,
        "cindex": final_test_cindex,
        "brier": final_test_brier,
        "nbll": final_test_nbll,
        "d_calibration": final_test_p_value,
        "mae_po": final_test_mae_po
    }
     

######################################################## Global Server Training ###########################################################  
if __name__ == '__main__': 
    args = get_args()
    
    dataset_config = DATASET_CONFIG[args.dataset] 
    args.n_parties = dataset_config["n_parties"]
    args.batch_size = dataset_config["batch_size"]
    
    num_folds = dataset_config["folds"] 
    if args.run_id < 0 or args.run_id >= num_folds:
        raise ValueError(f"{args.dataset} has {num_folds} folds. Received run_id={args.run_id}.")
      
    PROJECT_ROOT = Path(__file__).resolve().parent  
    log_dir =   PROJECT_ROOT / "results" / "logs" / args.setting.lower() / args.dataset / args.model 
    model_dir = PROJECT_ROOT / "checkpoints" / args.setting.lower() / args.dataset / args.model  
 
    mkdirs(log_dir)
    mkdirs(model_dir)
        
    data = prepare_survival_data(args) 
    eval_times = compute_eval_times(data, interval=args.interval)

    input_size = len(data.feature_names) 
    model = init_nets(input_size, n_parties=args.n_parties, args=args)

    if args.setting not in ["CL", "FL"]: 
        raise ValueError(f"Unsupported setting: {args.setting}")
      
    if args.setting == "CL": 
      model = init_nets(data.centralized.train.X.shape[1], 1, args)[0]  
      prepared_data = prepare_survival_data(args) 
      
      eval_times = compute_eval_times(prepared_data, interval=args.pseudo_perc_interval) 
      results = train_centralized(model=model, data=prepared_data.centralized, eval_times=eval_times, args=args)  
    else:  
      results = run_federated( model=model, clients=data.clients, eval_times=eval_times, args=args)  
 

