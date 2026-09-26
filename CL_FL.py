
import os
import traceback
import numpy as np
from federated.client import train_selected_clients 
from federated.server import fedavg, EarlyStopping
from federated.client import deepcopy, local_train_net, train_net_all 
from federated.client_utility import evaluate_global_model_on_clients
         
def run_centralized(args, data, model_spec, eval_times):
    """
    Run centralized training and evaluation.

    Parameters
    ----------
    args
        Experiment configuration.

    data
        PreparedSurvivalData containing centralized and client-level data.

    model_spec
        Model configuration from MODEL_REGISTRY.

    eval_times
        Evaluation time points used for survival metrics.

    Returns
    -------
    dict
        Final test metrics.
    """

    print("\n================ Centralized Training ================")

  
    # 1. Retrieve centralized train / validation / test data 
    train = data.centralized.train
    val = data.centralized.val
    test = data.centralized.test

    feature_names = data.feature_names  
    input_dim = train.X.shape[1]

    print(f"Train shape : {train.X.shape}")
    print(f"Val shape   : {val.X.shape}")
    print(f"Test shape  : {test.X.shape}")
    print(f"Input dim   : {input_dim}")
 
    # 2. Initialize AFT-Weibull model 
    model_class = model_spec["model"]

    model = model_class(
        input_dim=input_dim
    ).to(args.device)
 
    # 3. Train and evaluate 
    try:
      test_cindex, test_brier, test_nbll, test_p_value, test_mae_po  = train_net_all(0, model, train.X, train.time, train.event, val.X, val.time, val.event, test.X, test.time, test.event, eval_times, feature_names, args, args.device)

    except Exception as exc:

        print(
            f"[Centralized training error] "
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        test_cindex = np.nan
        test_brier = np.nan
        test_nbll = np.nan
        test_p_value = np.nan
        test_mae_po = np.nan
 
    # 4. Return common result format 
    results = {
        "val_c_index": np.nan,
        "c_idx": test_cindex,
        "brier": test_brier,
        "nbll": test_nbll,
        "p_value": test_p_value,
        "weighted_MAE_PO": test_mae_po,
        "comm_round_done": 1,
    }

    print("\nCentralized Test Results")
    print(f"C-index       : {test_cindex}")
    print(f"IBS           : {test_brier}")
    print(f"NBLL          : {test_nbll}")
    print(f"D-cal p-value : {test_p_value}")
    print(f"MAE-PO        : {test_mae_po}")

    return results
  
def run_federated(args, data, model_spec, eval_times):
    """
    Run federated training using weighted FedAvg.

    The function is model-independent. Model-specific training and
    prediction are accessed through model_spec.

    Parameters
    ----------
    args
        Experiment configuration.

    data
        PreparedSurvivalData containing all client datasets.

    model_spec
        Model specification obtained from MODEL_REGISTRY.

    eval_times
        Evaluation time points used for survival prediction and metrics.

    Returns
    -------
    dict
        Metrics corresponding to the best global validation round.
    """

    print("\n################ Federated Training ##################################")
    print(f"Model   : {args.model} Dataset : {args.dataset} Fold    : {args.run_id}")

    
    # 1. Federated dataset information 
    clients = data.clients
    n_clients = len(clients)

    if n_clients == 0:
        raise ValueError("No clients are available for federated training.")

    input_dim = clients[0].train.X.shape[1] 
    print(f"Number of clients : {n_clients} and dimension   : {input_dim}")
 
    # 2. Model factory
    model_class = model_spec["model"]  
     
    # 3. Initialize local client models
    client_models = {client_id: model_class(input_dim) for client_id in range(n_clients)}

    # 4. Initialize global model
    global_model = model_class(input_dim) 
    global_state = deepcopy(global_model.state_dict())

    # 5. Use identical initialization for all clients
    if args.is_same_initial: 
        for client_id in range(n_clients): 
            client_models[client_id].load_state_dict(global_state, strict=True)
  
    # 6. Server-side early stopping
    os.makedirs(args.modeldir, exist_ok=True) 
    checkpoint_name = f"{args.model}_fold_{args.run_id}_global_model.pt"  
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, directory=args.modeldir, checkpoint_pth=checkpoint_name)
 
    # 7. Initialize metrics 
    global_val_cindex = np.nan
    test_cindex = np.nan
    
    test_brier = np.nan
    test_nbll = np.nan
    
    test_p_value = np.nan
    test_mae_po = np.nan 
    comm_round_done = 0
 
    # 8. Communication rounds 
    for comm_round in range(args.comm_round): 
        print( f"Communication Round {comm_round + 1}/{args.comm_round}") 
        
        # 8.1 Select clients
        selected_clients = np.arange( n_clients)

        # 8.2 Broadcast current global model to clients
        global_state = deepcopy( global_model.state_dict())

        for client_id in selected_clients: 
            client_id = int(client_id)

            # Recreate model if client was previously skipped
            if client_models[client_id] is None: 
                client_models[client_id] = model_class(input_dim)

            client_models[ client_id ].load_state_dict( global_state, strict=True) 
            client_models[ client_id ].to(args.device)
 
        # 8.3 Local client training 
        client_models = train_selected_clients(
            models=client_models,
            selected_clients=selected_clients,
            data=data,
            model_spec=model_spec,
            eval_times=eval_times,
            args=args,
            device=args.device,
        )

      
        # 8.4 Keep clients that produced valid models 
        valid_clients = [ int(client_id) for client_id in selected_clients if client_models[int(client_id)] is not None] 
        print( f"Valid clients: {len(valid_clients)}/{len(selected_clients)}")

        if len(valid_clients) == 0: 
            print( "No valid client models were produced. Federated training stopped.") 
            break

        # 8.5 Client training-set sizes
        client_sizes = [len( clients[client_id].train.X ) for client_id in valid_clients] 
        
        # 8.6 Weighted FedAvg 
        models_to_aggregate = [ client_models[client_id] for client_id in valid_clients]  
        aggregated_state = fedavg(
            client_models=models_to_aggregate,
            client_sizes=client_sizes 
        )
 
        # 8.7 Update global model 
        global_model.load_state_dict( aggregated_state, strict=True) 
        global_model.to(args.device)

       
        # 8.8 Evaluate global model on all clients 
        evaluation_clients = list(range(n_clients))

        try:
            global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_mae_po  = evaluate_global_model_on_clients(global_model=global_model, client_ids=evaluation_clients, data=data, model_spec=model_spec, eval_times=eval_times,  device=args.device)

        except Exception as exc: 
            print(f"[Global evaluation error] {type(exc).__name__}: {exc}") 
            traceback.print_exc()

            global_val_cindex = np.nan
            test_cindex = np.nan
            test_brier = np.nan
            test_nbll = np.nan
            test_p_value = np.nan
            test_mae_po = np.nan

        # 8.9 Server-side early stopping
        improved = False 
        if global_val_cindex is not None and np.isfinite(global_val_cindex): 
            current_global_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in global_model.state_dict().items()
            }

            improved = early_stopping(
                global_val_cindex,
                current_global_state,
                round_idx=comm_round,
                test_metrics={
                    "test_c_index": test_cindex,
                    "test_brier": test_brier,
                    "test_nbll": test_nbll,
                    "test_p_value": test_p_value,
                    "test_MAE_PO": test_mae_po,
                },
            )

        else: 
            print( f"Round {comm_round + 1}: global validation C-index is undefined; server-side early stopping skipped.")

        # 8.10 Report round results
        if improved: 
            print( f"New best global model selected at round {comm_round + 1}")

        print(f"Validation C-index : {global_val_cindex} and Test C-index  : {test_cindex}") 
        print( f"Test IBS           : {test_brier} and Test NBLL  : {test_nbll}") 
        print(f"D-cal p-value      : {test_p_value} and Weighted MAE-PO  : {test_mae_po}")

        # Number of completed communication rounds
        comm_round_done =  comm_round + 1 
        
        # 8.11 Check server-side stopping criterion 
        if early_stopping.early_stop: 
            print( f"Server-side early stopping after {comm_round_done} communication rounds.") 
            break

    # 9. Select metrics from best validation round
    if np.isfinite( early_stopping.best_val_cindex):
        final_val_cindex =  early_stopping.best_val_cindex  
    else: 
        final_val_cindex =  global_val_cindex
 

    final_test_cindex =  early_stopping.best_test_cindex if early_stopping.best_test_cindex is not None else test_cindex 
    final_test_brier =  early_stopping.best_test_brier if early_stopping.best_test_brier is not None else test_brier 
    final_test_nbll =  early_stopping.best_test_nbll if early_stopping.best_test_nbll is not None else test_nbll 
    final_test_p_value =  early_stopping.best_test_p_value if early_stopping.best_test_p_value is not None else test_p_value 
    final_test_mae_po = early_stopping.best_test_MAE_PO if early_stopping.best_test_MAE_PO is not None else test_mae_po  
 
    # 10. Best communication round  
    if early_stopping.best_round is not None:
        best_round = early_stopping.best_round
    elif comm_round_done > 0:
        best_round = comm_round_done - 1
    else:
        best_round = None
    
    # 11. Final result 
    results = {
        "val_c_index": final_val_cindex,
        "c_idx": final_test_cindex,
        "brier": final_test_brier,
        "nbll": final_test_nbll,
        "p_value": final_test_p_value,
        "weighted_MAE_PO": final_test_mae_po,
        "best_round": best_round,
        "comm_round_done": comm_round_done,
    }

    print("\n####################### Final Federated Results#########################################" )

    print( f"Best round      : {best_round}") 
    print( f"Validation C-I  : {final_val_cindex} Test C-index    : {final_test_cindex}")  
    print( f"Test IBS        : {final_test_brier} Test NBLL       : {final_test_nbll}") 
    print( f"D-cal p-value   : {final_test_p_value} Weighted MAE-PO : {final_test_mae_po}")

    return results