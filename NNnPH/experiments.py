import numpy as np 
import torch 
import os 
from model import *
from local_clients import *
from utils import * 
from local_clients import local_train_net, train_net_all   
from server_utils import get_args, compute_eval_time  
from datasets.data_processing import prepare_survival_data,  compute_eval_times 
from model import init_nets  
from copy import deepcopy

def train_centralized(model, data, eval_times, args):   
      model = init_nets(data.centralized.train.X.shape[1], 1, args)[0]  
      prepared_data = prepare_survival_data(args) 
      
      eval_times = compute_eval_times( prepared_data, interval=args.pseudo_perc_interval) 
      results = train_centralized(model=model, data=prepared_data.centralized, eval_times=eval_times, args=args)
      
      test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = results
      print('>> All in Test Cindex: {}'.format(test_cindex))
      print('>> All in Test Brier: {}'.format(test_brier))
      print('>> All in Test NBLL: {}'.format(test_nbll))
      print('>> All in Test D-Calibration: {}'.format(test_p_value))
      print('>> All in Test Weighted MAE PO: {}'.format(test_weighted_MAE_PO))

def run_federated( model, clients, eval_times, args):  
       pass 
     

######################################################## Global Server Training ###########################################################  
if __name__ == '__main__': 
    args = get_args() 
    mkdirs(args.logdir)
    mkdirs(args.modeldir) 
        
    data = prepare_survival_data(args) 
    eval_times = compute_eval_times(data, interval=args.pseudo_perc_interval)

    input_size = len(data.feature_names) 
    model = init_nets(input_size, n_parties=args.n_parties, args=args)

    if args.setting not in ["CL", "FL"]: 
        raise ValueError(f"Unsupported setting: {args.setting}")
      
    if args.setting == "CL": 
        train_centralized( model=model, data=data.centralized, eval_times=eval_times, args=args ) 
    else:  
        run_federated( model=model, clients=data.clients, eval_times=eval_times, args=args) 


def another(): 
    print('dataset:', args.dataset)
    start = time.time()
    start_counter = time.perf_counter()

    num_durations = 40
    discretiser = Discretiser(num_durations, scheme='km')

    # Combined the datasets from clients for centralized training
    X_train_all = []
    train_time_all = []
    train_status_all = []

    X_val_all = []
    val_time_all = []
    val_status_all = []

    X_test_all = []
    test_time_all = []
    test_status_all = []

    X_train_local_non_std = []
    X_train_local_std = []
    train_time_local = []
    train_status_local = []
    train_data_length = []

    X_val_local_non_std = []
    X_val_local_std = []
    val_time_local = []
    val_status_local = []

    X_test_local_non_std = []
    X_test_local_std = []
    test_time_local = []
    test_status_local = []
    client_name_array = []

    pseudo_evaltime = compute_eval_time(args)
    for party_id in range(args.n_parties):

        # Load local dataset
        train_df, val_df, test_df, client_name, feature_list = get_local_dataset(args.dataset, args.partition, party_id, args.run_id, args)
        std_list = [c for c in feature_list if c.startswith('num_')]
        client_name_array.append(client_name)

        X_train = np.array(train_df.drop(columns=["time", "event"]), dtype='float32')
        X_val = np.array(val_df.drop(columns=["time", "event"]), dtype='float32')
        X_test = np.array(test_df.drop(columns=["time", "event"]), dtype='float32')

        train_time, train_status = train_df["time"], train_df["event"]
        val_time, val_status = val_df["time"], val_df["event"]
        test_time, test_status = test_df["time"], test_df["event"]

        train_time = np.asarray(train_time).reshape(-1)
        train_status = np.asarray(train_status).reshape(-1).astype(int)

        val_time = np.asarray(val_time).reshape(-1)
        val_status = np.asarray(val_status).reshape(-1).astype(int)

        test_time = np.asarray(test_time).reshape(-1)
        test_status = np.asarray(test_status).reshape(-1).astype(int)

        # Append the data from each of the clients
        X_train_local_non_std.append(X_train)
        train_time_local.append(train_time)
        train_status_local.append(train_status)

        X_val_local_non_std.append(X_val)
        val_time_local.append(val_time)
        val_status_local.append(val_status)

        X_test_local_non_std.append(X_test)
        test_time_local.append(test_time)
        test_status_local.append(test_status)

        X_train_loc = pd.DataFrame(X_train)
        X_train_loc.columns = feature_list

        X_val_loc = pd.DataFrame(X_val)
        X_val_loc.columns = feature_list

        X_test_loc = pd.DataFrame(X_test)
        X_test_loc.columns = feature_list 

        X_train_loc = np.array(X_train_loc, dtype='float32')
        X_val_loc = np.array(X_val_loc, dtype='float32')
        X_test_loc = np.array(X_test_loc, dtype='float32')

        X_train_local_std.append(X_train_loc)
        X_val_local_std.append(X_val_loc)
        X_test_local_std.append(X_test_loc)
        train_data_length.append(X_train_loc.shape[0])

    X_train_all = np.vstack(X_train_local_non_std)
    X_train_all = np.array(X_train_all)

    train_time_all = np.concatenate(train_time_local)
    train_time_all = np.array(train_time_all, dtype='float32')

    train_status_all = np.concatenate(train_status_local)
    train_status_all = np.array(train_status_all, dtype='int64')

    X_val_all = np.vstack(X_val_local_non_std)
    X_val_all = np.array(X_val_all)

    val_time_all = np.concatenate(val_time_local)
    val_time_all = np.array(val_time_all, dtype='float32')

    val_status_all = np.concatenate(val_status_local)
    val_status_all = np.array(val_status_all, dtype='int64')

    X_test_all = np.vstack(X_test_local_non_std)
    X_test_all = np.array(X_test_all)

    test_time_all = np.concatenate(test_time_local)
    test_time_all = np.array(test_time_all)

    test_status_all = np.concatenate(test_status_local)
    test_status_all = np.array(test_status_all)

    X_train_all = pd.DataFrame(X_train_all)
    X_train_all.columns = feature_list

    X_val_all = pd.DataFrame(X_val_all)
    X_val_all.columns = feature_list

    X_test_all = pd.DataFrame(X_test_all)
    X_test_all.columns = feature_list 
   
    X_train_all = np.array(X_train_all)
    X_val_all = np.array(X_val_all)
    X_test_all = np.array(X_test_all)

    y_train_all = (train_time_all, train_status_all)
    y_val_all = (val_time_all, val_status_all)

    y_train_trans_all = data_transform(y_train_all, discretiser, fit_transform=True)
    train_time_trans_all = y_train_trans_all[0]
    train_status_trans_all = y_train_trans_all[1]

    y_val_trans_all = data_transform(y_val_all, discretiser, fit_transform=False)
    val_time_trans_all = y_val_trans_all[0]
    val_status_trans_all = y_val_trans_all[1]

    # Create train and validation dataloader
    train_dataset = TensorDataset(torch.tensor(X_train_all), torch.tensor( train_time_trans_all), torch.tensor(train_status_trans_all))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    validate_dataset = TensorDataset(torch.tensor(X_val_all), torch.tensor(val_time_trans_all), torch.tensor(val_status_trans_all))
    val_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True)

    test_cindex, test_brier, test_nbll = 0.0, 0.0, 0.0
    
    def fix_cindex_direction(cindex):
      if np.isfinite(cindex) and cindex < 0.5:
          return 1.0 - cindex
      return cindex
######################################## Centralized Training ######################################

    if args.alg == 'all_in': 
        nets, local_model_meta_data, layer_type = init_nets(X_test_all.shape[1], args.dropout_p, 1, args, discretiser)

        try:
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = train_net_all(0, nets[0].to( args.device), train_dataloader, val_dataloader, args.epochs, args.lr, args.optimizer, train_time_all, train_status_all, X_val_all, val_time_all, val_status_all, X_test_all, test_time_all, test_status_all, pseudo_evaltime, discretiser, args, device=args.device)
        except Exception as e:
            print( f"[Exception in train_net_all method]:  {type(e).__name__}: {e}")
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan 

        print('>> All in Test Cindex: {}'.format(test_cindex))
        print('>> All in Test Brier: {}'.format(test_brier))
        print('>> All in Test NBLL: {}'.format(test_nbll))
        print('>> All in Test D-Calibration: {}'.format(test_p_value))
        print('>> All in Test Weighted MAE PO: {}'.format(test_weighted_MAE_PO))

          
############################################### FedAvg ##############################################################################
    elif args.alg == 'fedavg':
        print("Initializing nets")

     

        nets, local_model_meta_data, layer_type = init_nets(X_train_local_std[0].shape[1], args.dropout_p, args.n_parties, args, discretiser)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, directory=args.modeldir, discretiser=discretiser, checkpoint_pth=f"{args.model}_model_run_{args.run_id}.pt")

        global_models, global_model_meta_data, global_layer_type = init_nets(X_train_local_std[0].shape[1], 0, 1, args, discretiser)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        global_c_index, comm_round_done = 0.0, 0
        for comm_round in range(args.comm_round):
            print("in comm round:" + str(comm_round))
            print('communication round:', comm_round)

            memory = mem_mb()
            print("Before:", memory, "MB")

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            # 1. Server sends current global model to selected clients
            global_para = deepcopy(global_model.state_dict())   
            for idx in selected:
                if nets[idx] is None:
                    initialized_nets, _, _ = init_nets(
                        input_size=X_train_local_std[0].shape[1],
                        dropout_p=args.dropout_p,
                        n_parties=1,
                        args=args,
                        discretiser=discretiser,
                    )
                    nets[idx] = initialized_nets[0]

                nets[idx].load_state_dict(global_para, strict=True)
                nets[idx].to(args.device)

            print(f"device is {args.device}")
            
            # 2. Clients train locally with local early stopping
            nets = local_train_net(nets, client_name_array, selected, args, X_train_local_std, train_time_local, train_status_local, X_val_local_std, val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local, pseudo_evaltime, discretiser, train_data_length, comm_round, device=args.device)
            
            valid_clients = [client_id for client_id in selected if nets[client_id] is not None]
            if not valid_clients:
                logger.warning(f"Round {comm_round}: no valid client updates; retaining the previous global model.")
                continue
            
            # 3. Server aggregates trained client models 
            total_data_points = sum( train_data_length[client_id] for client_id in valid_clients )
            if total_data_points <= 0:
                logger.warning( f"Round {comm_round}: valid clients contain zero  training samples; retaining the global model.")
                continue 
            
            # fed_avg_freqs = [train_data_length[r] / total_data_points for r in valid_clients] 
            client_states = {
                client_id: {
                    key: value.detach().cpu().clone()
                    for key, value in nets[client_id].state_dict().items()
                }
                for client_id in valid_clients
            }

            # Use the largest client for non-floating buffers,
            # such as BatchNorm num_batches_tracked.
            largest_client = max(
                valid_clients,
                key=lambda client_id: train_data_length[client_id],
            )

            # Use the current global state as the reference.
            server_state = {
                key: value.detach().cpu().clone()
                for key, value in global_para.items()
            }

            new_global_para = {}

            for key, reference_value in server_state.items():
                if  reference_value.is_floating_point() or reference_value.is_complex() :
                    aggregated_value = torch.zeros_like(reference_value)

                    for client_id in valid_clients:
                        aggregation_weight = ( train_data_length[client_id] / total_data_points)
                        aggregated_value += ( client_states[client_id][key] * aggregation_weight )

                    new_global_para[key] = aggregated_value

                else:
                    # Do not average integer BatchNorm buffers.
                    new_global_para[key] = ( client_states[largest_client][key].clone())

            # Load the newly aggregated state into the global model.
            global_model.load_state_dict( new_global_para, strict=True)
            
            # 5. Evaluate the newly aggregated global model on clients
            eval_clients = list(range(args.n_parties))  
            try:
                global_val_cindex,  test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = evaluate_global_model_on_clients(
                    global_model,  eval_clients, X_val_local_std,  val_time_local, val_status_local,  X_test_local_std,  test_time_local,  test_status_local, train_time_local, train_status_local, pseudo_evaltime, discretiser, device=args.device
                )

            except Exception as e:
                print(f"[Exception in global evaluation]: {type(e).__name__}: {e}")
                traceback.print_exc()

                global_val_cindex = np.nan
                test_cindex = np.nan
                test_brier = np.nan
                test_nbll = np.nan
                test_p_value = np.nan
                test_weighted_MAE_PO = np.nan
            
            metrics_dict = {
                "c_idx": test_cindex,
                "brier": test_brier,
                "nbll": test_nbll,
                "p_value": test_p_value,
                "weighted_MAE_PO": test_weighted_MAE_PO
            } 
                
            # 6. Server early stopping should use the aggregated global model  
            current_global_para = {
                key: value.detach().cpu().clone()
                for key, value in global_model.state_dict().items()
            } 
            
            improved = False
            if (global_val_cindex is not None and np.isfinite(global_val_cindex)):
                improved = early_stopping(
                    global_val_cindex,
                    current_global_para,
                    round_idx=comm_round,
                    test_metrics={
                        "test_c_index": test_cindex,
                        "test_brier": test_brier,
                        "test_nbll": test_nbll,
                        "test_p_value": test_p_value,
                        "test_MAE_PO": test_weighted_MAE_PO,
                    },
                )
            else:
                logger.warning(f"Round {comm_round}: global validation C-index  is undefined; skipping early stopping.") 
            
            if improved:
              print(
                  f"New best global model at round {comm_round}: "
                  f"val_cindex={global_val_cindex:.4f}, "
                  f"test_cindex={test_cindex:.4f}, "
                  f"test_brier={test_brier:.4f}, "
                  f"test_nbll={test_nbll:.4f}, "
                  f"test_p_value={test_p_value:.4f}, "
                  f"test_weighted_MAE_PO={test_weighted_MAE_PO:.4f}"
              )
            
            print(">> Global Model Validation C-index:", global_val_cindex)
            print(">> Global Model Test Cindex:", test_cindex)
            print(">> Global Model Test Brier:", test_brier)
            print(">> Global Model Test NBLL:", test_nbll)
            print(">> Global Model D-calibration:", test_p_value)
            print(">> Global Model Weighted MAE PO:", test_weighted_MAE_PO) 

            comm_round_done = comm_round
            if early_stopping.early_stop:
                comm_round_done = comm_round
                print("Early stopping")
                break

            memory = mem_mb()
            print("After:", memory, "MB")

        # report the metrics from the best validation round, not the last round
        final_round = early_stopping.best_round if early_stopping.best_round is not None else comm_round
        final_val_cidx = early_stopping.best_val_cindex if np.isfinite(early_stopping.best_val_cindex) else global_val_cindex 
        final_test_cidx = early_stopping.best_test_cindex if early_stopping.best_test_cindex is not None else test_cindex  
        final_test_brier = early_stopping.best_test_brier if early_stopping.best_test_brier is not None else test_brier 
        final_test_nbll = early_stopping.best_test_nbll if early_stopping.best_test_nbll is not None else test_nbll 
        final_test_p_value = early_stopping.best_test_p_value if early_stopping.best_test_p_value is not None else test_p_value  
        final_test_weighted_MAE_PO = early_stopping.best_test_MAE_PO if early_stopping.best_test_MAE_PO is not None else test_weighted_MAE_PO 

    print("ARGS USED:", args.init_seed, args.dataset, args.partition, args.model, args.lr, args.batch_size)
    print("CSV:", os.path.join(args.logdir, "results.csv"))

    end = time.time()
    end_counter = time.perf_counter()

    total_time = end - start
    total_time_counter = end_counter - start_counter

    # total time taken
    print(f"Runtime of the {args.model} model is {end - start} sec")

    import builtins

    csv_path = os.path.join(args.logdir, "results_server_side_early_KM.csv")
    print(csv_path)

    epoch = args.epochs
    if args.alg == "all_in":
        comm_round_done = 1
        final_test_cidx = test_cindex
        final_test_brier = test_brier
        final_test_nbll = test_nbll
        final_test_p_value = test_p_value
        final_test_weighted_MAE_PO = test_weighted_MAE_PO

    row = {
        "seed": args.init_seed,
        "fold_id": int(args.run_id),
        "dataset": args.dataset,
        "partition": args.partition,
        "model": args.model,
        "c_idx": builtins.round(final_test_cidx, 4),
        "brier": builtins.round(final_test_brier, 4),
        "nbll": builtins.round(final_test_nbll, 4),
        "p_value": builtins.round( final_test_p_value, 4),
        "weighted_MAE_PO": builtins.round( final_test_weighted_MAE_PO, 4),
        "lr": builtins.round(float(args.lr), 5),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "dropout": float(args.dropout_p),
        "comm_round": int(args.comm_round),
        "time": builtins.round(float(total_time), 4),
        "time_counter": builtins.round(float(total_time_counter), 4),
        "memory": memory,
    }

    df = pd.DataFrame([row])
    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
