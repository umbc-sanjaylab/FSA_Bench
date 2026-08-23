import numpy as np
import json
import torch
import argparse
import os
from math import *
from sklearn.preprocessing import StandardScaler
from model import *
from local_clients import *
from utils import *
import builtins
import time
from local_clients import local_train_net, train_net_all
from sklearn.model_selection import StratifiedKFold
import os
import psutil
import traceback

proc = psutil.Process(os.getpid())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CoxPH', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='metabric', help='dataset used for training')
    parser.add_argument( '--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy')
    parser.add_argument('--Pseudo_value', type=str, default='non-DP', help='the pseudo value calculation strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=50, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg', help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication round')
    parser.add_argument('--patience', type=int, default=20, help='number of patience')
    parser.add_argument('--is_same_initial', type=int, default=1,  help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--run_id', type=int, default=0,  help="fold number of 5-fold cross validation")
    parser.add_argument('--dropout_p', type=float, required=False,  default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="/home/local/AD/il66977/benchmark_FSA/Fed_CoxPH/data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False,  default="/home/local/AD/il66977/benchmark_FSA/Fed_CoxPH/logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="/home/local/AD/il66977/benchmark_FSA/Fed_CoxPH/models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda:0',  help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str,  default='adam', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1,  help='the mu parameter for fedprox')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1.0,  help='Sample ratio for each communication round')
    parser.add_argument('--pseudo_perc_interval', type=int,  default=10,  help='Interval for pseudo values calculation')
    parser.add_argument('--epsilon', type=float, default=5.0, help='epsilon')

    args = parser.parse_args()
    return args


def init_nets(input_size, dropout_p, n_parties, args, discretiser):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        hidden_sizes = [32, 32]
        batch_norm = True

        if args.model == "LinearPH":
            net = CoxPH(dim_in=input_size, dim_out=len(discretiser.cuts), batch_norm=batch_norm)
        elif args.model == "NNph":
            net = MLPPH(dim_in=input_size, num_nodes=hidden_sizes, dim_out=len(discretiser.cuts), batch_norm=batch_norm, dropout=dropout_p)
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type 

def mem_mb():
    return proc.memory_info().rss / (1024**2)

def get_fold_index(party_id, X_data, status_data, args):
    fold_indices = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.init_seed + int(party_id))

    dummy_X = np.zeros((len(status_data), 1))  # optional: can also use X_data
    for _, test_idx in skf.split(X_data, status_data):
        fold_indices.append(test_idx)

    # Rolling cross validation
    num_folds = 5
    val_fold = (3 + args.run_id) % num_folds
    test_fold = (4 + args.run_id) % num_folds
    train_folds = [i for i in range(num_folds) if i not in (val_fold, test_fold)]

    train_idx = np.concatenate([fold_indices[i] for i in train_folds])
    val_idx = fold_indices[val_fold]
    test_idx = fold_indices[test_fold]

    return train_idx, val_idx, test_idx


def compute_eval_time(args):
    local_unique_times = []

    for client_id in range(args.n_parties):
        train_df, val_df, _, _, _ = get_local_dataset(args.dataset, args.partition, client_id, args.run_id, args)
        train_time, _ = train_df["time"], train_df["event"]

        val_time, _ = val_df["time"], val_df["event"]
        local_unique_times.append(np.unique(np.concatenate([train_time, val_time])))

    glob_unique_time = np.unique(np.concatenate(local_unique_times))
    pseudo_evaltime = np.percentile(glob_unique_time, np.arange(args.pseudo_perc_interval, 99, args.pseudo_perc_interval))
    return pseudo_evaltime

#############################################################################################################################
################################################### Global Server Training ###########################################################


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)

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
    tr_data_len = []

    pseudo_evaltime = compute_eval_time(args)
    for party_id in range(args.n_parties):

        # Load local dataset
        train_df, val_df, test_df, client_name, feature_list = get_local_dataset(args.dataset, args.partition, party_id, args.run_id, args)
        
        std_list = [c for c in feature_list if c.startswith('num_')]
        client_name_array.append(client_name)

        X_train = np.array(train_df.drop(columns=["time", "event"]), dtype='float32')
        X_val = np.array(val_df.drop(columns=["time", "event"]), dtype='float32')
        X_test = np.array(test_df.drop(columns=["time", "event"]), dtype='float32')
        tr_data_len.append(X_train.shape[0])

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

        # if std_list:
        #     scaler = StandardScaler().fit(X_train_loc.loc[:, std_list])
        #     X_train_loc.loc[:, std_list] = scaler.transform(X_train_loc.loc[:, std_list])
        #     X_val_loc.loc[:, std_list] = scaler.transform(X_val_loc.loc[:, std_list])
        #     X_test_loc.loc[:, std_list] = scaler.transform(X_test_loc.loc[:, std_list])

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

    # if std_list:
    #     scaler = StandardScaler().fit(X_train_all.loc[:, std_list])
    #     X_train_all.loc[:, std_list] = scaler.transform( X_train_all.loc[:, std_list])
    #     X_val_all.loc[:, std_list] = scaler.transform( X_val_all.loc[:, std_list])
    #     X_test_all.loc[:, std_list] = scaler.transform( X_test_all.loc[:, std_list])

    X_train_all = np.array(X_train_all)
    X_val_all = np.array(X_val_all)
    X_test_all = np.array(X_test_all)

    y_train_all = (train_time_all, train_status_all)
    y_val_all = (val_time_all, val_status_all) 
    y_train_trans_all = data_transform( y_train_all, discretiser, fit_transform=True)
    
    print("*"*100) 
    print(f"discretizer cuts length: {len(discretiser.cuts)}")
    print(f"discretizer cuts : {discretiser.cuts}") 
    print("*"*100)
    
    train_time_trans_all = y_train_trans_all[0]
    train_status_trans_all = y_train_trans_all[1]

    y_val_trans_all = data_transform( y_val_all, discretiser, fit_transform=False)
    val_time_trans_all = y_val_trans_all[0]
    val_status_trans_all = y_val_trans_all[1]

    # Create train and validation dataloader
    train_dataset = TensorDataset(torch.tensor(X_train_all), torch.tensor( train_time_trans_all), torch.tensor(train_status_trans_all))
    train_dataloader = DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True,  drop_last=False)

    validate_dataset = TensorDataset(torch.tensor(X_val_all), torch.tensor( val_time_trans_all), torch.tensor(val_status_trans_all))
    val_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)  # ,  drop_last=True) 
      
    def fix_cindex_direction(cindex):
        if np.isfinite(cindex) and cindex < 0.5:
            return 1.0 - cindex
        return cindex
                  
    test_cindex, test_brier, test_nbll = 0.0, 0.0, 0.0
######################################## Centralized Training ######################################

    if args.alg == 'all_in':

        memory = mem_mb()
        print("Before:", memory, "MB") 
        nets, local_model_meta_data, layer_type = init_nets(X_test_all.shape[1], args.dropout_p, 1, args, discretiser)
        
        try:
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = train_net_all(0, nets[0].to( args.device), train_dataloader, val_dataloader, args.epochs, args.lr, args.optimizer,  train_time_all, train_status_all, X_val_all, val_time_all, val_status_all, X_test_all, test_time_all, test_status_all, pseudo_evaltime, discretiser, args, device=args.device)
        except Exception as e:
            print(f"[Exception in train_net_all method]:  {type(e).__name__}: {e}")
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan

        print('>> All in Test Cindex: {}'.format(test_cindex))
        print('>> All in Test Brier: {}'.format(test_brier))
        print('>> All in Test NBLL: {}'.format(test_nbll))
        print('>> All in Test P-Value: {}'.format(test_p_value))
        print('>> All in Test Weighted MAE PO: {}'.format(test_weighted_MAE_PO))
        final_round = 1 

        memory = mem_mb()
        print("After:", memory, "MB")

        ############################################### FedAvg ##############################################################################
        
    elif args.alg == 'fedavg':
        print("Initializing nets")

        memory = mem_mb()
        print("Before:", memory, "MB")

        nets, local_model_meta_data, layer_type = init_nets( X_train_local_std[0].shape[1], args.dropout_p, args.n_parties, args, discretiser)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, directory=args.modeldir, discretiser=discretiser,  checkpoint_pth=f"{args.model}_model_run_{args.run_id}.pt")

        global_models, global_model_meta_data, global_layer_type = init_nets(X_train_local_std[0].shape[1], 0, 1, args, discretiser)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        global_c_index = 0.0
        comm_round_done = 0

        for comm_round in range(args.comm_round): 
            print(f'communication round: {comm_round}')

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
            nets = local_train_net(nets, args.modeldir, client_name_array, selected, args, X_train_local_std, train_time_local, train_status_local, X_val_local_std, val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local, pseudo_evaltime, discretiser, train_data_length, comm_round, device=args.device) 

            # 3. Server aggregates trained client models
            valid_clients = [client_id for client_id in selected if nets[client_id] is not None]
            if not valid_clients:
                logger.warning(
                    f"Round {comm_round}: no valid client updates; "
                    "retaining the previous global model."
                )
                continue
              
            total_data_points = sum(train_data_length[client_id] for client_id in valid_clients) 
            if total_data_points <= 0:
                logger.warning( f"Round {comm_round}: valid clients contain zero  training samples; retaining the global model.")
                continue
              
            new_global_para = None 
            for client_id in valid_clients:
                aggregation_weight = ( train_data_length[client_id] / total_data_points ) 
                net_para = nets[client_id].cpu().state_dict()

                if new_global_para is None:
                    new_global_para = {
                        key: value * aggregation_weight
                        for key, value in net_para.items()
                    }
                else:
                    for key, value in net_para.items():
                        new_global_para[key] += ( value * aggregation_weight)

            global_model.load_state_dict(new_global_para)  

            # 5. Evaluate the newly aggregated global model on clients
            eval_clients = list(range(args.n_parties))
            
            try:
                ( global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO ) = evaluate_global_model_on_clients( global_model, eval_clients, X_val_local_std, val_time_local,  val_status_local,  X_test_local_std,  test_time_local,  test_status_local,  train_time_local,  train_status_local, pseudo_evaltime, discretiser,  device=args.device)
            except Exception as e:
                print(f"[Exception in train_net method]:  {type(e).__name__}: {e}")
                traceback.print_exc()
                global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
            metrics_dict = {
                "c_idx": test_cindex,
                "brier": test_brier,
                "nbll": test_nbll,
                "p_value": test_p_value,
                "weighted_MAE_PO": test_weighted_MAE_PO
            } 
            
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
                    f"New best model selected at round {comm_round}: "
                    f"val_cindex={global_val_cindex:.4f}, "
                    f"test_cindex={test_cindex:.4f}, "
                    f"test_brier={test_brier:.4f}, "
                    f"test_nbll={test_nbll:.4f}, "
                    f"test_p_value={test_p_value:.4f}, "
                    f"test_weighted_MAE_PO={test_weighted_MAE_PO:.4f}"
                )
            
                
            comm_round_done = comm_round
            if early_stopping.early_stop:
                comm_round_done = comm_round
                print("Early stopping")
                break

            memory = mem_mb()
            print("After:", memory, "MB")

            print('>> Global Model Test Cindex: {}'.format(test_cindex))
            print('>> Global Model Test Brier: {}'.format(test_brier))
            print('>> Global Model Test NBLL: {}'.format(test_nbll))
            print('>> Global Model D-calibration: {}'.format(test_p_value))
            print('>> Global Model Weighted MAE PO: {}'.format(test_weighted_MAE_PO))
            
            

        # report the metrics from the best validation round, not the last round
        final_round = early_stopping.best_round if early_stopping.best_round is not None else comm_round
        final_val_cidx = early_stopping.best_val_cindex if np.isfinite(early_stopping.best_val_cindex) else global_val_cindex
        final_test_cidx =   early_stopping.best_test_cindex if early_stopping.best_test_cindex is not None else test_cindex
        final_test_brier = early_stopping.best_test_brier if early_stopping.best_test_brier is not None else test_brier
        final_test_nbll = early_stopping.best_test_nbll if early_stopping.best_test_nbll is not None else test_nbll 
        final_test_p_value = early_stopping.best_test_p_value if early_stopping.best_test_p_value is not None else test_p_value
        final_test_MAE_PO = early_stopping.best_test_MAE_PO if early_stopping.best_test_MAE_PO is not None else test_weighted_MAE_PO
        
        print("ARGS USED:", args.init_seed, args.dataset,
              args.partition, args.model, args.lr, args.batch_size)
        print("CSV:", os.path.join(args.logdir,  "results_server_side_early_stopping.csv"))

    end = time.time()
    end_counter = time.perf_counter()

    total_time = end - start
    total_time_counter = end_counter - start_counter

    if args.alg == "all_in":
        comm_round_done = 1
        final_test_cidx = test_cindex
        final_test_brier = test_brier
        final_test_nbll = test_nbll
        final_test_p_value = test_p_value
        final_test_MAE_PO = test_weighted_MAE_PO

    # total time taken
    print(
        f"Runtime of the {args.model} model is {end - start} sec")

    import builtins

    csv_path = os.path.join(args.logdir, "results_early_stopping.csv")
    print(csv_path)

    epoch = args.epochs
    if args.alg == "all_in":
        comm_round_done = 1
        final_test_cidx = test_cindex
        final_test_brier = test_brier
        final_test_nbll = test_nbll
        final_test_p_value = test_p_value
        final_test_MAE_PO = test_weighted_MAE_PO

    row = {
            "seed": args.init_seed,
            "fold_id": int(args.run_id),
            "dataset": args.dataset,
            "partition": args.partition,
            "model": args.model,
            "best_round": int(final_round) + 1 if final_round is not None else np.nan,  
            "comm_rounds_completed": int(comm_round_done) + 1,
            "configured_comm_rounds": int(args.comm_round),
            "c_idx": builtins.round(final_test_cidx, 4),
            "brier": builtins.round(final_test_brier, 4),
            "nbll": builtins.round(final_test_nbll, 4),
            "p_value": builtins.round(final_test_p_value, 4),
            "MAE_PO": builtins.round(final_test_MAE_PO, 4),
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
