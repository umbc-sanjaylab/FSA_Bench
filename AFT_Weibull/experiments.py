import traceback
import psutil
import time
from local_clients import local_train_net, train_net_all, evaluate_global_model_on_clients
from model import init_nets
from sklearn.preprocessing import StandardScaler
from math import *
import warnings
from utils import *
import os
import argparse
import numpy as np
import builtins 
import torch
from copy import deepcopy  
import gc
gc.collect()
torch.cuda.set_per_process_memory_fraction(0.15, 0)
torch.cuda.empty_cache()

warnings.filterwarnings('ignore')

proc = psutil.Process(os.getpid())

################################################### Get Arguments ###########################################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='FedPseudo', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='metabric', help='dataset used for training')
    parser.add_argument(  '--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=100, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=5, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg', help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=10, help='number of maximum communication round')
    parser.add_argument('--patience', type=int, default=20,  help='number of patience')
    parser.add_argument('--eval_every', type=int, default=5, help='compute C-index every N epochs')
    parser.add_argument('--is_same_initial', type=int, default=1,  help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--run_id', type=int, default=0, help="ID for 5-fold cross validation")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str,  default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--pseudo_perc_interval', type=int, default=10,  help='Interval for pseudo values calculation')
    parser.add_argument('--sensitivity', type=float, default=2.0, help='sensitivity')
    parser.add_argument('--epsilon', type=float, default=5.0, help='epsilon')
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy')

    args = parser.parse_args()
    return args


def mem_mb():
    return proc.memory_info().rss / (1024**2) 

def compute_eval_time(args):
    local_unique_times = []

    for client_id in range(args.n_parties):
        train_df, val_df, _, _, _ = get_local_dataset(args.dataset, args.partition, client_id, args.run_id, args)
        train_time, _ = train_df["time"], train_df["event"]

        val_time, _ = val_df["time"], val_df["event"]
        local_unique_times.append(np.unique(np.concatenate([train_time, val_time])))

    glob_unique_time = np.unique(np.concatenate(local_unique_times))
    pseudo_evaltime = np.percentile(glob_unique_time, np.arange(
        args.pseudo_perc_interval, 99, args.pseudo_perc_interval))
    return pseudo_evaltime

################################################### Global Server Training ###########################################################


if __name__ == '__main__':
    args = get_args()
    print(args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    print('dataset:', args.dataset)
    start = time.time()
    start_counter = time.perf_counter()

################################################### Data loading ##################################################################

    start = time.time()

    # Combined the datasets from clients for centralized training
    tr_data_len = []
    X_train_local_non_std = []
    X_train_local_std = []

    X_val_local_non_std = []
    X_val_local_std = []

    X_test_local_non_std = []
    X_test_local_std = []

    train_time_local = []
    train_status_local = []

    val_time_local = []
    val_status_local = []

    test_time_local = []
    test_status_local = []

    train_pseudo_local = []
    val_pseudo_local = []

    feature_list = None
    std_list = None
    global_unique_times = []

    client_name_array = []
    client_name = None

    pseudo_evaltime = compute_eval_time(args)
    print(pseudo_evaltime)

    for party_id in range(args.n_parties):
        train_df, val_df, test_df, client_name, feature_list = get_local_dataset(args.dataset, args.partition, party_id, args.run_id, args)
        std_list = [c for c in feature_list if c.startswith('num_')]
        client_name_array.append(client_name)

        column = 'status' if 'status' in train_df.columns else 'event'
        X_train = np.array(train_df.drop( columns=["time", column]), dtype='float32')
        X_val = np.array(val_df.drop( columns=["time", column]), dtype='float32')
        X_test = np.array(test_df.drop( columns=["time", column]), dtype='float32')

        train_time, train_status = train_df["time"], train_df[column]
        val_time, val_status = val_df["time"], val_df[column]
        test_time, test_status = test_df["time"], test_df[column]

        train_time = np.asarray(train_time).reshape(-1)
        train_status = np.asarray(train_status).reshape(-1).astype(int)

        val_time = np.asarray(val_time).reshape(-1)
        val_status = np.asarray(val_status).reshape(-1).astype(int)

        test_time = np.asarray(test_time).reshape(-1)
        test_status = np.asarray(test_status).reshape(-1).astype(int)

        X_train_loc = pd.DataFrame(X_train)
        X_train_loc.columns = feature_list

        X_val_loc = pd.DataFrame(X_val)
        X_val_loc.columns = feature_list

        X_test_loc = pd.DataFrame(X_test)
        X_test_loc.columns = feature_list

        # std_list = [c for c in feature_list if c.startswith('num_')]
        # if std_list:
        #     scaler = StandardScaler().fit(X_train_loc.loc[:, std_list])
        #     X_train_loc.loc[:, std_list] = scaler.transform( X_train_loc.loc[:, std_list])
        #     X_val_loc.loc[:, std_list] = scaler.transform( X_val_loc.loc[:, std_list])
        #     X_test_loc.loc[:, std_list] = scaler.transform( X_test_loc.loc[:, std_list])

        X_train_loc = np.array(X_train_loc, dtype='float32')
        X_val_loc = np.array(X_val_loc, dtype='float32')
        X_test_loc = np.array(X_test_loc, dtype='float32')

        X_train_central = np.array(X_train, dtype='float32')
        X_val_central = np.array(X_val, dtype='float32')
        X_test_central = np.array(X_test, dtype='float32')

        # training covariates, time, and status
        X_train_local_non_std.append(X_train_central)
        X_train_local_std.append(X_train_central)
        train_time_local.append(train_time)
        train_status_local.append(train_status)

        # training data length
        tr_data_len.append(X_train.shape[0])

        # validation covariates, time, and status
        X_val_local_non_std.append(X_val_central)
        X_val_local_std.append(X_val_loc)
        val_time_local.append(val_time)
        val_status_local.append(val_status)

        # test covariates, time, and status
        X_test_local_non_std.append(X_test_central)
        X_test_local_std.append(X_test_loc)
        test_time_local.append(test_time)
        test_status_local.append(test_status)

    # concatenation of training covariates
    X_train_global = np.vstack(X_train_local_non_std)
    X_train_global = np.array(X_train_global, dtype='float32')

    # concatenation of training time
    train_time_global = np.concatenate(train_time_local)
    train_time_global = np.array(train_time_global, dtype='float32')

    # concatenation of training status
    train_status_global = np.concatenate(train_status_local)
    train_status_global = np.array(train_status_global, dtype='float32')

    # concatenation of validation covariates
    X_val_global = np.vstack(X_val_local_non_std)
    X_val_global = np.array(X_val_global, dtype='float32')

    # concatenation of validation time
    val_time_global = np.concatenate(val_time_local)
    val_time_global = np.array(val_time_global, dtype='float32')

    # concatenation of validation status
    val_status_global = np.concatenate(val_status_local)
    val_status_global = np.array(val_status_global, dtype='float32')

    # concatenation of test covariates
    X_test_global = np.vstack(X_test_local_non_std)
    X_test_global = np.array(X_test_global, dtype='float32')

    # concatenation of test time
    test_time_global = np.concatenate(test_time_local)
    test_time_global = np.array(test_time_global, dtype='float32')

    # concatenation of test status
    test_status_global = np.concatenate(test_status_local)
    test_status_global = np.array(test_status_global, dtype='float32')

    # Standardize the continuous variables
    X_train_global = pd.DataFrame(X_train_global)
    X_train_global.columns = feature_list

    X_val_global = pd.DataFrame(X_val_global)
    X_val_global.columns = feature_list

    X_test_global = pd.DataFrame(X_test_global)
    X_test_global.columns = feature_list

    # Standardize the continuous variables
    # if std_list:
    #     scaler = StandardScaler().fit(X_train_global.loc[:, std_list])
    #     X_train_global.loc[:, std_list] = scaler.transform( X_train_global.loc[:, std_list])
    #     X_val_global.loc[:, std_list] = scaler.transform( X_val_global.loc[:, std_list])
    #     X_test_global.loc[:, std_list] = scaler.transform( X_test_global.loc[:, std_list])

    X_train_global = np.array(X_train_global, dtype='float32')
    X_val_global = np.array(X_val_global, dtype='float32')
    X_test_global = np.array(X_test_global, dtype='float32')

    print("X_val shape:", X_val_global.shape)
    print("X_test shape:", X_test_global.shape)

################################################### End Data loading ##################################################################

    
    def fix_cindex_direction(cindex):
        if np.isfinite(cindex) and cindex < 0.5:
            return 1.0 - cindex
        return cindex

############################################### Model Training & Evaluation ##############################################################################
    if args.alg == 'all_in':

        memory = mem_mb()
        print("Before:", memory, "MB")

        test_cindex, test_brier, test_nbll, test_p_value, test_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan
        nets = init_nets(input_size=X_train_global.shape[1], n_parties=args.n_parties, args=args)
        try:
            test_cindex, test_brier, test_nbll, test_p_value, test_MAE_PO = train_net_all(0, nets[0].to(args.device), X_train_global, train_time_global, train_status_global, X_val_global, val_time_global, val_status_global, X_test_global, test_time_global, test_status_global, pseudo_evaltime, feature_list, args, args.device)
        except Exception as e:
            print(  f"[Exception in train_net_all method]:  {type(e).__name__}: {e}")
            test_cindex, test_brier, test_nbll, test_p_value, test_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan
 

        final_test_cidx = test_cindex
        final_test_brier = test_brier
        final_test_nbll = test_nbll
        final_test_p_value = test_p_value
        final_test_weighted_MAE_PO = test_MAE_PO

        print('>> All in Test Cindex: {}'.format(test_cindex))
        print('>> All in Test Brier: {}'.format(test_brier))
        print('>> All in Test NBLL: {}'.format(test_nbll))
        print('>> All in Test P-Value: {}'.format(test_p_value))
        print('>> All in Test Weighted MAE PO: {}'.format(test_MAE_PO))

        memory = mem_mb()
        print("After:", memory, "MB")

    elif args.alg == 'fedavg':
        print(f"Initializing nets. Seed: {args.init_seed}")

        memory = mem_mb()
        print("Before:", memory, "MB")

        nets = init_nets(input_size=X_train_local_std[0].shape[1], n_parties=args.n_parties, args=args)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, directory=args.modeldir, checkpoint_pth=f'AFT_Weibull_model_run_{args.run_id}.tar')

        global_models = init_nets( input_size=X_train_local_std[0].shape[1],  n_parties=1, args=args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        global_val_cindex = 0.0
        test_cindex, test_brier, test_nbll, test_p_value, test_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan
        comm_round_done = 0

        for comm_round in range(args.comm_round):
            print("in comm round:" + str(comm_round))
            arr = np.arange(args.n_parties)

            # np.random.shuffle(arr)
            selected = arr  # arr[:int(args.n_parties * args.sample)]

            
            global_para = deepcopy(global_model.state_dict())
            for idx in selected:
                if nets[idx] is None:
                    nets[idx] = init_nets( input_size=X_train_local_std[0].shape[1], n_parties=1, args=args)[0]
                    
                nets[idx].load_state_dict( global_para, strict=True)
                nets[idx].to(args.device)
            
             # 2. Clients train locally with local early stopping 
            nets = local_train_net(nets, selected, args, X_train_local_std, train_time_local, train_status_local, X_val_local_std,  val_time_local, val_status_local, pseudo_evaltime, feature_list, device=args.device)
            
            # Keep only clients that produced valid local models.
            valid_clients = [client_id for client_id in selected if nets[client_id] is not None]
            total_data_points = sum(tr_data_len[client_id] for client_id in valid_clients)

            new_global_para = None 
            for client_id in valid_clients:
                aggregation_weight = ( tr_data_len[client_id] / total_data_points ) 
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
                global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_MAE_PO = evaluate_global_model_on_clients( global_model, eval_clients, X_val_local_std, val_time_local,  val_status_local,  X_test_local_std,  test_time_local,  test_status_local,  train_time_local,  train_status_local, pseudo_evaltime,  device=args.device)
            except Exception as e:
                print( f"[Exception in train_net method]:  {type(e).__name__}: {e}")
                traceback.print_exc()
                global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_MAE_PO= np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
            # test_cindex, test_brier, test_nbll
            improved = False

            if  global_val_cindex is not None and np.isfinite(global_val_cindex):
                
                current_global_para = {
                    key: value.detach().cpu().clone()
                    for key, value in global_model.state_dict().items()
                }

                improved = early_stopping(
                    global_val_cindex,
                    current_global_para,
                    round_idx=comm_round,
                    test_metrics={
                        "test_c_index": test_cindex,
                        "test_brier": test_brier,
                        "test_nbll": test_nbll,
                        "test_p_value": test_p_value,
                        "test_MAE_PO": test_MAE_PO,
                    },
                )
            
            else:
                print(f"Round {comm_round}: global validation C-index is undefined; skipping early stopping.")

            if improved:
                print(
                    f"New best model selected at round {comm_round}: "
                    f"val_cindex={global_val_cindex:.4f}, "
                    f"test_cindex={test_cindex:.4f}, "
                    f"test_brier={test_brier:.4f}, "
                    f"test_nbll={test_nbll:.4f}, "
                    f"test_p_value={test_p_value:.4f}, "
                    f"test_weighted_MAE_PO={test_MAE_PO:.4f}"
                ) 
            
            
            comm_round_done = comm_round
            if early_stopping.early_stop:
                comm_round_done = comm_round
                print("Early stopping")
                break

            print(
                f">> Weighted Test metrics: "
                f"C-index={test_cindex:.4f}, "
                f"Brier={test_brier:.6f}, "
                f"NBLL={test_nbll:.6f}"
                f"p-value={test_p_value:.6f}"
                f"MAE PO={test_MAE_PO:.6f}"  
            )

        # report the metrics from the best validation round, not the last round
        final_round = early_stopping.best_round if early_stopping.best_round is not None else comm_round
        final_val_cidx = early_stopping.best_val_cindex if np.isfinite(early_stopping.best_val_cindex) else global_val_cindex 
        final_test_cidx = early_stopping.best_test_cindex if early_stopping.best_test_cindex is not None else test_cindex  
        final_test_brier = early_stopping.best_test_brier if early_stopping.best_test_brier is not None else test_brier
        final_test_nbll = early_stopping.best_test_nbll if early_stopping.best_test_nbll is not None else test_nbll 
        final_test_p_value = early_stopping.best_test_p_value if early_stopping.best_test_p_value is not None else test_p_value 
        final_test_weighted_MAE_PO = early_stopping.best_test_MAE_PO if early_stopping.best_test_MAE_PO is not None else test_MAE_PO 
        
        final_val_cidx =  final_val_cidx
        final_test_cidx = final_test_cidx
          
######################################## Centralized Training ######################################

    if args.alg == 'local_training':
        print("Initializing nets")
        nets = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, X_train_local_std, train_pseudo_local, train_time_local, train_status_local, X_val_local_std, val_pseudo_local,
                        val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local, pseudo_evaltime, device=args.device)

    end = time.time()
    end_counter = time.perf_counter()

    total_time = end - start
    total_time_counter = end_counter - start_counter

    # total time taken
    print(f"Runtime of the {args.model} model is {end - start} sec")


    os.makedirs(args.logdir, exist_ok=True)
    csv_path = os.path.join(args.logdir, "results_server_side_early.csv")

    epoch = args.epochs
    if args.alg == "all_in":
        comm_round_done = 1
        # final_test_cidx = weighted_test_cindex
        # final_test_brier = weighted_test_brier
        # final_test_nbll = weighted_test_nbll

    row = {
        "seed": int(args.init_seed),
        "fold_id": int(args.run_id),
        "dataset": args.dataset,
        "partition": args.partition,
        "comm_round_done": comm_round_done,
        "model": args.model,
        "c_idx": builtins.round(float(final_test_cidx), 4),
        "brier": builtins.round(float(final_test_brier), 4),
        "nbll": builtins.round(float(final_test_nbll), 4),
        "p_value": builtins.round(float(final_test_p_value), 4),
        "weighted_MAE_PO": builtins.round(float(final_test_weighted_MAE_PO), 4),
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
