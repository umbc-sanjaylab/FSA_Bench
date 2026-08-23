import numpy as np
import argparse
import os
from math import *
from local_clients import *
from utils import *
from local_clients import train_net_all
import traceback
import builtins
import os
import psutil
from copy import deepcopy  
from sklearn.preprocessing import StandardScaler
from model import init_nets
import time
proc = psutil.Process(os.getpid())


def mem_mb():
    return proc.memory_info().rss / (1024**2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DeepSurv', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='metabric', help='dataset used for training')
    parser.add_argument( '--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo',  help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01,  help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5,  help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',  help='communication strategy: fedavg/fedprox')
    parser.add_argument('--epsilon', type=float, default=5.0, help='epsilon')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--patience', type=int, default=10, help='number of patience')
    parser.add_argument('--eval_every', type=int, default=5, help='compute C-index every N epochs')
    parser.add_argument('--is_same_initial', type=int, default=1,  help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--run_id', type=int, default=0,  help="Fold number in cross-fold validation")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="/home/local/AD/mrahman6/0.AAAI_2023/Data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5,  help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="/home/local/AD/mrahman6/0.AAAI_2023/Cox_CRA/logs/", help='Log directory path')
    parser.add_argument('--csvdir', type=str, required=False, default="/home/local/AD/mrahman6/0.AAAI_2023/Cox_CRA/csv/", help='CSV directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="/home/local/AD/mrahman6/0.AAAI_2023/Cox_CRA/models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0',  help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1,  help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0,  help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level',  help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0,   help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1,   help='Sample ratio for each communication round')
    parser.add_argument('--time_horizon_interval', type=float,  default=1.0, help='Time horizon interval (default: 1.0)')
    parser.add_argument('--pseudo_perc_interval', type=int,  default=10,  help='Interval for pseudo values calculation')
    args = parser.parse_args()
    return args


def compute_eval_time(args):
    local_unique_times = []

    for client_id in range(args.n_parties):
        train_df, val_df, _, _, _ = get_local_dataset(args.dataset, args.partition, client_id, args.run_id, args)
        train_time, _ = train_df["time"], train_df["event"]

        val_time, _ = val_df["time"], val_df["event"]
        local_unique_times.append(
            np.unique(np.concatenate([train_time, val_time])))

    glob_unique_time = np.unique(np.concatenate(local_unique_times))
    pseudo_evaltime = np.percentile(glob_unique_time, np.arange(
        args.pseudo_perc_interval, 99, args.pseudo_perc_interval))
    return pseudo_evaltime

#############################################################################################################################
################################################### Global Server Training ###########################################################


if __name__ == '__main__':
    args = get_args()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    mkdirs(args.logdir)
    mkdirs(args.modeldir)

    seed = args.init_seed
    print("#" * 100)
    print(f"Partitioning data: {args.dataset} {args.init_seed}")
    print('dataset:', args.dataset)

    start = time.time()
    start_counter = time.perf_counter()

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

    tr_data_len = []
    client_name_array = []
    client_name = None

    for client_id in range(args.n_parties):
        train_df, val_df, test_df, client_name, feature_list = get_local_dataset(args.dataset, args.partition, client_id, args.run_id, args)
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

        if client_id == 1:
            print(X_train_loc.head(2))

        X_train_loc = np.array(X_train_loc, dtype='float32')
        X_val_loc = np.array(X_val_loc, dtype='float32')
        X_test_loc = np.array(X_test_loc, dtype='float32')

        X_train_local_std.append(X_train_loc)
        X_val_local_std.append(X_val_loc)
        X_test_local_std.append(X_test_loc)
        train_data_length.append(X_train_loc.shape[0])

    print("data is initialzied")
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
    print("centralzied data is formed")

    X_train_all = np.vstack(X_train_all)
    X_train_all = np.array(X_train_all, dtype='float32')

    X_val_all = np.vstack(X_val_all)
    X_val_all = np.array(X_val_all, dtype='float32')

    X_test_all = np.vstack(X_test_all)
    X_test_all = np.array(X_test_all, dtype='float32')

    df_train1_all = pd.concat([pd.DataFrame(train_time_all), pd.DataFrame(train_status_all), pd.DataFrame(X_train_all)], axis=1)
    df_val1_all = pd.concat([pd.DataFrame(val_time_all), pd.DataFrame( val_status_all), pd.DataFrame(X_val_all)], axis=1)
    df_test1_all = pd.concat([pd.DataFrame(test_time_all), pd.DataFrame(test_status_all), pd.DataFrame(X_test_all)], axis=1)

    # Get the targets (time and status)
    def get_target(df): return (df.iloc[:, 0].values, df.iloc[:, 1].values)
    y_train_all = get_target(df_train1_all)
    y_val_all = get_target(df_val1_all)

    # Create the validation set
    val_all = X_val_all, y_val_all
     

    # Evaluation time points
    evaltime = compute_eval_time(args)

    ######################################## Centralized ######################################
    start = time.time()
    if args.alg == 'all_in':
      
        memory = mem_mb()
        print("Before:", memory, "MB")

        nets, local_model_meta_data, layer_type = init_nets(X_train_all.shape[1], args.dropout_p, 1, args)
        print("model is initialized")

        n_epoch = args.epochs
        try:
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = train_net_all( 0, nets[0], X_train_all, y_train_all, val_all, args.epochs, args.lr,  args.optimizer, train_time_all, train_status_all, X_val_all, val_time_all,  val_status_all, X_test_all, test_time_all, test_status_all, evaltime, args=args, device=args.device)
        except Exception as e:
            print("Exception in train_net_all")
            traceback.print_exc()
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan

        print('>> All in Test Cindex: {}'.format(test_cindex))
        print('>> All in Test Brier: {}'.format(test_brier))
        print('>> All in Test NBLL: {}'.format(test_nbll))
        print('>> All in Test D-calibration: {}'.format(test_p_value))
        print('>> All in Test Weighted MAE PO: {}'.format(test_weighted_MAE_PO))

        memory = mem_mb()
        print("After:", memory, "MB")

    ######################################## Centralized ######################################
    elif args.alg == 'fedavg':

        memory = mem_mb()
        print("Before:", memory, "MB")

        print("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets( X_train_local_std[0].shape[1], args.dropout_p, args.n_parties, args)

        # early_stopping = EarlyStopping(patience=args.patience, verbose=True, checkpoint_pth=f'{args.modeldir}/Fed_CoxPH_model_run_{args.run_id}.tar')
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, directory=args.modeldir, checkpoint_pth=f"{args.model}_model_run_{args.run_id}.pt")

        global_models, global_model_meta_data, global_layer_type = init_nets( X_train_local_std[0].shape[1], args.dropout_p, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        global_c_index = 0.0
        comm_round_done = 0

        test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = 0.0, 0.0, 0.0, 0.0, 0.0

        for comm_round in range(args.comm_round):
            print("in comm round:" + str(comm_round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)

            selected = arr[:int(args.n_parties * args.sample)] 
            global_para = deepcopy(global_model.cpu().state_dict())   
            
            for client_id in selected:
              if nets[client_id] is None: 
                  reinitialized_nets, global_model_meta_data, global_layer_type = init_nets( X_train_local_std[0].shape[1], args.dropout_p, 1, args)
                  nets[client_id] = reinitialized_nets[0]

              nets[client_id].load_state_dict(global_para) 
 
            nets = local_train_net( nets, client_name_array, selected, args, X_train_local_std, train_time_local, train_status_local, X_val_local_std,  val_time_local,  val_status_local, X_test_local_std, test_time_local, test_status_local, evaltime, train_data_length, comm_round, args.device)  

            valid_clients = [client_id for client_id in selected if nets[client_id] is not None]
            total_data_points = sum(tr_data_len[client_id] for client_id in valid_clients)
            
            new_global_para = None 
            for client_id in valid_clients:
                aggregation_weight = (tr_data_len[client_id] / total_data_points ) 
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
            
            try:
                global_c_index, test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO =  evaluate_global_model_on_clients(global_model, selected,  X_train_local_std, train_time_local, train_status_local, X_val_local_std, val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local,  evaltime, args, device="cpu") 
                
            except Exception as e:
                print("Exception in train_net_all")
                traceback.print_exc()
                global_c_index, test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
            evaluated_global_para = deepcopy(global_model.cpu().state_dict())
            
            # test_cindex, test_brier, test_nbll
            improved = early_stopping(
                global_c_index,
                evaluated_global_para,
                round_idx=comm_round,
                test_metrics={
                    "test_c_index": test_cindex,
                    "test_brier": test_brier,
                    "test_nbll": test_nbll,
                    "test_p_value": test_p_value,
                    "test_MAE_PO": test_weighted_MAE_PO,
                },
            )
 
            print(f"New best model selected at round {comm_round}: ", 
                      f"val_cindex={global_c_index:.4f}, ",
                      f"test_cindex={test_cindex:.4f}, ",
                      f"test_brier={test_brier:.4f}, ",
                      f"test_nbll={test_nbll:.4f}, ",
                      f"test_p_value={test_p_value:.4f}, ",
                      f"test_weighted_MAE_PO={test_weighted_MAE_PO:.4f}"
                      )

            if improved:
                    comm_round_done = comm_round

            if early_stopping.early_stop:
                comm_round_done = comm_round
                print("Early stopping")
                break 

            print('>> Global Model Test Cindex1: {}'.format(test_cindex))
            print('>> Global Model Test Brier2: {}'.format(test_brier))

            memory = mem_mb()
            print("After:", memory, "MB")

        print('>> Global Model Test Cindex1: {}'.format(test_cindex))
        print('>> Global Model Test Brier2: {}'.format(test_brier)) 

        # report the metrics from the best validation round, not the last round
        final_round = early_stopping.best_round if early_stopping.best_round is not None else comm_round
        final_val_cidx = early_stopping.best_val_cindex if np.isfinite(early_stopping.best_val_cindex) else global_c_index 
        final_test_cidx = early_stopping.best_test_cindex if early_stopping.best_test_cindex is not None  and np.isfinite(early_stopping.best_test_cindex) else test_cindex  
        final_test_brier = early_stopping.best_test_brier if early_stopping.best_test_brier is not None and np.isfinite(early_stopping.best_test_brier) else test_brier  
        
        final_test_nbll = early_stopping.best_test_nbll if early_stopping.best_test_nbll is not None and np.isfinite(early_stopping.best_test_nbll) else test_nbll  
        final_test_p_value = early_stopping.best_test_p_value if early_stopping.best_test_p_value is not None and np.isfinite(early_stopping.best_test_p_value) else test_p_value  
        final_test_weighted_MAE_PO = early_stopping.best_test_MAE_PO if early_stopping.best_test_MAE_PO is not None and np.isfinite(early_stopping.best_test_MAE_PO) else test_weighted_MAE_PO  
        test_cindex,  test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = final_test_cidx, final_test_brier, final_test_nbll, final_test_p_value, final_test_weighted_MAE_PO

    print("ARGS USED:", args.init_seed, args.dataset, args.partition, args.model, args.lr, args.batch_size)
    print("CSV:", os.path.join(args.datadir, "results.csv"))

    end = time.time()
    end_counter = time.perf_counter()

    total_time = end - start
    total_time_counter = end_counter - start_counter

    csv_path = os.path.join(args.logdir, "results.csv")
    print(csv_path)

    row = {
        "seed": args.init_seed,
        "fold_id": int(args.run_id),
        "dataset": args.dataset,
        "partition": args.partition,
        "model": args.model,
        "lr": builtins.round(float(args.lr), 4),
        "batch_size": args.batch_size,
        "c_idx": builtins.round(test_cindex, 4),
        "brier": builtins.round(test_brier, 4),
        "nbll": builtins.round(test_nbll, 4),
        "p_value": builtins.round(test_p_value, 4),
        "weighted_MAE_PO": builtins.round(test_weighted_MAE_PO, 4),
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

    # end time
    end = time.time()
    # total time taken
    print(f"Runtime of the {args.model} model is {end - start} sec")
######################################## Federated  Training ######################################
