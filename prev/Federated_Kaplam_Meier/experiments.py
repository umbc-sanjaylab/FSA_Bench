import numpy as np
import torch
import argparse
import os
from math import * 
from utils import *
import time 
from model import init_nets
from local_clients import train_net, local_train_net
import os
import psutil 
import traceback

proc = psutil.Process(os.getpid())


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DeepHit',  help='neural network used in training')
    parser.add_argument('--dataset', type=str,  default='metabric', help='dataset used for training')
    parser.add_argument( '--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64,  help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=50, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10,
                        help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50,
                        help='number of maximum communication round')
    parser.add_argument('--patience', type=int, default=5,
                        help='number of patience')
    parser.add_argument('--is_same_initial', type=int, default=1,
                        help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--run_id', type=int, default=0,
                        help="ID for 5-fold cross validation")
    parser.add_argument('--dropout_p', type=float, required=False,
                        default=0.1, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False,
                        default="/home/local/AD/mrahman6/0.AAAI_2023/Data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5,
                        help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False,
                        default="/home/local/AD/mrahman6/0.AAAI_2023/DeepHit/logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False,
                        default="/home/local/AD/mrahman6/0.AAAI_2023/DeepHit/models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run the program')
    parser.add_argument('--log_file_name', type=str,
                        default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str,
                        default='adam', help='the optimizer')
    parser.add_argument('--rho', type=float, default=0,
                        help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=0.8,
                        help='Sample ratio for each communication round')
    parser.add_argument('--pseudo_perc_interval', type=int,
                        default=10,  help='Interval for pseudo values calculation')
    parser.add_argument('--pseudo_perc_end_interval', type=int,
                        default=90,  help='Interval for pseudo values calculation')
    parser.add_argument('--epsilon', type=float, default=5.0, help='epsilon')
    parser.add_argument('--num_duration', type=int, default=50,
                        help='Number of interval in DeepHit')
    args = parser.parse_args()
    return args


################################################### Evaluation Time Points ###########################################################

def compute_eval_time(args): 
    local_clients_time = []
    local_clients_status = []

    for client_id in range(args.n_parties):
        train_df, val_df, _, _, _ = get_local_dataset(args.dataset, args.partition, client_id, args.run_id, args)
        train_time, train_status = train_df["time"], train_df["event"]
        val_time, val_status = val_df["time"], val_df["event"]

        local_clients_time.append(np.concatenate([train_time, val_time]))
        local_clients_status.append(np.concatenate([train_status, val_status]))
 

    glob_unique_time = np.unique(np.concatenate(local_clients_time))
    pseudo_evaltime = np.percentile(glob_unique_time, np.arange(args.pseudo_perc_interval, args.pseudo_perc_end_interval, args.pseudo_perc_interval))

    return pseudo_evaltime

################################################### Global Server Training ###########################################################


def mem_mb():
    return proc.memory_info().rss / (1024**2)


if __name__ == '__main__':
    args = get_args()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    print('dataset:', args.dataset)
    start = time.time()
    start_counter = time.perf_counter()

    # Combined the datasets from clients for centralized training
    train_time_all = []
    train_status_all = []

    val_time_all = []
    val_status_all = []

    test_time_all = []
    test_status_all = []

    train_time_local = []
    train_status_local = []

    val_time_local = []
    val_status_local = []

    test_time_local = []
    test_status_local = []
    train_data_length = []

    #######################################

    for party_id in range(args.n_parties): 
        train_df, val_df, test_df, client_name, feature_list = get_local_dataset(args.dataset, args.partition, party_id, args.run_id, args)
        std_list = [c for c in feature_list if c.startswith('num_')]

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
        train_time_local.append(train_time)
        train_status_local.append(train_status)

        val_time_local.append(val_time)
        val_status_local.append(val_status)

        test_time_local.append(test_time)
        test_status_local.append(test_status)
        train_data_length.append(train_time.shape[0])

    evaltime = compute_eval_time(args)
    train_time_all = np.concatenate(train_time_local)
    train_time_all = np.array(train_time_all, dtype='float32')

    train_status_all = np.concatenate(train_status_local)
    train_status_all = np.array(train_status_all, dtype='float32')

    val_time_all = np.concatenate(val_time_local)
    val_time_all = np.array(val_time_all, dtype='float32')

    val_status_all = np.concatenate(val_status_local)
    val_status_all = np.array(val_status_all, dtype='float32')

    test_time_all = np.concatenate(test_time_local)
    test_time_all = np.array(test_time_all, dtype='float32')

    test_status_all = np.concatenate(test_status_local)
    test_status_all = np.array(test_status_all, dtype='float32')

    ######################################## FedAvg ######################################
    if args.alg == 'all_in': 
        memory = mem_mb()
        print("Before:", memory, "MB")
         
        test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan
        try:  
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO  = train_net(train_time_all, train_status_all, val_time_all, val_status_all, test_time_all, test_status_all, evaltime)
        except Exception as e:
            print(f"[Exception in train_net_all method]:  {type(e).__name__}: {e}")
            traceback.print_exc()

        memory = mem_mb()
        print("After:", memory, "MB")

        logger.info('>> All in Test Cindex: {}'.format(test_cindex))
        logger.info('>> All in Test Brier: {}'.format(test_brier))
        logger.info('>> All in Test NLL: {}'.format(test_nbll)) 
        logger.info('>> All in Test P-Value: {}'.format(test_p_value))
        logger.info('>> All in Test Weighted MAE PO: {}'.format(test_weighted_MAE_PO)) 
        
        print('>> All in Test Cindex: {}'.format(test_cindex))
        print('>> All in Test Brier: {}'.format(test_brier)) 
        print('>> All in Test NLL:: {}'.format(test_cindex))
        print('>> All in Test P-Value: {}'.format(test_brier))  
        print('>> All in Test Weighted MAE PO: {}'.format(test_brier))

######################################## Centralized Training ######################################
 

    elif args.alg == 'fedavg':
        logger.info("Initializing nets")

        memory = mem_mb()
        print("Before:", memory, "MB")

        # def local_train_net(train_time_local, train_status_local, val_time_local, val_status_local, test_time_local, test_status_local, evaltime):
        
        try:
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = local_train_net(train_time_local, train_status_local, val_time_local, val_status_local, test_time_local, test_status_local, evaltime) 
        except Exception as e:
            print(f"[Exception in local_train_net method]:  {type(e).__name__}: {e}")
            traceback.print_exc()
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan

        print('>> Global Model Test Cindex1: {}'.format(test_cindex))
        print('>> Global Model Test Brier2: {}'.format(test_brier))

        memory = mem_mb()
        print("After:", memory, "MB")

    end = time.time()
    end_counter = time.perf_counter()

    total_time = end - start
    total_time_counter = end_counter - start_counter

    import builtins
    csv_path = os.path.join(args.logdir, "results_early.csv")

    row = {
        "seed": int(args.init_seed),
        "fold_id": int(args.run_id),
        "dataset": args.dataset,
        "partition": args.partition,
        "model": args.model,
        "c_idx": builtins.round(float(test_cindex), 4),
        "brier": builtins.round(float(test_brier), 4),
        "nbll": builtins.round(float(test_nbll), 4),
        "p_value": builtins.round(float(test_p_value), 4),
        "weighted_MAE_PO": builtins.round(float(test_weighted_MAE_PO), 4),
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
