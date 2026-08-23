import traceback
import psutil
import time
from sklearn.model_selection import StratifiedKFold
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri, pandas2ri, default_converter
from model import init_nets
from sklearn.preprocessing import StandardScaler
from math import *
import warnings
from copy import deepcopy
import rpy2.robjects as ro
from local_clients import train_net_all, local_train_net, evaluate_global_model_on_clients
from utils import *
import os
import logging
import argparse
import numpy as np 
import torch
import json
from datetime import datetime 
import gc
gc.collect()
torch.cuda.empty_cache()
warnings.filterwarnings('ignore')

proc = psutil.Process(os.getpid())

################################################### Get Arguments ###########################################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DeepPseudo',  help='neural network used in training')
    parser.add_argument('--dataset', type=str,  default='metabric', help='dataset used for training')
    parser.add_argument( '--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=100, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=5,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg', help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=10, help='number of maximum communication round')
    parser.add_argument('--patience', type=int, default=20, help='number of patience')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs for centralized settings')
    parser.add_argument('--is_same_initial', type=int, default=1,  help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--run_id', type=int, default=0, help="ID for 5-fold cross validation")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False,  default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False,  default="./models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda:0',  help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str,  default='adam', help='the optimizer')
    parser.add_argument('--sample', type=float, default=1,  help='Sample ratio for each communication round')
    parser.add_argument('--pseudo_perc_interval', type=int,  default=10,  help='Interval for pseudo values calculation')
    parser.add_argument('--sensitivity', type=float,  default=2.0, help='sensitivity')
    parser.add_argument('--epsilon', type=float, default=5.0, help='epsilon')
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy')

    args = parser.parse_args()
    return args


def pseudoFunction(data_py, eval_time_py):
    """Calculate the pseudo values for survival function.
    Arguments:
        data: A dataframe containing survival time and event status.
        evaltime: Evaluation times at which pseudo values are calcuated
    Returns:
        A dataframe of pseudo values for all subjects in the data at the evaluation time points. 
    """
    r = ro.r
    r.source(
        '/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepPseudo/Compute_pseudo_values.R')
    data_py = data_py.copy()

    with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter):
        data_r = ro.conversion.py2rpy(data_py)           # pandas -> R df
        eval_time_r = ro.conversion.py2rpy(eval_time_py)  # numpy -> R vector
        
        # call R inside context
        ps_r = r.pseudo_CRA(data_r, eval_time_r)
        ps_py = ro.conversion.rpy2py(ps_r)               # R -> numpy

    ps_py = np.asarray(ps_py)

    # Convert (N,1,T) -> (N,T)
    if ps_py.ndim == 3 and ps_py.shape[1] == 1:
        ps_py = ps_py[:, 0, :]

    return ps_py


def mem_mb():
    return proc.memory_info().rss / (1024**2)


def compute_eval_time(args):
    local_unique_times = []
    
    

    for client_id in range(args.n_parties):
        train_df, val_df, _, _, _ = get_local_dataset( args.dataset, args.partition, client_id, args.run_id, args)
        column = 'status' if 'status' in train_df.columns else 'event'
        train_time, _ = train_df["time"], train_df[column]

        val_time, _ = val_df["time"], val_df[column]
        local_unique_times.append( np.unique(np.concatenate([train_time, val_time])))

    glob_unique_time = np.unique(np.concatenate(local_unique_times))

    # generate EXACTLY N evaluation points
    percentiles = np.linspace(0, 100, args.pseudo_perc_interval + 2)[1:-1]
    pseudo_evaltime = np.percentile(glob_unique_time, percentiles) 
    return pseudo_evaltime

################################################### Global Server Training ###########################################################


if __name__ == '__main__':  
    args = get_args() 
    
    # logger file name 
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    argument_path = f"experiment_arguments-{timestamp}.json"
    log_path = f"experiment_log-{timestamp}.log"

    # Save experiment arguments
    with open(os.path.join(args.logdir, argument_path), "w") as f:
        json.dump(vars(args), f, indent=4, default=str) 

    # Remove old handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logger
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%m-%d %H:%M", filemode="w" )

    logger = logging.getLogger("experiment")
    logger.info(args)  

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    logger.info(f'dataset: {args.dataset}')
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
    logger.info(pseudo_evaltime)

    for party_id in range(args.n_parties):
        train_df, val_df, test_df, client_name, feature_list = get_local_dataset( args.dataset, args.partition, party_id, args.run_id, args)
        column = 'status' if 'status' in train_df.columns else 'event'
        
        feat_cols = train_df.drop(columns=["time", column]).columns.tolist()
        assert feat_cols == feature_list, "feature order mismatch"

        std_list = [c for c in feature_list if c.startswith('num_')]
        client_name_array.append(client_name) 
        
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

        # Create DataFrame with time and status for pseudo values calculation
        central_train_data = pd.DataFrame(pd.concat([pd.DataFrame(train_time), pd.DataFrame(train_status)], axis=1))
        central_train_data.columns = ['time', 'status']
        
        central_val_data = pd.DataFrame(  pd.concat([pd.DataFrame(val_time), pd.DataFrame(val_status)], axis=1))
        central_val_data.columns = ['time', 'status']
        
        # Calculate Pseudo values at prespecified evaltime
        central_data = pd.concat( [central_train_data, central_val_data], axis=0)
        pseudo_central = pseudoFunction(central_data, pseudo_evaltime)
         
        pseudo_train=pseudo_central[:len(central_train_data),:]
        pseudo_val=pseudo_central[len(central_train_data):,:]
        
        # pseudo_train = pseudoFunction(central_train_data, pseudo_evaltime)
        # pseudo_val = pseudoFunction(central_val_data, pseudo_evaltime)
        
        #########################################################################################

        # Convert pseudo values to numpy float32
        train_pseudo = np.array(pseudo_train, dtype='float32')
        val_pseudo = np.array(pseudo_val, dtype='float32')

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

        # Append the data from each of the clients
        X_train_local_non_std.append(X_train_central)
        X_train_local_std.append(X_train_loc)

        train_pseudo_local.append(train_pseudo)
        train_time_local.append(train_time)

        train_status_local.append(train_status)
        tr_data_len.append(X_train.shape[0])

        X_val_local_non_std.append(X_val_central)
        X_val_local_std.append(X_val_loc)

        val_pseudo_local.append(val_pseudo)
        val_time_local.append(val_time)
        val_status_local.append(val_status)

        X_test_local_non_std.append(X_test_central)
        X_test_local_std.append(X_test_loc)
        
        test_time_local.append(test_time)
        test_status_local.append(test_status)

    X_train_global = np.vstack(X_train_local_non_std)
    X_train_global = np.array(X_train_global, dtype='float32')

    train_time_global = np.concatenate(train_time_local)
    train_time_global = np.array(train_time_global, dtype='float32')

    train_status_global = np.concatenate(train_status_local)
    train_status_global = np.array(train_status_global, dtype='float32')

    X_val_global = np.vstack(X_val_local_non_std)
    X_val_global = np.array(X_val_global, dtype='float32')

    val_time_global = np.concatenate(val_time_local)
    val_time_global = np.array(val_time_global, dtype='float32')

    val_status_global = np.concatenate(val_status_local)
    val_status_global = np.array(val_status_global, dtype='float32')

    X_test_global = np.vstack(X_test_local_non_std)
    X_test_global = np.array(X_test_global, dtype='float32')

    test_time_global = np.concatenate(test_time_local)
    test_time_global = np.array(test_time_global, dtype='float32')

    test_status_global = np.concatenate(test_status_local)
    test_status_global = np.array(test_status_global, dtype='float32')

    # Create DataFrame with time and status for pseudo values calculation
    central_train_data_global = pd.DataFrame(pd.concat([pd.DataFrame(train_time_global), pd.DataFrame(train_status_global)], axis=1))
    central_train_data_global.columns = ['time', 'status']

    central_val_data_global = pd.DataFrame(pd.concat([pd.DataFrame(val_time_global), pd.DataFrame(val_status_global)], axis=1))
    central_val_data_global.columns = ['time', 'status']

    central_data_global = pd.concat([central_train_data_global, central_val_data_global])

    # Calculate Pseudo values at prespecified evaltime
    # pseudo_central_global = pseudoFunction(central_train_data_global, pseudo_evaltime)
    # ps_py = np.asarray(pseudo_central_global)
    # print(f"pseudo value array shape is {ps_py.shape}")
    
    pseudo_central_global=pseudoFunction(central_data_global, pseudo_evaltime) 
    pseudo_train_global=pseudo_central_global[:len(central_train_data_global),:]
    pseudo_val_global=pseudo_central_global[len(central_train_data_global):,:]
    
    # pseudo_train_global = pseudoFunction(central_train_data_global, pseudo_evaltime)
    # pseudo_val_global = pseudoFunction(central_val_data_global, pseudo_evaltime)

    # Convert pseudo values to numpy float32
    train_pseudo_global = np.array(pseudo_train_global, dtype='float32')
    val_pseudo_global = np.array(pseudo_val_global, dtype='float32')

    # print(train_pseudo_global)
    # print(val_pseudo_global)

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

    logger.info(f"X_val shape: {X_val_global.shape}")
    logger.info(f"X_test shape: {X_test_global.shape}")

    # Create train and validation dataloader
    train_dataloader = FastTensorDataLoader(torch.from_numpy(X_train_global.copy()), torch.from_numpy( train_pseudo_global.copy()), batch_size=args.batch_size, shuffle=True)
    val_dataloader = FastTensorDataLoader(torch.from_numpy(X_val_global.copy()), torch.from_numpy( val_pseudo_global.copy()), batch_size=args.batch_size, shuffle=False)

    print(pseudo_evaltime)

    # final_test_p_value = test_p_value
    # final_test_MAE_PO = test_weighted_MAE_PO

################################################### End Data loading ##################################################################
    if args.alg == 'all_in':
        memory = mem_mb()
        logger.info(f"Before: {memory} MB")
        logger.info(f"Device is {args.device}")

        nets = init_nets(input_size=X_train_global.shape[1], dropout_p=args.dropout_p, n_parties=1, args=args)

        try:
            test_cindex,  test_brier,  test_nbll, test_p_value, test_MAE_PO = train_net_all(0, nets[0], train_dataloader, val_dataloader, args.epochs, args.lr, args. optimizer,  train_time_global,  train_status_global, X_val_global, val_time_global, val_status_global,  X_test_global, test_time_global, test_status_global, pseudo_evaltime, args, logger, device=args.device)
        except Exception as e:
            logger.info(f"[Exception in train_net_all method]:  {type(e).__name__}: {e}")
            test_cindex,  test_brier,  test_nbll, test_p_value, test_MAE_PO  = np.nan, np.nan, np.nan, np.nan, np.nan

        memory = mem_mb()
        logger.info(f"After: {memory} MB") 
        logger.info(f"All in Test metrics: C-index={test_cindex:.4f}, Brier={test_brier:.4f} nbll: {test_nbll:.4f} p-value: {test_p_value:.4f} MAE-PO: {test_MAE_PO:.4f}")

    def fix_cindex_direction(cindex):
        if np.isfinite(cindex) and cindex < 0.5:
            return 1.0 - cindex
        return cindex
    
    
 ###################################################### Federated Average Algorithm #################################################### 
  
    if args.alg == 'fedavg':
        logger.info(f"Initializing nets. Seed: {args.init_seed}")

        memory = mem_mb()
        logger.info(f"Before: {memory} MB")

        nets = init_nets(input_size=X_train_local_std[0].shape[1], dropout_p=args.dropout_p, n_parties=args.n_parties, args=args)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, directory=args.modeldir, checkpoint_pth=f"DeepPseudo_global_model_{args.run_id}.pt")

        global_models = init_nets( input_size=X_train_local_std[0].shape[1], dropout_p=args.dropout_p, n_parties=1, args=args)
        global_model = global_models[0]

        global_para = deepcopy(global_model.state_dict()) 
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        global_val_cindex = 0.0
        test_cindex, test_brier, test_nbll = 0.0, 0.0, 0.0
        comm_round_done = 0

        for comm_round in range(args.comm_round):
            logger.info(f"in comm round: {str(comm_round)}")
            arr = np.arange(args.n_parties)

            # np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            
            # 1. Server sends current global model to selected clients
            global_para = deepcopy(global_model.state_dict())
 
            for idx in selected:
                nets[idx].load_state_dict(global_para)
            
            # 2. Clients train locally with local early stopping
            nets = local_train_net(client_name_array, nets, selected, args, X_train_local_std, train_pseudo_local, train_time_local, train_status_local, X_val_local_std, val_pseudo_local, val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local, pseudo_evaltime, tr_data_len, comm_round, logger, device=args.device)   

            # update global model using FedAvg algorithm from individual centers
            total_data_points = sum([tr_data_len[r] for r in selected])
            fed_avg_freqs = [tr_data_len[r] / total_data_points for r in selected]
            
            def state_dict_is_finite(state_dict):
                return all(
                    not tensor.is_floating_point()
                    or torch.isfinite(tensor).all().item()
                    for tensor in state_dict.values()
                )

            new_global_para = None
            for idx, client_id in enumerate(selected):
                net_para = {k: v.detach().cpu().clone() for k, v in nets[client_id].state_dict().items()}
                
                if not state_dict_is_finite(net_para):
                    logger.warning(f"Skipping client {client_id}: model contains NaN/Inf")
                    continue

                if new_global_para is None:
                    new_global_para = {k: net_para[k] * fed_avg_freqs[idx] for k in net_para}
                else:   
                    for key in net_para:
                        new_global_para[key] += net_para[key] * fed_avg_freqs[idx]

            global_model.load_state_dict(new_global_para) 
             
            # 5. Evaluate the newly aggregated global model on clients
            eval_clients = range(args.n_parties) 
            global_val_cindex, avg_test_cindex,  avg_test_brier,  avg_test_nbll, avg_test_p_value,  avg_test_MAE_PO = evaluate_global_model_on_clients(global_model, eval_clients, X_val_local_std,  val_time_local, val_status_local,  X_test_local_std,  test_time_local,  test_status_local, train_time_local, train_status_local, pseudo_evaltime, logger, device=args.device) 
            
            # test_cindex, test_brier, test_nbll
            improved = early_stopping(
                global_val_cindex,
                new_global_para,
                round_idx=comm_round,
                test_metrics={
                    "test_c_index": avg_test_cindex,
                    "test_brier": avg_test_brier,
                    "test_nbll": avg_test_nbll,
                    "test_p_value": avg_test_p_value,
                    "test_MAE_PO": avg_test_MAE_PO
                },
            )

            if improved:
                logger.info(
                    f"New best model selected at round {comm_round}: "
                    f"val_cindex={global_val_cindex:.4f}, "
                    f"test_cindex={avg_test_cindex:.4f}, "
                    f"test_brier={avg_test_brier:.4f}, "
                    f"test_nbll={avg_test_nbll:.4f}"
                    f"test_p_value={avg_test_p_value:.4f}"
                    f"test_MAE_PO={avg_test_MAE_PO:.4f}"
                )
                
            comm_round_done = comm_round
            if early_stopping.early_stop:
                comm_round_done = comm_round
                logger.info("Early stopping")
                break

            logger.info(
                f">> Weighted Test metrics: "
                f"C-index={avg_test_cindex:.4f}, "
                f"Brier={avg_test_brier:.6f}, "
                f"NBLL={avg_test_nbll:.6f}"
                f"P-Value={avg_test_p_value:.6f}"
                f"MAE-PO={avg_test_MAE_PO:.6f}"
            )

            memory = mem_mb()
            logger.info(f"After: {memory:.2f} MB")

        # report the metrics from the best validation round, not the last round
        final_round = early_stopping.best_round if early_stopping.best_round is not None else comm_round
        final_val_cidx =  early_stopping.best_val_cindex if np.isfinite(early_stopping.best_val_cindex) else global_val_cindex  
        
        final_test_cidx = early_stopping.best_test_cindex if early_stopping.best_test_cindex is not None else avg_test_cindex    
        final_test_brier = early_stopping.best_test_brier if early_stopping.best_test_brier is not None else avg_test_brier  
        final_test_nbll =  early_stopping.best_test_nbll if early_stopping.best_test_nbll is not None else avg_test_nbll  
        final_test_p_value = early_stopping.best_test_p_value if early_stopping.best_test_p_value is not None else avg_test_p_value
        final_test_MAE_PO =  early_stopping.best_test_MAE_PO if early_stopping.best_test_MAE_PO is not None else avg_test_MAE_PO
         
######################################## local Training ######################################

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
    logger.info(f"Runtime of the {args.model} model is {end - start} sec")

    import builtins
    import os

    os.makedirs(args.logdir, exist_ok=True)
    csv_path = os.path.join(args.logdir, "results_both_side_early.csv")

    epoch = args.epochs
    if args.alg == "all_in":
        comm_round_done = 1
        final_test_cidx = test_cindex
        final_test_brier = test_brier 
        final_test_nbll = test_nbll
        final_test_p_value = test_p_value
        final_test_MAE_PO = test_MAE_PO

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
        "MAE_PO": builtins.round(float(final_test_MAE_PO), 4),
        "lr": builtins.round(float(args.lr), 5),
        "batch_size": int(args.batch_size),
        "epochs": int(epoch),
        "dropout": float(args.dropout_p),
        "time": builtins.round(float(total_time), 4),
        "time_counter": builtins.round(float(total_time_counter), 4),
        "memory": memory,
    }

    df = pd.DataFrame([row])

    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
