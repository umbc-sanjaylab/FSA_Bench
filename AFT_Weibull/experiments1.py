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
from federated.aggregation import fedavg
from config import get_args
import numpy as np
from datasets.data_processing import prepare_survival_data, get_local_dataset
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
    data = prepare_survival_data(args)

    X_train = data.centralized.train.X
    train_time = data.centralized.train.time
    train_event = data.centralized.train.event

    X_val = data.centralized.val.X
    val_time = data.centralized.val.time
    val_event = data.centralized.val.event

    X_test = data.centralized.test.X
    test_time = data.centralized.test.time
    test_event = data.centralized.test.event

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
            valid_client_sizes = sum(tr_data_len[client_id] for client_id in valid_clients)

            

            
            valid_models = [nets[client_id] for client_id in valid_clients]
            global_state = fedavg(valid_models, valid_client_sizes) 
            global_model.load_state_dict(global_state)
             
            # 5. Evaluate the newly aggregated global model on clients
            eval_clients = list(range(args.n_parties))
            
            try:
                global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_MAE_PO = evaluate_global_model_on_clients(global_model, eval_clients, X_val_local_std, val_time_local,  val_status_local,  X_test_local_std,  test_time_local,  test_status_local,  train_time_local,  train_status_local, pseudo_evaltime,  device=args.device)
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
            global_state = fedavg([nets[client_id] for client_id in valid_clients], valid_client_sizes)             global_state = fedavg([nets[client_id] for client_id in valid_clients], valid_client_sizes) 