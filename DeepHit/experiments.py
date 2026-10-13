import numpy as np
import torch
import argparse
import os
from math import *
from local_clients import *
from local_clients import evaluate_global_model_on_clients
from utils import *
import time 
from model import init_nets
from local_clients import train_net_all, local_train_net
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import os
import json
from datetime import datetime 
import psutil
from copy import deepcopy  
from datasets.data_processing import prepare_survival_data
proc = psutil.Process(os.getpid()) 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DeepHit', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='metabric', help='dataset used for training')
    parser.add_argument( '--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64,  help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01,  help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=50,  help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg', help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication round')
    parser.add_argument('--patience', type=int, default=5,  help='number of patience')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--run_id', type=int, default=0,  help="ID for 5-fold cross validation")
    parser.add_argument('--dropout_p', type=float, required=False,  default=0.1, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False,  default="/home/local/AD/mrahman6/0.AAAI_2023/Data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5,   help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="/home/local/AD/mrahman6/0.AAAI_2023/DeepHit/logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False,  default="/home/local/AD/mrahman6/0.AAAI_2023/DeepHit/models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str,    default='adam', help='the optimizer')
    parser.add_argument('--rho', type=float, default=0,  help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=0.8, help='Sample ratio for each communication round')
    parser.add_argument('--pseudo_perc_interval', type=int,   default=10,  help='Interval for pseudo values calculation')
    parser.add_argument('--epsilon', type=float, default=5.0, help='epsilon')
    parser.add_argument('--num_duration', type=int, default=365, help='Number of interval in DeepHit')
    args = parser.parse_args()
    return args


################################################### Evaluation Time Points ###########################################################

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

if __name__ == '__main__':
    args = get_args()
    data = prepare_survival_data(args) 
    
    # logger file name  
    argument_path = f"experiment_arguments-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
    log_path = f"experiment_log-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

    # Save experiment arguments
    with open(os.path.join(args.logdir, argument_path), "w") as f:
        json.dump(vars(args), f, indent=4, default=str)

    # Remove old handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logger
    logging.basicConfig( filename=os.path.join(args.logdir, log_path), level=logging.INFO,  format="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%m-%d %H:%M", filemode="w" )
    logger = logging.getLogger("experiment")
    logger.info(args)

    feature_list = data.feature_names 
    X_train_local = [] 
    X_val_local = [] 
    X_test_local = []

    train_time_local = []
    train_status_local = []

    val_time_local = []
    val_status_local = []

    test_time_local = []
    test_status_local = []

    std_list = None 
    feature_list = None
    
    tr_data_len = []
    client_name_array = [] 
    local_unique_times = []
    client_censored_train = []

    for client_id in range(args.n_parties):
        client_data = data.clients[client_id]

        X_train = client_data.train.X
        train_time = client_data.train.time
        train_status = client_data.train.event

        X_val = client_data.val.X
        val_time = client_data.val.time
        val_status = client_data.val.event

        X_test = client_data.test.X
        test_time = client_data.test.time
        test_status = client_data.test.event

        X_train_local.append(X_train)
        train_time_local.append(train_time)
        train_status_local.append(train_status)

        X_val_local.append(X_val)
        val_time_local.append(val_time)
        val_status_local.append(val_status)

        X_test_local.append(X_test)
        test_time_local.append(test_time)
        test_status_local.append(test_status)
        
        tr_data_len.append(X_train.shape[0]) 
        local_unique_times.append(np.unique(np.concatenate([train_time, val_time])))
        
        num_censored_train = (train_time == 0).sum() 
        client_censored_train.append(num_censored_train) 
    
    evaltime = compute_eval_time(args)
    X_train_all = data.centralized.train.X  
    X_val_all = data.centralized.val.X  
    X_test_all = data.centralized.test.X

    train_time_all = data.centralized.train.time  
    train_status_all = data.centralized.train.event

    val_time_all = np.concatenate(val_time_local)  
    val_status_all = np.concatenate(val_status_local)  
    
    test_time_all = np.concatenate(test_time_local)  
    test_status_all = np.concatenate(test_status_local)  

    df_train_all = pd.concat([pd.DataFrame(train_time_all), pd.DataFrame(train_status_all), pd.DataFrame(X_train_all)], axis=1)
    df_val_all = pd.concat([pd.DataFrame(val_time_all), pd.DataFrame(val_status_all), pd.DataFrame(X_val_all)], axis=1)
    df_test_all = pd.concat([pd.DataFrame(test_time_all), pd.DataFrame(test_status_all), pd.DataFrame(X_test_all)], axis=1) 
    
    class LabTransform(LabTransDiscreteTime):
      def __init__(self, num_durations):
          super().__init__(num_durations)
          self.num_durations = num_durations

      def transform(self, durations, events):
        durations = np.asarray(durations, dtype=np.float64).reshape(-1)
        events = np.asarray(events).copy().reshape(-1)

        if len(durations) != len(events):
            raise ValueError("durations and events must have the same length.")

        # Handle a genuinely empty split.
        if len(durations) == 0:
            return ( np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

        if not np.isfinite(durations).all():
            raise ValueError("durations contain NaN or infinite values.")

        event_mask = events > 0

        # Pycox's transform fails when event_mask contains no True values.
        # Manually discretize an all-censored split using the shared cuts.
        if not np.any(event_mask):
            cuts = np.asarray(self.cuts)

            # Pycox uses right censoring for durations beyond the last cut.
            clipped_durations = np.minimum(durations, cuts[-1])

            # For censoring, use the cut at or immediately before the censoring time.
            duration_idx = (np.searchsorted( cuts, clipped_durations, side="right", ) - 1)
            duration_idx = np.clip( duration_idx, 0, len(cuts) - 1)
            transformed_events = np.zeros(len(events), dtype=np.int64)

            return ( duration_idx.astype(np.int64), transformed_events)

        # Normal path when the split contains at least one event.
        duration_idx, retained_events = super().transform( durations, event_mask)

        # Preserve the original event type, but mark observations
        # censored by the discretizer as event=0.
        events[retained_events == 0] = 0

        return (
            duration_idx.astype(np.int64),
            events.astype(np.int64),
        )

    # DeepHitSingle.label_transform standardize the durations 
    num_durations = args.num_duration  
    labtrans = LabTransform(num_durations)  
    
    y_train_all = labtrans.fit_transform(train_time_all, train_status_all.astype(int))
    y_val_all = labtrans.transform(val_time_all, val_status_all.astype(int))

    # Create the validation set
    val_all = (X_val_all, y_val_all)
    test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan 
    alpha, sigma = (0.2, 0.1)
    
    if args.alg not in ['all_in', 'fedavg']:
        raise ValueError(f"Unsupported algorithm: {args.alg}")
    
######################################## Centralized Training ######################################
    if args.alg == 'all_in':
        nets, local_model_meta_data, layer_type, alpha, sigma = init_nets( input_size=X_train_all.shape[1], censor_rate=client_censored_train, dropout_p=args.dropout_p, n_parties=1, args=args)  
        try:
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = train_net_all(alpha, sigma, 0, nets[0], X_train_all, y_train_all, val_all, args.epochs, args.lr, args.optimizer, train_time_all, train_status_all,  X_val_all, val_time_all, val_status_all, X_test_all, test_time_all, test_status_all, evaltime, labtrans, args, args.device)
        except Exception as e:
            logger.info(f"[Exception in train_net_all method]:  {type(e).__name__}: {e}")
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan
            traceback.print_exc()  

        logger.info('>> All in Test Cindex: {}'.format(test_cindex))
        logger.info('>> All in Test Brier: {}'.format(test_brier))
        logger.info('>> All in Test N BLL: {}'.format(test_nbll))
        logger.info('>> All in Test P_value: {}'.format(test_p_value))
        logger.info('>> All in Test Weighted MAE PO: {}'.format(test_weighted_MAE_PO)) 

    else:  
        logger.info("Initializing nets")   
        nets = init_nets( input_size=X_train_local[0].shape[1], censor_rate = client_censored_train, dropout_p=args.dropout_p, n_parties=args.n_parties, args=args)   
        
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, directory=args.modeldir, checkpoint_pth=f"DeepHit_model_run_{args.run_id}.pt")
        global_models = init_nets( input_size=X_train_local[0].shape[1], censor_rate=client_censored_train, dropout_p=args.dropout_p, n_parties=1, args=args)

        global_model = global_models[0] 
        global_para = deepcopy(global_model.state_dict())
        
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para) 

        global_val_cindex = 0.0
        test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = np.nan, np.nan, np.nan, np.nan, np.nan
        comm_round_done = 0

        for comm_round in range(args.comm_round):
            logger.info("in comm round:" + str(comm_round))
            print('communication round:', comm_round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            
            selected = arr[:int(args.n_parties * args.sample)]
            global_para = deepcopy(global_model.cpu().state_dict())  

            for client_id in selected:
              if nets[client_id] is None:
                  reinitialized_nets, _, _, _, _ = init_nets( input_size=X_train_local[client_id].shape[1], censor_rate=client_censored_train[client_id], dropout_p=args.dropout_p, n_parties=1, args=args)
                  nets[client_id] = reinitialized_nets[0]

              nets[client_id].load_state_dict(global_para) 

            nets = local_train_net(alpha, sigma, client_name_array, nets, selected, args, X_train_local, train_time_local, train_status_local, X_val_local, val_time_local, val_status_local, X_test_local, test_time_local, test_status_local, evaltime, labtrans, tr_data_len, comm_round, args.device)
            
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
            eval_clients = range(args.n_parties)

            global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = evaluate_global_model_on_clients(global_model, alpha, sigma, eval_clients, X_val_local, val_time_local, val_status_local, X_test_local, test_time_local, test_status_local, train_time_local, train_status_local,evaltime, labtrans, args, device=args.device)
            
            model_dict = {"model_state_dict": global_para,
                "duration_index": labtrans.cuts,
                "num_durations": args.num_duration,
                "alpha": alpha, "sigma": sigma,
                "input_size": X_train.shape[1],
                "dropout_p": args.dropout_p,
                }  
            
            # test_cindex, test_brier, test_nbll
            improved = early_stopping(
                global_val_cindex,
                model_dict,
                round_idx=comm_round_done,
                test_metrics={
                    "test_c_index": test_cindex,
                    "test_brier": test_brier,
                    "test_nbll": test_nbll,
                    "test_p_value": test_p_value,
                    "test_MAE_PO": test_weighted_MAE_PO
                },
            )

            if improved:
                print(
                    f"New best model selected at round {comm_round}: "
                    f"val_cindex={global_val_cindex:.4f}, "
                    f"test_cindex={test_cindex:.4f}, "
                    f"test_brier={test_brier:.4f}, "
                    f"test_nbll={test_nbll:.4f}",
                    f"test_p_value={test_p_value:.4f}, "
                    f"test_MAE_PO={test_weighted_MAE_PO:.4f}"
                )
                
            comm_round_done = comm_round
            if early_stopping.early_stop:
                comm_round_done = comm_round
                print("Early stopping")
                break 

            print(f'>> Global Model Test Cindex: {test_cindex:.4f}, Test Brier: {test_brier:.4f}, Test NBLL: {test_nbll:.4f}, Test P-Value: {test_p_value:.4f}, Test MAE-PO: {test_weighted_MAE_PO:.4f}') 
            
            # test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO  
            final_round = early_stopping.best_round if early_stopping.best_round is not None else comm_round
            final_val_cidx =  early_stopping.best_val_cindex if np.isfinite(early_stopping.best_val_cindex) else global_val_cindex
            final_test_cidx =  early_stopping.best_test_cindex if early_stopping.best_test_cindex is not None else test_cindex 
            final_test_brier =  early_stopping.best_test_brier if early_stopping.best_test_brier is not None  else test_brier
            final_test_nbll =  early_stopping.best_test_nbll if early_stopping.best_test_nbll is not None else test_nbll
            final_test_p_value =  early_stopping.best_test_p_value if early_stopping.best_test_p_value is not None else test_p_value
            final_test_MAE_PO =   early_stopping.best_test_MAE_PO if early_stopping.best_test_MAE_PO is not None else test_weighted_MAE_PO
            
    import builtins
    csv_path = os.path.join(args.logdir, "results_early.csv") 

    row = {
        "seed": int(args.init_seed),
        "fold_id": int(args.run_id),
        "dataset": args.dataset,
        "partition": args.partition,
        "model": args.model,
        "c_idx": builtins.round(float(final_test_cidx), 4),
        "brier": builtins.round(float(final_test_brier), 4),
        "nbll": builtins.round(float(final_test_nbll), 4),
        "p_value": builtins.round(float(final_test_p_value), 4),
        "MAE_PO": builtins.round(float(final_test_MAE_PO), 4),
        "lr": builtins.round(float(args.lr), 5),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "dropout": float(args.dropout_p),
        "comm_round": int(args.comm_round)
    }

    df = pd.DataFrame([row]) 
    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
 
######################################## Federated  Training ######################################
