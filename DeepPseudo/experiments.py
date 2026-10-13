import argparse, json
from pathlib import Path

from config import DATASET_CONFIG, save_result 
from datasets.data_processing import prepare_survival_data_and_time
 
from .CL import train_centralized
from .FL import run_federated
from .server.utils import create_experiment_logger, mkdirs

#################################### Arguments for Survival Model Experiments ############################################################ 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CoxPH', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='Aids2', help='dataset used for training')
    parser.add_argument( '--net_config', type=lambda x: list(map(int, x.split(', '))), default=[64, 32])
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy')
    parser.add_argument('--Pseudo_value', type=str, default='non-DP', help='the pseudo value calculation strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)') 
    parser.add_argument("--lr", type=float, default=1e-3, help='learning rate (default: 0.01)') 
    parser.add_argument("--epochs", type=int, default=250, help='number of local epochs') 
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg', help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication round')
    parser.add_argument('--patience', type=int, default=25, help='number of patience')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=5, help="Random seed")
    parser.add_argument('--run_id', type=int, default=0, help="fold number of 5-fold cross validation") 
    parser.add_argument("--dropout", type=float,  required=False, default=0.05, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./Fed_CoxPH/data/", help="Data directory") 
    parser.add_argument("--reg", type=float, default=5e-5, help="L2 regularization strength") 
    parser.add_argument('--logdir', type=str, required=False, default="./Fed_CoxPH/logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./Fed_CoxPH/models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1.0, help='Sample ratio for each communication round')
    parser.add_argument('--num_duration', type=int, default=40,  help='Interval for pseudo values calculation')
    parser.add_argument('--epsilon', type=float, default=5.0, help='epsilon')  
    parser.add_argument('--time_interval', type=int, default=10, help='Time interval for evaluation points')

    args = parser.parse_args()
    return args 
  
######################################################## Main Experiment ########################################################

if __name__ == "__main__":
    args = get_args()
    if args.partition not in ["CL", "FL"]:
        raise ValueError(f"Unsupported partition: {args.partition}")

    if args.dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Dataset-specific configuration
    dataset_config = DATASET_CONFIG[args.dataset]
    args.n_parties = dataset_config["n_clients"]
    args.batch_size = dataset_config["batch_size"]
    num_folds = dataset_config["n_folds"]

    # Output directories
    project_root = Path(__file__).resolve().parent
    args.logdir = str(project_root / "results" / "logs" / args.partition.lower() / args.dataset / args.model)
    args.modeldir = str(project_root / "checkpoints" / args.partition.lower() / args.dataset / args.model)

    mkdirs(args.logdir)
    mkdirs(args.modeldir)

    # Save experiment arguments
    argument_path = Path(args.logdir) / f"{args.model}_{args.dataset}_arguments.json"
    with open(argument_path, "w") as f:
        json.dump(vars(args), f, indent=4, default=str)

    # Cross-validation
    for fold in range(num_folds):
        args.run_id = fold

        # Configure logger for current fold
        logger = create_experiment_logger(args.logdir, args.partition, args.model, args.dataset, args.run_id) 
        logger.info("Running %s experiment | Model: %s | Dataset: %s | Fold: %d/%d", args.partition, args.model, args.dataset, fold + 1, num_folds)

        # Load preprocessed data and evaluation time for both CL & FL
        data, eval_times = prepare_survival_data_and_time(args)
        input_size = len(data.feature_names)

        # Initialize model and start experiment 
        if args.partition == "CL": 
            results = train_centralized(data=data.centralized, feature_list=data.feature_list, evaltime=eval_times, args=args)  
        else:
            results = run_federated(clients=data.clients, eval_times=eval_times, args=args)
        
        # def save_result(final_test_cidx, final_test_brier, final_test_nbll, final_test_p_value, final_test_weighted_MAE_PO, args): 
        save_result(results["best_round"], results['val_cindex'], results['cindex'], results['brier'], results['nbll'], results['d_calibration'], results['mae_po'], args) 
        logger.info("Fold %d results: %s", fold + 1, results) 