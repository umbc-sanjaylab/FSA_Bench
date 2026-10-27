import argparse, json
from pathlib import Path

from config import DATASET_CONFIG
from utils import save_result
from datasets.data_processing import prepare_survival_data_and_time
 
from .CL import train_centralized
from .FL import run_federated
from .utils import get_args 
from .server.utils import create_experiment_logger, mkdirs
 
  
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
    args.modeldir = str(project_root / "results" / "checkpoints" / args.partition.lower() / args.dataset / args.model)

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