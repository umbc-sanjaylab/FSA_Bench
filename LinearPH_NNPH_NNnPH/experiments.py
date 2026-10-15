import argparse
import json
from pathlib import Path

from config import DATASET_CONFIG, save_result 
from datasets.data_processing import prepare_survival_data_and_time
 
from .CL import train_centralized
from .FL import run_federated
from .server.utils import create_experiment_logger, mkdirs 

#################################### Arguments for Survival Model Experiments ############################################################ 

def get_args():
    parser = argparse.ArgumentParser(description="Run LinearPH, NNPH, and NNnPH survival experiments.") 
    parser.add_argument('--model', type=str, default='NNnPH', choices=['LinearPH', 'NNPH', 'NNnPH'], help='Survival model')
    parser.add_argument('--dataset', type=str, default='Aids2', help='Dataset used for training')
    parser.add_argument('--partition', type=str, default='FL', choices=['CL', 'FL'], help='Learning setting')

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--comm_round', type=int, default=50, help='Maximum number of communication rounds')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--init_seed', type=int, default=5, help='Random seed')
    parser.add_argument('--run_id', type=int, default=0, help='Cross-validation fold index')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout probability (default: 0.05)')
    parser.add_argument('--reg', type=float, default=5e-5, help='L2 regularization strength')
    parser.add_argument('--device', type=str, default='cuda:0', help='Execution device')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--sample', type=float, default=1.0, help='Client sampling ratio per communication round')
    parser.add_argument('--num_duration', type=int, default=40, help='Number of duration intervals')
    parser.add_argument('--time_interval', type=int, default=10, help='Percentile interval for evaluation times')

    return parser.parse_args()
  
######################################################## Main Experiment ########################################################

if __name__ == "__main__":
    # load the arguments and validate it 
    args = get_args() 
    if args.dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # load the dataset configuration and experiment hyperparameters 
    dataset_config = DATASET_CONFIG[args.dataset]
    args.n_parties = dataset_config["n_clients"]
    args.batch_size = dataset_config["batch_size"]
    num_folds = dataset_config["n_folds"]

    # configure the model directory and log directory
    project_root = Path(__file__).resolve().parent
    args.logdir = str(project_root / "results" / "logs" / args.partition.lower() / args.dataset / args.model)
    args.modeldir = str(project_root / "checkpoints" / args.partition.lower() / args.dataset / args.model)

    # create the directories  
    mkdirs(args.logdir)
    mkdirs(args.modeldir)

    # start the experiment for each fold
    for fold in range(num_folds):
        args.run_id = fold

        # store the experiment arguments in file 
        argument_path = Path(args.logdir) / f"{args.model}_{args.dataset}_{args.run_id}.json"
        with open(argument_path, "w") as f:
            json.dump(vars(args), f, indent=4, default=str)

        logger = create_experiment_logger(args.logdir, args.partition, args.model, args.dataset, args.run_id)
        try:
            # Log experiment details
            logger.info(  "Running %s experiment | Model: %s | Dataset: %s | Fold: %d/%d",
                                               args.partition, args.model, args.dataset, fold + 1, num_folds )

            # load data and start model training and evaluation
            data, eval_times = prepare_survival_data_and_time(args) 
            if args.partition == "CL":
                results = train_centralized( data=data.centralized, eval_times=eval_times, args=args, logger=logger )
            else:
                results = run_federated( clients=data.clients, eval_times=eval_times, args=args, logger=logger )

            # save the results for the current fold
            save_result(results["best_round"], results["val_cindex"], results["cindex"],
                              results["brier"], results["nbll"], results["d_calibration"], results["mae_po"], args) 
            logger.info("Fold %d results: %s", fold + 1, results)

        finally:
            for handler in logger.handlers[:]:
                handler.flush()
                handler.close()
                logger.removeHandler(handler)