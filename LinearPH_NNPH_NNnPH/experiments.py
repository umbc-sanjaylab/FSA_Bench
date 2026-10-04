import json   
from model import *   
from server.utils import create_experiment_logger, get_args
from datasets.data_processing import prepare_survival_data,  compute_eval_times 
from model import init_net 
from server.utils import mkdirs
from CL import train_centralized
from FL import run_federated 
from config import DATASET_CONFIG
from pathlib import Path    
  
######################################################## Main Experiment ########################################################

if __name__ == "__main__":
    args = get_args() 
    if args.setting not in ["CL", "FL"]:
        raise ValueError(f"Unsupported setting: {args.setting}")

    if args.dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Dataset-specific configuration
    dataset_config = DATASET_CONFIG[args.dataset]
    args.n_parties = dataset_config["n_parties"]
    args.batch_size = dataset_config["batch_size"]
    num_folds = dataset_config["folds"]

    # Output directories
    project_root = Path(__file__).resolve().parent
    args.logdir = str(project_root / "results" / "logs" / args.setting.lower() / args.dataset / args.model)
    args.modeldir = str(project_root / "checkpoints" / args.setting.lower() / args.dataset / args.model)

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
        logger = create_experiment_logger(args.logdir, args.setting, args.model, args.dataset, args.run_id) 
        logger.info("Running %s experiment | Model: %s | Dataset: %s | Fold: %d/%d", args.setting, args.model, args.dataset, fold + 1, num_folds)

        # Load preprocessed data and evaluation time for both CL & FL
        data = prepare_survival_data(args) 
        eval_times = compute_eval_times(data, interval=args.pseudo_perc_interval)
        input_size = len(data.feature_names)

        # Initialize model and start experiment
        model = init_net(input_size, args) 
        if args.setting == "CL":
            results = train_centralized(model=model, data=data.centralized, eval_times=eval_times, args=args)
        else:
            results = run_federated(model=model, clients=data.clients, eval_times=eval_times, args=args)

        logger.info("Fold %d results: %s", fold, results) 