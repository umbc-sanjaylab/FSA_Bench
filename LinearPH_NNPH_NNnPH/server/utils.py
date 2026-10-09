import argparse, torch 
import numpy as np 
import pandas as pd, os, logging  
from pathlib import Path   

#################################### Arguments for NNnPH Model ############################################################ 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CoxPH', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='metabric', help='dataset used for training')
    parser.add_argument( '--net_config', type=lambda x: list(map(int, x.split(', '))))
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
    parser.add_argument('--pseudo_perc_interval', type=int, default=10,  help='Interval for pseudo values calculation')
    parser.add_argument('--epsilon', type=float, default=5.0, help='epsilon')

    args = parser.parse_args()
    return args 
  
  
#################################### Early Stopping Criteria ############################################################ 

class EarlyStopping:
    """Stop training if validation C-index does not improve for `patience` rounds."""

    def __init__(self, patience=25, verbose=False, delta=0.0, directory=None, discretiser=None,  checkpoint_pth="checkpoint.pt"):
        self.patience = int(patience)
        self.verbose = verbose
        self.directory = directory

        self.delta = float(delta)
        self.checkpoint_pth = checkpoint_pth
        self.discretiser = discretiser

        self.counter = 0
        self.best_val_cindex = float("-inf")
        self.early_stop = False

        # new: keep metadata for the selected best round
        self.best_round = None
        self.best_test_cindex = None
        self.best_test_brier = None
        self.best_test_nbll = None
        self.best_test_p_value = None
        self.best_test_MAE_PO = None

    def __call__(self, val_c_index, param, round_idx=None, test_metrics=None):
        if val_c_index is None or not np.isfinite(val_c_index):
            if self.verbose:
                print(f"[EarlyStopping] Skipping invalid val_c_index = {val_c_index}")
            return False

        improved = val_c_index > self.best_val_cindex + self.delta 
        if improved:
            if self.verbose:
                print(f"Validation C-index improved " f"({self.best_val_cindex:.6f} -> {val_c_index:.6f}). Saving checkpoint...")

            self.best_val_cindex = val_c_index
            self.counter = 0
            self.best_round = round_idx

            if test_metrics is not None:
                self.best_test_cindex = test_metrics.get("test_c_index")
                self.best_test_brier = test_metrics.get("test_brier")
                self.best_test_nbll = test_metrics.get("test_nbll")
                self.best_test_p_value = test_metrics.get("test_p_value")
                self.best_test_MAE_PO = test_metrics.get("test_MAE_PO")

            self.save_checkpoint(param)
            return True

        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True

        return False
    
    def save_checkpoint(self, param):
        if param is None:
            return

        os.makedirs(self.directory, exist_ok=True) 
        torch.save({
                "model": param,
                "discretizer": self.discretiser
            }, os.path.join(self.directory, self.checkpoint_pth))

     

#################################### Early Stopping Criteria ############################################################ 

def create_experiment_logger(logdir, setting, model, dataset, run_id):
    logger_name = f"{setting}_{model}_{dataset}_fold_{run_id}"
    log_path = Path(logdir) / f"{model}_{dataset}_fold_{run_id}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%m-%d %H:%M"
    )

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
  
#################################### Make Directory ############################################################ 

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
   