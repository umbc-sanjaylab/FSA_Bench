from torch.utils.data import Dataset
import pandas as pd, numpy as np, torch

import rpy2.robjects as ro  
from rpy2.robjects import default_converter
from pathlib import Path

import argparse 
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri, pandas2ri, default_converter


#################################### Arguments for Survival Model Experiments ############################################################ 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DeepPseudo', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='dataDIVAT1', help='dataset used for training') 
    parser.add_argument('--partition', type=str, default='IID', help='the data partitioning strategy') 
    parser.add_argument('--batch_size', type=int, default=None, help='input batch size for training (default: 64)') 
    parser.add_argument("--lr", type=float, default=1e-3, help='learning rate (default: 0.01)') 
    parser.add_argument("--epochs", type=int, default=250, help='number of local epochs') 
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster') 
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication round')
    parser.add_argument('--patience', type=int, default=25, help='number of patience')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=5, help="Random seed")
    parser.add_argument('--run_id', type=int, default=0, help="fold number of 5-fold cross validation") 
    parser.add_argument("--dropout", type=float,  required=False, default=0.5, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory") 
    parser.add_argument("--reg", type=float, default=5e-5, help="L2 regularization strength") 
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer') 
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1.0, help='Sample ratio for each communication round') 
    parser.add_argument('--time_interval', type=int, default=10, help='Time interval for evaluation points')

    args = parser.parse_args()
    return args

################################################## Data Builder ###################################################### 

class DataBuilder(Dataset):
    def __init__(self, covariates, pseudo_value): 
        X_covariates_df = pd.DataFrame(covariates) 
        self.X_covariates = torch.from_numpy(np.array(X_covariates_df, dtype=np.float32)) 
        self.pseudo_value = pseudo_value 

    def __len__(self):
        return len(self.X_covariates)

    def __getitem__(self, idx):
       covariates = torch.as_tensor(self.X_covariates[idx], dtype=torch.float32)
       pseudo_value = torch.as_tensor(self.pseudo_value[idx], dtype=torch.float32)
       
       return covariates, pseudo_value

################################## Method for checking whether metric is finite ##################################################

def is_finite_metric(value):
    return (
        value is not None
        and np.isscalar(value)
        and np.isfinite(float(value))
    )
  

################################## Method for pseudo value computation ##################################################

def pseudoFunction(data_py, eval_time_py):
    """Calculate the pseudo values for survival function.
    Arguments:
        data: A dataframe containing survival time and event status.
        evaltime: Evaluation times at which pseudo values are calcuated
    Returns:
        A dataframe of pseudo values for all subjects in the data at the evaluation time points. 
    """
    PROJECT_ROOT = Path(__file__).resolve().parent  
    r = ro.r
    r.source(f'{PROJECT_ROOT}/Compute_pseudo_values.R')
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
