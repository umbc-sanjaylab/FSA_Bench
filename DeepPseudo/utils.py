from torch.utils.data import Dataset
import pandas as pd, numpy as np, torch

import rpy2.robjects as ro  
from rpy2.robjects import default_converter
from pathlib import Path

from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri, pandas2ri, default_converter

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
