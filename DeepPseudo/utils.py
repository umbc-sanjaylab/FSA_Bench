from torch.utils.data import Dataset
import pandas as pd, numpy as np, torch

################################################## Data Builder ###################################################### 

class DataBuilder(Dataset):
    def __init__(self, covariates, pseudo_value, feature_list):
        self.feature_list = feature_list
        X_covariates_df = pd.DataFrame(covariates, columns=feature_list)
        
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