from torch.utils.data import Dataset
import pandas as pd, numpy as np, torch

################################################## Data Builder ###################################################### 

class DataBuilder(Dataset):
    def __init__(self, covariates, time, status, feature_list):
        self.feature_list = feature_list
        X_covariates_df = pd.DataFrame(covariates, columns=feature_list)
        self.X_covariates = torch.from_numpy(np.array(X_covariates_df, dtype=np.float32))

        self.time = time
        self.status = status

    def __len__(self):
        return len(self.X_covariates)

    def __getitem__(self, idx):
        return  torch.as_tensor(self.X_covariates[idx], dtype=torch.float32), \
                torch.as_tensor(self.time[idx], dtype=torch.float32), \
                torch.as_tensor(self.status[idx], dtype=torch.float32)


################################## Method for checking whether metric is finite ##################################################

def is_finite_metric(value):
    return (
        value is not None
        and np.isscalar(value)
        and np.isfinite(float(value))
    )