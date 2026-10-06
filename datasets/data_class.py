import numpy as np, torch
from dataclasses import dataclass 
from torch.utils.data import Dataset

############################ Survival data: feature, time, and event status ###################################################### 

@dataclass
class SurvivalSplit:
    X: np.ndarray
    time: np.ndarray
    event: np.ndarray

############################ Survival data for Clients ###################################################### 

@dataclass
class ClientSurvivalData:
    client_id: int
    client_name: str
    
    train: SurvivalSplit
    val: SurvivalSplit
    test: SurvivalSplit

############################ Survival data for Centralized Partition ############################################### 
@dataclass
class CentralizedSurvivalData:
    train: SurvivalSplit
    val: SurvivalSplit
    test: SurvivalSplit

############################ Survival data for both Centralized & Federated Partition ################################## 

@dataclass
class PreparedSurvivalData:
    clients: list[ClientSurvivalData]
    centralized: CentralizedSurvivalData
    feature_names: list[str]


################################################## Data Builder ###################################################### 

class SurvivalDataset(Dataset):
    def __init__(self, X, time, event):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.time = torch.as_tensor(time, dtype=torch.float32)
        self.event = torch.as_tensor(event, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.time[idx], self.event[idx]