from dataclasses import dataclass
from typing import List
import numpy as np 
from torch.utils.data import Dataset  
import torch 

@dataclass
class SurvivalSplit:
    X: np.ndarray
    time: np.ndarray
    event: np.ndarray 

@dataclass
class ClientSurvivalData:
    client_id: int
    client_name: str                       

    train: SurvivalSplit
    val: SurvivalSplit
    test: SurvivalSplit
    
@dataclass
class CentralizedSurvivalData:
    train: SurvivalSplit
    val: SurvivalSplit
    test: SurvivalSplit
    
@dataclass
class PreparedSurvivalData:
    clients: List[ClientSurvivalData]
    centralized: CentralizedSurvivalData
    feature_names: List[str] 

class SurvivalDataset(Dataset):
    def __init__(self, X, time, event): 
        self.X = torch.as_tensor( X, dtype=torch.float32)
        self.time = torch.as_tensor( time, dtype=torch.float32)
        self.event = torch.as_tensor( event, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return  self.X[idx], self.time[idx], self.event[idx],
    