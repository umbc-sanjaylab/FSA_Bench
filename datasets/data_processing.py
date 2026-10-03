from dataclasses import dataclass
from typing import List
import numpy as np
from data_loader import get_local_dataset
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

def split_survival_dataframe(df):
    """
    Separate covariates, survival time, and event indicator.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing feature columns together with  'time' and 'event'.

    Returns
    -------
    SurvivalSplit
    """

    required_columns = {"time", "event"} 
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError( f"Missing required survival columns: {sorted(missing)}")  

    X = np.asarray( df.drop(columns=["time", "event"]), dtype=np.float32)
    time = np.asarray( df["time"], dtype=np.float32).reshape(-1)
    event = np.asarray( df["event"], dtype=np.int64).reshape(-1)

    return SurvivalSplit( X=X, time=time, event=event)

def prepare_survival_data(args):
    """
    Load and prepare local client data and construct the corresponding
    centralized dataset.

    This function is shared by all survival models.
    """

    clients = []
    feature_names = None

    for client_id in range(args.n_parties):
        train_df, val_df, test_df, client_name, current_features, = get_local_dataset( dataset_name=args.dataset, partition=args.partition, client_id=client_id, run_id=args.run_id)

        if feature_names is None:
            feature_names = current_features

        elif feature_names != current_features:
            raise ValueError( f"Feature mismatch for client {client_id}")

        client_data = ClientSurvivalData(
            client_id=client_id,
            client_name=client_name,
            train=split_survival_dataframe(train_df),
            val=split_survival_dataframe(val_df),
            test=split_survival_dataframe(test_df)
        )

        clients.append(client_data)

    centralized = build_centralized_data(clients)
    return PreparedSurvivalData( clients=clients, centralized=centralized, feature_names=feature_names)   
    
################################################## Centralized Data ######################################################

def build_centralized_data(clients):

    train = SurvivalSplit(
        X=np.vstack([client.train.X for client in clients]),
        time=np.concatenate( [client.train.time for client in clients] ).astype(np.float32),
        event=np.concatenate( [client.train.event for client in clients] ).astype(np.int64)
    )

    val = SurvivalSplit(
        X=np.vstack([client.val.X for client in clients]),
        time=np.concatenate([client.val.time for client in clients]).astype(np.float32),
        event=np.concatenate([client.val.event for client in clients]).astype(np.int64) 
    )

    test = SurvivalSplit(
        X=np.vstack([client.test.X for client in clients]),
        time=np.concatenate([client.test.time for client in clients]).astype(np.float32),
        event=np.concatenate([client.test.event for client in clients]).astype(np.int64)
    )

    return CentralizedSurvivalData(train=train, val=val, test=test)


################################################## Evaluation Time Points ############################################

def compute_eval_times(data, interval=10):
    times = []
    for client in data.clients: 
        times.append(client.train.time)
        times.append(client.val.time)

    global_times = np.unique(np.concatenate(times))
    return np.percentile(  global_times, np.arange(interval, 99, interval))


################################################## Data Builder ######################################################
  
class SurvivalDataset(Dataset):
    def __init__(self, X, time, event): 
        self.X = torch.as_tensor( X, dtype=torch.float32)
        self.time = torch.as_tensor( time, dtype=torch.float32)
        self.event = torch.as_tensor( event, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return  self.X[idx], self.time[idx], self.event[idx],