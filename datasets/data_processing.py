import  pandas as pd,  numpy as np   
from .data_loader import get_local_dataset, PROCESSED_DATA_DIR
from .data_loader import SurvivalSplit,  ClientSurvivalData
from .data_loader import PreparedSurvivalData, CentralizedSurvivalData 
from config import get_client_name_for_federated_datasets 
    
############################################# Simple Method  ##########################################################################
def get_centralized_dataset(dataset_name, run_id):
    base_dir = PROCESSED_DATA_DIR / dataset_name / "CL" / "0" / str(run_id)

    train_df = pd.read_csv(base_dir / "train.csv")
    val_df = pd.read_csv(base_dir / "val.csv")
    test_df = pd.read_csv(base_dir / "test.csv")

    return train_df, val_df, test_df
    
############################################# Simple Method  ########################################################################## 

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
    return SurvivalSplit(X=X, time=time, event=event)

############################################# Simple Method  ########################################################################## 

def prepare_survival_data_and_time(args, client_id=None):
    clients = []
    feature_names = None

    client_list = get_client_name_for_federated_datasets(args.dataset)
    if client_id is not None:
        client_list = [client_list[client_id]]
      
    for client_id in client_list:
        train_df, val_df, test_df, client_name, current_features = get_local_dataset(args.dataset, "FL", client_id, args.run_id)

        if feature_names is None:
            feature_names = current_features 
            
        elif feature_names != current_features:
            raise ValueError(f"Feature mismatch for client {client_id}")

        clients.append(
            ClientSurvivalData(
                client_id=client_id,
                client_name=client_name,
                train=split_survival_dataframe(train_df),
                val=split_survival_dataframe(val_df),
                test=split_survival_dataframe(test_df)
            )
        )

    train_df, val_df, test_df = get_centralized_dataset(args.dataset, args.run_id)

    centralized = CentralizedSurvivalData(
        train=split_survival_dataframe(train_df),
        val=split_survival_dataframe(val_df),
        test=split_survival_dataframe(test_df)
    )

    survival_data = PreparedSurvivalData(clients=clients, centralized=centralized, feature_names=feature_names) 
    eval_times =    compute_eval_times(survival_data, interval=args.time_interval)
    return survival_data, eval_times 

################################################## Evaluation Time Points ############################################ 

def compute_eval_times(data, interval=10, end=99):
    times = []
    for client in data.clients:
        times.append(client.train.time)
        times.append(client.val.time)

    global_times = np.unique(np.concatenate(times)) 
    percentiles = np.arange(interval, end, interval)
     
    return np.percentile(global_times, percentiles)