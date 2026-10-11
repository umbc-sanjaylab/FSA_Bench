import numpy as np, pandas as pd
from .data_class import SurvivalSplit, ClientSurvivalData
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_CONFIG
from .data_class import PreparedSurvivalData, CentralizedSurvivalData

######################################### Read configuration of dataset ######################################### 

def get_dataset_config(dataset_name):
    """Return metadata associated with a federated dataset."""

    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DATASET_CONFIG[dataset_name]
  
######################################### Read dataset properties ############################################## 

def dataset_properties(dataset_name):
    """Return metadata associated with a federated dataset."""

    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_config = DATASET_CONFIG[dataset_name]

    region_column = dataset_config["region_column"]
    n_folds = dataset_config["n_folds"]
    n_clients = dataset_config["n_clients"]
    return region_column, n_folds, n_clients 
  
######################################### Read data from CSV file #############################################

def load_raw_dataset(dataset_name):
    """Load a raw survival dataset."""

    file_path = RAW_DATA_DIR / f"{dataset_name}.csv"
    if not file_path.exists():
        raise FileNotFoundError( f"Dataset not found: {file_path}" )

    return pd.read_csv(file_path)

######################################### Read train, val, and test data for a specific client ################
 
def get_local_dataset( dataset_name, partition, client_id, run_id):
    """
    Load train/validation/test data for one client.

    Returns
    -------
    train_df : pd.DataFrame
    val_df : pd.DataFrame
    test_df : pd.DataFrame
    client_name : str
    feature_list : list[str]
    """ 
    base_dir =  PROCESSED_DATA_DIR /  dataset_name / partition / str(client_id) / str(run_id) 

    train_path = base_dir / "train.csv"
    val_path = base_dir / "val.csv"
    test_path = base_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    feature_list = [ col for col in train_df.columns if col not in ("time", "event")]
    return train_df, val_df, test_df, str(client_id), feature_list  
  
  
  ############################################# Simple Method  ##########################################################################
def get_centralized_dataset(dataset_name, run_id):
      base_dir = PROCESSED_DATA_DIR / dataset_name / "centralized" / "0" / str(run_id)
  
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
      return SurvivalSplit( X=X, time=time, event=event)
   
def prepare_survival_data(args):
      clients = []
      feature_names = None
  
      for client_id in range(args.n_parties):
          train_df, val_df, test_df, client_name, current_features = get_local_dataset(args.dataset, "federated", client_id, args.run_id)
  
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
  
      return PreparedSurvivalData(clients=clients, centralized=centralized, feature_names=feature_names)  
       
  
  ################################################## Evaluation Time Points ############################################
  
def compute_eval_times(data, interval=10):
      times = []
      for client in data.clients: 
          times.append(client.train.time)
          times.append(client.val.time)
  
      global_times = np.unique(np.concatenate(times))
      return np.percentile(  global_times, np.arange(interval, 99, interval))
  
  
  ################################################## Data Builder ######################################################
    