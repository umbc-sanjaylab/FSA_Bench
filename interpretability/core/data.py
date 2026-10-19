import pandas as pd, numpy as np 
from sksurv.util import Surv
from datasets.data_loader import get_centralized_dataset
from .config import  TIME_COL, EVENT_COL, SEED
from .utils import select_survival_background, pseudo_eval_times
from datasets.data_processing import prepare_survival_data_for_client 
from config import PROCESSED_DATA_DIR

def split_xy(df, feature_cols):
    X_df = df[feature_cols].astype("float32") 
    y_sksurv = Surv.from_arrays( event=df[EVENT_COL].astype(bool).to_numpy(), time=df[TIME_COL].to_numpy(dtype="float32"))  # For SurvSHAP
    y_pycox = ( df[TIME_COL].to_numpy(dtype="float32"), df[EVENT_COL].to_numpy(dtype="int64"))  # For PyCox CoxPH 
    return X_df, y_sksurv, y_pycox
  
def prepare_fold_data(dataset, fold):
    train_df, val_df, test_df = get_centralized_dataset(dataset, fold) 
    feature_cols = [col for col in train_df.columns if col not in (TIME_COL, EVENT_COL)]
    times = pseudo_eval_times(pd.concat([train_df[TIME_COL], val_df[TIME_COL]], ignore_index=True))

    background_idx = select_survival_background(train_df, background_size=100, random_state=SEED)
    background_df = train_df.iloc[background_idx]

    X_bg, y_bg, _ = split_xy(background_df, feature_cols)
    X_test, y_test, _ = split_xy(test_df, feature_cols)
    X_train, _, y_train_pycox = split_xy(train_df, feature_cols)

    return feature_cols, times, X_bg, y_bg, X_test, y_test, X_train, y_train_pycox

def prepare_site_data(dataset, site, fold):
    clients = prepare_survival_data_for_client(dataset, fold)  
    client = next((c for c in clients if c.client == site), None)
    train_df, val_df, test_df = client.train , client.val, client.test
     
    feature_cols = [col for col in train_df.columns if col not in (TIME_COL, EVENT_COL)]
    background_idx = select_survival_background(train_df, background_size=100, random_state=SEED)
    background_df = train_df.iloc[background_idx]

    X_bg, y_bg, _ = split_xy(background_df, feature_cols)
    X_test, y_test, _ = split_xy(test_df, feature_cols)
    X_train, _, y_train_pycox = split_xy(train_df, feature_cols)

    return train_df, val_df, test_df  
  
  
def prepare_site_eval_times(dataset, sites, fold): 
    times = [] 
    
    for site in sites:
        train_df, val_df, _= prepare_site_data(dataset,  site, fold)  
        times.append(train_df[TIME_COL].to_numpy())
        times.append(val_df[TIME_COL].to_numpy())

    unique_times = np.unique(np.concatenate(times))
    return np.percentile(unique_times, np.arange(10, 99, 10))  

