import pandas as pd
from sksurv.util import Surv
from datasets.data_loader import get_centralized_dataset
from .config import  TIME_COL, EVENT_COL, SEED
from .utils import select_survival_background, pseudo_eval_times

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