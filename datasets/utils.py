import os 
import numpy as np  
from .preprocessing.missing_value import preprocess_missing
from .preprocessing.encoding import encode_with_schema 
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split 
from collections import Counter

################################################## Five fold cross validation ############################################

def five_fold_cross_validation(dataset_name, centralized_data, no_of_fold, run_id):
    skf = StratifiedKFold( n_splits=no_of_fold,  shuffle=True, random_state=1234 )

    event_col = "event" if "event" in centralized_data.columns else "status" 
    status_data =  np.asarray(centralized_data[event_col], dtype="float32").reshape(-1).astype(int) 

    folds = []
    for _, test_idx in skf.split(centralized_data, status_data):
        folds.append(test_idx)
    
    fold_id = run_id % no_of_fold 
    test_fold_id = fold_id
    val_fold_id = (fold_id + 1) % no_of_fold

    test_idxs = folds[test_fold_id]
    val_idxs = folds[val_fold_id]

    train_idxs = []
    for i in range(no_of_fold):
        if i != test_fold_id and i != val_fold_id:
            train_idxs.extend(folds[i])

    return train_idxs, val_idxs, test_idxs

################################# Check whether Split can be Stratified  ############################################  

def check_stratified_split(train_val_status, n_train_val, val_ratio=0.2):
    classes, counts = np.unique(train_val_status, return_counts=True)
    n_classes = len(classes)

    # Basic validation size: 20% of train+val
    val_size = int(np.ceil(val_ratio * n_train_val))

    # Validation must have at least 1 sample
    val_size = max(1, val_size)

    # Validation cannot take all samples
    val_size = min(val_size, n_train_val - 1)

    # For stratification, validation size must be >= number of classes
    stratified_val_size = val_size

    if n_classes > 1:
        stratified_val_size = max(val_size, n_classes)
        stratified_val_size = min(stratified_val_size, n_train_val - 1)

    can_stratify_train_val = (
        n_classes > 1
        and counts.min() >= 2
        and stratified_val_size >= n_classes
        and (n_train_val - stratified_val_size) >= n_classes
    )

    return can_stratify_train_val, val_size, stratified_val_size
  
  
################################################## Three fold cross validation ############################################  

def three_fold_cross_validation(dataset_name, data, no_of_fold, run_id):
    event_col = "event" if "event" in data.columns else "status"
    status_data = np.asarray(data[event_col]).reshape(-1).astype(int) 
    class_counts = Counter(status_data)

    # If any class has fewer samples than number of folds, StratifiedKFold is unsafe
    if min(class_counts.values()) < no_of_fold:
        splitter = KFold( n_splits=no_of_fold, shuffle=True, random_state=1234)
        folds = [test_idx for _, test_idx in splitter.split(data)]
    else:
        splitter = StratifiedKFold( n_splits=no_of_fold, shuffle=True, random_state=1234)
        folds = [test_idx for _, test_idx in splitter.split(data, status_data)]

    test_fold_id = run_id % no_of_fold
    test_idx = folds[test_fold_id]
    
    train_val_idx = np.setdiff1d(np.arange(len(data)), test_idx)
    train_val_status = status_data[train_val_idx]
    
    # Step 2: safe train/validation split
    n_train_val = len(train_val_idx) 
    if n_train_val < 2:
        raise ValueError(f"Dataset: {dataset_name} Train+val set has only {n_train_val} samples after test split.")
    
    can_stratify_train_val, val_size, stratified_val_size = check_stratified_split(train_val_status, n_train_val)

    # Stratify train/val only if every class has at least 2 samples
    if can_stratify_train_val:
        train_idx, val_idx = train_test_split( train_val_idx, test_size=stratified_val_size, random_state=1234,  shuffle=True, stratify=train_val_status )
    else:
        train_idx, val_idx = train_test_split( train_val_idx,  test_size=val_size,  random_state=1234,  shuffle=True,  stratify=None)

    return train_idx, val_idx, test_idx 

################################################## save train/val/test split #################################################

def normalize_train_val_test_data(train_data, val_data, test_data):
    """
    Fit StandardScaler using train data and transform the train, val and test data.
    Arguments: 
          train_data, val_data, test_data: pandas DataFrames
    Returns:
          train_data, val_data, test_data: pandas DataFrames with normalized numerical features
    """
    
    cols_to_remove = ['time', 'event']
    new_col_name = train_data.columns.tolist()
    
    feature_list = [x for x in new_col_name if x not in cols_to_remove]
    std_list = [c for c in feature_list if c.startswith('num_')]
    
    if len(std_list) == 0:
        return train_data, val_data, test_data
     
    # Convert num_ columns to float by replacing columns directly
    train_data[std_list] = train_data[std_list].astype(np.float64)
    val_data[std_list] = val_data[std_list].astype(np.float64)
    test_data[std_list] = test_data[std_list].astype(np.float64)
    
    scaler = StandardScaler().fit(train_data.loc[:, std_list])
    train_data.loc[:, std_list] = scaler.transform(train_data.loc[:, std_list])
    val_data.loc[:, std_list] = scaler.transform(val_data.loc[:, std_list])
    test_data.loc[:, std_list] = scaler.transform(test_data.loc[:, std_list]) 
    
    return train_data, val_data, test_data 
    
    
################################################## save train/val/test split #################################################

def _save_split(train_data, val_data, test_data, out_dir,
                global_cats_dict, bin_cols_name, multi_cols_name):
    """
    Impute, encode, and save train/val/test.
    """
    os.makedirs(out_dir, exist_ok=True)

    train_data, val_data, test_data = preprocess_missing(
        train_data, val_data, test_data
    )

    train_data, val_data, test_data = encode_with_schema(
        train_data,
        val_data,
        test_data,
        global_cats_dict,
        bin_cols_name,
        multi_cols_name,
    )

    train_data.to_csv(f"{out_dir}/train.csv", index=False)
    val_data.to_csv(f"{out_dir}/val.csv", index=False)
    test_data.to_csv(f"{out_dir}/test.csv", index=False) 