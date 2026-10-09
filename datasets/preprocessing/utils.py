import os 
import numpy as np  
from .missing_value import preprocess_missing
from .encoding import encode_with_schema 
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import numpy as np
from collections import Counter


############################### Federated Partitions ####################################################### 
  
def federated_partition(data, num_centers, region_column):
    """
    Split data according to region across each center
    Arguments:
        data -- combined data
        num_centers -- number of centers
        region_column -- region column of the dataframe. 
    Returns:
        Dict with center_id : indices of data assigned to center
    """

    # Get unique regions
    unique_values = sorted(data[region_column].unique())

    # Optional check
    if num_centers != len(unique_values):
        print(f"Warning: num_centers ({num_centers}) != number of regions ({len(unique_values)})")

    dict_center_idxs = {}
    for _, region in enumerate(unique_values):

        # Get indices for this region and store it in a dictionary
        region_idxs = data.index[data[region_column] == region].tolist()
        dict_center_idxs[region] = region_idxs

    return dict_center_idxs

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

################################################## Split Train/Val/Test data into 5 folds ##########################################

def _partition_split(split_data, partition, n_parties, seed_no, column=None):
    """
    Arguments: 
          Partition a given split: train / val / test.
    Returns 
          dict: client_id -> row indices relative to split_data
    """
    if partition == "federated":
        return federated_partition(split_data, n_parties, column)

    else:
        raise ValueError(f"Unknown partition: {partition}")

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
    
################################################## highly correlated features #################################################

def remove_highly_correlated_features(
    train_df,
    val_df,
    test_df,
    dataset_name,
    run_id,
    output_dir,
    time_column="time",
    event_column="event",
    threshold=0.8,
): 

    os.makedirs(output_dir, exist_ok=True)

    protected_cols = {time_column, event_column, "status"}

    merged_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    # Only true continuous numerical features
    num_features = [
        col for col in merged_df.columns
        if col.startswith("num_") and col not in protected_cols
    ]

    corr_df = merged_df[num_features].copy()

    corr_matrix = corr_df.corr()
    abs_corr_matrix = corr_matrix.abs()

    upper_triangle = abs_corr_matrix.where(
        np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(bool)
    )

    high_corr_pairs = []
    features_to_remove = set()

    for col in upper_triangle.columns:
        high_corr = upper_triangle[col][upper_triangle[col] > threshold]

        for row, abs_corr_value in high_corr.items():
            real_corr_value = corr_matrix.loc[row, col]
            high_corr_pairs.append((row, col, real_corr_value))

            # Keep the feature with fewer missing values
            row_missing = merged_df[row].isna().mean()
            col_missing = merged_df[col].isna().mean()

            if row_missing <= col_missing:
                features_to_remove.add(col)
            else:
                features_to_remove.add(row)

    txt_path = os.path.join(
        output_dir,
        f"{dataset_name}_fold_{run_id}_correlated_features.txt"
    )

    with open(txt_path, "w") as txt_file:
        txt_file.write(f"Dataset: {dataset_name}\n")
        txt_file.write(f"Fold: {run_id}\n")
        txt_file.write(f"Threshold: |corr| > {threshold}\n")
        txt_file.write("Applied only to num_ continuous numerical features\n")
        txt_file.write("=" * 70 + "\n\n")

        if len(high_corr_pairs) == 0:
            txt_file.write(f"No numerical feature pairs found with |corr| > {threshold}\n")
        else:
            txt_file.write(f"Numerical feature pairs with |corr| > {threshold}:\n\n")

            for row, col, corr_value in high_corr_pairs:
                txt_file.write(f"{row}  <-->  {col} : corr = {corr_value:.4f}\n")

        txt_file.write("\nFeatures selected for removal:\n")
        if len(features_to_remove) == 0:
            txt_file.write("None\n")
        else:
            for feature in sorted(features_to_remove):
                txt_file.write(f"{feature}\n")

    train_df = train_df.drop(columns=list(features_to_remove), errors="ignore")
    val_df = val_df.drop(columns=list(features_to_remove), errors="ignore")
    test_df = test_df.drop(columns=list(features_to_remove), errors="ignore")

    print(
        f"Removed correlated numerical features for {dataset_name}, "
        f"fold {run_id}: {sorted(features_to_remove)}"
    )
    print(f"Saved correlation report: {txt_path}")

    return train_df, val_df, test_df