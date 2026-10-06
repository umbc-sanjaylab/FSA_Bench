import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler 

# Preprocess the numerical covariates with missing values  
def knn_impute_numeric(train_data, val_data, test_data, n_neighbors=3, weights="uniform"):
    train = train_data.copy()
    val   = val_data.copy()
    test  = test_data.copy()

    num_cols = [c for c in train.columns if c.startswith("num_")]
    if not num_cols:
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    # cols_with_nan = [col for col in num_cols if train[col].isna().any()]
    
    # Only select columns that actually have missing values    
    cols_with_nan = [
        col for col in num_cols
        if train[col].isna().any() or val[col].isna().any() or test[col].isna().any()
    ]

    # If no missing values → skip
    if not cols_with_nan:
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    # Handle edge case: all-NaN columns (IMPORTANT)
    all_nan_cols = [col for col in cols_with_nan if train[col].isna().all()]
    valid_cols   = [col for col in cols_with_nan if col not in all_nan_cols]

    # Fill completely NaN columns with 0 (or median)
    for col in all_nan_cols:
        train[col] = train[col].fillna(0)
        val[col]   = val[col].fillna(0)
        test[col]  = test[col].fillna(0)

    # If no valid columns left → return
    if not valid_cols:
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    # If only 1 column → fallback to mean (KNN not meaningful)
    if len(valid_cols) == 1:
        col = valid_cols[0]
        mean_val = train[col].mean()

        train[col] = train[col].fillna(mean_val)
        val[col]   = val[col].fillna(mean_val)
        test[col]  = test[col].fillna(mean_val)

    else:
        # Apply KNN only on valid columns
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)

        train[valid_cols] = imputer.fit_transform(train[valid_cols])

        if len(val) > 0:
            val[valid_cols] = imputer.transform(val[valid_cols])

        if len(test) > 0:
            test[valid_cols] = imputer.transform(test[valid_cols])

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

# Preprocess the categorical covariates with missing values
def mode_impute_categorical(train_data, val_data, test_data):
    train = train_data.copy()
    val   = val_data.copy()
    test  = test_data.copy()

    # find out the categorical columns in the dataset
    fac_cols = [c for c in train.columns if c.startswith("fac_")]
    if not fac_cols:
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    # Only select columns that actually contain missing values
    cols_to_impute = []
    for col in fac_cols:
        has_nan = train[col].isna().any()
        has_missing_str = (train[col] == "missing").any()
        
        if has_nan or has_missing_str:
            cols_to_impute.append(col)

    # If there is no categorical column with missing values, return the original dataframes without modification
    if not cols_to_impute:
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
      
    # Fit modes on TRAIN only
    modes = {}
    for col in cols_to_impute:
        # treat "missing" as NaN before computing mode
        temp = train[col].replace("missing", np.nan)
        mode_vals = temp.mode(dropna=True)
        modes[col] = mode_vals.iloc[0] if not mode_vals.empty else np.nan 

    # Transform train/val/test using train modes
    for col in cols_to_impute:
        fill_val = modes[col]
        train[col] = train[col].replace("missing", np.nan).fillna(fill_val)
        val[col]   = val[col].replace("missing", np.nan).fillna(fill_val)
        test[col]  = test[col].replace("missing", np.nan).fillna(fill_val)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
 
# Preprocess the covariates with missing values
def preprocess_missing(train_data, val_data, test_data, n_neighbors=3, weights="uniform"):
    """
    Wrapper: mode-impute categorical (fit on train), then KNN-impute numerical (fit on train).
    Returns: train_clean, val_clean, test_clean
    """
    train, val, test = mode_impute_categorical(train_data, val_data, test_data)
    train, val, test = knn_impute_numeric(train, val, test, n_neighbors=n_neighbors, weights=weights)
    return train, val, test