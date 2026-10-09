from .preprocessing.encoding import build_global_cat_schema 
from .preprocessing.dataset_cleaning import clean_dataset 
from .data_loader import load_raw_dataset, dataset_properties
     
import pandas as pd, numpy as np, os
from config import PROCESSED_DATA_DIR, DATASET_CONFIG 
from .utils import five_fold_cross_validation, three_fold_cross_validation
from .utils import _save_split, normalize_train_val_test_data
 
############################### Federated Partitions among clients with folds ###############################################################   
def federated_partition_with_fold(dataframe, dataset_name, region_column, n_folds, dict_center_idxs, global_cats_dict, bin_cols_name, multi_cols_name):
    """
    Split data among federated clients across cross-validation folds and store
    the corresponding train/validation/test splits.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Combined survival dataset.
    dataset_name : str
        Name of the dataset.
    region_column : str
        Column used to identify federated clients.
    n_folds : int
        Number of cross-validation folds.
    dict_center_idxs : dict
        Mapping from client IDs to their row indices.
    global_cats_dict : dict
        Global categorical feature schema.
    bin_cols_name : list
        Binary categorical feature names.
    multi_cols_name : list
        Multi-class categorical feature names.

    Returns
    -------
    dict
        Mapping from client IDs to their train/validation/test splits for each fold.
    """

    # Create the root directory for the processed dataset
    processed_dataset_directory = PROCESSED_DATA_DIR / dataset_name
    
    # Process each federated client independently 
    for center_id, center_indices in dict_center_idxs.items(): 
      center_indices = np.array(center_indices)
      
      # Extract data associated with the current client 
      client_data = dataframe.loc[center_indices].reset_index(drop=True) 
      client_dict = {'train': [], 'val': [], 'test': []}  
      
      # Create train/val/test splits for each cross-validation fold
      for run_id in range(n_folds):
          if n_folds == 3:
              train_idxs, val_idxs, test_idxs = three_fold_cross_validation(dataframe, client_data, n_folds, run_id)
          elif n_folds == 5:
              train_idxs, val_idxs, test_idxs = five_fold_cross_validation(dataframe, client_data, n_folds, run_id)
          else:
              raise ValueError(f"Unsupported number of folds: {n_folds}")
          
          # Extract train/val/test data and remove the client partition column
          train_data = client_data.loc[train_idxs].drop(columns=region_column, errors="ignore")
          val_data = client_data.loc[val_idxs].drop(columns=region_column, errors="ignore")
          test_data = client_data.loc[test_idxs].drop(columns=region_column, errors="ignore") 
          
          # Normalize and save the client-wise train/val/test splits   
          dataset_directory = processed_dataset_directory /  "FL" / str(center_id) / str(run_id)
          scaled_train_data, scaled_val_data, scaled_test_data = normalize_train_val_test_data(train_data.copy(), val_data.copy(), test_data.copy())
          _save_split(scaled_train_data, scaled_val_data, scaled_test_data, dataset_directory, global_cats_dict, bin_cols_name, multi_cols_name)  
          
          # Store unscaled client splits for constructing centralized data
          client_dict['train'].append(train_data)
          client_dict['val'].append(val_data)
          client_dict['test'].append(test_data)
      
      # Store all cross-validation splits for the current client
      dict_center_idxs[center_id] = client_dict
    
    # Combine client-wise splits to construct centralized data
    clients = dict_center_idxs.keys()  
    for run_id in range(n_folds):
      
      # Collect train/val/test splits from all clients for the current fold
      pd_train, pd_val, pd_test = [], [], [] 
      for client_id in clients:
          pd_train.append(dict_center_idxs[client_id]["train"][run_id])
          pd_val.append(dict_center_idxs[client_id]["val"][run_id])
          pd_test.append(dict_center_idxs[client_id]["test"][run_id])
      
      # Combine corresponding client splits into centralized datasets
      pd_train_data = pd.concat(pd_train).reset_index(drop=True)
      pd_val_data = pd.concat(pd_val).reset_index(drop=True)
      pd_test_data = pd.concat(pd_test).reset_index(drop=True)
      
      # Normalize and save the centralized train/val/test splits 
      dataset_directory = f"{processed_dataset_directory}/CL/0/{run_id}"
      scaled_pd_train_data, scaled_pd_val_data, scaled_pd_test_data = normalize_train_val_test_data(pd_train_data.copy(),  pd_val_data.copy(), pd_test_data.copy())
      _save_split(scaled_pd_train_data, scaled_pd_val_data, scaled_pd_test_data, dataset_directory, global_cats_dict, bin_cols_name, multi_cols_name) 
       
    return dict_center_idxs 

################################################## Dataset List Loader ########################################################  
def  prepare_datasets():
    """
    Load dataset from our repository and stores the centralized 
                      and client-wise splits in the processed_data directory. 
    """

    # Federated survival datasets configured for preprocessing
    federated_datasets = DATASET_CONFIG.keys()  
    for dataset_name in federated_datasets:
        df = load_raw_dataset(dataset_name)    
        
        # extract dataset properties and clean the dataset
        region_column, no_of_fold, _= dataset_properties(dataset_name)
        df = clean_dataset(df, dataset_name, region_column) 

        # Create the directory for the processed dataset if it doesn't exist 
        dataset_directory = PROCESSED_DATA_DIR / dataset_name 
        os.makedirs(dataset_directory, exist_ok=True)
        if df is None:
            print(f" Survival data {dataset_name} processing is skipped due to insufficient unique samples.")
            continue

        # Build global categorical schema and partition the dataset among centers
        global_cats_dict, bin_cols_name, multi_cols_name = build_global_cat_schema(df)  
        
        # Get unique regions
        unique_values = sorted(df[region_column].unique())
        dict_center_idxs = {}
        
        # Split data according to region across each center
        for _, region in enumerate(unique_values): 
              region_idxs = df.index[df[region_column] == region].tolist()
              dict_center_idxs[region] = region_idxs 
        
        # Distribute data among clients and store the train/val/test CSVs in file storage
        federated_partition_with_fold(df, dataset_name, region_column, no_of_fold, dict_center_idxs, global_cats_dict, bin_cols_name, multi_cols_name) 
        print(f" Survival data {dataset_name} processing is done.")
  
################################################## Entry Point (Threaded Execution) ##########################################

if __name__ == "__main__":
    prepare_datasets() 