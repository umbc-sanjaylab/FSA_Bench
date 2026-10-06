import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from .utils import five_fold_cross_validation, three_fold_cross_validation, _save_split, normalize_train_val_test_data  
 
#################################### Get number of clients and folds ########################################################
def region_wise_cross_validation(centralized_data, no_of_fold, run_id):
    skf = StratifiedKFold(n_splits=no_of_fold,  shuffle=True, random_state=1234)

    event_col = "event" if "event" in centralized_data.columns else "status"
    status_data =  np.asarray(centralized_data[event_col], dtype="float32").reshape(-1).astype(int)

    train_idx_array, val_idx_array, test_idx_array = [], [], []
    for train_val_idx, test_idx in skf.split(centralized_data, status_data):
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2,  random_state=1234,  shuffle=True, stratify=status_data[train_val_idx])

        train_idx_array.append(train_idx)
        val_idx_array.append(val_idx)
        test_idx_array.append(test_idx)

    fold_id = run_id % no_of_fold
    return  train_idx_array[fold_id], val_idx_array[fold_id], test_idx_array[fold_id] 
 
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
  
############################### Federated Partitions among clients with folds ###############################################################  

def federated_partition_with_fold(data_directory, data, region_column, fold, dict_center_idxs, global_cats_dict, bin_cols_name, multi_cols_name):
    """
    split data among client with #folds and store it
    Arguments:
        data -- combined data
        dict_center_idxs -- number of centers
    """ 
 
    for center_id, center_indices in dict_center_idxs.items():
        
      center_indices = np.array(center_indices)
      client_data = data.loc[center_indices].reset_index(drop=True)
      
      client_dict = {
        'train': [], 
        'val': [], 
        'test': []
      }  
      
      for run_id in range(fold): 
          if fold == 3: 
              train_idxs, val_idxs, test_idxs = three_fold_cross_validation(data, client_data, fold, run_id) 
          elif fold == 5: 
              train_idxs, val_idxs, test_idxs = five_fold_cross_validation(data, client_data, fold, run_id) 
          
          train_data = client_data.loc[train_idxs].drop(columns=region_column, errors="ignore")
          val_data = client_data.loc[val_idxs].drop(columns=region_column, errors="ignore")
          test_data = client_data.loc[test_idxs].drop(columns=region_column, errors="ignore") 
           
          dataset_directory = f"{data_directory}/federated/{center_id}/{run_id}" 
          scaled_train_data, scaled_val_data, scaled_test_data = normalize_train_val_test_data(train_data.copy(), val_data.copy(), test_data.copy())
          _save_split(scaled_train_data, scaled_val_data, scaled_test_data, dataset_directory, global_cats_dict, bin_cols_name, multi_cols_name)  
          
          client_dict['train'].append(train_data)
          client_dict['val'].append(val_data)
          client_dict['test'].append(test_data)
            
      dict_center_idxs[center_id] = client_dict
    
    clients = dict_center_idxs.keys()  
    for run_id in range(fold):
      
      pd_train, pd_val, pd_test = [], [], [] 
      for c1 in clients: 
        pd_train.append(dict_center_idxs[c1]['train'][run_id])
        pd_val.append(dict_center_idxs[c1]['val'][run_id])
        pd_test.append(dict_center_idxs[c1]['test'][run_id]) 
      
      pd_train_data = pd.concat(pd_train).reset_index(drop=True)
      pd_val_data = pd.concat(pd_val).reset_index(drop=True)
      pd_test_data = pd.concat(pd_test).reset_index(drop=True)
      
      dataset_directory = f"{data_directory}/centralized/0/{run_id}"
      scaled_pd_train_data, scaled_pd_val_data, scaled_pd_test_data = normalize_train_val_test_data(pd_train_data.copy(),  pd_val_data.copy(), pd_test_data.copy())
      _save_split(scaled_pd_train_data, scaled_pd_val_data, scaled_pd_test_data, dataset_directory, global_cats_dict, bin_cols_name, multi_cols_name) 
       
    return dict_center_idxs
 