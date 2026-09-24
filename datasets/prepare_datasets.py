from typing import List
import pandas as pd 
import os  
import threading  
from federated import process_federated_datasets, federated_partition, federated_partition_with_fold, information_for_distributed_dataset
from encoding import build_global_cat_schema  


################################################## Read Raw CSV #############################################################

def get_dataframe(name: str) -> pd.DataFrame:
    """
    Read a dataset CSV from the raw_data directory
    Arguments:
        name -- dataset file name (without extension)
    Returns:
        DataFrame loaded from /datasets/raw_data/{name}.csv
    """
    df = pd.read_csv(f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/raw_data/{name}.csv")
    return df
  

################################################## Federated data Partition #########################################################

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
        print(
            f"Warning: num_centers ({num_centers}) != number of regions ({len(unique_values)})")

    dict_center_idxs = {}
    for _, region in enumerate(unique_values):

        # Get indices for this region and store it in a dictionary
        region_idxs = data.index[data[region_column] == region].tolist()
        dict_center_idxs[region] = region_idxs

    return dict_center_idxs 



################################################## Client-wise Dataset Export #################################################


def distribute_data_among_clients(dataset_name, dataset_directory,  centralized_data, global_cats_dict, bin_cols_name, multi_cols_name, n_parties=5, no_of_fold=5, seed_no=1234, column=None, federated_flag=False):
    """
    New fair split pipeline:

    full dataset
        -> create global folds once
        -> centralized uses global train/val/test
        -> IID splits global train/val/test separately across clients
        -> non-IID splits global train/val/test separately across clients
    """ 
     
    print(dataset_name, n_parties)
    centralized_data = centralized_data.reset_index(drop=True)
    
    if federated_flag:
        region_column, no_of_fold, num_centers = information_for_distributed_dataset(dataset_name) 
        dict_center_idxs = federated_partition(centralized_data, num_centers, region_column)
        federated_partition_with_fold(dataset_directory, centralized_data, region_column, no_of_fold, dict_center_idxs, global_cats_dict, bin_cols_name, multi_cols_name)
     

################################################## Dataset List Loader ########################################################  
def get_dataset_names() -> List[str]:
    """
    Load dataset from our repository and stores the centralized, IID, non-IID, and  client-wise splits in the processed_data directory.
    Arguments:
          None
    Returns:
          None 
    """

    # Manually added SurvBenchmark datasets (raw CSVs already present) 
    federated_datasets = ["TCGA", "Aids2", "LeukSurv", "Pbc3", "Dialysis", "d.oropha.rec", "cgd", "uis",  "scania", "seer"]  
    dataset_name, column, prefix = None, None, None

    # Process and export SurvBenchmark datasets
    for name in federated_datasets:
        df = get_dataframe(name)
        no_of_clients = 0  
        
        column, no_of_fold, no_of_clients = information_for_distributed_dataset(name)
        df = process_federated_datasets(df, name, column)
        prefix = "federated_data"

        dataset_name = name
        dataset_directory = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/processed_data/{prefix}/{dataset_name}"
        os.makedirs(dataset_directory, exist_ok=True)

        if df is None:
            print(f" Survival data {name} processing is skipped due to insufficient unique samples.")
            continue

        # Distribute among clients and write train/val/test CSVs
        global_cats_dict, bin_cols_name, multi_cols_name = build_global_cat_schema(df)
        distribute_data_among_clients(dataset_name, dataset_directory, df, global_cats_dict, bin_cols_name, multi_cols_name, n_parties=no_of_clients, no_of_fold=no_of_fold, seed_no=1234, column=column, federated_flag=True)

        print(f" Survival data {name} processing is done.")

    return None 
  
################################################## Entry Point (Threaded Execution) ##########################################

if __name__ == "__main__":
    # Run main_job in a separate thread (useful if integrating into a larger pipeline)
    t = threading.Thread(target=get_dataset_names, name="dataset-thread")
    
    t.start()
    t.join()
