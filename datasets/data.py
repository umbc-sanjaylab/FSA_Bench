from typing import List
import pandas as pd 
import os 
import numpy as np
import threading
from preprocess.large_sampled import *
from preprocess.small_sampled import process_small_datasets
from preprocess.federated import process_federated_datasets, federated_partition, federated_partition_with_fold, information_for_distributed_dataset
from preprocess.omics import process_high_dimensional_datasets
from preprocess.high_censored import information_for_high_censored_dataset, process_high_censored_datasets
from censoring import *
from preprocess.encoding import encode_with_schema, build_global_cat_schema  
from utils import _partition_split, _save_split, make_k_fold_cross_validation, five_fold_cross_validation, three_fold_cross_validation, normalize_train_val_test_data


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
    
    small_datasets = ["veteran", "follic", "pbc", "Lung_data", "Melanoma_clinical_data", "hepatoCellular",  "glioma", "wpbc", "ova", "Unemployment", "Melanoma", "micro.censure", "e1684", "zinc", "stagec", "nki70",  "mgus", "whas500", "ovarian", "breast", "pharmacoSmoking", "burn", "cost", "GBSG2", "FRTCS", "vlbw"]
    
    small_datasets = ["wpbc", "Melanoma", "micro.censure",  "zinc",  "nki70",  "breast",  "burn"]
    high_censored_datasets = ["dataDIVAT1", "dataDIVAT2", "dataDIVAT3", "divorce", "hdfail", "flchain", "prostateSurvival", "nwtco","actg"]
     
    print(dataset_name, n_parties)
    centralized_data = centralized_data.reset_index(drop=True)
    
    if federated_flag:
        region_column, no_of_fold, num_centers = information_for_distributed_dataset(dataset_name) 
        dict_center_idxs = federated_partition(centralized_data, num_centers, region_column)
        federated_partition_with_fold(dataset_directory, centralized_data, region_column, no_of_fold, dict_center_idxs, global_cats_dict, bin_cols_name, multi_cols_name)
    else:
        partition_list = ["centralized", "iid", "non-iid"]
        
        for run_id in range(no_of_fold):
            print(f"\nProcessing global fold: {run_id}")

            global_train_idx, global_val_idx, global_test_idx = None, None, None  
            if dataset_name in small_datasets:
                global_train_idx, global_val_idx, global_test_idx = three_fold_cross_validation(dataset_name, centralized_data.copy(), no_of_fold, run_id) 
            elif dataset_name in high_censored_datasets:
                global_train_idx, global_val_idx, global_test_idx = five_fold_cross_validation(dataset_name, centralized_data.copy(), no_of_fold, run_id) 
            else:
                global_train_idx, global_val_idx, global_test_idx = make_k_fold_cross_validation(centralized_data,  no_of_fold=no_of_fold, run_id=run_id)

            global_train = centralized_data.iloc[global_train_idx].copy().reset_index(drop=True)
            global_val = centralized_data.iloc[global_val_idx].copy().reset_index(drop=True)
            global_test = centralized_data.iloc[global_test_idx].copy().reset_index(drop=True) 
            print( f"Global fold {run_id}: train={global_train.shape}, val={global_val.shape}, test={global_test.shape}")  
            
            
            for partition in partition_list: 
              
              # Step 2: Save centralized split directly 
              if partition == "centralized":
                  train_c = global_train.copy()
                  val_c = global_val.copy()
                  test_c = global_test.copy()
                  
                  out_dir = f"{dataset_directory}/centralized/0/{run_id}"
                  train_data, val_data, test_data = normalize_train_val_test_data(train_c, val_c, test_c)
                  _save_split(  train_c,  val_c,  test_c,  out_dir,  global_cats_dict,   bin_cols_name,   multi_cols_name) 
                  
              # Step 3: Save IID and Non-IID  splits
              else:
                  print(f"Processing partition: {partition}, fold: {run_id}") 
                  train_partitions = _partition_split( global_train.copy(), partition, n_parties, seed_no, column )
                  val_partitions = _partition_split(global_val.copy(), partition, n_parties, seed_no, column)
                  test_partitions = _partition_split(global_test.copy(), partition, n_parties, seed_no, column)

                  client_ids = range(n_parties)
                  for client_id in client_ids:
                    
                      train_idx = train_partitions[client_id]
                      val_idx = val_partitions[client_id]
                      test_idx = test_partitions[client_id]

                      train_data = global_train.iloc[train_idx].copy()
                      val_data = global_val.iloc[val_idx].copy()
                      test_data = global_test.iloc[test_idx].copy()  

                      print(f"{partition} | client {client_id} | fold {run_id}: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}") 
                      out_dir = f"{dataset_directory}/{partition}/{client_id}/{run_id}"
                      
                      train_data, val_data, test_data = normalize_train_val_test_data(train_data, val_data, test_data)
                      _save_split( train_data, val_data, test_data, out_dir, global_cats_dict, bin_cols_name, multi_cols_name) 


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
    omics_datasets = ["ovarian1data", "Ovarian2_data", "GE1_data", "GE2_data", "GE3_data", "GE4_data", "GE5_data", "GE6_data",
                      "Melanoma_itraq_data", "Melanoma_nano_data", "gse4335", "AML_Bull",  "dataOvarian1", "gse3143", "gse1992", "MCLcleaned", "metabric-hd", "vdv", "chop", "phpl04K8a", "Bergamaschi"]

    large_datasets = ["UnempDur", "grace", "Framingham", "support2",  "acath",  "TRACE", "metabrick", "rdata",  "rott2"]  
    high_censored_datasets = ["dataDIVAT1", "dataDIVAT2", "dataDIVAT3", "divorce", "hdfail", "flchain", "prostateSurvival", "nwtco","actg"]   
    small_datasets = ["veteran", "follic", "pbc", "Lung_data", "Melanoma_clinical_data", "hepatoCellular",  "glioma", "wpbc", "ova", "Unemployment", "Melanoma", "micro.censure", "e1684", "zinc", "stagec", "nki70",  "mgus", "whas500", "ovarian", "breast", "pharmacoSmoking", "burn", "cost", "GBSG2", "FRTCS", "vlbw"] 
    federated_datasets = ["TCGA", "Aids2", "LeukSurv", "Pbc3", "Dialysis", "d.oropha.rec", "cgd", "uis",  "scania", "seer"] 
    
    small_datasets = ["wpbc", "Melanoma", "micro.censure",  "zinc",  "nki70",  "breast",  "burn"]
    dataset_name, column, prefix = None, None, None

    # Process and export SurvBenchmark datasets
    for name in high_censored_datasets:
        df = get_dataframe(name)
        no_of_clients = 0

        if name in omics_datasets:
            df, no_of_clients, no_of_fold = process_high_dimensional_datasets(df, name)
            prefix = "omics_data"

        elif name in small_datasets:
            df, no_of_clients, no_of_fold = process_small_datasets(df, name)
            prefix = "small_data"
          
        elif name in large_datasets:
            df, no_of_clients, no_of_fold = process_large_datasets(df, name) 
            prefix = "large_data"

        elif name in high_censored_datasets:
            df, no_of_fold, no_of_clients = process_high_censored_datasets(df, name) 
            prefix = "high_censored_data"

        elif name in federated_datasets:
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
        distribute_data_among_clients(dataset_name, dataset_directory, df, global_cats_dict, bin_cols_name, multi_cols_name, n_parties=no_of_clients, no_of_fold=no_of_fold, seed_no=1234, column=column, federated_flag=False)

        print(f" Survival data {name} processing is done.")

    return None

################################################################################################################################


################################################## Entry Point (Threaded Execution) ##########################################

if __name__ == "__main__":
    # Run main_job in a separate thread (useful if integrating into a larger pipeline)
    t = threading.Thread(target=get_dataset_names, name="dataset-thread")
    
    t.start()
    t.join()
