import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from utils import five_fold_cross_validation, three_fold_cross_validation, _save_split, normalize_train_val_test_data
import os 


################################################## Federated dataset information ################################################ 

def information_for_distributed_dataset(dataset_name):
    column, no_of_fold, no_of_clients = "", 0, 0
    match dataset_name:
        case "TCGA":
            column, no_of_fold, no_of_clients = "Region", 5, 7
        case "Aids2":
            column, no_of_fold, no_of_clients = "fac_state", 5, 4
        case "LeukSurv":
            column, no_of_fold, no_of_clients = "fac_district", 3, 24
        case "Pbc3":
            column, no_of_fold, no_of_clients = "fac_unit", 3, 6
        case "Dialysis":
            column, no_of_fold, no_of_clients = "fac_center", 3, 67
        case "d.oropha.rec":
            column, no_of_fold, no_of_clients = "fac_inst", 3, 6
        case "cgd":
            column, no_of_fold, no_of_clients = "fac_hos_cat", 3, 4
        case "uis":
            column, no_of_fold, no_of_clients = "fac_site", 5, 2
        case "cancer":
            column, no_of_fold, no_of_clients = "fac_inst", 1, 19
        case "scania":
            column, no_of_fold, no_of_clients = "fac_parish", 5, 5
        case "seer":
            column, no_of_fold, no_of_clients = "Area_3", 5, 3
    return column, no_of_fold, no_of_clients

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

############################################### Preprocess Federated Survival Datasets #####################################################


def process_federated_datasets(df, file_name, column):

    # A study of survival of patients from 17 cancer
    if file_name == "TCGA":

        # rename of the covariates and status variable
        df.rename(columns={"status": "event", "age_at_index": "num_age_at_index",
                  "age_at_diagnosis": "num_age_at_diagnosis"}, inplace=True)

        #  rename the columns with binary suffix
        remove_cols = ['event', 'num_age_at_index',
                       'num_age_at_diagnosis', 'time', column]
        fac_columns = [col for col in df.columns if col not in remove_cols]
        df.rename(
            columns={c: f"fac_{c}_binary" for c in fac_columns}, inplace=True)

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # common multi-centric datasets
    elif file_name in ["Aids2", "Pbc3"]:

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0"], errors="ignore")

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # LeukSurv dataset
    elif file_name == "LeukSurv":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0",
                     "num_xcoord", "num_ycoord"], errors="ignore")

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # Dialysis dataset
    elif file_name == "Dialysis":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0",
                     "num_begin"], errors="ignore")

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # carcinoma of the oropharynx
    elif file_name == "d.oropha.rec":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0"], errors="ignore")

        # rename the covariates with _ordinal suffix
        fac_cols = ["fac_treatm", "fac_grade", "fac_cond",
                    "fac_site", "fac_tstage", "fac_nstage"]
        df = df.rename(columns={col: f"{col}_ordinal" for col in fac_cols})

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # Chronic Granulomatous Disease that weakens white blood cells
    elif file_name == "cgd":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0",
                     "fac_center"], errors="ignore")

        # Ordinal encoding of the covariate and map the encoding to the column
        agestr_dict = {
            "fac_treat": {'placebo': 0, 'rIFN-g': 1},
            "fac_sex": {'female': 0, 'male': 1},
            "fac_inherit": {"X-linked": 0, "autosomal": 1},
            "fac_steroids": {'N': 1, 'Y': 0},
            "fac_propylac": {'N': 1, 'Y': 0}
        }

        for key, value in agestr_dict.items():
            df[key] = df[key].map(value)

        # rename the columns with ordinal suffix
        rename_dict = {"fac_treat": "fac_treat_binary", "fac_sex": "fac_sex_binary",
                       "fac_propylac": "fac_propylac_binary", "fac_inherit": "fac_inherit_binary", "fac_steroids": "fac_steroids_binary"}
        df = df.rename(columns=rename_dict)

        # cast the type of the covariates to numeric
        for col in rename_dict.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # Study of drug addicts undergoing treatment by University of Illinois
    elif file_name == "uis":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0"], errors="ignore")

        # binary encoding and rename the covariate with _binary suffix
        df['fac_treat'] = df['fac_treat'].map({'short': 0, 'long': 1})
        df = df.rename(columns={'fac_treat': 'fac_treat_binary'})

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # Study of old age life histories above age 50 at rural area of scania, southern Sweden
    elif file_name == "scania":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0",
                     "num_birthdate"], errors="ignore")

        # Ordinal encoding of the covariate and map the encoding to the column
        agestr_dict = {
            "fac_sex": {'female': 0, 'male': 1},
            "fac_immigrant": {'no': 1, 'yes': 0},
            "fac_ses": {'lower': 1, 'upper': 0}
        }

        for key, value in agestr_dict.items():
            df[key] = df[key].map(value)

        # rename the columns with ordinal suffix
        rename_dict = {"fac_ses": "fac_ses_binary",
                       "fac_sex": "fac_sex_binary", "fac_immigrant": "fac_immigrant_binary"}
        df = df.rename(columns=rename_dict)

        # cast the type of the covariates to numeric
        for col in rename_dict.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # reset index of dataframe
        df = df.reset_index(drop=True)

    #  A study of cancer statistics by the SEER program in the United States.
    elif file_name == "seer":

        # drop unnecessary columns
        df = df.drop(columns=["pid", "Unnamed: 0"], errors="ignore")

        # rename the columns with ordinal suffix
        rename_dict = {"status": "event", "Ageatdiagnosis": "num_Ageatdiagnosis", "Race_ord": "fac_Race_ordinal", "Marital_3": "fac_Marital_3_ordinal",
                       "Histology_ord": "fac_Histology_ordinal", "Grade_3": "fac_Grade_3_ordinal", "Tumor_size_ord": "fac_Tumor_size_ordinal",
                       "T.stage_ord": "fac_T_stage_ordinal", "N_stage_ord": "fac_N_stage_ordinal", "M_stage_ord": "fac_M_stage_ordinal",
                       "RT_ord": "fac_RT_ordinal", "Chemo_di": "fac_Chemo_di_ordinal", "Surg_ord": "fac_Surg_ordinal"}
        df = df.rename(columns=rename_dict)

        # cast the type of the covariates to numeric
        for col in rename_dict.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # reset index of dataframe
        df = df.reset_index(drop=True)

    # Make compeing risks events binary (1=event occurred, 0=censored/other event)
    if "event" in df.columns:
        df["event"] = (df["event"] == 1).astype("int64")

    return df

###############################################################################################################################################
