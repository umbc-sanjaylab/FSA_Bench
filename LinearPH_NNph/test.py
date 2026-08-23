from utils import get_local_dataset
import pandas as pd 
import numpy as np

def get_local_dataset(dataset, partition, client_id, run_id, datadir):
    if partition == 'centralized':
        client_id = 0

    client_name = None
    dataset_directory = f"{datadir}/{dataset}/{partition}/{client_id}/{run_id}/"
    client_name = client_id

    train_df = pd.read_csv(dataset_directory + "train.csv")
    val_df = pd.read_csv(dataset_directory + "val.csv")
    test_df = pd.read_csv(dataset_directory + "test.csv")

    cols_to_remove = ['time', 'event']
    new_col_name = train_df.columns.tolist()
    feature_list = [x for x in new_col_name if x not in cols_to_remove]

    return train_df, val_df, test_df, client_name, feature_list

n_parties = 1
datadir = "/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/processed_data/1small_data/" 
dataset = "micro.censure"
partition = "centralized"
run_id = 0

for party_id in range(n_parties):

        # Load local dataset
        train_df, val_df, test_df, client_name, feature_list = get_local_dataset(
            dataset, partition, party_id, run_id, datadir)
        
        X_train = np.array(train_df.drop(
            columns=["time", "event"]), dtype='float32')
        X_val = np.array(val_df.drop(
            columns=["time", "event"]), dtype='float32')
        X_test = np.array(test_df.drop(
            columns=["time", "event"]), dtype='float32')
        
        train_time, train_status = train_df["time"], train_df["event"]
        val_time, val_status = val_df["time"], val_df["event"]
        test_time, test_status = test_df["time"], test_df["event"]

        train_time = np.asarray(train_time).reshape(-1)
        train_status = np.asarray(train_status).reshape(-1).astype(int)

        val_time = np.asarray(val_time).reshape(-1)
        val_status = np.asarray(val_status).reshape(-1).astype(int)

        test_time = np.asarray(test_time).reshape(-1)
        test_status = np.asarray(test_status).reshape(-1).astype(int)
        
        print("Train events:", train_status.sum())
        print("Val events:", val_status.sum())
        print("Test events:", test_status.sum())

        print("Train shape:", X_train.shape)
        print("Val shape:", X_val.shape)
        print("Test shape:", X_test.shape)
        
        print("NaN train:", np.isnan(X_train).sum())
        print("NaN val:", np.isnan(X_val).sum())
        print("NaN test:", np.isnan(X_test).sum())

        print("Inf train:", np.isinf(X_train).sum())
        print("Inf val:", np.isinf(X_val).sum())
        print("Inf test:", np.isinf(X_test).sum())

        print("Train time min/max:", train_time.min(), train_time.max())
        print("Val time min/max:", val_time.min(), val_time.max())
        print("Test time min/max:", test_time.min(), test_time.max())

        print("Unique train event times:", len(np.unique(train_time[train_status == 1])))
        print("Unique test event times:", len(np.unique(test_time[test_status == 1])))