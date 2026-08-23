import numpy as np
import logging
import os
import pandas as pd
from datetime import datetime 
from pathlib import Path

def set_logger_config(logdir):
    os.makedirs(logdir, exist_ok=True) 
    log_path = ( f"zero_event_report-" f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")

    # Remove old handlers to avoid duplicate log messages
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    logging.basicConfig( filename=os.path.join(logdir, log_path), level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%m-%d %H:%M", filemode="w" ) 
    return logging.getLogger("experiment")


def log_local_split_statistics( logger, dataset_name, partition_name, client_name, run_id, split_name, client_data ):
    time_col = "time"

    if "event" in client_data.columns:
        event_col = "event"
    elif "status" in client_data.columns:
        event_col = "status"
    else:
        raise ValueError(
            f"Neither 'event' nor 'status' column exists for "
            f"dataset={dataset_name}, partition={partition_name}, "
            f"client={client_name}, fold={run_id}, split={split_name}"
        )

    if time_col not in client_data.columns:
        raise ValueError(
            f"'time' column does not exist for "
            f"dataset={dataset_name}, partition={partition_name}, "
            f"client={client_name}, fold={run_id}, split={split_name}"
        )

    times = np.asarray(client_data[time_col]).reshape(-1)
    statuses = (
        np.asarray(client_data[event_col])
        .reshape(-1)
        .astype(int)
    )

    number_of_events = np.sum(statuses == 1)
    number_of_censored = np.sum(statuses == 0)

    # Log only when the split has zero events
    if number_of_events == 0:
        logger.warning(
            f"ZERO EVENTS: "
            f"dataset={dataset_name}, "
            f"partition={partition_name}, "
            f"client={client_name}, "
            f"fold={run_id}, "
            f"split={split_name}, "
            f"n={len(statuses)}, "
            f"events={number_of_events}, "
            f"censored={number_of_censored}, "
            f"min_time={times.min() if len(times) else None}, "
            f"max_time={times.max() if len(times) else None}"
        )
        logger.warning("*"*60)

    return number_of_events == 0


def inspect_client_data(file_path, client_id, df, feature_names=None):
    time_col = "time"

    if "event" in df.columns:
        event_col = "event"
    elif "status" in df.columns:
        event_col = "status"
        
    X = np.array(df.drop(columns=["time", "event"]), dtype='float32')
    durations, events = df["time"], df["event"] 
    
    X = np.asarray(X, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64).reshape(-1)
    events = np.asarray(events).reshape(-1)

    logger.info(f"\n--- Client {client_id} data check ---")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"NaN in X: {np.isnan(X).sum()}")
    logger.info(f"Inf in X: {np.isinf(X).sum()}")
    logger.info(f"NaN in durations: {np.isnan(durations).sum()}")
    logger.info(f"Inf in durations: {np.isinf(durations).sum()}")
    logger.info(f"Non-positive durations: {np.sum(durations <= 0)}")
    logger.info(f"Event distribution: {np.unique(events, return_counts=True)}")

    std = np.nanstd(X, axis=0)
    zero_variance_indices = np.where(std < 1e-12)[0]

    if feature_names is not None:
        zero_variance_features = [ feature_names[index] for index in zero_variance_indices  ]
        logger.info(f"Zero-variance features: {zero_variance_features}")
    else:
        logger.info(f"Zero-variance column indices: {zero_variance_indices}")

    if not np.isfinite(X).all():
        raise ValueError(f"Client {client_id}: X contains NaN or Inf")

    if not np.isfinite(durations).all():
        raise ValueError(f"Client {client_id}: durations contain NaN or Inf")

    if np.any(durations <= 0):
        invalid_positions = np.flatnonzero(durations <= 0)

        if invalid_positions.size > 0:
            invalid_values = durations[invalid_positions]

            raise ValueError(
                f"Path: {file_path}\n"
                f"Client: {client_id}\n"
                f"Weibull model requires durations > 0.\n"
                f"Invalid positions: {invalid_positions.tolist()}\n"
                f"Invalid duration values: {invalid_values.tolist()}"
            )
            
    if not np.isin(events, [0, 1]).all():
        raise ValueError(
            f"Client {client_id}: event must contain only 0 and 1"
        )

if __name__ == "__main__":
    prefix = "small_data"

    base_directory =  "/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets" 
    partition_list = [ "centralized", "federated"] #,"iid", "non-iid"]

    small_datasets = [ "veteran", "follic", "pbc", "Lung_data", "hepatoCellular", "wpbc", "ova", "Unemployment", "Melanoma", "micro.censure", "e1684", "zinc", "stagec", "nki70", "mgus", "whas500", "pharmacoSmoking", "burn", "cost", "GBSG2", "FRTCS", "vlbw", "glioma", "breast", "ovarian", "Melanoma_clinical_data" ] 
    total_fold = [ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 ] 
    total_batch_size = [ 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64 ]
    number_of_clients = [ 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 2, 2, 2, 2 ]
    
    prefix = "federated_data"
    federated_datasets = [ "Aids2" ] 
    total_fold = [ 5] 
    total_batch_size = [ 64]
    number_of_clients = [4]
    federated_flag = True 

    assert len(federated_datasets) == len(total_fold)
    assert len(federated_datasets) == len(number_of_clients)

    # Configure one consolidated log file
    logdir = f"{base_directory}/log/{prefix}/zero_event_report"
    logger = set_logger_config(logdir)

    total_zero_event_splits = 0

    for dataset_index, dataset_name in enumerate(federated_datasets):
        dataset_directory = ( f"{base_directory}/processed_data/{prefix}/{dataset_name}")

        temp_df = pd.read_csv(f"{dataset_directory}/centralized/0/0/train.csv")
        col_name = temp_df.drop(columns=["time", "event"]).columns
        
        fold_count = total_fold[dataset_index]
        dataset_client_count = number_of_clients[dataset_index]

        for partition in partition_list:
            client_count = ( 1 if partition == "centralized" else dataset_client_count )
            
            if partition == "centralized": 
                client_list = range(client_count)  
            elif federated_flag: 
                federated_directory = Path(dataset_directory) / "federated"
                client_list = sorted( folder.name for folder in federated_directory.iterdir() if folder.is_dir() )
                 
            else: 
                client_list = range(client_count)
            
            for client in client_list:
                for run_id in range(fold_count):
                    out_dir = f"{dataset_directory}/{partition}/{client}/{run_id}"
                      
                    split_files = {
                        "train": f"{out_dir}/train.csv",
                        "val": f"{out_dir}/val.csv",
                        "test": f"{out_dir}/test.csv"
                    }

                    for split_name, file_path in split_files.items():
                        split_data = pd.read_csv(file_path) 
                        split_data["time"] = pd.to_numeric(split_data["time"], errors="raise").astype(np.float64)
                        zero_mask = split_data["time"].eq(0)
                        
                        if zero_mask.any():
                            zero_indices = split_data.index[zero_mask].tolist()
                            zero_count = int(zero_mask.sum())

                            logger.warning(
                                "Replacing %d zero duration(s) with 0.5: "
                                "dataset=%s, partition=%s, client=%s, "
                                "fold=%s, split=%s, indices=%s, path=%s",
                                zero_count,
                                dataset_name,
                                partition,
                                client,
                                run_id,
                                split_name,
                                zero_indices,
                                file_path,
                            )

                            # Replace zero with 0.5
                            split_data.loc[zero_mask, "time"] = 0.5

                            # Save the corrected dataframe to the same CSV
                            split_data.to_csv(file_path, index=False)
                        
                        
                         
                       
                        inspect_client_data(file_path, client, split_data, feature_names=col_name) 

                        has_zero_events = log_local_split_statistics( logger=logger, dataset_name=dataset_name, partition_name=partition, client_name=client, run_id=run_id, split_name=split_name, client_data=split_data )

                        if has_zero_events:
                            total_zero_event_splits += 1

    logger.info(f"Scan completed. Total zero-event splits={total_zero_event_splits}" )

    print( f"Scan completed. Found {total_zero_event_splits} " f"split(s) with zero events." )
    print(f"Report directory: {logdir}")