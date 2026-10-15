import os
import pandas as pd
import logging  
from datetime import datetime 

#################################### Regions name for Federated Datasets ############################################################  

def get_client_name_for_federated_datasets(dataset):
    dataset_with_region = {
        "TCGA": ["South", "West", "Midwest", "Northeast", "Europe", "Canada", "Other"],
        "Aids2": ['NSW', 'Other', 'QLD', 'VIC'],
        "LeukSurv": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        "Pbc3": ['London', 'Copenhagen', 'Hvidovre', 'Barcelona', 'Munich', 'Lyon'],
        "Dialysis": [120, 128, 184, 217, 344, 561, 562, 641, 741, 743, 917, 918, 936, 954, 983, 984, 1029, 1039,
                     1048, 1051, 1053, 1061, 1064, 1070, 1071, 1072, 1093, 1100, 1159, 1372, 1437, 1654, 1692,
                     1699, 2026, 2210, 2211, 2242, 2320, 2351, 2611, 2632, 2811, 2844, 2987, 3064, 3596, 3675,
                     4122, 4157, 4163, 4779, 4825, 5089, 5135, 5419, 5423, 5432, 5481, 5495, 5513, 5514, 5543,
                     5653, 5681, 5692, 5768],
        "d.oropha.rec": [2, 5, 4, 6, 3, 1],
        "cgd": ['Europe:Amsterdam', 'Europe:other', 'US:NIH', 'US:other'],
        "uis": ["A", "B"],
        "scania":  [1, 2, 3, 4, 5],
        "seer": [1, 2, 3]
    }

    region_list = dataset_with_region[dataset]
    return region_list

   
#################################### Store Experiment Result ############################################################ 
 
def save_result(best_round, val_cindex, cindex, brier, nbll, d_calibration, mae_po, args):
    row = {
        "seed": args.init_seed,
        "fold_id": int(args.run_id),
        "dataset": args.dataset,
        "partition": args.partition,
        "model": args.model,
        "best_round": best_round,
        "val_cindex": val_cindex, 
        "c_idx": round(cindex, 4),
        "brier": round(brier, 4),
        "nbll": round(nbll, 4),
        "d_calibration": round(d_calibration, 4),
        "weighted_MAE_PO": round(mae_po, 4),
        "lr": round(float(args.lr), 5),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "dropout": float(args.dropout) 
    }

    df = pd.DataFrame([row])
    csv_path = os.path.join(args.logdir, f"results_{args.model}.csv")
    
    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)  


#########################  ##################################################
def create_partition_logger(dataset, model_name):
    log_directory = os.path.join( "log", "partition_level", dataset, model_name)
    os.makedirs(log_directory, exist_ok=True)

    log_path = os.path.join(log_directory, "partition_level_all_runs.log")
    logger_name = f"partition_level.{dataset}.{model_name}"
    logger = logging.getLogger(logger_name)
    
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter( fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler) 
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("")
    logger.info("#" * 90)
    logger.info(f"STARTING NEW RUN: run_id={run_id}, dataset={dataset}, model={model_name}")
    logger.info("#" * 90)

    return logger