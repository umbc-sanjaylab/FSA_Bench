import os 
import logging 
import argparse
import warnings 
import numpy as np 
import pandas as pd  
import torchtuples as tt
from pathlib import Path
from sksurv.util import Surv
from datetime import datetime 
from pycox.models import CoxPH
from .folds.score import global_importance
from scipy.stats import spearmanr, kendalltau
from lifelines.exceptions import ApproximationWarning
from .folds.config import TIME_COL, EVENT_COL,   SEED, B 
from .folds.utils import  load_central_csv, load_discretizer, pseudo_eval_times 
from .folds.utils import  get_data_model_directory_partition_wise
from .folds.utils import select_survival_background, load_model_partition
warnings.filterwarnings("ignore", category=ApproximationWarning)  

def split_xy(df, feature_cols):
    X_df = df[feature_cols].astype("float32") 
    y_sksurv = Surv.from_arrays( event=df[EVENT_COL].astype(bool).to_numpy(), time=df[TIME_COL].to_numpy(dtype="float32"))  # For SurvSHAP
    y_pycox = ( df[TIME_COL].to_numpy(dtype="float32"), df[EVENT_COL].to_numpy(dtype="int64"))  # For PyCox CoxPH 
    return X_df, y_sksurv, y_pycox
  
def ranking(ckpt_path, data_name, model_name, fold, chunk_size, logger): 
    global discretiser
    data_dir   = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/processed_data/federated_data/{data_name}/centralized"
    
    train_df = load_central_csv(data_dir, "train", fold)  
    val_df = load_central_csv(data_dir, "val", fold)
    test_df = load_central_csv(data_dir, "test", fold)

    feature_cols = [ column for column in train_df.columns if column not in (TIME_COL, EVENT_COL)]
    times = pseudo_eval_times( pd.concat( [train_df[TIME_COL], val_df[TIME_COL]], ignore_index=True)) 
    
    background_positions = select_survival_background(train_df, background_size = 100, random_state= 42)
    sampled_train_Df = train_df.iloc[background_positions]
     
    X_train_df, y_train_sksurv, _= split_xy(sampled_train_Df, feature_cols)
    X_test_df, y_test_sksurv, _ = split_xy(test_df, feature_cols)
    X_full_train_df, _, y_full_train_pycox = split_xy(train_df, feature_cols)
     
    in_features  =  len(feature_cols)
    out_features = 10 if model_name == "DeepPseudo" else 9   
    if model_name == "NNnPH": 
        model, _ = load_model_partition(model_name, ckpt_path, in_features=in_features, out_features=out_features, discretiser=discretiser) 
    else: 
        model, discretiser = load_model_partition(model_name, ckpt_path, in_features=in_features, out_features=out_features)
     
    if model_name == "Fed_CoxPH":                          
        opt = tt.optim.Adam(lr=1e-3, weight_decay=5e-5)
        net = CoxPH(model, opt)                       
        net.compute_baseline_hazards(X_full_train_df.to_numpy(dtype="float32"), y_full_train_pycox)
        logger.info("CoxPH object loaded successfully")
    else: 
        net = model 
    
    important_score = global_importance(fold, model_name, net, X_test_df, y_test_sksurv, background=X_train_df, y_background=y_train_sksurv, n_explain=None, times=times, discretizer=discretiser, B=B, method="sampling", random_state=SEED, features=feature_cols, chunk_size=chunk_size, num_workers=4, devices=[ "cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    logger.info("SurvSHAP completed for fold %d", fold)
    return  important_score, feature_cols


def validate_importance(scores, partition, expected_features):
    """Validate and reorder importance scores by feature name."""

    if not isinstance(scores, pd.Series):
        raise TypeError( f"{partition} importance must be a pandas Series, but received {type(scores)}")

    if not scores.index.is_unique:
        duplicates = scores.index[scores.index.duplicated()].tolist()
        raise ValueError(f"{partition} contains duplicate features: {duplicates}")

    missing_features = set(expected_features) - set(scores.index)
    unexpected_features = set(scores.index) - set(expected_features)

    if missing_features or unexpected_features:
        raise ValueError(f"{partition} feature-index mismatch. Missing={sorted(missing_features)}, unexpected={sorted(unexpected_features)}")

    scores = pd.to_numeric(scores, errors="raise")
    scores = scores.reindex(expected_features)
    scores.name = partition

    if not np.isfinite(scores.to_numpy()).all():
        invalid_features = scores.index[ ~np.isfinite(scores.to_numpy()) ].tolist()
        raise ValueError(f"{partition} contains NaN or infinite importance values for features: {invalid_features}")
    
    return scores


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

def get_args(): 
    parser = argparse.ArgumentParser( description="Run site-level interpretability analysis.")
    parser.add_argument("-d", "--dataset-group", type=int, required=True, choices=[1, 2, 3, 4, 5], help="Dataset group to run:")

    args = parser.parse_args()
    d = args.dataset_group
    return d 
  
if __name__=="__main__": 
    dataset = "Pbc3" 
    global discretiser
     
    d = get_args()  
    
    if d == 1: 
        dataset_name = "scania"
    elif d == 2: 
        dataset_name = "uis"
    elif d == 3: 
        dataset_name = "d.oropha.rec" 
    elif d == 4:
        dataset_name = "Dialysis"
    else: 
        dataset_name = "Aids2"
    
    models_name = ["FedPAttn", "FedPDNN", "FedPLSTM", "AFT_Weibull", "Fed_CoxPH", "LinearPH", "NNPH", "NNnPH", "DeepHit",  "DeepPseudo"]   
    data_dictionary = {
      "seer": {
            "folds": [0], # list(range(5)), 
            "chunk_size": 1500
        }, 
      "Pbc3":{
            "folds": list(range(3)), 
            "chunk_size": 100 
        }, 
      "LeukSurv":{
            "folds": list(range(3)), 
            "chunk_size": 100 
        },
      "seer": {
        "folds":  list(range(5)),
        "chunk_size": 100
      }, 
      "Pbc3": {
        "folds": list(range(3)),
        "chunk_size": 100  
      },
      "LeukSurv": {
        "folds": list(range(3)),
        "chunk_size": 100 
      }, 
      "cgd":{
        "folds": list(range(3)), 
        "chunk_size": 100 
      }, 
      "scania":{
        "folds": list(range(5)), 
        "chunk_size": 100 
      }, 
      "uis":{
        "folds": list(range(5)), 
        "chunk_size": 100 
      }, 
      "d.oropha.rec":{
        "folds": list(range(3)), 
        "chunk_size": 100 
      },   
      "Dialysis":{
        "folds": list(range(3)), 
         "chunk_size": 100 
      }, 
      "Aids2":{
        "folds": list(range(5)), 
         "chunk_size": 100 
      }    
    } 
     
    
    for model_name in models_name:
        rows = []
        logger = create_partition_logger(dataset, model_name)

        logger.info("=" * 70)
        logger.info("Starting partition-level analysis: dataset=%s, model=%s", dataset, model_name)

        folds = data_dictionary[dataset]["folds"]
        chunk_size = data_dictionary[dataset]["chunk_size"] 
        
        # Use the actual fold values, not range(len(folds)).
        for fold in folds:
            CL_Model_Path, FL_Model_Path =  get_data_model_directory_partition_wise( dataset, model_name, fold=fold) 

            cl_path = Path(CL_Model_Path).resolve()
            fl_path = Path(FL_Model_Path).resolve()
            
            if not cl_path.exists():
                raise FileNotFoundError( f"CL checkpoint path does not exist: {cl_path}")

            if not fl_path.exists():
                raise FileNotFoundError( f"FL checkpoint path does not exist: {fl_path}")

            if cl_path == fl_path:
                raise RuntimeError( f"CL and FL paths are identical for fold={fold}: {cl_path}")
 
            logger.info(f"Fold {fold} checkpoint paths:\nCL: {cl_path}\nFL: {fl_path}")

            if model_name == "NNnPH":
                discretiser = load_discretizer(FL_Model_Path)
 
            # Keep the original importance scores.
            cl_scores, cl_features = ranking(CL_Model_Path, dataset, model_name, fold, chunk_size, logger)
            fl_scores, fl_features = ranking(FL_Model_Path, dataset, model_name, fold, chunk_size, logger)

            if cl_features != fl_features:  
                 raise ValueError(f"CL and FL feature columns differ for fold={fold}.")
            
            cl_scores = validate_importance(cl_scores, "CL", cl_features)
            fl_scores = validate_importance(fl_scores, "FL", cl_features)

            # Align CL and FL explicitly by feature name.
            paired = pd.concat( [cl_scores, fl_scores], axis=1, join="outer")

            # Missing values here usually indicate mismatched feature names.
            if paired.isna().any().any():
                missing_in_cl = paired.index[paired["CL"].isna()].tolist()
                missing_in_fl = paired.index[paired["FL"].isna()].tolist()

                raise ValueError( f"Feature mismatch for fold={fold}. Missing in CL={missing_in_cl}; missing in FL={missing_in_fl}")
              
            if len(paired) < 2:
                logger.warning(f"Fold={fold}: Fewer than two features are available; correlation is undefined.")
                spearman_coeff = np.nan
                kendall_tau_coeff = np.nan

            elif paired["CL"].nunique() < 2 or paired["FL"].nunique() < 2:
                logger.warning(f"Fold={fold}: Correlation is undefined because at least one partition has constant importance scores. CL_unique={paired['CL'].nunique()},  FL_unique={paired['FL'].nunique()}")
                spearman_coeff = np.nan
                kendall_tau_coeff = np.nan

            else:
                spearman_coeff = spearmanr(paired["CL"], paired["FL"]).correlation
                kendall_tau_coeff = kendalltau(paired["CL"],paired["FL"]).correlation

            # Store ranks only for inspection.
            paired["CL_rank"] = paired["CL"].rank(method="average", ascending=False)
            paired["FL_rank"] = paired["FL"].rank( method="average", ascending=False)

            identical_scores = np.allclose( paired["CL"], paired["FL"], rtol=1e-7, atol=1e-10)
            identical_ranks = np.array_equal( paired["CL_rank"].to_numpy(), paired["FL_rank"].to_numpy())
 
            logger.info(f"Fold {fold} feature comparison:\n{paired.sort_values('CL_rank').to_string()}") 
            logger.info(f"Fold={fold}, Spearman={spearman_coeff:.6f}, Kendall tau={kendall_tau_coeff:.6f}, identical_scores={identical_scores}, identical_ranks={identical_ranks}")

            rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "CLvsFL_spearman": spearman_coeff,
                    "CLvsFL_kendall": kendall_tau_coeff,
                    "identical_scores": identical_scores,
                    "identical_ranks": identical_ranks,
                }
            )

        clfl = pd.DataFrame(rows)
        logger.info( "Fold-level partition results:\n%s", clfl.to_string(index=False))
        
        correlation_columns = ["CLvsFL_spearman", "CLvsFL_kendall"]
        valid_counts = clfl[correlation_columns].notna().sum()
        mean_result = clfl[correlation_columns].mean()

        logger.info(f"Valid correlation counts:\n{valid_counts.to_string()}")

        if (valid_counts < len(folds)).any():
            logger.warning(f"Some correlations are undefined. Total folds={len(folds)}; valid counts:\n {valid_counts.to_string()}")

        logger.info( f"Mean partition-level correlations:\n {mean_result.to_string()}")