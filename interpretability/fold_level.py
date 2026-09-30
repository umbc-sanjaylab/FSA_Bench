import argparse
import itertools

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import StratifiedShuffleSplit

from pathlib import Path
import torchtuples as tt
from pycox.models import CoxPH

from interpretability.folds.config import TIME_COL, EVENT_COL, SEED, B
from interpretability.folds.score import global_importance
from interpretability.folds.utils import (
    load_fold_csv,
    get_data_model_directory_fold_level,
    load_model_site,
    split_xy,
    pseudo_eval_times,
    create_model_logger   
)

from config import MODELS 
from utils import get_args, close_logger, get_folds


def select_survival_background( train_df: pd.DataFrame, background_size: int = 200, random_state: int = 42, ) -> np.ndarray:
    """
    Return positional indices for a representative survival background.

    Stratification uses:
        # event status
        # survival-time quantile
    """
    n_train = len(train_df)
    if n_train == 0:
        raise ValueError("train_df is empty.")

    # If the requested background includes every row, no sampling is needed.
    if background_size >= n_train:
          return np.arange(n_train)
      
    event = train_df[EVENT_COL].astype(int).reset_index(drop=True)
    time = train_df[TIME_COL].astype(float).reset_index(drop=True)

    # Rank first so repeated survival times do not cause qcut problems.
    number_of_bins = min(5, time.nunique())

    if number_of_bins >= 2:
        time_bin = pd.qcut( time.rank(method="first"), q=number_of_bins, labels=False, duplicates="drop")
        strata =  event.astype(str) + "_" + time_bin.astype(str)
    else:
        strata = event.astype(str)

    splitter = StratifiedShuffleSplit( n_splits=1, train_size=background_size, random_state=random_state)
    try:
        n_train = len(train_df)
        background_positions, _ = next( splitter.split( np.zeros(n_train), strata,)) 
    except ValueError: 
        try:
            background_positions, _ = next( splitter.split( np.zeros(n_train), event)) 
        except ValueError: 
            rng = np.random.default_rng(random_state)
            background_positions = rng.choice( n_train, size=background_size, replace=False, )

    return np.sort(background_positions)  

def fold_importance(logger, data_dir, dataset, model_name, client, fold, feature_cols, times, in_features, out_features): 
    test_df  = load_fold_csv(data_dir, "test",  fold=fold)
    train_df = load_fold_csv(data_dir, "train", fold=fold) 
    
    logger.info( "Fold %d sizes: train=%d,  test=%d", fold, len(train_df), len(test_df))  
    background_positions = select_survival_background(train_df, background_size = 100, random_state= 42)
    
    sampled_train_df = train_df.iloc[background_positions]
    logger.info("Fold %d background size: %d", fold, len(sampled_train_df))
   
    # global CKPT_TMPL  
    ckpt_path = get_data_model_directory_fold_level(model_name, dataset, fold, client=client)  
    logger.info( "Fold %d checkpoint: %s", fold, ckpt_path)
     
    X_test,  y_test_sksurv,  _ = split_xy(test_df, feature_cols)
    X_train, y_train_sksurv, y_train_pycox = split_xy(sampled_train_df, feature_cols) 
    
    model, discretiser = load_model_site(model_name, ckpt_path, in_features=in_features, out_features=out_features)
    
    net = model 
    if model_name == "Fed_CoxPH":
        opt = tt.optim.Adam(lr=1e-3, weight_decay=5e-5) 
        net = CoxPH(model, opt) 
        net.compute_baseline_hazards(X_train.to_numpy(dtype="float32"), y_train_pycox)
 
    logger.info( "Starting SurvSHAP: fold=%d, model=%s, test=%d", fold, model_name, len(X_test))   
    important_score = global_importance(fold, model_name, net, X_test, y_test_sksurv, background=X_train, y_background=y_train_sksurv, n_explain=None, times=times, discretizer=discretiser, B=B, method="sampling", random_state=SEED, features=feature_cols, chunk_size=1500, num_workers=4, devices=[ "cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    logger.info("SurvSHAP completed for fold %d", fold)
    return  important_score  

def get_centralized_data_dir(dataset):
    PROJECT_ROOT = Path(__file__).resolve().parents[1] 
    PROCESSED_DATA_DIR =  PROJECT_ROOT / "datasets" / "processed_data" / "federated_data"
    return PROCESSED_DATA_DIR / dataset  / "centralized" 

def calculate_importance(dataset, model_name, folds):  
    data_dir = get_centralized_data_dir(dataset) 
    print(data_dir)   
    
    # fold loop
    imps = {}
    logger = create_model_logger(dataset, model_name, log_folder="fold_level") 
    
    logger.info("Data directory: %s", data_dir)
    logger.info("Folds: %s", folds)

    ref = load_fold_csv(data_dir, "train", 0)
    feature_cols = [c for c in ref.columns if c not in (TIME_COL, EVENT_COL)]

    logger.info("Number of features: %d", len(feature_cols))
    out_features = 10 if model_name == "DeepPseudo" else 9  

    
    for fold in folds:  
        logger.info("=" * 70)
        logger.info("Starting fold %d", fold)
        
        train_times = load_fold_csv( data_dir, "train", fold )[TIME_COL].to_numpy() 
        val_times = load_fold_csv( data_dir, "val", fold )[TIME_COL].to_numpy()

        all_train_val_times = np.concatenate([ train_times, val_times]) 
        times = pseudo_eval_times(all_train_val_times, pseudo_interval=10)  
         
        important_score = fold_importance( logger, data_dir, dataset, model_name, client=0, fold=fold, feature_cols=feature_cols, times=times,  in_features=len(feature_cols),  out_features=out_features)
        imps[fold] =  important_score.reindex(feature_cols)  
        
        logger.info("Completed fold %d", fold)
        logger.info( "Fold %d global feature importance:\n%s", fold, important_score.sort_values( ascending=False ).to_string())
        
        # Ensure records are visible immediately.
        for handler in logger.handlers:
            handler.flush()  
    
    return imps, feature_cols, logger

def calculate_spearman_correlation(imps, feature_cols, folds, logger):
    pairs = []

    for a, b in itertools.combinations(folds, 2):
        x = imps[a].reindex(feature_cols)
        y = imps[b].reindex(feature_cols)

        valid = x.notna() & y.notna()
        rho = spearmanr(x[valid], y[valid]).correlation
        tau = kendalltau(x[valid], y[valid]).correlation 
        pairs.append((a, b, rho, tau))

    corr_df = pd.DataFrame( pairs, columns=["fold_a", "fold_b", "spearman", "kendall"]).dropna() 
    print(corr_df)
    logger.info( "Pairwise correlations:\n%s", corr_df.to_string(index=False))
    
    print(f"mean spearman={corr_df['spearman'].mean():.3f} and mean kendall={corr_df['kendall'].mean():.3f}")
    logger.info(f"mean spearman={corr_df['spearman'].mean():.3f} and mean kendall={corr_df['kendall'].mean():.3f}")

def stability_computation(imps, feature_cols, model_name, logger): 
    I = pd.DataFrame(imps).reindex(feature_cols)          # features x folds, raw mean|SurvSHAP|
    I = I / I.sum(axis=0)                                 # per-fold L1 normalize -> relative shares

    mu    = I.mean(axis=1)
    sigma = I.std(axis=1, ddof=1)                         # across folds
    cv    = np.divide(sigma, mu, out=np.full_like(mu, np.nan), where=mu > 0)
    stab  = (1.0 - cv).clip(lower=0.0, upper=1.0)         # per-feature stability

    m = mu > 0                                            # drop features that are 0 in every fold
    w = (mu[m] / mu[m].sum())                             # importance weights
    model_stability_weighted = float((w * stab[m]).sum())
    model_stability_mean = float(stab[m].mean())

    sdf = pd.DataFrame({"mu": mu, "sigma": sigma, "cv": cv, "stability": stab}).sort_values("mu", ascending=False)
    print(f"[{model_name}] stability  weighted={model_stability_weighted:.3f}  unweighted={model_stability_mean:.3f}")
    logger.info(f"[{model_name}] stability  weighted={model_stability_weighted:.3f}  unweighted={model_stability_mean:.3f}")
    
    print(sdf)
    logger.info( "Feature stability:\n%s", sdf.to_string()) 
            
if __name__ == "__main__": 
    args = get_args( "Run fold-level interpretability analysis.")
    dataset = args.dataset
    
    folds = get_folds(dataset) 
    for model_name in MODELS: 
        imps, feature_cols, logger = calculate_importance( dataset, model_name, folds) 
        calculate_spearman_correlation(imps, feature_cols, folds, logger) 

        stability_computation(imps, feature_cols, model_name, logger)  
        logger.info("Experiment completed successfully")  
        close_logger(logger)