import numpy as np, itertools
import pandas as pd
import torchtuples as tt

from pycox.models import CoxPH
from scipy.stats import kendalltau, spearmanr

from config import MODELS
from .lib.config import B, DATASET_CHUNKS, SEED
from .lib.data import prepare_fold_data
from .lib.score import global_importance
from .lib.utils import (
    close_logger,
    create_model_logger,
    get_args,
    get_folds,
    get_partition_checkpoint_paths,
    load_model_partition
) 

#################################### Fold Feature Importance ####################################

def fold_importance(dataset, model_name, fold, fold_data, chunk_size, logger):
    feature_cols, times, X_bg, y_bg, X_test, y_test, X_train, y_train_pycox = fold_data 
    _, checkpoint_path = get_partition_checkpoint_paths(dataset, model_name, fold)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FL checkpoint not found: {checkpoint_path}")

    logger.info("Fold %d checkpoint: %s", fold, checkpoint_path)

    out_features = 10 if model_name == "DeepPseudo" else 9
    model, discretiser = load_model_partition(model_name, checkpoint_path, len(feature_cols), out_features)

    if model_name == "Fed_CoxPH":
        model = CoxPH(model, tt.optim.Adam(lr=1e-3, weight_decay=5e-5))
        model.compute_baseline_hazards(X_train.to_numpy(dtype="float32"), y_train_pycox)

    importance = global_importance(
        fold, model_name, model, X_test, y_test,
        background=X_bg, y_background=y_bg, times=times,
        discretizer=discretiser, B=B, method="sampling",
        random_state=SEED, features=feature_cols,
        chunk_size=chunk_size, num_workers=4,
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )

    importance = importance.reindex(feature_cols)
    if not np.isfinite(importance.to_numpy()).all():
        raise ValueError(f"Fold {fold} contains invalid feature-importance values.")

    logger.info("Fold %d feature importance:\n%s", fold, importance.sort_values(ascending=False).to_string())
    return importance
 

#################################### Calculate Fold Importance ####################################

def calculate_importance(dataset, model_name, folds, fold_data, chunk_size, logger):
    feature_cols = fold_data[folds[0]][0]
    imps = {}
    for fold in folds:
        logger.info("=" * 70)
        logger.info("Starting fold %d", fold)

        if fold_data[fold][0] != feature_cols:
            raise ValueError(f"Feature columns differ for fold {fold}.")
        imps[fold] = fold_importance(  dataset, model_name, fold,  fold_data[fold], chunk_size, logger)
        
    
    return imps, feature_cols 
 
#################################### Fold Correlation ####################################

def calculate_fold_correlations(imps, feature_cols, folds, logger):
    rows = []
    for fold_a, fold_b in itertools.combinations(folds, 2):
        x = imps[fold_a].reindex(feature_cols)
        y = imps[fold_b].reindex(feature_cols)
        valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)

        if valid.sum() < 2 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
            spearman, kendall = np.nan, np.nan
        else:
            spearman = spearmanr(x[valid], y[valid]).correlation
            kendall = kendalltau(x[valid], y[valid]).correlation

        rows.append((fold_a, fold_b, spearman, kendall))

    corr_df = pd.DataFrame(rows, columns=["fold_a", "fold_b", "spearman", "kendall"])
    logger.info("Pairwise fold correlations:\n%s", corr_df.to_string(index=False))
    
    logger.info( "Mean Spearman=%.3f | Mean Kendall=%.3f", corr_df["spearman"].mean(), corr_df["kendall"].mean()) 
    return corr_df   

#################################### Feature Stability ####################################

def stability_computation(imps, feature_cols, model_name, logger):
    importance = pd.DataFrame(imps).reindex(feature_cols)
    column_sums = importance.sum(axis=0)

    if (column_sums <= 0).any():
        raise ValueError(f"{model_name} contains a fold with zero total importance.")

    importance = importance.div(column_sums, axis=1) 
    mu = importance.mean(axis=1)
    sigma = importance.std(axis=1, ddof=1)
    
    cv = sigma / mu.replace(0, np.nan)
    stability = (1 - cv).clip(0, 1)

    valid = mu > 0
    weights = mu[valid] / mu[valid].sum()

    weighted_stability = float((weights * stability[valid]).sum())
    mean_stability = float(stability[valid].mean())

    stability_df = pd.DataFrame({
        "mu": mu,
        "sigma": sigma,
        "cv": cv,
        "stability": stability
    }).sort_values("mu", ascending=False)

    logger.info( "[%s] stability | weighted=%.3f | unweighted=%.3f", model_name, weighted_stability, mean_stability)
    logger.info("Feature stability:\n%s", stability_df.to_string())
    return stability_df  

#################################### Main ####################################
if __name__ == "__main__":
    args = get_args("Run fold-level interpretability analysis.")

    # extract parameters from the arguments 
    dataset = args.dataset
    folds = get_folds(dataset)
    chunk_size = DATASET_CHUNKS[dataset]

    # Prepare each fold once and reuse it for every model.
    fold_data = {fold: prepare_fold_data(dataset, fold) for fold in folds}

    for model_name in MODELS:
        logger = create_model_logger(dataset, model_name, log_folder="fold_level")

        try:
            logger.info( "Starting fold-level analysis | Dataset: %s | Model: %s", dataset, model_name) 
            imps, feature_cols = calculate_importance( dataset, model_name, folds, fold_data, chunk_size, logger )

            calculate_fold_correlations(imps, feature_cols, folds, logger)
            stability_computation(imps, feature_cols, model_name, logger) 
            logger.info("Experiment completed successfully") 
        finally:
            close_logger(logger)