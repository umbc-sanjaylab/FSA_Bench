import argparse, warnings 
import numpy as np
import pandas as pd
import torchtuples as tt
from pycox.models import CoxPH
from scipy.stats import kendalltau, spearmanr 
from lifelines.exceptions import ApproximationWarning

from .lib.config import B, DATASET_CHUNKS, DATASET_FOLDS, SEED  
from .lib.data import prepare_fold_data 
from .lib.score import global_importance
from utils import create_partition_logger
from .lib.utils import get_partition_checkpoint_paths 
from .lib.utils import get_folds, load_model_partition 
warnings.filterwarnings("ignore", category=ApproximationWarning)   

def ranking(checkpoint_path, model_name, fold, fold_data, chunk_size, logger):
    feature_cols, times, X_bg, y_bg, X_test, y_test, X_train, y_train_pycox = fold_data

    model, discretiser = load_model_partition(
        model_name, checkpoint_path,
        in_features=len(feature_cols),
        out_features=10 if model_name == "DeepPseudo" else 9
    )

    if model_name == "Fed_CoxPH":
        net = CoxPH(model, tt.optim.Adam(lr=1e-3, weight_decay=5e-5))
        net.compute_baseline_hazards(X_train.to_numpy(dtype="float32"), y_train_pycox)
    else:
        net = model

    importance = global_importance(
        fold, model_name, net, X_test, y_test,
        background=X_bg, y_background=y_bg, times=times,
        discretizer=discretiser, B=B, method="sampling",
        random_state=SEED, features=feature_cols,
        chunk_size=chunk_size, num_workers=4,
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )

    logger.info("SurvSHAP completed for fold %d", fold)
    return importance 


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

  
######################### Compare feature importance for both CL & FL Settings ##################################################
def compare_importance(cl_scores, fl_scores, fold, logger):
    paired = pd.concat([cl_scores.rename("CL"), fl_scores.rename("FL")], axis=1)

    if len(paired) < 2 or paired["CL"].nunique() < 2 or paired["FL"].nunique() < 2:
        spearman, kendall = np.nan, np.nan
        logger.warning("Fold %d: feature-rank correlation is undefined.", fold)
    else:
        spearman = spearmanr(paired["CL"], paired["FL"]).correlation
        kendall = kendalltau(paired["CL"], paired["FL"]).correlation

    paired["CL_rank"] = paired["CL"].rank(method="average", ascending=False)
    paired["FL_rank"] = paired["FL"].rank(method="average", ascending=False)

    identical_scores = np.allclose(paired["CL"], paired["FL"], rtol=1e-7, atol=1e-10)
    identical_ranks = np.array_equal(paired["CL_rank"].to_numpy(), paired["FL_rank"].to_numpy())

    logger.info("Fold %d feature comparison:\n%s", fold, paired.sort_values("CL_rank").to_string())

    return {
        "fold": fold,
        "CLvsFL_spearman": spearman,
        "CLvsFL_kendall": kendall,
        "identical_scores": identical_scores,
        "identical_ranks": identical_ranks
    }


####################################### Arguments Method ####################################################################

def get_args(description): 
    parser = argparse.ArgumentParser( description=description) 
    parser.add_argument("-d", "--dataset", required=True, choices=DATASET_FOLDS.keys(), help="Dataset to analyze.") 
    return parser.parse_args()  
  
if __name__=="__main__":  
    args = get_args("Compare centralized and federated feature importance.")
    
    dataset =  args.dataset 
    folds = get_folds(dataset)   
    
    chunk_size = DATASET_CHUNKS[dataset]
    models_name = ["FedPAttn", "FedPDNN", "FedPLSTM", "AFT_Weibull", "Fed_CoxPH", "LinearPH", "NNPH", "NNnPH", "DeepHit",  "DeepPseudo"]   
    
    for model_name in models_name:
        rows = []
        logger = create_partition_logger(dataset, model_name)

        logger.info("=" * 70)
        logger.info("Starting partition-level analysis: dataset=%s, model=%s", dataset, model_name) 
        
        # Use the actual fold values, not range(len(folds)).
        for fold in folds:
            cl_path, fl_path = get_partition_checkpoint_paths(dataset, model_name, fold)

            if not cl_path.exists():
                raise FileNotFoundError(f"CL checkpoint not found: {cl_path}")
            if not fl_path.exists():
                raise FileNotFoundError(f"FL checkpoint not found: {fl_path}")

            logger.info("Fold %d checkpoints:\nCL: %s\nFL: %s", fold, cl_path, fl_path)

            fold_data = prepare_fold_data(dataset, fold)

            cl_scores = ranking(cl_path, model_name, fold, fold_data, chunk_size, logger)
            fl_scores = ranking(fl_path, model_name, fold, fold_data, chunk_size, logger)
            
            feature_cols = fold_data[0]
            cl_scores = validate_importance(cl_scores, "CL", feature_cols)
            fl_scores = validate_importance(fl_scores, "FL", feature_cols)

            result = compare_importance(cl_scores, fl_scores, fold, logger)
            result["model"] = model_name
            rows.append(result)

        clfl = pd.DataFrame(rows)
        logger.info( "Fold-level partition results:\n%s", clfl.to_string(index=False))
        
        correlation_columns = ["CLvsFL_spearman", "CLvsFL_kendall"]
        valid_counts = clfl[correlation_columns].notna().sum()
        mean_result = clfl[correlation_columns].mean()

        logger.info(f"Valid correlation counts:\n{valid_counts.to_string()}")

        if (valid_counts < len(folds)).any():
            logger.warning(f"Some correlations are undefined. Total folds={len(folds)}; valid counts:\n {valid_counts.to_string()}")

        logger.info( f"Mean partition-level correlations:\n {mean_result.to_string()}")