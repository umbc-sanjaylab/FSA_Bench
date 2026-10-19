import itertools
import numpy as np
import pandas as pd
import torchtuples as tt

from pycox.models import CoxPH
from scipy.stats import kendalltau, spearmanr

from config import MODELS
from ..core.score import global_importance
from ..config import B, DATASET_CHUNKS, SEED
from ..core.data import prepare_site_data, prepare_site_eval_times
from ..core.utils import (
    create_site_logger,
    get_args, get_folds, 
    load_model_partition,
    close_logger, get_sites, 
    get_partition_checkpoint_paths,
)

  
############################################# calculate feature importance for a particular client ########################################
   
def site_importance(model, discretiser, model_name, fold, site, site_data, times, chunk_size, logger):
    logger.info( f"\nComputing site-level SurvSHAP: " f"dataset={dataset}, model={model_name}, " f"site={site}, fold={fold}" )
    feature_cols, X_bg, y_bg, X_test, y_test, X_train, y_train_pycox = site_data
    
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

    importance = importance.reindex(feature_cols) 
    if not np.isfinite(importance.to_numpy()).all():
        raise ValueError(f"Site {site}, fold {fold} contains invalid feature importance values.")

    logger.info("Site %s feature importance:\n%s", site, importance.sort_values(ascending=False).to_string())
    return importance
  
############################################# calculate correlations between clients for a particular fold ########################################
def calculate_site_correlations(imps, sites, fold, logger):
    rows = [] 
    for site_a, site_b in itertools.combinations(sites, 2):
        x, y = imps[site_a], imps[site_b]
        valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)

        if valid.sum() < 2 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
            spearman, kendall = np.nan, np.nan
        else:
            spearman = spearmanr(x[valid], y[valid]).correlation
            kendall = kendalltau(x[valid], y[valid]).correlation

        rows.append((fold, site_a, site_b, spearman, kendall, int(valid.sum())))

    result = pd.DataFrame( rows,
        columns=["fold", "site_a", "site_b", "spearman", "kendall", "valid_features"]
    )

    logger.info("Fold %d site correlations:\n%s", fold, result.to_string(index=False))
    return result
  
############################################# calculate Jaccard Value for top K features ########################################
   
def topk(series, k=10):
    return set(series.nlargest(k).index)

def jaccard(a, b):
    union = a | b
    if not union:
        return np.nan
    else: 
        return len(a & b) / len(union) 

# def compute_Jaccard_value(imps, sites,  fold, k, logger):
def calculate_site_jaccard(imps, sites, fold, k, logger):
    rows = [] 
    for site_a, site_b in itertools.combinations(sites, 2):
        score = jaccard(topk(imps[site_a], k), topk(imps[site_b], k))
        rows.append((fold, site_a, site_b, score))
      
    result = pd.DataFrame(rows, columns=["fold", "site_a", "site_b", "jaccard"])
    logger.info("Fold %d top-%d Jaccard:\n%s", fold, k, result.to_string(index=False))
    return result

def determine_jaccard_k(number_of_features, top_fraction=0.20, minimum_k=2):
    if number_of_features < 2:
        raise ValueError("Jaccard analysis requires at least two features.")

    proportional_k = int(np.ceil(top_fraction * number_of_features))
    return min(number_of_features - 1, max(minimum_k, proportional_k))
 

####################################### Arguments Method ####################################################################

if __name__ == "__main__":
    out_features = 9 
    args = get_args( "Run site-level interpretability analysis.")
    
    dataset =  args.dataset 
    folds = get_folds(dataset)  
    
    sites = get_sites(dataset)
    chunk_size = DATASET_CHUNKS[dataset]  
    
    site_data = {} 
    eval_times = {} 
    
    for fold in folds: 
      fold_data = {} 
      
      eval_times[fold] = prepare_site_eval_times(dataset, sites, fold) 
      for site in sites:  
        fold_data[site] = prepare_site_data(dataset, site, fold) 
      
      site_data[fold] = fold_data   
    
    for model_name in MODELS:
        logger = create_site_logger(dataset, model_name)
        all_correlations = []
        all_jaccard = [] 
        
        try:
            for fold in folds:   
                feature_cols = site_data[fold][sites[0]][0]
                
                _, checkpoint_path = get_partition_checkpoint_paths(dataset, model_name, fold) 
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"FL checkpoint not found: {checkpoint_path}")
                
                model, discretiser = load_model_partition(model_name, checkpoint_path,
                    in_features=len(feature_cols),  out_features=10 if model_name == "DeepPseudo" else 9
                )

                imps = {}
                rank_dataframe = {}

                for site in sites: 
                    imps[site] = site_importance( 
                                      model, discretiser, model_name, fold, site,
                                      site_data[fold][site], eval_times[fold], chunk_size, logger 
                                  )
                
                corr_df = calculate_site_correlations(imps, sites, fold, logger)

                k = determine_jaccard_k(len(feature_cols), top_fraction=0.40)
                jaccard_df = calculate_site_jaccard(imps, sites, fold, k, logger)

                all_correlations.append(corr_df)
                all_jaccard.append(jaccard_df)
 
            # Overall summary across every fold.
            correlation_df = pd.concat(all_correlations, ignore_index=True)
            jaccard_df = pd.concat(all_jaccard, ignore_index=True) 
            
            logger.info(
                "Overall mean Spearman=%.4f | Kendall=%.4f | Jaccard=%.4f",
                correlation_df["spearman"].mean(),
                correlation_df["kendall"].mean(),
                jaccard_df["jaccard"].mean()
            ) 

        finally:
            close_logger(logger)
        