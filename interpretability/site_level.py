import os 
import logging  
import itertools
import argparse
import numpy as np 
import pandas as pd
import torchtuples as tt
from pycox.models import CoxPH
from datetime import datetime 
from collections import Counter
from itertools import combinations
from predict import predict_surv_matrix
from scipy.stats import spearmanr, kendalltau
from fold.config import TIME_COL, EVENT_COL, SEED, B 
from fold.utils import  load_site_csv, load_central_site_csv, split_xy
from fold.utils import  pseudo_eval_times, get_data_model_directory_site_level
from fold.utils import get_Discretizer, select_survival_background, pseudo_eval_times, load_model_site
from fold.score import global_importance 


def create_site_logger(dataset, model_name):
    log_directory = os.path.join( "log", "site_level", dataset, model_name)
    os.makedirs(log_directory, exist_ok=True)

    # All folds and future executions are stored here.
    log_path = os.path.join(log_directory, "site_level_all_runs.log")

    logger_name = f"site_level.{dataset}.{model_name}"
    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove handlers left by an earlier initialization.
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    # Append instead of overwriting earlier runs 
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler) 
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("")
    logger.info("#" * 90)
    logger.info(f"STARTING NEW RUN: run_id={run_id}, dataset={dataset}, model={model_name}")
    logger.info("#" * 90)

    return logger, run_id, log_path 

def site_importance(data_dir, logger, dataset, model_name, site, checkpoint_path, fold, feature_cols, in_features, out_features, times, discretiser):
    print( f"\nComputing site-level SurvSHAP: " f"dataset={dataset}, model={model_name}, " f"site={site}, fold={fold}" )
    
    logger.info("=" * 70)
    logger.info(f"Starting site {site}: dataset={dataset}, model={model_name}, fold={fold}") 
    
    test_df = load_site_csv(data_dir,  "test", site, fold)
    train_df = load_site_csv(data_dir, "train", site, fold)
    logger.info(f"Site {site} sizes: train={len(train_df)}, test={len(test_df)}")
    
    background_positions = select_survival_background( train_df, background_size=100, random_state=42)
    sampled_train_df = train_df.iloc[ background_positions].copy() 
    logger.info(f"Site {site} background size: {len(sampled_train_df)}")
    
    X_test_df, y_test_sksurv, _ = split_xy( test_df, feature_cols)
    X_train_df, y_train_sksurv, y_train_pycox = split_xy( sampled_train_df, feature_cols)

    X_bg = X_train_df.copy()
    y_bg = y_train_sksurv.copy()
    logger.info(f"Site {site} checkpoint: {checkpoint_path}")
     
    global_model, loaded_discretizer = load_model_site(model_name, checkpoint_path, in_features=in_features, out_features=out_features)

    # Prefer a discretizer loaded with the checkpoint when available.
    if loaded_discretizer is not None:
        prediction_discretizer = loaded_discretizer
    else:
        prediction_discretizer = discretiser

    if model_name == "Fed_CoxPH":
        optimizer = tt.optim.Adam( lr=1e-3, weight_decay=5e-5, )

        net = CoxPH(global_model, optimizer)
        net.compute_baseline_hazards( X_train_df.to_numpy(dtype="float32"), y_train_pycox, )

        global_model = net 
        print( f"CoxPH object loaded for site {site}" )
        logger.info(f"CoxPH object loaded for site {site}")
    
    logger.info(f"Starting SurvSHAP for site {site} test observations={len(X_test_df)}") 

    importance = global_importance(fold=fold, model_name=model_name, model=global_model, X=X_test_df, y=y_test_sksurv, background=X_bg, y_background=y_bg, n_explain=None, times=times, discretizer=prediction_discretizer, B=B, method="sampling", random_state=SEED, features=feature_cols, chunk_size=1500, num_workers=4, devices=[ "cuda:0", "cuda:1", "cuda:2", "cuda:3", ], )

    logger.info(f"Completed SurvSHAP for site {site}")
    logger.info(f"Site {site} global feature importance: {importance.sort_values(ascending=False).to_string()} \n")

    for handler in logger.handlers:
        handler.flush()
        
    return importance
   
def topk(series, k=10):
    return set( series .dropna() .sort_values(ascending=False).head(k).index) 

def jac(set_a, set_b):
    union = set_a | set_b

    if not union:
        return np.nan

    return len(set_a & set_b) / len(union) 

def compute_Jaccard_value(imps, sites, logger, fold, k):
    rows = []

    for site_a, site_b in combinations(sites, 2):
        set_a = topk(imps[site_a], k)
        set_b = topk(imps[site_b], k)

        coefficient = jac(set_a, set_b)
        rows.append((fold, site_a, site_b, coefficient))

    jaccard_df = pd.DataFrame(rows, columns=["fold", "site_a", "site_b", "jaccard"])

    # Remove invalid Jaccard values before calculating the mean.
    valid_jaccard_df = jaccard_df.dropna(subset=["jaccard"]).copy()

    # This handles both an empty DataFrame and all-NaN values.
    if valid_jaccard_df.empty:
        logger.warning(f"No valid Jaccard similarities computed for fold={fold}")
        return valid_jaccard_df

    mean_jaccard = valid_jaccard_df["jaccard"].mean()

    frequency = Counter(feature for site in sites for feature in topk(imps[site], k))

    frequency_df = pd.DataFrame(sorted(frequency.items(), key=lambda item: -item[1]), columns=["feature", "site_frequency"])

    print(f"\nPairwise top-{k} Jaccard similarities for fold {fold}:")
    print(valid_jaccard_df.to_string(index=False))
    print(f"Fold {fold} mean top-{k} Jaccard={mean_jaccard:.4f}")

    logger.info(f"Pairwise top-{k} Jaccard similarities for fold {fold}:\n{valid_jaccard_df.to_string(index=False)}")
    logger.info(f"Fold Jaccard summary: fold={fold}, k={k}, comparisons={len(valid_jaccard_df)}, mean_Jaccard={mean_jaccard:.6f}")
    logger.info(f"Top-feature site frequency for fold {fold}:\n{frequency_df.to_string(index=False)}")

    return valid_jaccard_df

def determine_jaccard_k(number_of_features, top_fraction=0.20, minimum_k=2):
    if number_of_features < 2:
        raise ValueError("Jaccard analysis requires at least two features.")

    proportional_k = int(np.ceil(top_fraction * number_of_features))
    return min(number_of_features - 1, max(minimum_k, proportional_k))

def get_args(): 
    parser = argparse.ArgumentParser( description="Run site-level interpretability analysis.")
    parser.add_argument("-d", "--dataset-group", type=int, required=True, choices=[1, 2, 3, 4, 5], help="Dataset group to run:")

    args = parser.parse_args()
    d = args.dataset_group
    return d 

if __name__ == "__main__":
    out_features = 9
    dataset_name = "seer"
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
    
    data_dictionary = {
      "seer": {
        "folds":  list(range(5)),
        "sites" : [1, 2, 3]
      }, 
      "Pbc3": {
        "folds": list(range(3)),
        "sites" : ["Barcelona", "Copenhagen", "Hvidovre", "London", "Lyon", "Munich"]
      },
      "LeukSurv": {
        "folds": list(range(3)),
        "sites" : list(range(1, 25, 1)) 
      }, 
      "cgd":{
        "folds": list(range(3)), 
        "sites" :  [ "Europe:Amsterdam", "Europe:other", "US:NIH", "US:other"]
      }, 
      "scania":{
        "folds": list(range(5)), 
        "sites" : list(range(1, 6, 1)) 
      }, 
      "uis":{
        "folds": list(range(5)), 
        "sites" : ['A', 'B']
      }, 
      "d.oropha.rec":{
        "folds": list(range(3)), 
        "sites" : list(range(1, 7, 1)) 
      },   
      "Dialysis":{
        "folds": list(range(3)), 
        "sites" : [ 120, 128, 184, 217, 344, 561, 562, 641, 741, 743, 917, 918, 936, 954, 983, 984, 1029, 1039, 1048, 1051, 1053, 1061, 1064, 1070, 1071, 1072, 1093, 1100, 1159, 1372, 1437, 1654, 1692, 1699, 2026, 2210, 2211, 2242, 2320, 2351, 2611, 2632, 2811, 2844, 2987, 3064, 3596, 3675, 4122, 4157, 4163, 4779, 4825, 5089, 5135, 5419, 5423, 5432, 5481, 5495, 5513, 5514, 5543, 5653, 5681, 5692, 5768 ]
      }, 
      "Aids2":{
        "folds": list(range(5)), 
        "sites" : ["NSW", "QLD", "VIC", "Other"] 
      },   
    }   
    
    models_name = ["FedPAttn", "FedPDNN", "FedPLSTM", "Fed_CoxPH", "LinearPH", "NNPH", "NNnPH", "DeepHit",  "DeepPseudo", "AFT_Weibull"]   
    # if n == 1: 
    #     models_name = ["FedPAttn", "FedPDNN", "FedPLSTM"]
    # elif n == 2: 
    #     models_name = ["AFT_Weibull", "Fed_CoxPH", "DeepPseudo", "DeepHit"]
    # else: 
    #     models_name = ["LinearPH", "NNPH", "NNnPH"] 
    

    data_dir = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/processed_data/federated_data/{dataset_name}/federated"   
    
    for model_name in models_name:
        out_features = 10 if model_name == "DeepPseudo" else 9
        logger, run_id, log_path = create_site_logger( dataset=dataset_name, model_name=model_name)

        folds = data_dictionary[dataset_name]["folds"]
        sites = data_dictionary[dataset_name]["sites"]

        # Stores correlations from every fold.
        all_pairs = []
        all_jaccard_results = []
        jaccard_d = 0 
        
        try:
            for fold in folds:
                logger.info("")
                logger.info("=" * 90)
                logger.info("STARTING FOLD: run_id=%s, model=%s, fold=%d", run_id, model_name, fold)  
                checkpoint_path = get_data_model_directory_site_level(model_name, dataset_name, fold)

                imps = {}
                rank_dataframe = {}

                for site in sites: 
                    ref = load_central_site_csv(data_dir, "train",  site, fold=fold)
                    ref_val = load_central_site_csv(data_dir, "val",  site, fold=fold)
                
                    feature_cols = [column for column in ref.columns if column not in (TIME_COL, EVENT_COL)]
                    in_features = len(feature_cols)
                
                    discretiser = get_Discretizer(fold)
                    times = pseudo_eval_times(pd.concat([ref[TIME_COL], ref_val[TIME_COL]], ignore_index=True)) 
                    importance = site_importance(
                        data_dir=data_dir,
                        logger=logger,
                        dataset=dataset_name,
                        model_name=model_name,
                        site=site,
                        checkpoint_path=checkpoint_path,
                        fold=fold,
                        feature_cols=feature_cols,
                        in_features=in_features,
                        out_features=out_features,
                        times=times,
                        discretiser=discretiser,
                    )

                    imps[site] = importance.reindex(feature_cols) 
                    rank_dataframe[site] = imps[site].rank( ascending=False, method="average")

                ranks = pd.DataFrame(rank_dataframe)
                fold_pairs = []

                for site_a, site_b in itertools.combinations(sites, 2):
                    valid = (ranks[site_a].notna() & ranks[site_b].notna())

                    number_valid = int(valid.sum()) 
                    if number_valid < 2:
                        logger.warning(f"Insufficient valid features: fold={fold}, site_a={site_a}, site_b={site_b}, valid_features={number_valid}")
                        continue
                    
                    site_a_ranks = ranks.loc[valid, site_a]
                    site_b_ranks = ranks.loc[valid, site_b]

                    if site_a_ranks.nunique() < 2 or site_b_ranks.nunique() < 2:
                        logger.warning(f"Constant feature rankings: fold={fold}, site_a={site_a}, site_b={site_b}; correlation is undefined.")
                        continue
                    
                    spearman = spearmanr(ranks.loc[valid, site_a], ranks.loc[valid, site_b]).correlation
                    kendall = kendalltau(ranks.loc[valid, site_a], ranks.loc[valid, site_b]).correlation
                    result = (fold, site_a, site_b, spearman, kendall, number_valid)

                    fold_pairs.append(result)
                    all_pairs.append(result)

                    print(f"Model={model_name}, fold={fold}, sites={site_a} vs {site_b}, Spearman={spearman:.4f}, Kendall tau={kendall:.4f}") 
                    logger.info(f"Site correlation: fold={fold}, site_a={site_a}, site_b={site_b}, valid_features={number_valid}, Spearman={spearman:.4f}, Kendall_tau={kendall:.4f}")

                fold_correlation_df = pd.DataFrame(fold_pairs, columns=[ "fold", "site_a", "site_b", "spearman", "kendall", "valid_features"]) 
                if fold_correlation_df.empty:
                    logger.warning( "No valid site correlations for fold=%d", fold)
                else:
                    fold_mean_spearman =  fold_correlation_df["spearman"].mean() 
                    fold_mean_kendall = fold_correlation_df["kendall"].mean() 
                    
                    logger.info(f"Pairwise correlations for fold {fold}:\n{fold_correlation_df.to_string(index=False)}") 
                    logger.info(f"Fold summary: fold={fold}, mean_Spearman={fold_mean_spearman:.6f}, mean_Kendall_tau={fold_mean_kendall:.6f}")

                # Must remain inside the fold loop because `imps` contains results for the current fold only.
                logger.info(f"Computing Jaccard coefficients for fold={fold}") 
                
                jaccard_d = determine_jaccard_k(len(feature_cols), top_fraction=0.40, minimum_k=2) 
                fold_jaccard_df = compute_Jaccard_value(
                    imps=imps,
                    sites=sites,
                    logger=logger,
                    fold=fold,
                    k= jaccard_d
                ) 
                
                if not fold_jaccard_df.empty:
                    all_jaccard_results.append(fold_jaccard_df)

                logger.info(f"COMPLETED FOLD: run_id={run_id}, model={model_name}, fold={fold}")

            # Overall summary across every fold.
            correlation_df = pd.DataFrame(all_pairs, columns=["fold", "site_a", "site_b", "spearman", "kendall", "valid_features"]) 
            if correlation_df.empty:
                logger.warning( "No valid site correlations were computed for any fold.")
            else:
                logger.info( "ALL-FOLD PAIRWISE CORRELATIONS:\n%s", correlation_df.to_string(index=False))

                mean_spearman = correlation_df["spearman"].mean()
                mean_kendall = correlation_df["kendall"].mean()

                logger.info(f"OVERALL SUMMARY: run_id={run_id}, model={model_name}, folds={len(folds)}, comparisons={len(correlation_df)}, mean_Spearman={mean_spearman:.6f}, mean_Kendall_tau={mean_kendall:.6f}")
                per_fold_summary = ( correlation_df .groupby("fold")[["spearman", "kendall"]].mean().reset_index())

                logger.info(f"PER-FOLD MEAN CORRELATIONS:\n{per_fold_summary.to_string(index=False)}")
            
            if not all_jaccard_results:
                logger.warning("No valid Jaccard similarities were computed for any fold.")
            else:
                all_jaccard_df = pd.concat(all_jaccard_results, ignore_index=True)
                per_fold_jaccard = all_jaccard_df.groupby("fold", as_index=False).agg(mean_jaccard=("jaccard", "mean"), number_of_pairs=("jaccard", "count")) 

                overall_mean_jaccard = per_fold_jaccard["mean_jaccard"].mean()
                overall_sd_jaccard = per_fold_jaccard["mean_jaccard"].std(ddof=1)

                logger.info(f"ALL-FOLD PAIRWISE JACCARD SIMILARITIES:\n{all_jaccard_df.to_string(index=False)}")
                logger.info(f"PER-FOLD MEAN JACCARD:\n{per_fold_jaccard.to_string(index=False)}")
                logger.info(f"OVERALL JACCARD SUMMARY: run_id={run_id}, model={model_name}, k={jaccard_d}, folds={len(per_fold_jaccard)}, mean_Jaccard={overall_mean_jaccard:.6f}, SD_across_folds={overall_sd_jaccard:.6f}")

                print("\nPer-fold mean Jaccard similarity:")
                print(per_fold_jaccard.to_string(index=False)) 
                print(f"\nOverall top-{jaccard_d} Jaccard similarity:  {overall_mean_jaccard:.4f} ±  {overall_sd_jaccard:.4f}")
                
                logger.info(f"RUN COMPLETED SUCCESSFULLY: run_id={run_id}")
                logger.info("#" * 90)

        except Exception:
            logger.exception(f"RUN FAILED: run_id={run_id}, dataset={dataset_name}, model={model_name}")
            raise

        finally:
            for handler in logger.handlers[:]:
                handler.flush()
                handler.close()
                logger.removeHandler(handler)

        print(f"Log saved to: {log_path}")
        