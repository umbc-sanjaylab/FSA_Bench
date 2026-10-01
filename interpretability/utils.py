
import numpy as np 
import pandas as pd 
import concurrent.futures 
from survshap import SurvivalModelExplainer, ModelSurvSHAP
from lifelines import KaplanMeierFitter

import os, argparse, warnings  
from pathlib import Path 
from .config import DATASET_FOLDS, MODELS

####################################### Arguments Method ####################################################################

def get_args(description): 
    parser = argparse.ArgumentParser( description=description) 
    parser.add_argument("-d", "--dataset", required=True, choices=DATASET_FOLDS.keys(), help="Dataset to analyze.") 
    return parser.parse_args()  

####################################### Folds Method ####################################################################

def get_folds(dataset): 
    return list(range(DATASET_FOLDS[dataset])) 

####################################### logger close ####################################################################

def close_logger(logger): 
    for handler in logger.handlers[:]:
        handler.flush() 
        handler.close()
        logger.removeHandler(handler) 
 

####################################### partition wise configuration ####################################################################

def get_data_model_directory_partition_wise(model, data, fold, client): 
    PROJECT_ROOT = Path(__file__).resolve().parent  
    model_name = model 
    
    if model  not in MODELS:
        raise ValueError(f"Unsupported model: {model_name}") 
    
    if model in ["FedPDNN", "FedPLSTM", "FedPAttn"]: 
        model_name = "FedPseudo_Flare"   
    
    cent_tmpl=  PROJECT_ROOT / model_name / "models" /  "CL" / data / f"{model_name}_{fold}_global.pt" 
    fed_tmpl =  PROJECT_ROOT / model_name / "models" /  "FL" / data / f"{model_name}_{fold}_{client}.pt" 
    
    return str(cent_tmpl), str(fed_tmpl) 

####################################### Site-Level Configuration ###################################

def get_data_model_directory_site_level(model, data, fold, client):    
    PROJECT_ROOT = Path(__file__).resolve().parent  
    model_name = model 
    
    if model  not in MODELS:
        raise ValueError(f"Unsupported model: {model_name}") 
    
    if model in ["FedPDNN", "FedPLSTM", "FedPAttn"]: 
        model_name = "FedPseudo_Flare"  
    
    ckpt_tmpl =   PROJECT_ROOT / model_name / "models" /  "FL" / data / f"{model_name}_{fold}_{client}.pt" 
    return str(ckpt_tmpl)  

####################################### Fold-Level Configuration ###################################

def get_data_model_directory_fold_level(model, data, fold):  
    PROJECT_ROOT = Path(__file__).resolve().parent   
    model_name = model
    
    if model_name  not in MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if model in ["FedPDNN", "FedPLSTM", "FedPAttn"]: 
        model_name = "FedPseudo_Flare"  
    
    ckpt_tmpl =  PROJECT_ROOT / model_name / "models" /  "CL" / data / f"{model_name}_{fold}_global.pt" 
    return str(ckpt_tmpl)  

############################################# Discretizer ######################################################################### 

class Discretiser():
    """
    Arguments:
        cuts -- number of cuts to make.
        scheme -- either equidistant or km
    """

    def __init__(self, cuts, scheme='equidistant'):
        self._cuts = cuts
        self._scheme = scheme
        self.cuts = None

    def fit(self, durations, events):
        durations = durations.astype(np.float64)
        if self._scheme == 'equidistant':
            self.cuts = np.linspace(0, durations.max(), self._cuts, dtype=np.float64)
        elif self._scheme == 'km':
            kmf = KaplanMeierFitter()
            kmf.fit(durations, events)
            
            last_duration = durations.max()
            percentiles = np.linspace(kmf.predict(last_duration), kmf.predict(0), self._cuts, dtype=np.float64)
            self.cuts = np.unique([kmf.percentile(p) for p in percentiles][::-1])
        else:
            raise ValueError(f"Unsupported scheme: {self._scheme}") 

    def transform(self, durations, events):
        # -1 so that idx_durations starts from 0
        # +events so that uncensored values never have t=0?
        # clipping it so that max index falls within [0,...,self.cuts - 1]
        idx_durations = np.digitize(durations, self.cuts) - 1 + events
        idx_durations = idx_durations.clip(0, len(self.cuts) - 1)
        return (idx_durations, events)

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        durations, events = self.transform(durations, events)
        return (durations, events)

    @property
    def dim_out(self):
        if self.cuts is None:
            raise ValueError("Need to call `fit` before this is accessible.")
        return len(self.cuts) 

def data_transform(y, discretiser, fit_transform=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fit_transform:
            y_trans = discretiser.fit_transform(*y)
        else:
            y_trans = discretiser.transform(*y)

    return y_trans
  
  
################################################## Get Input Feature ######################################################


def get_input_feature(args):
    if args.partition == "federated":
        client_name = get_client_name_for_federated_datasets( args.dataset, client_id=0)
        dataset_directory = f"{args.datadir}/{args.dataset}/{args.partition}/{client_name}/{args.run_id}/"
    else:
        dataset_directory = f"{args.datadir}/{args.dataset}/{args.partition}/0/{args.run_id}/"

    train_df = pd.read_csv(dataset_directory + "train.csv")
    X_train = np.array(train_df.drop(columns=["time", "event"]), dtype='float32')
    return X_train.shape[1]

######################################### Region from Federated Dataset by Client ID ########################################


def get_client_name_for_federated_datasets(dataset, client_id):
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
    return region_list[client_id]


def global_importance1(fold, model, X, y, background=None,  y_background=None, n_explain=100, times=None,  adapter="sksurv", predict_surv_matrix=None,  B=25,  method="sampling", random_state=42, features=None):
    """
    Returns a pandas Series: feature -> global importance (aggregated |SurvSHAP(t)|), sorted descending.

    model        : fitted survival model
    X, y         : full test data (X DataFrame, y sksurv structured array)
    background   : rows used as SHAP background (default: all of X). Keep small
                   (50-100) -- kernel SHAP cost scales with it.
    n_explain    : how many individuals to explain & average over.
    times        : shared time grid. Default = unique event times in y.
    adapter      : "sksurv" | "pycox" | "matrix".
    method       : "kernel" | "sampling".  
    """
    if times is None:
        time_field = y.dtype.names[1]
        times = np.unique(y[time_field])

    if adapter == "sksurv":
        psf, pchf = adapter_sksurv(times)
    elif adapter == "pycox":
        psf, pchf = adapter_pycox(times)
    elif adapter == "matrix":
        if predict_surv_matrix is None:
            raise ValueError("adapter='matrix' needs predict_surv_matrix=...")
        psf, pchf = adapter_from_surv_matrix(predict_surv_matrix, times)
    else:
        raise ValueError("adapter must be sksurv | pycox | matrix") 
    
    # Background data 
    bg = background.copy()
    y_bg = y_background
    
    # SurvivalExplainer and SurvSHAP
    explainer = SurvivalModelExplainer(model, bg, y_bg, psf)
    explain_X =  X.copy()  # .sample( n=min(n_explain, len(X)), random_state=random_state)
    
    
    if method == "kernel":  
        ms = ModelSurvSHAP(calculation_method=method, B=B, random_state=random_state)
    else: 
        ms = ModelSurvSHAP(calculation_method=method, random_state=random_state)
        
    # if explain_start is not None or explain_end is not None:
    #     explain_X = explain_X.iloc[explain_start:explain_end]
    
    # Split data into chunks
    def split_dataframe(df, chunk_size):
        return [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Function to compute SHAP for a chunk
    def process_chunk(chunk):
        # shap_values = explainer.shap_values(chunk[features])
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          ms.fit(explainer, explain_X, timestamps=times)

        shap_values = (ms.result.set_index("variable_name")["aggregated_change"].abs().sort_values(ascending=False))
        shap_val = pd.DataFrame(shap_values, columns=features, index=chunk.index)
        top_predictor = shap_val.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
        return top_predictor
    
    # Parallel execution
    def compute_shap_parallel(explain_X, chunk_size=100, num_workers=8):
        chunks = split_dataframe(explain_X, chunk_size)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_chunk, chunks))
        return pd.concat(results)
    
    return compute_shap_parallel(explain_X, chunk_size=100, num_workers=8)   

def adapter_sksurv(times):
    """For scikit-survival models. Re-grids onto a shared time grid."""
    def psf(model, data):
        return np.array([StepFn(times, [fn(t) for t in times])
                         for fn in model.predict_survival_function(data)])

    def pchf(model, data):
        return np.array([StepFn(times, [fn(t) for t in times])
                         for fn in model.predict_cumulative_hazard_function(data)])
    return psf, pchf

class StepFn:
    """Minimal survival/CHF curve object that survshap accepts."""
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def __call__(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.clip(np.searchsorted(self.x, t, side="right") - 1,
                      0, len(self.y) - 1)
        return self.y[idx]


def adapter_pycox(times):
    """
    For pycox-style models (DeepSurv, DeepHit, ... and likely your PseudoNAM
    if it exposes predict_surv_df). `model.predict_surv_df(x)` returns a
    DataFrame indexed by time, one column per observation.
    Adjust `to_input(data)` if your model wants tensors/np arrays.
    """
    def to_input(data):
        return data.values.astype("float32") if hasattr(data, "values") else \
               np.asarray(data, dtype="float32")

    def psf(model, data):
        surv = model.predict_surv_df(to_input(data))     # index=time, cols=obs
        grid = surv.index.to_numpy()
        S = surv.to_numpy().T                             # rows=obs, cols=time
        # interpolate each obs onto the shared `times` grid
        out = []
        for row in S:
            yv = np.interp(times, grid, row, left=1.0, right=row[-1])
            out.append(StepFn(times, yv))
        return np.array(out)

    def pchf(model, data):
        sfs = psf(model, data)
        # CHF = -log(S), guarded against log(0)
        return np.array([StepFn(f.x, -np.log(np.clip(f.y, 1e-8, 1.0)))
                         for f in sfs])
    return psf, pchf


def adapter_from_surv_matrix(predict_surv_matrix, times):
    """
    Most general escape hatch. You provide a function
        predict_surv_matrix(model, data) -> array [n_obs, len(times)]
    of survival probabilities on `times`. Works for ANY model.
    """
    def psf(model, data):
        S = np.asarray(predict_surv_matrix(model, data), dtype=float)
        return np.array([StepFn(times, row) for row in S])

    def pchf(model, data):
        S = np.asarray(predict_surv_matrix(model, data), dtype=float)
        return np.array([StepFn(times, -np.log(np.clip(row, 1e-8, 1.0)))
                         for row in S])
    return psf, pchf
  
def run_fold_on_two_gpus( client, fold, feature_cols, times, in_features, out_features, ):
    number_of_test_instances = len(load_fold_csv("test", fold=fold))

    midpoint = ( number_of_test_instances + 1 ) // 2 
    jobs = [ { "client": client, "fold": fold, "feature_cols": feature_cols, "times": times, "in_features": in_features, "out_features": out_features, "device": "cuda:1", "explain_start": 0, "explain_end": midpoint, "background_size": 100, }, { "client": client, "fold": fold, "feature_cols": feature_cols, "times": times, "in_features": in_features, "out_features": out_features, "device": "cuda:2", "explain_start": midpoint, "explain_end": number_of_test_instances, "background_size": 100, }]

    context = mp.get_context("spawn") 
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=2,
        mp_context=context,
    ) as executor:

        futures = [ executor.submit( fold_importance, **job, ) for job in jobs ]
        shard_results = [ future.result() for future in futures ]

    total_instances = sum(
        shard_size
        for shard_size, _ in shard_results
    )

    weighted_sum = sum(
        shard_importance * shard_size
        for shard_size, shard_importance
        in shard_results
    )

    full_importance = (
        weighted_sum / total_instances
    )

    return (
        full_importance
        .sort_values(ascending=False)
        .rename("global_importance")
    )