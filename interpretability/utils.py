
import numpy as np 
import pandas as pd 
import concurrent.futures 
from survshap import SurvivalModelExplainer, ModelSurvSHAP
from lifelines import KaplanMeierFitter

import os, argparse, warnings  
from pathlib import Path 
from .config import DATASET_FOLDS,  DATASET_CLIENTS

from .model import FedPDNN, FedPLSTM, FedPAttn, DeepPseudo
from .model import WeibullTimeToEventModel, Fed_CoxPH 
from .model import LinearPH, NNPH, NNnPH, DeepHit_DNN

import pandas as pd 
import numpy as np 
import os, torch, logging  

from sksurv.util import Surv
import torchtuples as tt
from .config import TIME_COL, EVENT_COL, MODEL_DIRECTORIES

from lifelines import KaplanMeierFitter
from pycox.models import DeepHit 
from DeepHit.model import build_deephit_net 
from sklearn.model_selection import StratifiedShuffleSplit 


def select_survival_background( train_df: pd.DataFrame, background_size: int = 200, random_state: int = 42) -> np.ndarray:
    """
    Return positional indices for a representative survival background.

    Stratification uses:
        event status + survival-time quantile
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


####################################### partition wise configuration #################################################################### 

def get_partition_checkpoint_paths(dataset, model_name, fold):
    project_root = Path(__file__).resolve().parents[1] 
    if model_name not in MODEL_DIRECTORIES:
        raise ValueError(f"Unsupported model: {model_name}")

    model_root = project_root / MODEL_DIRECTORIES[model_name] 
    cl_model_path = model_root / "checkpoints" / "cl" / dataset / model_name / f"{model_name}_CL_{fold}.pt"  
    fl_model_path = model_root / "checkpoints" / "fl" / dataset / model_name / f"{model_name}_global_{fold}.pt"  
    return cl_model_path, fl_model_path

################################ Model Directory for Site Level ############################################################ 

def get_data_model_directory_site_level(model_name, data, fold):
    CKPT_TMPL = None 
    PROJECT_ROOT = Path(__file__).resolve().parent  
    if model_name in ["FedPDNN", "FedPLSTM", "FedPAttn"]: 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/{model_name}/{data}/{model_name}_global_{fold}.pt"
    elif model_name == "AFT_Weibull": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/{model_name}_model_run_{fold}.tar"
    elif model_name == "DeepHit": 
         CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/{model_name}_model_run_{fold}.pt"
    elif model_name == "LinearPH": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/LinearPH_model_run_{fold}.pt"
    elif model_name == "NNPH": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/NNph_model_run_{fold}.pt"
    elif model_name == "Fed_CoxPH": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/CoxPH_model_run_{fold}.pt"
    elif model_name == "NNnPH": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/NNnPH_model_run_{fold}.pt"
    elif model_name == "DeepPseudo": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/DeepPseudo_global_model_{fold}.pt"
    else:
        raise ValueError("model name did not match with our repository")
         
    return CKPT_TMPL

####################################### Discretizer ####################################################################

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
            ValueError
        return self

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
      
################################################# Get Discretizer ################################################################################

def get_Discretizer(fold): 
    PROJECT_ROOT = Path(__file__).resolve().parent  
    data_dir = f"{PROJECT_ROOT}/datasets/processed_data/federated_data/seer/centralized" 
    train_df = load_fold_csv(data_dir, "train", fold)

    train_time_all = train_df[TIME_COL].to_numpy()
    train_status_all = train_df[EVENT_COL].to_numpy().astype(int) 
    y_train_all = (train_time_all, train_status_all)

    discretiser = Discretiser(40, scheme="km")
    discretiser.fit(train_df[TIME_COL].to_numpy(), train_df[EVENT_COL].to_numpy())   
    return discretiser

####################################### Split Dataframe into feature and labels ####################################################################
 
def load_discretizer(ckpt_path1): 
    ckpt = torch.load(ckpt_path1,  weights_only=False)
    discretizer_ley = "discretizer" if "discretizer" in ckpt else "discretiser"
    discretiser = ckpt[discretizer_ley]  
    return discretiser  
 
def load_model_partition(model_name, checkpoint_path, in_features, out_features=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    discretiser = checkpoint.get("discretizer", checkpoint.get("discretiser"))

    if model_name == "DeepHit":
        duration_index = checkpoint["duration_index"]
        net = build_deephit_net(in_features, len(duration_index))
        net.load_state_dict(checkpoint["model"])

        model = DeepHit(net,
            tt.optim.Adam(lr=1e-3),
            alpha=checkpoint["alpha"],
            sigma=checkpoint["sigma"],
            duration_index=duration_index
        )

        model.net.eval()
        return model, None

    if model_name in ["LinearPH", "NNPH", "NNnPH"] and discretiser is None:
        raise ValueError(f"{model_name} checkpoint does not contain a discretizer: {checkpoint_path}")

    model = build_model(model_name, in_features, out_features, discretiser)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, discretiser 
  
# def load_model(model_name, ckpt_path1, in_features, out_features):
#     discretiser = None 
#     ckpt = torch.load(ckpt_path1,  weights_only=False)
    
#     if model_name == "DeepHit":   
#         state = ckpt["model_state_dict"]
#         opt = tt.optim.Adam(lr=1e-3, weight_decay=5e-5)
#         model = build_model(model_name, in_features, out_features)
        
#         duration_index = ckpt["duration_index"]
#         alpha, sigma = ckpt["alpha"], ckpt["sigma"]
#         model = DeepHit(model, opt, alpha=alpha, sigma=sigma, duration_index=duration_index)
        
#     elif model_name in [ "LinearPH", "NNPH"]:
#         state = ckpt["model"]
#         discretiser = ckpt["discretizer"] 
#         model = build_model(model_name, in_features, out_features, discretiser) 
#         state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()} 
#         model.load_state_dict(state) 
#     else:
#         model = build_model(model_name, in_features, out_features, discretiser) 
#         state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt  
#         state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()}
#         model.load_state_dict(state)
#     return model, discretiser


  
# def load_model_site(model_name, ckpt_path1, in_features, out_features):
#     discretiser = None 
#     ckpt = torch.load(ckpt_path1,  weights_only=False)
    
#     if model_name == "DeepHit":   
#         state = ckpt["model_state_dict"]
#         opt = tt.optim.Adam(lr=1e-3, weight_decay=5e-5)
#         model = build_model(model_name, in_features, out_features)
        
#         duration_index = ckpt["duration_index"]
#         alpha, sigma = ckpt["alpha"], ckpt["sigma"]
#         model = DeepHit(model, opt, alpha=alpha, sigma=sigma, duration_index=duration_index)
        
#     elif model_name in ["NNnPH"]: 
        
#         if isinstance(ckpt, dict) and "model" in ckpt:
#             state = ckpt["model"]
#             discretizer_ley = "discretizer" if "discretizer" in ckpt else "discretiser"
#             discretiser = ckpt[discretizer_ley] 
#         elif isinstance(ckpt, dict) and ckpt and all(torch.is_tensor(value) for value in ckpt.values()):
#             state = ckpt
#         else:       
#             state = None 
        
#         model = build_model(model_name, in_features, out_features, discretiser) 
#         state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()} 
#         model.load_state_dict(state)
        
#     elif model_name in [ "LinearPH", "NNPH"]:
#         state = ckpt["model"]
#         discretiser = ckpt["discretizer"] 
#         model = build_model(model_name, in_features, out_features, discretiser) 
#         state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()} 
#         model.load_state_dict(state)
    
#     else:
#         model = build_model(model_name, in_features, out_features, discretiser) 
#         state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt  
#         state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()}
#         model.load_state_dict(state)
#     return model, discretiser


  
# def load_model_partition(model_name, ckpt_path1, in_features, out_features, discretiser = None): 
#     ckpt = torch.load(ckpt_path1,  weights_only=False)
    
#     if model_name == "DeepHit":   
#         state = ckpt["model_state_dict"]
#         opt = tt.optim.Adam(lr=1e-3, weight_decay=5e-5)
#         model = build_model(model_name, in_features, out_features)
        
#         duration_index = ckpt["duration_index"]
#         alpha, sigma = ckpt["alpha"], ckpt["sigma"]
#         model = DeepHit(model, opt, alpha=alpha, sigma=sigma, duration_index=duration_index)
          
#     elif model_name in [ "LinearPH", "NNPH", "NNnPH"]:
#         state = ckpt["model"]
#         discretiser = ckpt["discretizer"] 
#         model = build_model(model_name, in_features, out_features, discretiser) 
#         state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()} 
#         model.load_state_dict(state)
    
#     else:
#         model = build_model(model_name, in_features, out_features, discretiser) 
#         state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt  
#         state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()}
#         model.load_state_dict(state)
#     return model, discretiser
 
  
####################################### Split Dataframe into feature and labels ####################################################################

def split_xy(df, feature_cols):
    X_df = df[feature_cols].astype("float32") 
    y_sksurv = Surv.from_arrays( event=df[EVENT_COL].astype(bool).to_numpy(), time=df[TIME_COL].to_numpy(dtype="float32"))  # For SurvSHAP
    y_pycox = ( df[TIME_COL].to_numpy(dtype="float32"), df[EVENT_COL].to_numpy(dtype="int64"))  # For PyCox CoxPH 
    return X_df, y_sksurv, y_pycox

####################################### load fold csv ################################################################################################

def load_fold_csv(data_dir, split, fold):                     
    return pd.read_csv(os.path.join(data_dir, "0", str(fold), f"{split}.csv"))  

def load_central_csv(data_dir, split, fold=-1):
    return pd.read_csv(os.path.join(data_dir, "0", str(fold), f"{split}.csv")) 
  
def load_site_csv(data_dir, split, site, fold=-1):
    return pd.read_csv(os.path.join(data_dir, str(site), str(fold), f"{split}.csv"))  

def load_central_site_csv(central_data_dir, split, site, fold=-1):
    return pd.read_csv(os.path.join(central_data_dir, str(site), str(fold), f"{split}.csv"))  

def split_xy(df, feature_cols):
    X_df = df[feature_cols].astype("float32")
    y_sksurv = Surv.from_arrays( event=df[EVENT_COL].astype(bool).to_numpy(), time=df[TIME_COL].to_numpy(dtype="float32"))  # For SurvSHAP
    y_pycox = ( df[TIME_COL].to_numpy(dtype="float32"), df[EVENT_COL].to_numpy(dtype="int64"))  # For PyCox CoxPH 
    return X_df, y_sksurv, y_pycox 

####################################### evaluation time point calculation ##############################################################################

def pseudo_eval_times(time_values, pseudo_interval=10):
    """Replicates your Pseudo_Evaltime: percentiles of the unique event/observed times."""
    uniq = np.unique(np.asarray(time_values, dtype=float))
    times = np.percentile(uniq, np.arange(10, 99, pseudo_interval))
    return times    
  
####################################### Create Logger folder per model  #################################################################################

def create_model_logger(dataset, model_name, log_folder="fold_level"):
    log_dir = os.path.join("log", log_folder, dataset, model_name)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "fold_level_all.log")

    logger = logging.getLogger(f"experiment.{dataset}.{model_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Prevent duplicate handlers.
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%m-%d %H:%M",
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info( "Starting fold-level experiment: dataset=%s, model=%s", dataset, model_name) 
    return logger 

 

####################################### Model Selection ###################################################################################################

def build_model(name, in_features, out_features,  discretiser=None):
    if name == "FedPDNN":
        return FedPDNN(dataset="seer", in_features=in_features, out_features=out_features, dropout=0.0, hidden_size=0, num_layers=0)
    elif name == "FedPLSTM":
        return FedPLSTM(dataset="seer", in_features=in_features,  out_features=out_features, dropout=0.0, hidden_size=128, num_layers=2)
    elif name == "FedPAttn":
        return FedPAttn(dataset="seer", in_features=in_features, out_features=out_features, dropout=0.0, hidden_size=128, num_layers=2)
    elif name == "DeepPseudo":
        return DeepPseudo(in_features=in_features, out_features=out_features, num_event=1, dropout=0.0)  
    elif name == "AFT_Weibull":
        return WeibullTimeToEventModel(num_input_features=in_features)
    elif name == "Fed_CoxPH":
        return Fed_CoxPH(in_features=in_features, out_features=1, output_bias=False)
    elif name == "LinearPH":
        return LinearPH(dim_in=in_features, dim_out=len(discretiser.cuts), batch_norm=True) 
    elif name == "NNPH":
        return NNPH(dim_in=in_features, num_nodes=[32, 32], dim_out=len(discretiser.cuts), batch_norm=True, dropout=0.05)  
    elif name == "NNnPH":
        # return NNPH(dim_in=in_features, num_nodes=[32, 32], dim_out=len(discretiser.cuts), batch_norm=True, dropout=0.05)  
        return NNnPH(dim_in=in_features, num_nodes= [32, 32], dim_out=len(discretiser.cuts), batch_norm=True, dropout=0.05)
    elif name == "DeepHit":
        return DeepHit_DNN(in_features=in_features, num_nodes_shared=[128, 64], num_nodes_indiv=[32], num_risks=1, out_features=out_features, dropout=0.05)    
    else: 
        raise ValueError(name) 
      
####################################### load model checkpoint ###################################################################################################

def get_data_model_directory_fold_level(model_name, data, fold, client=0):
    CKPT_TMPL = None 
    PROJECT_ROOT = Path(__file__).resolve().parent  
    if model_name in ["FedPDNN", "FedPLSTM", "FedPAttn"]: 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/{model_name}/{data}/{model_name}_global_{fold}.pt"
    elif model_name == "AFT_Weibull": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/AFT_Weibull_model_run_{fold}.tar"
    elif model_name == "DeepHit": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/DeepHit_model_run_{fold}.pt"
    elif model_name == "LinearPH": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/LinearPH_model_run_{fold}.pt"
    elif model_name == "NNPH": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/NNph_model_run_{fold}.pt"
    elif model_name == "Fed_CoxPH": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/CoxPH_model_run_{fold}.pt"
    elif model_name == "NNnPH": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/NNnPH_model_run_{fold}.pt"
    elif model_name == "DeepPseudo": 
        CKPT_TMPL = f"{PROJECT_ROOT}/models/federated_data/federated/{data}/DeepPseudo_global_model_{fold}.pt"
    else:
        raise ValueError("model name did not match with our repository") 
    return CKPT_TMPL 
 
####################################### Arguments Method ####################################################################

def get_args(description): 
    parser = argparse.ArgumentParser( description=description) 
    parser.add_argument("-d", "--dataset", required=True, choices=DATASET_FOLDS.keys(), help="Dataset to analyze.") 
    return parser.parse_args()  

####################################### Folds Method ####################################################################

def get_folds(dataset): 
    return list(range(DATASET_FOLDS[dataset])) 
  

####################################### Method for getting sites ####################################################################

def get_sites(dataset): 
    return DATASET_CLIENTS[dataset] 

####################################### logger close ####################################################################

def close_logger(logger): 
    for handler in logger.handlers[:]:
        handler.flush() 
        handler.close()
        logger.removeHandler(handler)   

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