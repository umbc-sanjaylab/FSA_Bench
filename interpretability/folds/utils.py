from .model import FedPDNN, FedPLSTM, FedPAttn, DeepPseudo, WeibullTimeToEventModel, Fed_CoxPH, LinearPH, NNPH, NNnPH, DeepHit_DNN
import pandas as pd 
import numpy as np 
import os, torch, logging  
from sksurv.util import Surv
import torchtuples as tt
from .config import TIME_COL, EVENT_COL
from lifelines import KaplanMeierFitter
from pycox.models import DeepHit 
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

def get_data_model_directory_partition_wise(dataset, model_name, fold):
    CENT_TMPL, FED_TMPL = None, None 
    if model_name in ["FedPDNN", "FedPLSTM", "FedPAttn"] : 
        CENT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/FedPseudo_Flare/models/federated_data/centralized/{dataset}/{dataset}/{model_name}_global_{fold}.pt"
        FED_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/FedPseudo_Flare/models/federated_data/federated/{dataset}/{model_name}/{dataset}/{model_name}_global_{fold}.pt"
    elif model_name == "AFT_Weibull": 
        CENT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/AFT_Weibull/models/federated_data/centralized/{dataset}/{model_name}_{fold}_0.pt"
        FED_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/AFT_Weibull/models/federated_data/federated/{dataset}/{model_name}_model_run_{fold}.tar"
    elif model_name == "DeepHit": 
        CENT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepHit/models/federated_data/centralized/{dataset}/{model_name}_{fold}_0.pt"
        FED_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepHit/models/federated_data/federated/{dataset}/{model_name}_model_run_{fold}.pt"
    elif model_name == "LinearPH": 
        CENT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/LinearPH_NNph/models/federated_data/centralized/{dataset}/Centralized_LinearPH_{dataset}_{fold}_0.pt"
        FED_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/LinearPH_NNph/models/federated_data/federated/{dataset}/LinearPH_model_run_{fold}.pt"
    elif model_name == "NNPH": 
        CENT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/LinearPH_NNph/models/federated_data/centralized/{dataset}/Centralized_NNph_{dataset}_{fold}_0.pt"
        FED_TMPL =  f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/LinearPH_NNph/models/federated_data/federated/{dataset}/NNph_model_run_{fold}.pt"
    elif model_name == "Fed_CoxPH": 
        CENT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/Fed_CoxPH/models/federated_data/centralized/{dataset}/Fed_CoxPH_{fold}_0.pt"
        FED_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/Fed_CoxPH/models/federated_data/federated/{dataset}/CoxPH_model_run_{fold}.pt"
    elif model_name == "NNnPH": 
        CENT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/NNnPH/models/federated_data/centralized/{dataset}/Centralized_NNnPH_{dataset}_{fold}_0.pt"
        FED_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/NNnPH/models/federated_data/federated/{dataset}/NNnPH_model_run_{fold}.pt"
    elif model_name == "DeepPseudo":   
        CENT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepPseudo/models/federated_data/centralized/{dataset}/DeepPseudo_{fold}_0.pt"
        FED_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepPseudo/models/federated_data/federated/{dataset}/DeepPseudo_global_model_{fold}.pt"
    else: 
        raise ValueError("model name did not match with our repository")
    return CENT_TMPL, FED_TMPL

################################ Model Directory for Site Level ############################################################ 

def get_data_model_directory_site_level(model_name, data, fold):
    CKPT_TMPL = None 
    if model_name in ["FedPDNN", "FedPLSTM", "FedPAttn"] : 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/FedPseudo_Flare/models/federated_data/federated/{data}/{model_name}/{data}/{model_name}_global_{fold}.pt"
    elif model_name == "AFT_Weibull": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/AFT_Weibull/models/federated_data/federated/{data}/{model_name}_model_run_{fold}.tar"
    elif model_name == "DeepHit": 
         CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepHit/models/federated_data/federated/{data}/{model_name}_model_run_{fold}.pt"
    elif model_name == "LinearPH": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/LinearPH_NNph/models/federated_data/federated/{data}/LinearPH_model_run_{fold}.pt"
    elif model_name == "NNPH": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/LinearPH_NNph/models/federated_data/federated/{data}/NNph_model_run_{fold}.pt"
    elif model_name == "Fed_CoxPH": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/Fed_CoxPH/models/federated_data/federated/{data}/CoxPH_model_run_{fold}.pt"
    elif model_name == "NNnPH": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/NNnPH/models/federated_data/federated/{data}/NNnPH_model_run_{fold}.pt"
    elif model_name == "DeepPseudo": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepPseudo/models/federated_data/federated/{data}/DeepPseudo_global_model_{fold}.pt"
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
    data_dir = "/home/il66977@ad.umbc.edu/Benchmarking_FSA/datasets/processed_data/federated_data/seer/centralized" 
    train_df = load_fold_csv(data_dir, "train", fold)

    train_time_all = train_df[TIME_COL].to_numpy()
    train_status_all = train_df[EVENT_COL].to_numpy().astype(int) 
    y_train_all = (train_time_all, train_status_all)

    discretiser = Discretiser(40, scheme="km")
    discretiser.fit(train_df[TIME_COL].to_numpy(), train_df[EVENT_COL].to_numpy())   
    return discretiser

####################################### Split Dataframe into feature and labels ####################################################################

def load_model(model_name, ckpt_path1, in_features, out_features):
    discretiser = None 
    ckpt = torch.load(ckpt_path1,  weights_only=False)
    
    if model_name == "DeepHit":   
        state = ckpt["model_state_dict"]
        opt = tt.optim.Adam(lr=1e-3, weight_decay=5e-5)
        model = build_model(model_name, in_features, out_features)
        
        duration_index = ckpt["duration_index"]
        alpha, sigma = ckpt["alpha"], ckpt["sigma"]
        model = DeepHit(model, opt, alpha=alpha, sigma=sigma, duration_index=duration_index)
        
    elif model_name in [ "LinearPH", "NNPH"]:
        state = ckpt["model"]
        discretiser = ckpt["discretizer"] 
        model = build_model(model_name, in_features, out_features, discretiser) 
        state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()} 
        model.load_state_dict(state) 
    else:
        model = build_model(model_name, in_features, out_features, discretiser) 
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt  
        state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()}
        model.load_state_dict(state)
    return model, discretiser

def load_discretizer(ckpt_path1): 
    ckpt = torch.load(ckpt_path1,  weights_only=False)
    discretizer_ley = "discretizer" if "discretizer" in ckpt else "discretiser"
    discretiser = ckpt[discretizer_ley]  
    return discretiser

  
def load_model_site(model_name, ckpt_path1, in_features, out_features):
    discretiser = None 
    ckpt = torch.load(ckpt_path1,  weights_only=False)
    
    if model_name == "DeepHit":   
        state = ckpt["model_state_dict"]
        opt = tt.optim.Adam(lr=1e-3, weight_decay=5e-5)
        model = build_model(model_name, in_features, out_features)
        
        duration_index = ckpt["duration_index"]
        alpha, sigma = ckpt["alpha"], ckpt["sigma"]
        model = DeepHit(model, opt, alpha=alpha, sigma=sigma, duration_index=duration_index)
        
    elif model_name in ["NNnPH"]: 
        
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            discretizer_ley = "discretizer" if "discretizer" in ckpt else "discretiser"
            discretiser = ckpt[discretizer_ley] 
        elif isinstance(ckpt, dict) and ckpt and all(torch.is_tensor(value) for value in ckpt.values()):
            state = ckpt
        else:       
            state = None 
        
        model = build_model(model_name, in_features, out_features, discretiser) 
        state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()} 
        model.load_state_dict(state)
        
    elif model_name in [ "LinearPH", "NNPH"]:
        state = ckpt["model"]
        discretiser = ckpt["discretizer"] 
        model = build_model(model_name, in_features, out_features, discretiser) 
        state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()} 
        model.load_state_dict(state)
    
    else:
        model = build_model(model_name, in_features, out_features, discretiser) 
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt  
        state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()}
        model.load_state_dict(state)
    return model, discretiser


  
def load_model_partition(model_name, ckpt_path1, in_features, out_features, discretiser = None): 
    ckpt = torch.load(ckpt_path1,  weights_only=False)
    
    if model_name == "DeepHit":   
        state = ckpt["model_state_dict"]
        opt = tt.optim.Adam(lr=1e-3, weight_decay=5e-5)
        model = build_model(model_name, in_features, out_features)
        
        duration_index = ckpt["duration_index"]
        alpha, sigma = ckpt["alpha"], ckpt["sigma"]
        model = DeepHit(model, opt, alpha=alpha, sigma=sigma, duration_index=duration_index)
        
    elif model_name in ["NNnPH"]: 
        
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            discretizer_ley = "discretizer" if "discretizer" in ckpt else "discretiser"
            discretiser = ckpt[discretizer_ley] 
        elif isinstance(ckpt, dict) and ckpt and all(torch.is_tensor(value) for value in ckpt.values()):
            state = ckpt
        else:       
            state = None 
        
        model = build_model(model_name, in_features, out_features, discretiser) 
        state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()} 
        model.load_state_dict(state)
        
    elif model_name in [ "LinearPH", "NNPH"]:
        state = ckpt["model"]
        discretiser = ckpt["discretizer"] 
        model = build_model(model_name, in_features, out_features, discretiser) 
        state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()} 
        model.load_state_dict(state)
    
    else:
        model = build_model(model_name, in_features, out_features, discretiser) 
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt  
        state = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in state.items()}
        model.load_state_dict(state)
    return model, discretiser
 
  
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
    if model_name in ["FedPDNN", "FedPLSTM", "FedPAttn"]: 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/FedPseudo_Flare/models/federated_data/federated/{data}/{model_name}/{data}/{model_name}_global_{fold}.pt"
    elif model_name == "AFT_Weibull": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/AFT_Weibull/models/federated_data/federated/{data}/AFT_Weibull_model_run_{fold}.tar"
    elif model_name == "DeepHit": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepHit/models/federated_data/federated/{data}/DeepHit_model_run_{fold}.pt"
    elif model_name == "LinearPH": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/LinearPH_NNph/models/federated_data/federated/{data}/LinearPH_model_run_{fold}.pt"
    elif model_name == "NNPH": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/LinearPH_NNph/models/federated_data/federated/{data}/NNph_model_run_{fold}.pt"
    elif model_name == "Fed_CoxPH": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/Fed_CoxPH/models/federated_data/federated/{data}/CoxPH_model_run_{fold}.pt"
    elif model_name == "NNnPH": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/NNnPH/models/federated_data/federated/{data}/NNnPH_model_run_{fold}.pt"
    elif model_name == "DeepPseudo": 
        CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepPseudo/models/federated_data/federated/{data}/DeepPseudo_global_model_{fold}.pt"
    else:
        raise ValueError("model name did not match with our repository") 
    return CKPT_TMPL


# def get_data_model_directory_fold_level(model_name, data, fold, client=0):
#     CKPT_TMPL = None 
#     if model_name in ["FedPDNN", "FedPLSTM", "FedPAttn"]: 
#         CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/FedPseudo_Flare/models/federated_data/centralized/{data}/{data}/{model_name}_global_{fold}.pt"
#     elif model_name == "AFT_Weibull": 
#         CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/AFT_Weibull/models/federated_data/centralized/{data}/{model_name}_{fold}_{client}.pt"
#     elif model_name == "DeepHit": 
#         CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepHit/models/federated_data/centralized/{data}/{model_name}_{fold}_0.pt"
#     elif model_name == "LinearPH": 
#         CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/LinearPH_NNph/models/federated_data/centralized/{data}/Centralized_{model_name}_{data}_{fold}_0.pt"
#     elif model_name == "NNPH": 
#         CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/LinearPH_NNph/models/federated_data/centralized/{data}/Centralized_NNph_{data}_{fold}_0.pt"
#     elif model_name == "Fed_CoxPH": 
#         CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/Fed_CoxPH/models/federated_data/centralized/{data}/Fed_CoxPH_{fold}_0.pt"
#     elif model_name == "NNnPH": 
#         CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/NNnPH/models/federated_data/centralized/{data}/Centralized_NNnPH_{data}_{fold}_0.pt"
#     elif model_name == "DeepPseudo": 
#         CKPT_TMPL = f"/home/il66977@ad.umbc.edu/Benchmarking_FSA/DeepPseudo/models/federated_data/centralized/{data}/DeepPseudo_{fold}_0.pt"
#     else:
#         raise ValueError("model name did not match with our repository")
         
#     return CKPT_TMPL

