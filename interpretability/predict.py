import torch 
import numpy as np 
import pandas as pd 

import torchtuples as tt
from pycox.models import CoxPH

DEVICE = "cuda:1"
SEED   = 42

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


##################################### Survival Model Inference Methods ##################################### 
def predict_surv_matrix(model, X_df, evaltime, model_name, discretizer = None,  device="cuda:1"):
    """Return S(t|x) as [n_obs, len(times)] on the model's 9 pseudo-eval times.
    Seeds before the forward so FedPAttn's random attention is deterministic."""
    
    result = None 
    if model_name == "DeepPseudo":
        result = predict_surv_matrix_deepPseudo(model, X_df, device=device)
    elif model_name in ["FedPDNN", "FedPLSTM", "FedPAttn"]:
        result = predict_surv_matrix_FedPseudo(model, X_df, device=device)
    elif model_name == "AFT_Weibull":
        result = predict_surv_matrix_weibull(model, X_df, evaltime, device=device)
    elif model_name == "Fed_CoxPH":
        result = predict_surv_matrix_fed_coxph(model, X_df, evaltime)
    elif model_name in ["LinearPH", "NNPH"]:
        result = predict_surv_matrix_LinearPH_NNPH(model, X_df, evaltime, discretizer)
    elif model_name in ["NNnPH"]:
        result = predict_surv_matrix_NNnPH(model, X_df, evaltime, discretizer)
    elif model_name in ["DeepHit"]:
        result = predict_surv_matrix_deephit(model, X_df, evaltime, discretizer)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return np.array([StepFn(evaltime, row) for row in result])

##################################### Predict method of FedPseudo Models #####################################

def predict_surv_matrix_FedPseudo(model, X_df, device="cuda:1"):
    """Return S(t|x) as [n_obs, len(times)] on the model's 9 pseudo-eval times.
    Seeds before the forward so FedPAttn's random attention is deterministic."""
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
      
    x = torch.tensor(np.asarray(X_df, dtype=np.float32), dtype=model_dtype, device=model_device)
    torch.manual_seed(SEED)
    model.eval()
    
    with torch.no_grad():
        out = model(x)
    S = out.detach().cpu().numpy()
    return S 

##################################### Predict method of DeepPseudo Models #####################################

def predict_surv_matrix_deepPseudo(model, X_df, device="cuda:1"):
    """Return S(t|x) as [n_obs, len(times)] on the model's 9 pseudo-eval times.
    Seeds before the forward so FedPAttn's random attention is deterministic."""
    x = torch.tensor(np.asarray(X_df, dtype=np.float32), device=device)
    torch.manual_seed(SEED)
    
    with torch.no_grad():
        out = model(x)
        
    out_array = out.detach().cpu().numpy()
    if out_array.ndim == 3 and out_array.shape[1] == 1:
            out_array = out_array.squeeze(axis=1)
            
    return out_array 


##################################### Predict method of AFT Weibull Models #####################################

def predict_surv_matrix_weibull(model, X_df, evaltime, device="cuda:1"):
    """Return S(t|x) as [n_obs, len(times)] on the model's 9 pseudo-eval times.
    Seeds before the forward so FedPAttn's random attention is deterministic."""
    X_test = torch.tensor(np.asarray(X_df, dtype=np.float32), device=device)
    model.to(device)
    
    torch.manual_seed(SEED)
    with torch.no_grad():
        log_params_test = model(X_test).detach().cpu().numpy()
    
    log_scale = log_params_test[:, 0]
    log_shape = log_params_test[:, 1]

    scale = np.exp(log_scale)
    shape = np.exp(log_shape)

    eps = 1e-12
    scale = np.clip(scale, eps, 1e12)
    shape = np.clip(shape, eps, 1e12)

    surv_test_np = np.exp(- (evaltime[np.newaxis, :] / scale[:, np.newaxis]) ** shape[:, np.newaxis])
    surv_test_np = np.clip(surv_test_np, eps, 1.0) 
    return surv_test_np


##################################### Predict method of Federated CoxPH Models #####################################

def predict_surv_matrix_fed_coxph(model, X_df, evaltime):
    """Return S(t|x) as [n_obs, len(times)] on the model's 9 pseudo-eval times.
    Seeds before the forward so FedPAttn's random attention is deterministic."""
    X = np.asarray(X_df, dtype=np.float32) 
    surv = model.predict_surv_df(X)                 # [n_times, n_obs], index = baseline time grid
    
    index_surv = np.asarray(surv.index)
    if index_surv.size == 0:
        return np.full((X.shape[0], len(evaltime)), np.nan)
      
    new_idx = []                                    # step lookup: largest baseline time <= each evaltime
    for t in evaltime:
        pos = int(np.clip(np.searchsorted(index_surv, t, side="right") - 1, 0, len(index_surv) - 1))
        new_idx.append(index_surv[pos])
    S = surv.loc[new_idx].to_numpy().T              # -> [n_obs, len(evaltime)]
    return np.clip(S, 1e-12, 1.0) 


##################################### LinearPH & NNPH Model Inference #####################################


def predict_surv(model, input=None):
    hazard = model(input).sigmoid()
    surv = (1 - hazard).log().cumsum(1).exp()
    return surv.cpu().detach().numpy()
 
def surv_const_pdf(s, sub=10):
    """Basic method for constant PDF interpolation that use `self.model.predict_surv`.
    Arguments:
        input {np.ndarray} -- Input to net.

    Returns:
        np.ndarray -- Predictions
    """
    s = torch.from_numpy(s)
    n, m = s.shape

    diff = (s[:, 1:] - s[:, :-1]).contiguous().view(-1,
                                                    1).repeat(1, sub).view(n, -1)
    rho = torch.linspace(0, 1, sub+1)[:-1].contiguous().repeat(n, m-1)
    s_prev = s[:, :-1].contiguous().view(-1, 1).repeat(1, sub).view(n, -1)
    surv = torch.zeros(n, int((m-1)*sub + 1))
    surv[:, :-1] = diff * rho + s_prev
    surv[:, -1] = s[:, -1]
    return surv.detach().numpy()

# def surv_const_pdf_df(s, cuts, sub=10):
#     surv = surv_const_pdf(s, sub)
#     grid = cuts
    
#     subgrid = [np.linspace(start, end, num=sub+1)[:-1] for start, end in zip(grid[:-1], grid[1:])] 
#     subgrid = [inner for outer in subgrid for inner in outer]
#     subgrid += [grid[-1]]   
#     return pd.DataFrame(surv, subgrid)

def surv_const_pdf_df(s, cuts, sub=10):
    surv = surv_const_pdf(s, sub)          # (n_samples, n_subgrid_times)
    grid = cuts
    subgrid = [np.linspace(start, end, num=sub+1)[:-1]
               for start, end in zip(grid[:-1], grid[1:])]
    subgrid = [inner for outer in subgrid for inner in outer]
    subgrid += [grid[-1]]
    
    return pd.DataFrame(surv, columns=subgrid)  # ← columns=, not positional arg
  

##############################  LinearPH & NNPH Inference  ############################################### 

def predict_surv_matrix_LinearPH_NNPH(model, X_df, evaltime, discretizer):
    X_test = np.asarray(X_df, dtype=np.float32)
    model.eval().to(DEVICE)

    sp_test = predict_surv(model, torch.tensor(X_test).to(DEVICE))  # (n_samples, n_model_cuts)

    n_cuts_model = sp_test.shape[1]      # e.g. 40 (model output)
    n_cuts_disc  = len(discretizer.cuts) # e.g. 39 (discretizer)

    # Align: trim whichever is longer so both match
    n_cuts = min(n_cuts_model, n_cuts_disc)
    sp_test_aligned = sp_test[:, :n_cuts]       # trim model output
    cuts_aligned    = discretizer.cuts[:n_cuts]  # trim cuts

    # print(f"[DEBUG] sp_test: {sp_test.shape} → aligned: {sp_test_aligned.shape}, cuts: {n_cuts}")

    surv_df = surv_const_pdf_df(sp_test_aligned, cuts_aligned)
    return surv_df.values

##############################  NNnPH Inference  ###########################################################  


def predict_surv_matrix_NNnPH(model, X_df, evaltime, discretizer):
    X_test = np.asarray(X_df, dtype=np.float32)
    model.eval().to(DEVICE)
     
    # Predict the survival probability for test set
    sp_test = predict_surv(model, torch.tensor(X_test).to(DEVICE)) 
    surv = surv_const_pdf_df(sp_test, discretizer.cuts) 
    return surv.values
  
  
#############################  DeepHit Inference  ###########################################################  
  
def predict_surv_matrix_deephit(model, X_df, evaltime, discretizer):
    # ── Convert DataFrame to numpy first ──
    X_test = np.asarray(X_df, dtype=np.float32)
    X_test_tensor = torch.tensor(X_test, device=DEVICE) 
    cif = model.predict_cif(X_test)  
    return cif[0]
