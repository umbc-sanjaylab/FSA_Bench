import traceback

import numpy as np
from lifelines import KaplanMeierFitter
import torch, warnings    
from torch.utils.data import DataLoader, TensorDataset   
import pandas as pd 
from NNnPH.utils import Concordance
from pycox.evaluation import EvalSurv  
from torch.utils.data import DataLoader, TensorDataset
import torch 

##################################### Optimizer from Arguments ############################################################### 

def build_optimizer(model, args): 
    parameters = filter( lambda p: p.requires_grad, model.parameters()) 
    if args.optimizer == "adam":
        return torch.optim.Adam( parameters, lr=args.lr, weight_decay=args.reg) 
    elif args.optimizer == "amsgrad":
        return torch.optim.Adam( parameters, lr=args.lr, weight_decay=args.reg, amsgrad=True) 
    elif args.optimizer == "sgd":
        return torch.optim.SGD( parameters, lr=args.lr, momentum=args.rho, weight_decay=args.reg)  
    else: 
      raise ValueError( f"Unsupported optimizer: {args.optimizer}") 
    
##################################### Fit the training target to transform validation target ################################### 

def data_transform(y, discretiser, fit_transform=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fit_transform:
            y_trans = discretiser.fit_transform(*y)
        else:
            y_trans = discretiser.transform(*y) 
    return y_trans
  
##################################### Client training and validation data preparation ############################################ 

def prepare_client_dataloaders(client, args):
  
    discretiser = Discretiser( args.num_duration, scheme="km") 
    train_time_trans, train_event_trans = data_transform( ( client.train.time, client.train.event ), discretiser, fit_transform=False) 
    val_time_trans, val_event_trans = data_transform((client.val.time, client.val.event ), discretiser, fit_transform=False) 

    train_dataset = TensorDataset(torch.as_tensor(client.train.X, dtype=torch.float32), torch.as_tensor(train_time_trans), torch.as_tensor(train_event_trans)) 
    val_dataset = TensorDataset(torch.as_tensor(client.val.X, dtype=torch.float32), torch.as_tensor(val_time_trans), torch.as_tensor(val_event_trans))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
     
    return train_loader, val_loader, discretiser

################################################## Concodance Index metrics ######################################################  
  
def surv_const_pdf(s, sub=10):
    """Basic method for constant PDF interpolation that use `self.model.predict_surv`.
    Arguments:
        input {np.ndarray} -- Input to net 
    Returns:
        np.ndarray -- Predictions
    """
    s = torch.from_numpy(s)
    n, m = s.shape
    diff = (s[:, 1:] - s[:, :-1]).contiguous().view(-1, 1).repeat(1, sub).view(n, -1)
    rho = torch.linspace(0, 1, sub+1)[:-1].contiguous().repeat(n, m-1)
    s_prev = s[:, :-1].contiguous().view(-1, 1).repeat(1, sub).view(n, -1)
    surv = torch.zeros(n, int((m-1)*sub + 1))
    surv[:, :-1] = diff * rho + s_prev
    surv[:, -1] = s[:, -1]
    return surv.detach().numpy() 

def surv_const_pdf_df(s, cuts, sub=10):
    surv = surv_const_pdf(s, sub)
    grid = cuts
    subgrid = [np.linspace(start, end, num=sub+1)[:-1] for start, end in zip(grid[:-1], grid[1:])]
    subgrid = [inner for outer in subgrid for inner in outer]
    subgrid += [grid[-1]]
    return pd.DataFrame(surv.transpose(), subgrid)   
  
def predict_surv(model, input=None):
    hazard = model(input).sigmoid()
    surv = (1 - hazard).log().cumsum(1).exp()
    return surv.cpu().detach().numpy()
  
 

################################################## Discretizer ######################################################

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
            self.cuts = np.linspace( 0, durations.max(), self._cuts, dtype=np.float64)
        elif self._scheme == 'km':
            kmf = KaplanMeierFitter()
            kmf.fit(durations, events)
            last_duration = durations.max()
            percentiles = np.linspace(kmf.predict( last_duration), kmf.predict(0), self._cuts, dtype=np.float64)
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
  
  
################################################## Concodance Index metrics ######################################################


def Concordance(model, x, durations, events, discretiser, device="cpu"):
    """
    Attributes:
    ----------
        model: Model

        x: Covariates
        durations: Event status array

        events: Observed times array
        evaltime: Prespecified evaluation times

    Returns
    -------
        Time-dependent C-index
    """

    model.to(device)
    model.eval()
    x = x.to(device)

    # Predict the survival probability for test set
    sp = predict_surv(model, x.to(device))
    surv = surv_const_pdf_df(sp, discretiser.cuts)

    cindex = np.nan
    # EvalSurv function from pycox package.
    ev = EvalSurv(surv, durations, events, censor_surv='km')

    try:
        cindex = ev.concordance_td()
        if not np.isfinite(cindex):
            cindex = np.nan
            print("[C-index WARNING] NaN/Inf returned.")
    except ZeroDivisionError as e:
        print(
            f"[Exception in evaluation method for concordance index score computation]:  {type(e).__name__}: {e}")
        cindex = np.nan
        traceback.print_exc()
    return cindex

############################################# Prepare the dataloader #########################################################################

def prepare_nnnph_dataloaders(data, args): 
    discretiser = Discretiser( args.num_duration, scheme="km") 

    # Fit discretiser using training target & use it to transform validation target
    train_time_trans, train_event_trans = data_transform( ( data.train.time, data.train.event ), discretiser, fit_transform=True)  
    val_time_trans, val_event_trans = data_transform( ( data.val.time, data.val.event ), discretiser, fit_transform=False )

    train_dataset = TensorDataset(torch.as_tensor(  data.train.X, dtype=torch.float32 ), torch.as_tensor(train_time_trans), torch.as_tensor(train_event_trans)) 
    val_dataset = TensorDataset(torch.as_tensor( data.val.X, dtype=torch.float32 ), torch.as_tensor(val_time_trans), torch.as_tensor(val_event_trans))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) 
    return  train_loader, val_loader, discretiser

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
            raise ValueError("Invalid scheme")
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
      



def is_finite_metric(value):
    return  value is not None and np.isscalar(value) and np.isfinite(float(value))  

############################################# Evaluation of Trained method  #################################################################


def Evaluation(model, train_time, train_status, X_test, test_time, test_status, evaltime, discretiser, device: str = ''):
    """
    Attributes:
    ----------
        model: Trained DeepPseudo Model

        train_time: Training time array
        train_status: Training status array

        X_test: Covariates in test set
        test_time: Event status array for test set
        test_status: Observed times array for test set

        evaltime: Prespecified evaluation times
        discretizer: object for discretizing time
        device: string for cpu or gpu indicator

    Returns
    -------
        Time-dependent C-index, Integrated Brier Score and Integrated nbll score
    """

    model.eval()
    model.to(device)

    # Predict the survival probability for test set
    sp_test = predict_surv(model, torch.tensor(X_test).to(device))
    surv = surv_const_pdf_df(sp_test, discretiser.cuts)
    
    # EvalSurv function from pycox package.
    ev = EvalSurv(surv, test_time, test_status, censor_surv='km')

    # time-dependent concordance index.
    try:
        cindex = ev.concordance_td()
        if not np.isfinite(cindex):
            print("[C-index WARNING] NaN/Inf returned.")
            cindex = np.nan
    except Exception as e:
        print(f"[C-index ERROR] {type(e).__name__}: {e}")
        cindex = np.nan

    # integrated brier score computation.
    try:
        brier = ev.integrated_brier_score(evaltime)
        if not np.isfinite(brier):
            print("[C-index WARNING] NaN/Inf returned.")
            brier = np.nan
    except Exception as e:
        print(f"[Brier ERROR] {type(e).__name__}: {e}")
        brier = np.nan

    # integrated brier score computation.
    try:
        nbll = ev.integrated_nbll(evaltime)
        if not np.isfinite(nbll):
            print("[C-index WARNING] NaN/Inf returned.")
            nbll = np.nan
    except Exception as e:
        print(f"[Brier ERROR] {type(e).__name__}: {e}")
        nbll = np.nan

    # --- SurvivalEvaluator (SurvivalEval) ---
    p_value = np.nan
    weighted_MAE_PO = np.nan

    try:
        test_time_arr = np.asarray(test_time,   dtype=float)
        test_status_arr = np.asarray(test_status, dtype=float)
        train_time_arr = np.asarray(train_time,  dtype=float)
        train_status_arr = np.asarray(train_status, dtype=float)
        evaltime_arr = np.asarray(evaltime,    dtype=float)

        surv_interp = pd.DataFrame(
            index=evaltime_arr,
            columns=surv.columns,
            dtype=float
        )

        for t in evaltime_arr:
            idx = surv.index.searchsorted(t, side='right') - 1
            idx = max(0, min(idx, len(surv.index) - 1))
            surv_interp.loc[t] = surv.iloc[idx].values

        from SurvivalEVAL.Evaluator import SurvivalEvaluator
        # Transpose to (n_samples, n_eval_times)
        surv_for_eval = surv_interp.values.T.astype(float)

        evaluator = SurvivalEvaluator(
            surv_for_eval,    # (n_samples, n_eval_times)
            evaltime_arr,     # (n_eval_times,)
            test_time_arr,    # (n_samples,)
            test_status_arr,  # (n_samples,)
            train_time_arr,   # (n_train_samples,)
            train_status_arr,  # (n_train_samples,)
        )

        # --- D-Calibration ---
        try:
            p_val_result, _ = evaluator.d_calibration()
            p_value = float(p_val_result) if np.isfinite(
                p_val_result) else np.nan
        except Exception as e:
            print(f"[D-Calibration WARNING] {e}")

        # --- Weighted Pseudo-Observation MAE ---
        try:
            mae_result = evaluator.mae(method='Pseudo_obs', weighted=True)
            weighted_MAE_PO = float(mae_result) if np.isfinite(
                mae_result) else np.nan
        except Exception as e:
            print(f"[Weighted MAE WARNING] {e}")

    except Exception as e:
        # FIX 3: Use traceback instead of print(..., exc_info=True)
        import traceback
        print(f"[SurvivalEvaluator] Failed to initialize: {e}")
        traceback.print_exc()

    print(
        f"[Evaluation] cindex={cindex:.4f}, brier={brier:.4f}, nbll={nbll:.4f}, "
        f"p_value={p_value}, weighted_MAE_PO={weighted_MAE_PO}"
    )

    return cindex, brier, nbll, p_value, weighted_MAE_PO
     