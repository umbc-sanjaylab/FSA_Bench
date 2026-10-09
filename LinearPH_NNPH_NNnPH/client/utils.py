import numpy as np, traceback
from torch import Tensor
from SurvivalEVAL.Evaluator import SurvivalEvaluator 
from lifelines import KaplanMeierFitter
import torch, warnings    
from torch.utils.data import DataLoader, TensorDataset   
import pandas as pd  
import torch.nn.functional as F 
from pycox.evaluation import EvalSurv  
from torch.utils.data import DataLoader, TensorDataset

import logging 
logger = logging.getLogger("experiment")

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

def prepare_client_dataloaders(client, discretiser, args):  
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


################################################## Negative Partial Log-likelihood Function ######################################################

def negative_llh(phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Arguments:
        phi -- N x T. Estimates in (-inf, inf) where hazard = sigmoid(phi). 
        idx_durations -- N. Event times for each individual represented as indices based on discretisation.
        events -- N. Indicator of event (1.) or censoring (0.) for each individual. Floats.
    Returns:
        Negative LLH as a 1dim Tensor
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`." +
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`," +
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    # stack events and idx_durations to get Nx1 Tensor
    # .view(-1, 1) -- infer first dimension to get second dimension as size=1
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    # create NxT zero Tensor and put in each row at the idx_duration whether the event occurred
    # censored ppl have row of zeros, uncensored ppl have row with a single one
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events.float())
    # with logits to pass phi through a sigmoid first -- recall it's unconstrained
    bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')
    # sum the loss until failure time:
    # first sum until each time
    # then gather the summed loss at relevant failure time
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)
  
def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    raise ValueError(
        f"`reduction` = {reduction} is not valid. Use 'none', or 'mean'.")
  
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
 
    # EvalSurv function from pycox package.
    ev = EvalSurv(surv, durations, events, censor_surv='km') 
    try:
        cindex = ev.concordance_td()
        if not np.isfinite(cindex):
            cindex = np.nan
            logger.warning("[C-index WARNING] NaN/Inf returned.")
    except ZeroDivisionError as e:
        logger.exception(f"[Exception in evaluation method for concordance index score computation]:  {type(e).__name__}: {e}")
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

    def __init__(self, num_duration, scheme='equidistant'):
        self._cuts = num_duration
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

############################################# Metrics value test  #################################################################
 
def safe_metric(metric_name, metric_fn):
    try: 
        value = float(metric_fn()) 
        if np.isfinite(value):
            return value

        logger.warning("%s returned NaN/Inf.", metric_name) 
    except Exception:
        logger.exception("%s computation failed.", metric_name)

    return np.nan

############################################# Evaluation of Trained method  #################################################################

def Evaluation(model, train_time, train_status, X_test, test_time, test_status, evaltime, discretiser, device="cpu"):
    """
    Evaluate a trained survival model using discrimination, calibration,
    and prediction-error metrics.

    Returns
    -------
    cindex, brier, nbll, p_value, weighted_MAE_PO
    """

    model.eval()
    model.to(device)

    # Predict survival probabilities
    X_test_tensor = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    sp_test = predict_surv(model, X_test_tensor)  
    surv = surv_const_pdf_df(sp_test, discretiser.cuts)
     
    # PyCox evaluation
    ev = EvalSurv(surv, test_time, test_status, censor_surv="km")

    cindex = safe_metric("C-index", ev.concordance_td)
    brier = safe_metric("Integrated Brier Score", lambda: ev.integrated_brier_score(evaltime))
    nbll = safe_metric("Integrated NBLL", lambda: ev.integrated_nbll(evaltime))

    # SurvivalEVAL metrics
    p_value = np.nan
    weighted_MAE_PO = np.nan

    try:
        test_time_arr = np.asarray(test_time, dtype=float)
        test_status_arr = np.asarray(test_status, dtype=float)
        train_time_arr = np.asarray(train_time, dtype=float)
        train_status_arr = np.asarray(train_status, dtype=float)
        evaltime_arr = np.asarray(evaltime, dtype=float)

        surv_interp = pd.DataFrame(index=evaltime_arr, columns=surv.columns, dtype=float)

        for t in evaltime_arr:
            idx = surv.index.searchsorted(t, side="right") - 1
            idx = max(0, min(idx, len(surv.index) - 1))
            surv_interp.loc[t] = surv.iloc[idx].values

        surv_for_eval = surv_interp.values.T.astype(float)

        evaluator = SurvivalEvaluator(
            surv_for_eval,
            evaltime_arr,
            test_time_arr,
            test_status_arr,
            train_time_arr,
            train_status_arr
        )

        p_value = safe_metric("D-Calibration", lambda: evaluator.d_calibration()[0])
        weighted_MAE_PO = np.nan 
        if np.sum(train_status) > 0:
            weighted_MAE_PO = safe_metric("Weighted MAE PO", lambda: evaluator.mae( method="Pseudo_obs", weighted=True))
        else:
            logger.warning( "Weighted MAE PO skipped because the training split has no events." )

    except Exception:
        logger.exception("SurvivalEvaluator initialization failed.")

    logger.info(
        "Evaluation | C-index: %s | Brier: %s | NBLL: %s | D-calibration: %s | Weighted MAE PO: %s",
        cindex, brier, nbll, p_value, weighted_MAE_PO
    )

    return cindex, brier, nbll, p_value, weighted_MAE_PO 
  
  # SurvivalEvaluator
     