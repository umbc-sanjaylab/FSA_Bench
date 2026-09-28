import builtins
import os
import logging
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import random
from torch.utils.data import TensorDataset, DataLoader
from model import *
from math import sqrt
from pycox.evaluation import EvalSurv
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import time
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
import warnings
import numpy as np
import traceback
from lifelines import KaplanMeierFitter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from SurvivalEVAL.Evaluator import SurvivalEvaluator
torch.cuda.set_per_process_memory_fraction(0.15, 0)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


#################################### Early Stopping Criteria ############################################################


class EarlyStopping:
    """Stop training if validation C-index does not improve for `patience` rounds."""

    def __init__(self, patience=25, verbose=False, delta=0.0, directory=None, discretiser=None, checkpoint_pth="checkpoint.pt"):
        self.patience = int(patience)
        self.verbose = verbose
        self.directory = directory
        self.delta = float(delta)
        self.logger = logging.getLogger("fedPseudo.server")
        self.checkpoint_pth = checkpoint_pth

        self.counter = 0
        self.best_val_cindex = float("-inf")
        self.early_stop = False
        self.discretiser=discretiser

        # new: keep metadata for the selected best round
        self.best_round = None
        self.best_test_cindex = None
        self.best_test_brier = None
        self.best_test_nbll = None
        self.best_test_p_value = None
        self.best_test_MAE_PO = None

    def __call__(self, val_c_index, param, round_idx=None, test_metrics=None):
        if val_c_index is None or not np.isfinite(val_c_index):
            if self.verbose:
                self.logger.info(f"[EarlyStopping] Skipping invalid val_c_index = {val_c_index}")
            return False

        improved = val_c_index > self.best_val_cindex + self.delta

        if improved:
            if self.verbose:
                self.logger.info(
                    f"Validation C-index improved "
                    f"({self.best_val_cindex:.6f} -> {val_c_index:.6f}). Saving checkpoint..."
                )

            self.best_val_cindex = val_c_index
            self.counter = 0
            self.best_round = round_idx

            if test_metrics is not None:
                self.best_test_cindex = test_metrics.get("test_c_index")
                self.best_test_brier = test_metrics.get("test_brier")
                self.best_test_nbll = test_metrics.get("test_nbll")
                self.best_test_p_value = test_metrics.get("test_p_value")
                self.best_test_MAE_PO = test_metrics.get("test_MAE_PO")

            self.save_checkpoint(param)
            return True

        self.counter += 1
        if self.verbose:
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True

        return False

    def save_checkpoint(self, param):
        if param is not None:
            os.makedirs(self.directory, exist_ok=True)
            torch.save(
              {
                "model": param, 
                "discretizer": self.discretiser   
              }
              , os.path.join( self.directory, self.checkpoint_pth))

######################################### Convert Torch Tensor to NumPy Array ####################################################


def to_1d_numpy(x):
    # torch tensor -> numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # pandas Series/DataFrame column -> numpy
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()

    # list/others -> numpy
    x = np.asarray(x)
    return x.reshape(-1)

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
    ev = EvalSurv(surv, durations, events, censor_surv='km')    # EvalSurv function from pycox package.

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

################################################## Get Local Dataset ######################################################


def get_local_dataset(dataset, partition, client_id, run_id, args):
    if partition == 'centralized':
        client_id = 0

    client_name = None
    if partition == "federated":
        client_name = get_client_name_for_federated_datasets(dataset, client_id)
        dataset_directory = f"{args.datadir}/{dataset}/{partition}/{client_name}/{run_id}/"
    else:
        dataset_directory = f"{args.datadir}/{dataset}/{partition}/{client_id}/{run_id}/"
        client_name = client_id

    train_df = pd.read_csv(dataset_directory + "train.csv")
    val_df = pd.read_csv(dataset_directory + "val.csv")
    test_df = pd.read_csv(dataset_directory + "test.csv")

    cols_to_remove = ['time', 'event']
    new_col_name = train_df.columns.tolist()
    feature_list = [x for x in new_col_name if x not in cols_to_remove]

    return train_df, val_df, test_df, client_name, feature_list

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
            self.cuts = np.linspace(
                0, durations.max(), self._cuts, dtype=np.float64)
        elif self._scheme == 'km':
            kmf = KaplanMeierFitter()
            kmf.fit(durations, events)
            last_duration = durations.max()
            percentiles = np.linspace(kmf.predict(
                last_duration), kmf.predict(0), self._cuts, dtype=np.float64)
            self.cuts = np.unique([kmf.percentile(p)
                                  for p in percentiles][::-1])
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

############################################# Discretizer #########################################################################


def surv_at_times(surv, evaltime):
    idx = np.searchsorted(surv.index, evaltime)
    idx = idx.clip(0, len(surv.index)-1)
    return surv.iloc[idx]

############################################# Discretizer #########################################################################


def predict_surv(model, input=None):
    hazard = model(input).sigmoid()
    surv = (1 - hazard).log().cumsum(1).exp()
    return surv.cpu().detach().numpy()

############################################# Discretizer #########################################################################


def predict_surv_df(self, model, cuts, input=None):
    surv = self.predict_surv(model, input)
    return pd.DataFrame(surv.transpose(), cuts)


############################################# Discretizer #########################################################################

def data_transform(y, discretiser, fit_transform=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fit_transform:
            y_trans = discretiser.fit_transform(*y)
        else:
            y_trans = discretiser.transform(*y)

    return y_trans


############################################# Discretizer #########################################################################


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

############################################# Discretizer #########################################################################


def surv_const_pdf_df(s, cuts, sub=10):
    surv = surv_const_pdf(s, sub)
    grid = cuts
    subgrid = [np.linspace(start, end, num=sub+1)[:-1]
               for start, end in zip(grid[:-1], grid[1:])]
    subgrid = [inner for outer in subgrid for inner in outer]
    subgrid += [grid[-1]]
    return pd.DataFrame(surv.transpose(), subgrid)

############################################# Store local training metrics #########################################################################


def store_results_in_csv(args, metrics_dict):
    csv_path = os.path.join(args.logdir, "results_local_training.csv")
    print(csv_path)

    row = {
        "seed": args.init_seed,
        "fold_id": int(args.run_id),
        "dataset": args.dataset,
        "partition": args.partition,
        "model": args.model,
        "c_idx": builtins.round(metrics_dict["c_idx"], 4),
        "brier": builtins.round(metrics_dict["brier"], 4),
        "nbll": builtins.round(metrics_dict["nbll"], 4),
        "p_value": builtins.round(metrics_dict["p_value"], 4),
        "weighted_MAE_PO": builtins.round(metrics_dict["weighted_MAE_PO"], 4),
        "lr": builtins.round(float(args.lr), 5),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "dropout": float(args.dropout_p)
    }

    df = pd.DataFrame([row])
    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)


############################################# Discretizer #########################################################################


def local_max_time(client_time):
    max_time = np.max(client_time)
    return max_time


############################################# Evaluation of Trained method  #################################################################

def Evaluation(model, train_time, train_status, X_test, test_time, test_status, evaltime, discretiser, device: str = ''):
    """
    Attributes:
    ----------
        model: Trained Model
        train_time: Event time array for train set
        train_status: Event status array for train set

        X_test: Covariates in test set
        test_time: Observed times array for test set
        test_status: Event status array for test set

        evaltime: Prespecified evaluation times
        device: Device to run the model on

    Returns
    -------
        cindex, brier, nbll, p_value, weighted_MAE_PO
    """

    model.eval()
    model.to(device)

    # Predict survival probabilities for test set
    sp_test = predict_surv(model, torch.tensor(X_test).to(device))
    # shape: (n_cuts, n_samples)
    surv = surv_const_pdf_df(sp_test, discretiser.cuts)
    
    print("sp_test NaN:", np.isnan(sp_test).sum())
    print("sp_test Inf:", np.isinf(sp_test).sum())
    print("surv NaN:", np.isnan(surv.values).sum())
    print("surv Inf:", np.isinf(surv.values).sum())
    print("surv shape:", surv.shape)
    print("surv index:", surv.index[:10])
    print(surv.iloc[:5, :5])

    # EvalSurv (pycox)
    ev = EvalSurv(surv, test_time, test_status, censor_surv='km')

    # --- C-index ---
    try:
        cindex = ev.concordance_td()
        if not np.isfinite(cindex):
            print("[C-index WARNING] NaN/Inf returned.")
            cindex = np.nan
    except Exception as e:
        print(f"[C-index ERROR] {type(e).__name__}: {e}")
        cindex = np.nan

    # --- Integrated Brier Score ---
    try:
        brier = ev.integrated_brier_score(evaltime)
        if not np.isfinite(brier):
            print("[Brier WARNING] NaN/Inf returned.")
            brier = np.nan
    except Exception as e:
        print(f"[Brier ERROR] {type(e).__name__}: {e}")
        brier = np.nan

    # --- Integrated NBLL ---
    try:
        nbll = ev.integrated_nbll(evaltime)
        if not np.isfinite(nbll):
            print("[NBLL WARNING] NaN/Inf returned.")
            nbll = np.nan
    except Exception as e:
        print(f"[NBLL ERROR] {type(e).__name__}: {e}")
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

    # FIX 4: Return all computed metrics
    return cindex, brier, nbll, p_value, weighted_MAE_PO


############################################# Discretizer #########################################################################

def get_trainable_parameters(net):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # logger.info("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    N = 0
    for params in paramlist:
        N += params.numel()
        # logger.info("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset +
                numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset += numel
    # logger.info("get trainable x:", X)
    return X


############################################# Discretizer #########################################################################


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset += numel


############################################# Discretizer #########################################################################


def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model

############################################# Discretizer #########################################################################


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if (type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif (type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    return

############################################# Discretizer #########################################################################


# scale the loss to make it data size invariant
def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    raise ValueError(
        f"`reduction` = {reduction} is not valid. Use 'none', or 'mean'.")


############################################# Discretizer #########################################################################

def negative_llh(phi: Tensor, idx_durations: Tensor, events: Tensor,
                 reduction: str = 'mean') -> Tensor:
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


################################ cumulative count of the target column based on a date column  #######################################################


def get_cumulative(table, groupby, date, name, key=None):
    """
    Action: 
       1. first project only targets and date column
       2. collapse based on key to remove duplicates in the working table
       3.  DOES not remove duplicates in original input table

    Keyword arguments:
      table -- description
    Return: return_description
    """

    df = table.copy()
    start = df.shape[0]

    if type(groupby) is not list:
        groupby = [groupby]

    filterby = groupby + [date]
    # collapse the dataframe based on keys
    if key is not None:
        if type(key) is not list:
            key = [key]
        filterby = filterby + key
    temp = df.drop_duplicates(key)[filterby]

    # counts number of records having the same values in groupby + [date]
    # reset_index to turn the Series into a DataFrame with columns filled in
    temp = temp.groupby(filterby, dropna=False).size(
    ).reset_index(name='no_of_records')

    # drop records without date to exclude from cumsum
    # cumulatively sum over number of records
    # dropna=False AND left join to keep all nans in the left table
    temp[name] = temp.dropna(subset=[date]).sort_values(
        by=date, ascending=True).groupby(groupby)['no_of_records'].cumsum()

    df = pd.merge(df, temp, on=filterby, how='left')
    df = df.drop(columns=['no_of_records'])

    end = df.shape[0]
    print("lost: ", start - end)
    return df


############################################# Get Indicator Method #########################################################################


def get_indicator(table, target, groupby, date, name, key=None):

    df = table.copy()
    start = df.shape[0]

    if type(groupby) is not list:
        groupby = [groupby]

    filterby = groupby + [date, target]
    # collapse the dataframe based on keys
    if key is not None:
        if type(key) is not list:
            key = [key]
        filterby = filterby + key
    temp = df.drop_duplicates(key)[filterby]

    temp['temp1'] = temp[target].astype(float)
    temp['temp2'] = temp.sort_values(by='temp1', ascending=False).sort_values(
        by=date, ascending=True).groupby(groupby)['temp1'].cumsum()
    # print(temp)
    # temp = temp.drop(columns=target)
    temp[name] = (temp.dropna(subset=[date, target])
                  ['temp2'] > 0).astype('bool')

    df = pd.merge(df, temp, on=filterby, how='left')

    df = df.drop(columns=['temp1', 'temp2'])

    end = df.shape[0]
    print("lost: ", start - end)
    return df

############################################# Merge #########################################################################


def merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
    left = left.merge(right, how, on, left_on, right_on, left_index,
                      right_index, sort, suffixes, copy, indicator, validate)
    drop = ['_r' in c for c in left.columns]
    return left.drop(columns=left.columns[drop])

############################################# Aggregate Method #####################################################################


def add_aggregate(table, what, target, groupby, name, key=None):

    df = table.copy()
    start = df.shape[0]

    if groupby is not list:
        groupby = [groupby]

    filterby = groupby + [target]

    if key is not None:
        if type(key) is not list:
            key = [key]
        filterby = filterby + key

    df = df.drop_duplicates(key)

    nonnulls = ~df[target].isnull()
    df[name] = df[nonnulls].groupby(groupby)[target].transform(what)
    df[name] = df[[name] + groupby].groupby(groupby)[name].transform('max')

    end = df.shape[0]
    print("lost: ", start - end)

    return df

####################################### returns the column as a DataFrame ###########################################################


def mean_impute(df, column):
    return df[column].fillna(df[column].mean()).round(decimals=0)

###################################### returns the column as a DataFrame ###########################################################


def replace(df, column, markers, target):
    if type(markers) is not list:
        markers = [markers]
    count = df[column].isin(markers).sum()
    X = df[column].replace(markers, target)
    return X, count


################################################ spits out numpy array ##########################################################

def scale(df, method='minmax'):
    if method == 'minmax':
        scaler = StandardScaler()  # MinMaxScaler()
    else:
        pass
    return scaler.fit_transform(df)
