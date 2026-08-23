import numpy as np
import pandas as pd
import torch
import random
from model import *
from pycox.evaluation import EvalSurv
import torch.nn as nn
# from fastai.tabular.all import *
# from fastcore.utils import *
import torch
import numpy as np
import os
import builtins
import traceback
from SurvivalEVAL.Evaluator import SurvivalEvaluator


#################################### Initialize Random Seeds ############################################################

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#################################### Early Stopping Criteria ############################################################


class EarlyStopping:
    """Stop training if validation C-index does not improve for `patience` rounds."""

    def __init__(self, patience=25, verbose=False, delta=0.0, directory=None, checkpoint_pth="checkpoint.pt"):
        self.patience = int(patience)
        self.verbose = verbose
        self.directory = directory
        self.delta = float(delta)
        self.checkpoint_pth = checkpoint_pth

        self.counter = 0
        self.best_val_cindex = float("-inf")
        self.early_stop = False

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
                print(f"[EarlyStopping] Skipping invalid val_c_index = {val_c_index}")
            return False
        
        # if val_c_index is None or not np.isfinite(val_c_index):
        #     self.counter += 1
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        #     return False

        improved = val_c_index > self.best_val_cindex + self.delta

        if improved:
            if self.verbose:
                print(
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
        # if self.verbose:
        #     print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True

        return False

    def save_checkpoint(self, param):
        if param is not None:
            os.makedirs(self.directory, exist_ok=True)
            torch.save(param, os.path.join(self.directory, self.checkpoint_pth))


#################################### Tensor Dataloader ############################################################
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
         
        
        for t in tensors: 
            print(t.shape[0]) 
             
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


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
        client_name = get_client_name_for_federated_datasets(
            dataset, client_id)
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

#################################### Check whether there is any comparable paris ###############################################


def has_comparable_pairs(durations, events):
    """
    Attributes:
    ----------
        durations: Observed times array
        events: Event status array
    Returns
    -------
        Boolean flag whether there is comparable paris
    """

    durations = np.asarray(durations).reshape(-1)
    events = np.asarray(events).reshape(-1)

    n = len(durations)
    comparable = 0

    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            if durations[i] < durations[j]:
                comparable += 1

    return comparable > 0

#################################### Compute Concordance Score from validation data ###############################################

def Concordance(model, x, durations, events, evaltime, device="cpu"):
    """
    Attributes:
    ----------
        model: Model
        x: Covariates
        durations: Observed times array
        events: Event status array
        evaltime: Prespecified evaluation times

    Returns
    -------
        Time-dependent C-index (or np.nan if not computable)
    """

    model = model.to(device)
    model.eval()
    x = x.to(device)

    durations = np.asarray(durations).reshape(-1)
    events = np.asarray(events).reshape(-1)
    evaltime = np.asarray(evaltime)

    # --- guards: not enough data / no events / no comparable pairs ---
    if len(durations) < 2:
        return np.nan

    n_events = np.sum(events == 1)
    if n_events == 0:
        return np.nan

    if not has_comparable_pairs(durations, events):
        return np.nan

    # --- model prediction ---
    with torch.no_grad():
        out = model(x)
    out_np = out.detach().cpu().numpy()                 # raw model output

    # Remove singleton event dim: (n_samples, 1, n_evaltime) -> (n_samples, n_evaltime)
    if out_np.ndim == 3 and out_np.shape[1] == 1:
        out_np = out_np.squeeze(axis=1)

    if out_np.shape[0] != len(durations):
        raise ValueError(
            f"Sample mismatch: out_np has {out_np.shape[0]} rows, "
            f"but durations has length {len(durations)}"
        )
    if out_np.shape[1] != len(evaltime):
        raise ValueError(
            f"Time mismatch: out_np has {out_np.shape[1]} columns, "
            f"but evaltime has length {len(evaltime)}"
        )

    # ---- CRITICAL: model predicts F(t) (event prob, increasing in t).
    # EvalSurv needs S(t) (survival, decreasing in t). Invert.
    # GATE THIS on the diagnostic: if model(x)[0] DECREASES across evaltimes,
    # it is already S(t) and you must NOT invert (set surv_np = out_np).
    surv_np = 1.0 - out_np                              # F(t) -> S(t)

    y_pred = pd.DataFrame(surv_np.T, index=evaltime)    # (n_evaltime, n_samples)
    ev = EvalSurv(y_pred, durations, events, censor_surv="km")

    try:
        cindex = float(ev.concordance_td())
        if not np.isfinite(cindex):
            print("[C-index WARNING] NaN/Inf returned.")
            return np.nan
        return cindex
    except ZeroDivisionError:
        print("[C-index WARNING] Zero comparable pairs in validation set.")
        return np.nan

########################################## Evaluation using Test Data ################################################################
def Evaluation(model, train_time, train_status, X_test, test_time, test_status, evaltime, logger, device="cpu"):
    """
    Returns:
        cindex, brier, nbll, p_value, weighted_MAE_PO
    If any metric fails, returns np.nan for that metric.
    """

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    model = model.to(device)
    model.eval()

    ev = None

    # --- Build survival DataFrames ---
    try:
        with torch.no_grad():
            sp_test = model(X_test_tensor)
            
        out_array = sp_test.cpu().detach().numpy()   # raw model output

        # Remove singleton event dimension: (n_samples, 1, n_evaltime) -> (n_samples, n_evaltime)
        if out_array.ndim == 3 and out_array.shape[1] == 1:
            out_array = out_array.squeeze(axis=1)

        # Sanity: model emits one value per (sample, evaltime)
        if out_array.shape[0] != len(test_time):
            raise ValueError(f"Sample mismatch: out_array has {out_array.shape[0]} rows, but test_time has length {len(test_time)}")
        if out_array.shape[1] != len(evaltime):
            raise ValueError(f"Time mismatch: out_array has {out_array.shape[1]} columns, but evaltime has length {len(evaltime)}")

        # ---- CRITICAL: model predicts F(t) (event prob, increasing in t).
        # EvalSurv needs S(t) (survival, decreasing in t). Invert.
        # GATE THIS on the diagnostic: if model(x)[0] DECREASES across evaltimes,
        # it is already S(t) and you must NOT invert (set surv_array = out_array).
        surv_array = 1.0 - out_array                 # F(t) -> S(t),  (n_samples, n_evaltime)

        # pycox EvalSurv: expects (n_evaltime, n_samples), evaltime as the row index
        surv_for_evsurv = pd.DataFrame(surv_array.T, index=evaltime)

        # SurvivalEvaluator (survivaleval): expects (n_samples, n_evaltime), evaltime as columns
        surv_for_eval = pd.DataFrame(surv_array, columns=np.asarray(evaltime, dtype=float))

        ev = EvalSurv(surv_for_evsurv, test_time, test_status, censor_surv='km')

    except Exception as e:
        logger.error(f"[Evaluation] Failed to build survival frames: {e}")
        traceback.print_exc()
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # --- C-index ---
    try:
        cindex = float(ev.concordance_td())
        if not np.isfinite(cindex):
            logger.info("[C-index ERROR] concordance_td returned NaN/Inf")
            cindex = np.nan
    except Exception as e:
        logger.error(f"[C-index ERROR] {type(e).__name__}: {e}")
        cindex = np.nan

    # --- Integrated Brier Score ---
    try:
        brier = float(ev.integrated_brier_score(evaltime))
        if not np.isfinite(brier):
            logger.info("[Brier ERROR] integrated_brier_score returned NaN/Inf")
            brier = np.nan
    except Exception as e:
        logger.error(f"[Brier ERROR] {type(e).__name__}: {e}")
        brier = np.nan

    # --- Integrated NBLL ---
    try:
        nbll = float(ev.integrated_nbll(evaltime))
        if not np.isfinite(nbll):
            logger.error("[NBLL ERROR] integrated_nbll returned NaN/Inf")
            nbll = np.nan
    except Exception as e:
        logger.error(f"[NBLL ERROR] {type(e).__name__}: {e}")
        nbll = np.nan

    # --- SurvivalEvaluator (D-Calibration & Weighted MAE) ---
    test_time_arr    = np.asarray(test_time,    dtype=float).reshape(-1)
    test_status_arr  = np.asarray(test_status,  dtype=int).reshape(-1)
    train_time_arr   = np.asarray(train_time,   dtype=float).reshape(-1)
    train_status_arr = np.asarray(train_status, dtype=int).reshape(-1)

    logger.info(
        f"[SurvivalEvaluator shapes] "
        f"surv={surv_for_eval.shape}, "          # expect (n_test_samples, n_evaltime)
        f"evaltime={np.asarray(evaltime).shape}, "
        f"test_time={test_time_arr.shape}, "
        f"train_time={train_time_arr.shape}"
    )

    p_value = np.nan
    weighted_MAE_PO = np.nan

    try:
        evaluator = SurvivalEvaluator(
            surv_for_eval,                         # (n_samples, n_evaltime)
            np.asarray(evaltime, dtype=float),     # (n_evaltime,)
            test_time_arr,
            test_status_arr,
            train_time_arr,
            train_status_arr,
        )

        try:
            p_val_result, _ = evaluator.d_calibration()
            p_value = float(p_val_result) if np.isfinite(p_val_result) else np.nan
        except Exception as e:
            logger.error(f"[D-Calibration WARNING] {e}")

        try:
            mae_result = evaluator.mae(method='Pseudo_obs', weighted=True)
            weighted_MAE_PO = float(mae_result) if np.isfinite(mae_result) else np.nan
        except Exception as e:
            logger.error(f"[Weighted MAE WARNING] {e}")

    except Exception as e:
        logger.error(f"[SurvivalEvaluator] Failed to initialize: {e}")
        traceback.print_exc()

    logger.info( f"[Evaluation] cindex={cindex}, brier={brier}, nbll={nbll}, p_value={p_value}, weighted_MAE_PO={weighted_MAE_PO}")
    return cindex, brier, nbll, p_value, weighted_MAE_PO

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

########################################## Parameters from trained model ################################################################


def get_trainable_parameters(net):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # logger.info("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    N = 0
    for params in paramlist:
        N += params.numel()
    X = torch.empty(N, dtype=torch.float64)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset +
                numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset += numel
    return X

########################################## Parameters to PyTorch Model ################################################################


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

########################################## Store the PyTorch Model in storage ############################################################


def save_model(model, model_index, args):
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

########################################## Retrieve the PyTorch Model from storage ########################################################


def load_model(model, model_index, device="cpu"):
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model


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

########################################## Pseudo Value based Loss Function ##############################################################


def pseudo_loss(output, target):
    loss = torch.mean((target-output)**2)
    return loss
