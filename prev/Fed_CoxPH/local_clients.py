import numpy as np
import copy
from math import *
# from torch.utils.tensorboard import SummaryWriter
import torchtuples as tt
from utils import *
import traceback
from pycox.models import CoxPH
from utils import negative_llh

####################################  Training Centralized CoxPH Model  ##################################################################### 
def train_net_all(net_id, model, X_train, y_train, val_data, epochs, lr, optimizer,  train_time, train_status,  X_val, val_time, val_status, X_test, test_time, test_status, evaltime, args, device): 
    print('Training network %s' % str(net_id))
    print('Training network %s' % str(net_id))

    # pick optimizer properly
    if optimizer.lower() == "adam":
        opt = tt.optim.Adam(lr=args.lr, weight_decay=args.reg)
    elif optimizer.lower() == "sgd":
        opt = tt.optim.SGD(lr=args.lr, weight_decay=args.reg)

    ckpt_path = os.path.join( args.modeldir,  f"fed_coxPh_{args.dataset}_seed{args.init_seed}") 
    callbacks = [tt.callbacks.EarlyStopping(patience=args.patience, file_path=ckpt_path, load_best=True, rm_file=True)]
    verbose = False

    net = CoxPH(model, opt)
    log = None 
    if np.sum(val_status == 1) == 0: 
          log = net.fit( X_train, y_train, batch_size=args.batch_size, epochs=epochs, callbacks=None, verbose=verbose)
    else:  
        log = net.fit(X_train, y_train, batch_size=args.batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose, val_data=val_data, val_batch_size=args.batch_size)
    
    print(f"the training for centralized partition: {log}") 
    
    model_name="Fed_CoxPH"
    os.makedirs(args.modeldir, exist_ok=True)
    model_path = os.path.join(args.modeldir, f"{model_name}_{args.run_id}_{net_id}.pt")
    torch.save(net.net.state_dict(), model_path)

    # Compute baseline hazard
    test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation(net, train_time, train_status, X_train, y_train, X_test, test_time, test_status, evaltime, device)
    return test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO 

###################################################  Training local CoxPH Model  ################################################################  
def train_net(net_id, model, X_train, y_train, val_data, epochs, lr, optimizer,   train_time, train_status, X_val, val_time, val_status, X_test, test_time, test_status, evaltime, comm_round, args, device):
    print('Training network %s' % str(net_id))
    verbose = False
    
    if np.sum(train_status == 1) == 0:
        return None

    # Build a model
    opt = tt.optim.Adam(lr=args.lr, weight_decay=args.reg)
    net = CoxPH(model, opt) 
    
    ckpt_path = os.path.join( args.modeldir,  f"fed_coxPh_{args.dataset}_seed{args.init_seed}") 
    callbacks = [tt.callbacks.EarlyStopping(patience=args.patience, file_path=ckpt_path, load_best=True, rm_file=True)]
    
    log = None 
    validation_cindex = 0.0
    
    print(f"At client {net_id} and communication round is {comm_round}, using global model, validation C-Index is {validation_cindex}")  
    if np.sum(val_status == 1) == 0: 
        log = net.fit( X_train, y_train, batch_size=args.batch_size, epochs=epochs, callbacks=None, verbose=verbose)
    else:  
        log = net.fit(X_train, y_train, batch_size=args.batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose, val_data=val_data, val_batch_size=args.batch_size)
    
    print(f"the training for centralized partition: {log}")  
    os.makedirs(args.modeldir, exist_ok=True)
    model_path = os.path.join( args.modeldir, f"{args.model}_{args.run_id}_{net_id}.pt")
    torch.save(net.net.state_dict(), model_path)
      
    return net.net
  
    # validation_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO

###################################### Aggregate Local Models ##############################################################################
def local_train_net( nets, client_name_array,  selected,   args,   X_train_local_std, train_time_local, train_status_local, X_val_local_std, val_time_local,  val_status_local,
    X_test_local_std,  test_time_local, test_status_local, evaltime,  tr_data_len, comm_round,  device):
    """
    Train local client models and aggregate validation/test metrics safely.

    Returns
    -------
    global_c_index : float
        Sample-weighted global validation C-index over valid clients.
    avg_cindex1 : float
        Sample-weighted global test C-index over valid clients.
    avg_brier1 : float
        Sample-weighted global test Brier score over valid clients.
    avg_nbll1 : float
        Sample-weighted global test NBLL over valid clients.
    """ 

    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        X_train = X_train_local_std[net_id]
        train_time = train_time_local[net_id]
        train_status = train_status_local[net_id]

        X_val = X_val_local_std[net_id]
        val_time = val_time_local[net_id]
        val_status = val_status_local[net_id]

        X_test = X_test_local_std[net_id]
        test_time_arr = test_time_local[net_id]
        test_status_arr = test_status_local[net_id]

        df_train1 = pd.concat([pd.DataFrame(train_time), pd.DataFrame( train_status), pd.DataFrame(X_train)],axis=1,)
        df_val1 = pd.concat( [pd.DataFrame(val_time), pd.DataFrame(val_status), pd.DataFrame(X_val)], axis=1,)
        df_test1 = pd.concat( [pd.DataFrame(test_time_arr), pd.DataFrame(test_status_arr), pd.DataFrame(X_test)], axis=1,)

        def get_target(df): 
              return (df.iloc[:, 0].values, df.iloc[:, 1].values)
            
        y_train1 = get_target(df_train1)
        y_val1 = get_target(df_val1)
        
        test_time, test_status = get_target(df_test1)
        val1 = (X_val, y_val1)
        n_epoch = args.epochs

        # try:
        nets[net_id] = train_net( client_name_array[net_id], net, X_train,  y_train1,  val1, n_epoch,  args.lr,  args.optimizer,  train_time, train_status,  X_val, val_time,  val_status,  X_test, test_time, test_status,  evaltime, comm_round, args, args.device) 
         
    return nets
 

############################ evaluate global model on clients #########################################################
def evaluate_global_model_on_clients( global_model, selected,   X_train_local_std, train_time_local, train_status_local,  X_val_local_std, val_time_local, val_status_local,
    X_test_local_std, test_time_local, test_status_local, pseudo_evaltime, args, device="cpu",):
 
    global_model.to(device)
    global_model.eval()

    # ── accumulators ────────────────────────────────────────────────────────
    total_val_samples  = 0
    weighted_val_cindex = 0.0

    test_acc = {
        "cindex": {"weighted_sum": 0.0, "n": 0},
        "brier":  {"weighted_sum": 0.0, "n": 0},
        "nbll":   {"weighted_sum": 0.0, "n": 0},
        "MAE_PO": {"weighted_sum": 0.0, "n": 0},
    }
    
    test_p_values = [] 
    val_dropped  = []
    test_dropped = {k: [] for k in test_acc}

    # FIX 3: get_target defined once, outside the loop
    def get_target(df):
        return (df.iloc[:, 0].values, df.iloc[:, 1].values)

    def _accumulate(metric_key, value, weight, client_id):
        if value is not None and np.isfinite(value):
            test_acc[metric_key]["weighted_sum"] += float(value) * weight
            test_acc[metric_key]["n"] += weight
        else:
            test_dropped[metric_key].append(client_id)

    # ── per-client loop ──────────────────────────────────────────────────────
    for net_id in selected:
        X_train      = X_train_local_std[net_id]
        train_time   = np.asarray(train_time_local[net_id])
        train_status = np.asarray(train_status_local[net_id])

        X_val      = X_val_local_std[net_id]
        val_time   = np.asarray(val_time_local[net_id])
        val_status = np.asarray(val_status_local[net_id])

        X_test      = X_test_local_std[net_id]
        test_time   = np.asarray(test_time_local[net_id])
        test_status = np.asarray(test_status_local[net_id])

        n_val  = len(val_time)
        n_test = len(test_time)

        # FIX 4: y_train built once and reused for BOTH calls
        y_train = (train_time, train_status)

        # FIX 5: one CoxPH wrapper per client iteration (shared underlying model)
        opt = tt.optim.Adam(lr=args.lr, weight_decay=args.reg)
        net = CoxPH(global_model, opt)

        ############################### validation ####################################################
        try:
            val_cindex, _, _, _, _ = Evaluation( net, train_time, train_status, X_train, y_train, X_val, val_time, val_status,  pseudo_evaltime,  device, )
        except Exception as e:
            print(f"[Client {net_id}] Validation failed: {type(e).__name__}: {e}")
            val_cindex = np.nan
 
        try:
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation(net,train_time,train_status,X_train, y_train, X_test, test_time, test_status,pseudo_evaltime, device) 
        except Exception as e:
            print(f"[Client {net_id}] Test evaluation failed: {type(e).__name__}: {e}")
            test_cindex = test_brier = test_nbll = test_p_value = test_weighted_MAE_PO = np.nan

        print(
            f"[Client {net_id}] "
            f"val_n={n_val}, val_events={int(np.sum(val_status))}, "
            f"val_cindex={val_cindex}, "
            f"test_n={n_test}, test_events={int(np.sum(test_status))}, "
            f"test_cindex={test_cindex}"
        )

        # ── accumulate validation ────────────────────────────────────────────
        if val_cindex is not None and np.isfinite(val_cindex):
            weighted_val_cindex += float(val_cindex)  * len(val_time) 
            total_val_samples   +=  len(val_time)
        else:
            val_dropped.append(net_id)

        ########## accumulate test #################################
        _accumulate("cindex", test_cindex,           n_test, net_id)
        _accumulate("brier",  test_brier,            n_test, net_id)
        _accumulate("nbll",   test_nbll,             n_test, net_id)
        _accumulate("MAE_PO", test_weighted_MAE_PO,  n_test, net_id)

        if test_p_value is not None and np.isfinite(test_p_value):
              test_p_values.append(test_p_value)

    # ── finalise validation ──────────────────────────────────────────────────
    if total_val_samples > 0:
        global_val_cindex = weighted_val_cindex / total_val_samples
    else:
        print("[Warning] No valid validation C-index from any client.")
        global_val_cindex = np.nan

    # ── finalise test ────────────────────────────────────────────────────────
    def _mean(key):
        acc = test_acc[key]
        return acc["weighted_sum"] / acc["n"] if acc["n"] > 0 else np.nan

    
    avg_test_cindex  = _mean("cindex")
    avg_test_brier   = _mean("brier")
    avg_test_nbll    = _mean("nbll")
    avg_test_MAE_PO  = _mean("MAE_PO")
    avg_test_p_value = np.nanmedian(test_p_values) if test_p_values else np.nan

    # ── auditability log ─────────────────────────────────────────────────────
    n_selected = len(list(selected))
    if val_dropped:
        print(f"[Aggregation] Val: dropped {len(val_dropped)}/{n_selected} clients (NaN): {val_dropped}")
    for k, dropped in test_dropped.items():
        if dropped:
            print(f"[Aggregation] Test {k}: dropped {len(dropped)}/{n_selected} clients (NaN): {dropped}")
 
    return (
        global_val_cindex,
        avg_test_cindex,
        avg_test_brier,
        avg_test_nbll,
        avg_test_p_value,
        avg_test_MAE_PO,
    )
 
    