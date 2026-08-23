import numpy as np
from pycox.models import DeepHit
from math import *
import torchtuples as tt
from utils import *
from model import *
import numpy as np
import torchtuples as tt
import traceback

################  Compute c-index for model training  #################################
def compute_deephit_cindex(model, X,  durations,  events, evaltime):
    """
    Compute the time-dependent C-index using the same procedure
    for validation and test datasets.
    """

    # DeepHit cumulative incidence function
    cif = model.predict_cif(X)
    eval_times_arr = np.asarray(evaltime, dtype=float)
    duration_index = np.asarray(model.duration_index, dtype=float)

    # Retain evaluation times covered by the model
    valid_eval_times = eval_times_arr[eval_times_arr >= duration_index[0]]  
    pos = np.searchsorted(duration_index,  valid_eval_times,  side="right") - 1
    pos = np.clip(pos,  0,  len(duration_index) - 1)

    # Remove duplicate positions
    pos = np.unique(pos)
    snapped_eval_times = duration_index[pos]
    cif_event_one = pd.DataFrame(np.asarray(cif[0]), index=duration_index)

    cif_for_eval_times = cif_event_one.iloc[pos].copy()
    cif_for_eval_times.index = snapped_eval_times

    # S(t) = 1 - CIF(t)
    surv_for_eval_times = 1.0 - cif_for_eval_times 
    evaluator = EvalSurv(surv_for_eval_times,  np.asarray(durations),  np.asarray(events), censor_surv="km")

    # Same call used in your Evaluation function
    cindex = float(evaluator.concordance_td()) 
    return cindex


################  Training DeepHit Model with "Centralized" framework in Local Client  #################################
def train_net_all(alpha, sigma, net_id, model, X_train, y_train, val_data, epochs, lr, optimizer, train_time, train_status,  X_val, val_time, val_status, X_test, test_time, test_status, evaltime, labtrans, args, device):
    logger.info('Training network %s' % str(net_id))
    print('Training network %s' % str(net_id)) 

    best_weights_path = os.path.join( args.modeldir,   f"DeepHit_{args.run_id}_{net_id}_best.pt")    
    # callbacks = [tt.callbacks.EarlyStopping(patience=args.patience), tt.callbacks.BestWeights(file_path=best_weights_path)]
    
    
    def get_validation_cindex():
      cindex = compute_deephit_cindex(model=net, X=X_val, durations=val_time,  events=val_status, evaltime=evaltime)
      logger.info( "Network %s validation C-index: %.6f", net_id, cindex)
      return cindex
    
    callbacks = [
      tt.callbacks.EarlyStopping(
        get_score=get_validation_cindex,
        minimize=False,
        patience=args.patience,
        file_path=best_weights_path,
        checkpoint_model=True,
        load_best=True,
        rm_file=False
      )
    ]
    
    verbose = False

    if optimizer.lower() == "adam":
        opt = tt.optim.Adam(lr=lr, weight_decay=args.reg)
    elif optimizer.lower() == "sgd":
        opt = tt.optim.SGD(lr=lr, weight_decay=args.reg)

    net = DeepHit(model.to(device), opt, alpha=alpha, sigma=sigma, duration_index=labtrans.cuts)
    log = net.fit(X_train, y_train, args.batch_size, epochs, callbacks, verbose, val_data=val_data, val_batch_size=args.batch_size) 
    
    ckpt_path = os.path.join(args.modeldir, f"DeepHit_{args.run_id}_0.pt")
    os.makedirs(args.modeldir, exist_ok=True)  
    torch.save({"model_state_dict": net.net.state_dict(),
                  "duration_index": labtrans.cuts,
                  "num_durations": args.num_duration,
                  "alpha": alpha, "sigma": sigma,
                  "input_size": X_train.shape[1],
                  "dropout_p": args.dropout_p,
                  }, ckpt_path)

    test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = Evaluation( net, train_time, train_status, X_test, test_time, test_status, evaltime, labtrans, device)
    return test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO


################  Training DeepPseudo Model with "FedAvg" framework in Local Client  #################################

def train_net(alpha, sigma, net_id, model, X_train, y_train, val_data, epochs, lr, optimizer,  train_time, train_status, X_val, val_time, val_status, X_test, test_time, test_status, evaltime, labtrans, comm_round, args, device):  

    logger.info('Training network %s' % str(net_id)) 
    ckpt_path = os.path.join( args.modeldir,  f"deephit_{args.dataset}_seed{args.init_seed}")
    
    callbacks = [tt.callbacks.EarlyStopping(patience=args.patience, file_path=ckpt_path, load_best=True, rm_file=True)]
    model_path = os.path.join( args.modeldir, f"{args.model}_{args.run_id}_{net_id}.pt")

    if optimizer.lower() == "adam":
        opt = tt.optim.Adam(lr=lr, weight_decay=args.reg)
    elif optimizer.lower() == "sgd":
        opt = tt.optim.SGD(lr=lr, weight_decay=args.reg)

    net = DeepHit(model.to(device), opt, alpha=alpha, sigma=sigma, duration_index=labtrans.cuts)  
    verbose = False
    if np.sum(val_status == 1) == 0: 
        log = net.fit( X_train, y_train, batch_size=args.batch_size, epochs=epochs, callbacks=None, verbose=verbose)
    else:  
        log = net.fit(X_train, y_train, batch_size=args.batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose, val_data=val_data, val_batch_size=args.batch_size)
         
    print(log) 
    os.makedirs(args.modeldir, exist_ok=True)
    torch.save({"model_state_dict": net.net.state_dict(),
                  "duration_index": labtrans.cuts,
                  "num_durations": args.num_duration,
                  "alpha": alpha, "sigma": sigma,
                  "input_size": X_train.shape[1],
                  "dropout_p": args.dropout_p,
                  }, model_path )
    
    return net.net      


def local_train_net(alpha, sigma, client_name_array, nets, selected, args, X_train_local_std, train_time_local, train_status_local, X_val_local_std, val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local, evaltime, labtrans, train_data_length, comm_round, device): 

    for net_id, net in nets.items():
        client_name = client_name_array[net_id]
        if net_id not in selected:
            continue

        # move the model to cuda device:
        net.to(device)

        X_train = X_train_local_std[net_id]
        train_time = train_time_local[net_id]
        train_status = train_status_local[net_id]

        X_val = X_val_local_std[net_id]
        val_time = val_time_local[net_id]
        val_status = val_status_local[net_id]

        X_test = X_test_local_std[net_id]
        test_status = test_status_local[net_id]
        test_time = test_time_local[net_id]

        train_time = np.array(train_time, dtype='float32')
        train_status = np.array(train_status, dtype='int64')

        val_time = np.array(val_time, dtype='float32')
        val_status = np.array(val_status, dtype='int64') 
        
        y_train = labtrans.transform(train_time, train_status.astype(int))
        if np.sum(val_status == 1) > 0:
           y_val = labtrans.transform(val_time, val_status.astype(int))

        # Create the validation set
        val_data = (X_val, y_val)
         
        deephit_net  = train_net(alpha, sigma, net_id, net, X_train, y_train, val_data, args.epochs, args.lr,  args.optimizer,  train_time, train_status, X_val, val_time, val_status, X_test, test_time, test_status, evaltime, labtrans, comm_round, args, device)
        
        nets[net_id] = deephit_net
    
    return nets
 

############################################### Evaluate Global Model ##################################################################### 

def evaluate_global_model_on_clients( net, alpha, sigma, selected, X_val_local_std, val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local, train_time_local, train_status_local, pseudo_evaltime, labtrans, args, device):  
    
    opt = tt.optim.Adam(lr=args.lr, weight_decay=args.reg)
    global_model = DeepHit(net.to(device), opt, alpha=alpha, sigma=sigma, duration_index=labtrans.cuts)  

    # --- validation accumulators (sample-weighted) ---
    total_val_samples = 0
    weighted_val_cindex = 0.0

    # --- test accumulators (sample-weighted, per metric) --- 
    test_acc = {
        "cindex": {"weighted_sum": 0.0, "n": 0},
        "brier":  {"weighted_sum": 0.0, "n": 0},
        
        "nbll":   {"weighted_sum": 0.0, "n": 0},
        "MAE_PO": {"weighted_sum": 0.0, "n": 0},
    }
    
    # p-values are NOT averaged numerically; collect per-client for reporting
    test_p_values = [] 

    # --- bookkeeping for auditability (IID vs non-IID drop comparison) ---
    val_dropped = []
    test_dropped = {k: [] for k in test_acc}

    def _accumulate(metric_key, value, weight, client_id):
        """Add a per-client metric into its weighted accumulator if finite."""
        if value is not None and np.isfinite(value):
            test_acc[metric_key]["weighted_sum"] += float(value) * weight
            test_acc[metric_key]["n"] += weight
        else:
            test_dropped[metric_key].append(client_id)
    
    
    for net_id in selected:
        X_val = X_val_local_std[net_id]
        val_time = np.asarray(val_time_local[net_id])
        val_status = np.asarray(val_status_local[net_id])

        X_test = X_test_local_std[net_id]
        test_time = np.asarray(test_time_local[net_id])
        test_status = np.asarray(test_status_local[net_id])

        train_time = np.asarray(train_time_local[net_id])
        train_status = np.asarray(train_status_local[net_id])

        n_val = len(val_time)
        n_test = len(test_time)
        
        try:
            val_cindex, _, _, _, _ = Evaluation(global_model,  train_time, train_status, X_val, val_time, val_status, pseudo_evaltime, labtrans, device) 
        except Exception as e:
            print("Exception in train_net_all")
            traceback.print_exc()
            val_cindex = np.nan
        
        try:
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO  = Evaluation(global_model, train_time, train_status, X_test, test_time, test_status, pseudo_evaltime, labtrans, device) 
        except Exception as e:
            print("Exception in train_net_all")
            traceback.print_exc()
            test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO  = np.nan, np.nan, np.nan, np.nan, np.nan 

        print( f"[Client {net_id}] val_n={n_val}, val_events={int(np.sum(val_status))}, " f"val_cindex={val_cindex}, " f"test_n={n_test}, test_events={int(np.sum(test_status))}, " f"test_cindex={test_cindex}" ) 
    
        # --- validation: sample-weighted ---
        if val_cindex is not None and np.isfinite(val_cindex):
            weighted_val_cindex += float(val_cindex) * n_val
            total_val_samples += n_val 
        else:
            val_dropped.append(net_id) 
        
        # --- test: sample-weighted, per metric, weight = n_test ---
        _accumulate("cindex", test_cindex, n_test, net_id)
        _accumulate("brier",  test_brier, n_test, net_id)
        _accumulate("nbll",   test_nbll, n_test, net_id)
        _accumulate("MAE_PO", test_weighted_MAE_PO, n_test, net_id)

        # p-value: collected, not numerically averaged
        if test_p_value is not None and np.isfinite(test_p_value):
            test_p_values.append(test_p_value)

    ######### finalize validation ###################################
    if total_val_samples > 0:
        global_val_cindex = weighted_val_cindex / total_val_samples
    else:
        print("[Warning] No valid validation C-index from any client.")
        global_val_cindex = np.nan

    ############### finalize test (sample-weighted means) ############
    def _weighted_mean(metric_key):
        acc = test_acc[metric_key]
        return acc["weighted_sum"] / acc["n"] if acc["n"] > 0 else np.nan

    avg_test_cindex = _weighted_mean("cindex") 
    avg_test_brier = _weighted_mean("brier")
    avg_test_nbll = _weighted_mean("nbll")
    avg_test_MAE_PO = _weighted_mean("MAE_PO")

    # p-value: median is a placeholder summary; report the list separately if needed
    avg_test_p_value = np.nanmedian(test_p_values) if len(test_p_values) > 0 else np.nan

    # --- auditability: log how many / which clients were dropped ---
    n_selected = len(list(selected))
    if val_dropped:
        print(f"[Aggregation] Val: dropped {len(val_dropped)}/{n_selected} clients (NaN): {val_dropped}")
        
    for k, dropped in test_dropped.items():
        if dropped:
            print(f"[Aggregation] Test {k}: dropped {len(dropped)}/{n_selected} clients (NaN): {dropped}")

    return global_val_cindex, avg_test_cindex, avg_test_brier, avg_test_nbll, avg_test_p_value, avg_test_MAE_PO
       

############################################################################################################################################
