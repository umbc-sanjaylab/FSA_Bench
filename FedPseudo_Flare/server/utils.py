import numpy as np
import os
import torch
import pandas as pd


#####################################  Evaluation Time Computation  ########################################

def Pseudo_Evaltime(global_full_table, pseudo_interval):
    """
    Arguments:
        global_full_table: The full table containing global survival data.
        pseudo_interval: Interval for percentile of time at which pseudo values will be calculated.

    Return:
        A vector of the evaluation time points at which pseudo values will be calculated.
    """
    global_unique_times = np.unique(global_full_table[:, 0])
    PseudoEvalTime = np.percentile(global_unique_times, np.arange(10, 99, pseudo_interval))
    return PseudoEvalTime

#####################################  Global Survival Curve Probabilities  ########################################

# global survival function calculation
def global_survival_probability(global_partial_matrix, is_DP):
    global_full_matrix = np.array(global_partial_matrix.copy())
    r_last = global_full_matrix[0, 1]
    d_last = global_full_matrix[0, 2]
    c_last = global_full_matrix[0, 3]

    for j in range(1, global_full_matrix.shape[0]):
        r = r_last - d_last - c_last
        d = global_full_matrix[j, 2]
        c = global_full_matrix[j, 3]
        if r < 0:
            global_full_matrix[j, 1] = 0
        else:
            global_full_matrix[j, 1] = r
        if d < 0:
            global_full_matrix[j, 2] = 0
        if c < 0:
            global_full_matrix[j, 3] = 0
        r_last = r
        d_last = d
        c_last = c

    if is_DP:
        indices = np.where(global_full_matrix[:, 1] <= global_full_matrix[:, 2])[0]
        if indices.size == 0:
            global_full_matrix = global_full_matrix
        else:
            idx = indices[0]
            global_full_matrix[idx:, 1] = global_full_matrix[idx-1, 1]
            global_full_matrix[idx:, 2] = 0

        global_full_matrix[global_full_matrix < 0] = 0

    # Calculate global survival probabilities using product rule formula
    global_surv = np.cumprod(1-(global_full_matrix[:, 2]/(global_full_matrix[:, 1]+0.000000001)))
    return global_full_matrix, global_surv
  
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
                print(
                    f"[EarlyStopping] Skipping invalid val_c_index = {val_c_index}")
            return False

        improved = val_c_index > self.best_val_cindex + self.delta

        if improved:
            if self.verbose:
                print(f"Validation C-index improved ({self.best_val_cindex:.6f} -> {val_c_index:.6f}). Saving checkpoint...")

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
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True

        return False

    def save_checkpoint(self, param):
        if param is not None:
            os.makedirs(self.directory, exist_ok=True)
            torch.save(param, os.path.join(self.directory, self.checkpoint_pth))
          
#################################### Check whether metric is valid or not ############################################################  
def _is_valid(value): 
        return value is not None and np.isscalar(value) and np.isfinite(float(value))  
      

#################################### Store result in CSV File #########################################################################  

def save_results_to_csv(final_val_cidx, final_test_cidx, final_test_brier, final_test_nbll, final_test_p_value, final_test_MAE_PO, final_round, args):
    csv_path = os.path.join(args.logdir, "results.csv") 
    
    row = {
        "comm_round": final_round,
        "seed": int(args.seed),
        "fold_id": int(args.run_id),
        "dataset": args.dataset,
        "partition": args.partition,
        "model": args.model,
        "val_c_idx": round(float(final_val_cidx), 4),
        "c_idx": round(float(final_test_cidx), 4),  
        "brier": round(float(final_test_brier), 4),  
        "nbll": round(float(final_test_nbll), 4),  
        "p_value": round(float(final_test_p_value), 4),  
        "MAE_PO": round(float(final_test_MAE_PO), 4),  
        "lr": round(float(args.lr), 5),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs)
    }

    df = pd.DataFrame([row])
    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
            
             