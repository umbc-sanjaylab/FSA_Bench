import numpy as np
import pandas as pd 

################################################### Function For initializing networks ##################################################

def init_net(args):  
    if args.model == "Kaplan_Meier":
        net = Kaplan_Meier()
    else:
        print("not supported yet")
        exit(1)
    return net    
  
############################# Kaplan Meier Model ############################################################################################  

class Kaplan_Meier:
    def __init__(self):
        self.timeline_ = None
        self.survival_ = None

    def fit(self, time, event):
        local_matrix = local_partial_matrix(time, event)
        return self.fit_from_partial_matrix(local_matrix)

    def fit_from_partial_matrix(self, partial_matrix):
        full_matrix, survival = global_survival_probability(partial_matrix)

        self.timeline_ = full_matrix[:, 0]
        self.survival_ = survival

        return self

    def predict_survival(self):
        if self.survival_ is None:
            raise ValueError("Kaplan_Meier must be fitted first.")

        return self.timeline_, self.survival_ 
      

#####################################  Global Survival Curve Probabilities  ########################################

def global_survival_probability(global_partial_matrix):
    """
    Arguments:
        global_partial_matrix: aggregated Kaplan-Meier Components for all clients

    Return:
        A partial table with KM estimator components.
    """

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

    # Calculate global survival probabilities using product rule formula
    global_surv = np.cumprod( 1-(global_full_matrix[:, 2]/(global_full_matrix[:, 1]+0.000000001)))
    return global_full_matrix, global_surv  

#################################### Compute elememts olf local matrix #####################################################
def local_partial_matrix(client_time, client_status):
    """
    Arguments:
        client_time: array-like, time of events or censoring for the client
        client_status: array-like, event status (1 for event, 0 for censoring) for the client

    Return:
        A new dataframe with time, number of patients at risk, number of events, and number of censored observations.
    """
    
    # time points and event status  
    time = np.asarray(client_time).reshape(-1)
    event = np.asarray(client_status).reshape(-1).astype(int)

    # sanity check for input arrays
    if len(time) != len(event):
        raise ValueError("Time and event arrays must have the same length.")

    if not np.isin(event, [0, 1]).all():
        raise ValueError(f"Event status must contain only 0/1, but found {np.unique(event)}.")

    # Unique event/censoring times
    timeline, inverse = np.unique(time, return_inverse=True)

    # Number of events and censored observations at each time
    di = np.bincount(inverse, weights=(event == 1).astype(int), minlength=len(timeline) ).astype(int) 
    ci = np.bincount(inverse, weights=(event == 0).astype(int), minlength=len(timeline) ).astype(int)

    # Number at risk immediately before each time point
    removed = di + ci
    nrisk = len(time) - np.concatenate(([0], np.cumsum(removed[:-1])))

    # client's Kaplan-Meier component representation  
    local_matrix = pd.DataFrame({
        "time": timeline,
        "nrisk": nrisk,
        "di": di,
        "ci": ci
    }) 
    
    # keep only initial risk count 
    local_matrix.loc[1:, "nrisk"] = 0 
    return local_matrix

###################################### Aggregate local client's Kaplan-Meier Components ########################################################

def aggregate_local_matrices(local_matrices):
    if not local_matrices:
        raise ValueError("No local Kaplan-Meier matrices were provided.")

    # Total number of subjects across all clients
    total_nrisk = sum(matrix.iloc[0]["nrisk"] for matrix in local_matrices)

    # Combine event/censor counts at common time points
    global_partial_matrix =  pd.concat(local_matrices, ignore_index=True).groupby("time", as_index=False)[["di", "ci"]].sum().sort_values("time").reset_index(drop=True) 

    # Server only needs the initial global risk count
    global_partial_matrix["nrisk"] = 0
    global_partial_matrix.loc[0, "nrisk"] = total_nrisk 
    return global_partial_matrix[["time", "nrisk", "di", "ci"]]