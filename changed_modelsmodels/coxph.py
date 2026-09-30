import torch 
from torch import nn 
import numpy as np 
from pycox.models.loss import CoxPHLoss
from pycox.models import CoxPH as PyCoxPH

############################# Build CoxPH Model ###############################################

def build_coxph(input_size, args):
    return CoxPHNetwork( in_features=input_size, out_features=1) 
    
############################# CoxPH Model Architecture ########################################

class CoxPHNetwork(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(CoxPHNetwork, self).__init__() 
        self.net =  nn.Linear(in_features, out_features, bias=False)
          
    def forward(self, x):
        return self.net(x)
       
 ############################# Train CoxPH #####################################################

def train_epoch( train_loader, model, optimizer, device):
    model.train()
    model.to(device)

    loss_fn = CoxPHLoss() 
    total_loss = 0.0
    total_events = 0

    for X, time, event in train_loader: 
        X = X.to( device=device, dtype=torch.float32) 
        time = time.to( device=device, dtype=torch.float32) 
        
        event = event.to(device=device, dtype=torch.float32)  
        num_events = int(event.sum().item())

        # Cox partial likelihood cannot be computed
        # for a batch containing no events.
        if num_events == 0:
            continue

        optimizer.zero_grad() 
        log_risk = model( X ).reshape(-1) 
        loss = loss_fn( log_risk, time, event)

        loss.backward()
        optimizer.step()

        total_loss += ( loss.item() * num_events) 
        total_events += num_events

    if total_events == 0:
        return np.nan 
    return  total_loss / total_events 
   
############################# Validation Loss #################################################

def validation_loss( val_loader, model, device="cpu"):
    model.eval()
    model.to(device)

    loss_fn = CoxPHLoss() 
    total_loss = 0.0
    total_events = 0

    with torch.no_grad(): 
        for X, time, event in val_loader: 
            X = X.to( device=device, dtype=torch.float32) 
            time = time.to( device=device, dtype=torch.float32) 
            
            event = event.to( device=device, dtype=torch.float32) 
            num_events = int( event.sum().item() )

            if num_events == 0:
                continue

            log_risk = model(X).reshape(-1) 
            loss = loss_fn( log_risk, time, event)

            total_loss +=  loss.item() * num_events 
            total_events += num_events

    if total_events == 0:
        return np.nan
    else: 
        return  total_loss / total_events 
 
############################# Fit Baseline Hazard #############################################

def fit_baseline_hazard( train_loader, model, device="cpu"):
    model.eval()
    model.to(device)

    X_list = []
    time_list = []
    event_list = []

    for X, time, event in train_loader: 
        X_list.append( X.detach().cpu().numpy()) 
        time_list.append( time.detach().cpu().numpy()) 
        event_list.append( event.detach().cpu().numpy())

    if len(X_list) == 0:
        return False

    X_train = np.concatenate( X_list, axis=0) 
    train_time = np.concatenate( time_list ) 
    train_event = np.concatenate( event_list )

    if np.sum(train_event) == 0:
        return False

    cox_model = PyCoxPH(model, device=device) 
    cox_model.compute_baseline_hazards( X_train, ( train_time, train_event )) 
    baseline_hazards =  cox_model.baseline_hazards_ 

    # Store locally on this model instance.
    # These are NOT trainable/federated parameters.
    model.baseline_times_ =  baseline_hazards.index.to_numpy(dtype=float) 
    model.baseline_cum_hazard_ =  baseline_hazards.cumsum().to_numpy(dtype=float) 
    return True
     
############################# Predict Survival ###############################################

def predict_survival( model, X, eval_times, device):
    if not hasattr( model, "baseline_times_"):
        raise RuntimeError( "Baseline hazard is not available. Call fit_baseline_hazard() first.")

    model.eval()
    model.to(device)

    X = torch.as_tensor( X, dtype=torch.float32, device=device) 
    with torch.no_grad(): 
        log_risk =  model(X).reshape(-1).cpu().numpy() 

    relative_risk = np.exp(np.clip( log_risk, -50, 50)) 
    baseline_times =   model.baseline_times_  
    
    baseline_cum_hazard = model.baseline_cum_hazard_  
    eval_times = np.asarray(eval_times, dtype=float) 
    indices = np.searchsorted( baseline_times, eval_times, side="right" ) - 1

    # This mirrors your previous Evaluation()
    # behavior of selecting the preceding
    # available baseline time.
    indices = np.clip( indices, 0, len(baseline_times) - 1 ) 
    H0 = baseline_cum_hazard[ indices ]

    survival = np.exp( -relative_risk[:, None] * H0[None, :] ) 
    return np.clip( survival, 1e-12, 1.0 )