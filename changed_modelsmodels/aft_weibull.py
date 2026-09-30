from typing import Callable,  Union
import torch.nn as nn
import torch
import gc
import numpy as np
gc.collect()
torch.cuda.empty_cache()

from torchsurv.loss.weibull import neg_log_likelihood_weibull
ModuleType = Union[str, Callable[..., nn.Module]] 


############################# Build AFT_Weibull Model ############################################################################################

def build_aft_weibull(input_size, args):
    return WeibullTimeToEventModel(input_size)
  
############################# AFT_Weibull Model Architecture  ###################################################################################

class WeibullTimeToEventModel(nn.Module):
    def __init__(self, num_input_features):
        super(WeibullTimeToEventModel, self).__init__() 
        self.phi = nn.Parameter(torch.tensor([0.0], dtype=torch.float))

        # Important: negative sign because log_scale = -psi when phi=0 and linear≈0
        self.psi = nn.Parameter( torch.tensor([-torch.log(torch.tensor(100.0))], dtype=torch.float)  ) 
        self.linear = nn.Linear(num_input_features, 1, bias=False)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        log_params = torch.zeros( batch_size, 2, dtype=torch.float, device=inputs.device) 
        log_params[:, 1] = self.phi
        log_params[:, 0] = ( -self.linear(inputs) - self.psi * torch.exp(-self.phi)).view(-1)

        return log_params  
      
########################## Method for training the model  ##################################### 
def train_epoch(train_loader, model, optimizer,  device):
    model.train()
    model.to(device)

    total_loss = 0.0
    num_samples = 0

    for X, time, event in train_loader:
        X = X.to(device)
        time = time.to(device)
        event = event.to(device).bool() 

        log_params = model(X) 
        loss = neg_log_likelihood_weibull( log_params, event, time)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        batch_loss = neg_log_likelihood_weibull( log_params, event, time, reduction="sum").item()

        total_loss += batch_loss
        num_samples += len(X)

    if num_samples == 0:
        return np.nan

    return total_loss / num_samples


########################## Method for evaluating the model ####################################################### 


def evaluate(val_loader, weibull_model, device="cpu"): 
    weibull_model.eval()
    weibull_model.to(device)

    with torch.no_grad():
        val_loss = torch.tensor(0.0, dtype=torch.float, device=device)
        num_points = 0

        for X_batch, Y_batch, D_batch in val_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            D_batch = D_batch.to(device).bool()

            log_hazards_batch = weibull_model(X_batch)
            val_loss += neg_log_likelihood_weibull( log_hazards_batch, D_batch, Y_batch, reduction='sum')
            num_points += X_batch.size(0)

        if num_points == 0:
            return np.nan

        val_loss = float(val_loss / num_points)

    return val_loss

########################## Method for predicting survival functions #####################################

def predict_survival(model, X, eval_times, device): 
    model.eval()
    model.to(device)

    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        log_params = model(X).cpu().numpy()

    log_scale = log_params[:, 0]
    log_shape = log_params[:, 1]

    scale = np.exp(log_scale)
    shape = np.exp(log_shape) 
    eps = 1e-12

    scale = np.clip(scale, eps, 1e12)
    shape = np.clip(shape, eps, 1e12) 
    eval_times = np.asarray(eval_times)

    survival = np.exp(-( eval_times[np.newaxis, :] / scale[:, np.newaxis] ) ** shape[:, np.newaxis])
    return np.clip( survival, eps, 1.0)
