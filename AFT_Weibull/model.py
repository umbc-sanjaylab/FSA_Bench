import torch.nn as nn
import torch 
import numpy as np
from torchsurv.loss.weibull import neg_log_likelihood_weibull

############################# AFT_Weibull ############################################################################################

class AFT_Weibull(nn.Module):
    def __init__(self, num_input_features):
        super(AFT_Weibull, self).__init__() 
        self.phi = nn.Parameter(torch.tensor([0.0], dtype=torch.float))

        # Important: negative sign because log_scale = -psi when phi=0 and linear≈0
        self.psi = nn.Parameter( torch.tensor([-torch.log(torch.tensor(100.0))], dtype=torch.float)) 
        self.linear = nn.Linear(num_input_features, 1, bias=False)

    def forward(self, inputs):
        batch_size = inputs.size(0) 
        log_params = torch.zeros(batch_size, 2, dtype=torch.float, device=inputs.device)

        log_params[:, 1] = self.phi
        log_params[:, 0] = (-self.linear(inputs) - self.psi * torch.exp(-self.phi) ).view(-1)

        return log_params

################################################### Function For initializing networks ##################################################

def init_net(input_size, args):  
    if args.model == "AFT_Weibull":
        net = AFT_Weibull(input_size)
    else:
        print("not supported yet")
        exit(1)
    return net   
  
########################## Function for training the model  #####################################


def train(train_loader, weibull_model, optimizer, device="cpu"):
    weibull_model.train()
    weibull_model.to(device)

    total_loss = 0.0
    num_points = 0

    for X_batch, Y_batch, D_batch in train_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        D_batch = D_batch.to(device).bool()

        # skip fully censored batch
        if torch.sum(D_batch).item() == 0:
            continue

        log_hazards_batch = weibull_model(X_batch) 
        loss_batch = neg_log_likelihood_weibull(log_hazards_batch, D_batch, Y_batch)

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        batch_loss_sum = neg_log_likelihood_weibull(log_hazards_batch, D_batch, Y_batch, reduction='sum').item() 
        total_loss += batch_loss_sum
        num_points += X_batch.size(0)

    if num_points == 0:
        return np.nan

    train_loss = total_loss / num_points
    return train_loss

########################## Function for evaluating the model ##############################################################


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

            # skip fully censored batches
            if torch.sum(D_batch).item() == 0:
                continue

            log_hazards_batch = weibull_model(X_batch)
            val_loss += neg_log_likelihood_weibull(log_hazards_batch, D_batch, Y_batch, reduction='sum')
            num_points += X_batch.size(0)

        if num_points == 0:
            return np.nan

        val_loss = float(val_loss / num_points)

    return val_loss
