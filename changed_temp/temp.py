import numpy as np
import torch
from models.coxph import cox_ph_loss

############################# Training #########################################################

def train_epoch( train_loader, model, optimizer, device):

    model.train()
    model.to(device)

    X_list = []
    time_list = []
    event_list = []

    for X, time, event in train_loader: 
        X_list.append( X.to(device)) 
        time_list.append( time.to(device)) 
        event_list.append(event.to(device))

    if len(X_list) == 0:
        return np.nan

    X = torch.cat( X_list, dim=0 ) 
    time = torch.cat( time_list, dim=0 ) 
    event = torch.cat( event_list, dim=0 ).bool()

    if event.sum() == 0:
        return np.nan

    optimizer.zero_grad() 
    log_risk = model(X).reshape(-1)

    loss = cox_ph_loss(log_risk, time, event) 
    loss.backward()

    optimizer.step() 
    return loss.item()