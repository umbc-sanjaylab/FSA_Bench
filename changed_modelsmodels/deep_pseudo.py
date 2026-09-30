import torch
from torch import nn 
import numpy as np

############################# Build DeepPseudo Model ##########################################

def build_deep_pseudo(input_size, args):
    return DeepPseudo( in_features=input_size, out_features=9, num_event=1, dropout=args.dropout)

############################# DeepPseudo Model Architecture #####################################    
class DeepPseudo(nn.Module):
    def __init__(self, in_features, out_features, num_event, dropout):
        super().__init__()
        self.out_features = out_features #number of classes
        self.num_event = num_event #number of classes
        self.surv_net = nn.Sequential(
            nn.Linear(in_features, 128), nn.SELU(),nn.Dropout(p=dropout),       
            nn.Linear(128, 64), nn.SELU(),nn.Dropout(p=dropout),                      
            nn.Linear(64, 64), nn.SELU(),nn.Dropout(p=dropout),
            nn.Linear(64, 32), nn.SELU(),nn.Dropout(p=dropout),
            nn.Linear(32, 32), nn.SELU(),nn.Dropout(p=dropout),
            nn.Linear(32, out_features*num_event), nn.Sigmoid())
 
    def forward(self, input):
        output = self.surv_net(input)
        final_output=torch.reshape(output, (-1, self.num_event, self.out_features))
        return final_output
  
############################# Train DeepPseudo ###############################################

def train_epoch( train_loader, model, optimizer, device):
    model.train()
    model.to(device)

    total_loss = 0.0
    num_batches = 0

    for X, pseudo_target in train_loader: 
        X = X.to( device=device, dtype=torch.float32) 
        pseudo_target = pseudo_target.to( device=device, dtype=torch.float32)

        prediction = model(X) 
        loss = pseudo_loss( prediction, pseudo_target)

        optimizer.zero_grad( set_to_none=True)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return np.nan

    return total_loss / num_batches

############################# Validation Loss ################################################

def validation_loss( val_loader, model, device="cpu"):
    model.eval()
    model.to(device)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad(): 
        for X, pseudo_target in val_loader: 
            X = X.to( device=device, dtype=torch.float32 ) 
            pseudo_target = pseudo_target.to( device=device, dtype=torch.float32)

            prediction = model(X) 
            loss = pseudo_loss( prediction, pseudo_target)

            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        return np.nan

    return total_loss / num_batches
 
############################# Predict Survival ###############################################

def predict_survival( model, X, eval_times, device):
    model.eval()
    model.to(device)

    X = torch.as_tensor( X, dtype=torch.float32, device=device) 
    with torch.no_grad():
        output = model(X)

    output =  output.detach().cpu().numpy()
     
    # Remove the singleton event dimension. DeepPseudo output: (n_samples, 1, n_eval_times) 
    if output.ndim == 3 and output.shape[1] == 1:
        output = output.squeeze(axis=1)

    if output.ndim != 2:
        raise ValueError( f"Expected DeepPseudo output with 2 dimensions after squeezing, but received shape {output.shape}.")

    eval_times = np.asarray( eval_times, dtype=float) 
    if output.shape[1] != len(eval_times):
        raise ValueError( f"Model produced {output.shape[1]} time outputs, but {len(eval_times)} evaluation times were provided.")

    
    # S(t) = 1 - F(t); F(t) is the cumulative event probability predicted by DeepPseudo.
    survival = 1.0 - output 
    return np.clip( survival, 1e-12, 1.0)