import torch 
import torch.nn as nn   

################################################### Function For initializing networks ##################################################
 
def init_net(input_size, eval_times, args):    
    if args.model == "DeepPseudo":
        net = DeepPseudo(input_size, out_features = len(eval_times), num_event = 1, dropout = args.dropout) 
    else:
        print("not supported yet")
        exit(1)
    return net  

############################# DeepPseudo#####################################    
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

########################## Function for training the model  #####################################
 
def train(dataloader, model, loss_fn, optimizer, device="cpu"):
    # Total number of observation divided by batch size
    num_batches = len(dataloader)
    total_loss = 0

    model.train()
    model.to(device)

    for batch, (X, y) in enumerate(dataloader):
        # x is covariates and y is the pseudo values in the batch
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    total_loss /= num_batches
    return total_loss

########################## Function for evaluating the model ##############################################################


def evaluate(dataloader, model, loss_fn, device="cpu"):
    model.eval()
    model.to(device)

    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            # x is covarites and y is the pseudo values in the batch
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += float(loss_fn(pred, y).item())

    test_loss /= num_batches

    return test_loss


