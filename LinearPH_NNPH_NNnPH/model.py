from torch import nn
import torch 

############################# NNnPH Model Initializer #####################################

def init_net(input_size, args):  
    if args.model == "LinearPH":
        net = LinearPH( dim_in=input_size, dim_out=args.num_duration, batch_norm=True)
    elif args.model == "NNph":
        net = NNPH( dim_in=input_size, hidden_sizes=[32, 32], dim_out=args.num_duration, batch_norm=True, dropout=args.dropout)      
    if args.model == "NNnPH":
        net = NNnPHNetwork(dim_in=input_size, hidden_sizes=[32, 32], dim_out=args.num_duration, batch_norm=True, dropout=args.dropout)
    else:
        print("not supported yet")
        exit(1) 
    return net 

############################# LinearPH Model Architecture #####################################

class LinearPH(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__() 
        
        self.linear1 = nn.Linear( dim_in, 1, bias=False) 
        self.linear2 = nn.Linear( 1, dim_out, bias=True)
         
        nn.init.ones_( self.linear2.weight) 
        nn.init.normal_( self.linear2.bias, mean=0.0, std=0.5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x) 
        return x
      
############################# NNPH Model Architecture ##################################### 

class NNPH(nn.Module):
    def __init__( self, dim_in, hidden_sizes, dim_out, batch_norm=True, dropout=0.0):
        super().__init__()

        layers = []
        prev = dim_in 
        for hidden_size in hidden_sizes: 
            linear = nn.Linear( prev, hidden_size, bias=True) 
            nn.init.kaiming_normal_( linear.weight, nonlinearity="relu")

            layers.append(linear)
            layers.append(nn.ReLU())

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev = hidden_size

        # Proportional hazards layer
        linear1 = nn.Linear(prev, 1, bias=False) 
        linear2 = nn.Linear( 1, dim_out, bias=True)

        nn.init.ones_(linear2.weight) 
        nn.init.normal_(linear2.bias, mean=0.0, std=0.5)

        layers.append(linear1)
        layers.append(linear2)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  


############################# NNnPH Model Architecture ########################################

class NNnPHNetwork(nn.Module):
    def __init__( self, dim_in, hidden_sizes, dim_out, batch_norm=True, dropout=0.0):
        super().__init__()

        layers = []
        prev = dim_in

        for hidden_size in hidden_sizes: 
            linear = nn.Linear( prev, hidden_size, bias=True) 
            nn.init.kaiming_normal_( linear.weight, nonlinearity="relu")

            layers.append(linear)
            layers.append(nn.ReLU())

            if batch_norm:
                layers.append( nn.BatchNorm1d(hidden_size))

            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev = hidden_size

        layers.append(nn.Linear( prev, dim_out, bias=True)) 
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) 
      
      
# #################################### FedPseudo################################################# 

def train(dataloader, model, loss_fn, optimizer, device1: str = 'cpu'):
    device = device1
    # print(f"device is {device}")

    # Total number of observation divided by batch size
    num_batches = len(dataloader)
    model.to(device)
    model.train()

    total_loss = 0
    for batch, (X, time, status) in enumerate(dataloader):
        X, time, status = X.to(device), time.to(
            device), status.to(device)  # x is covariates
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, time, status)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= num_batches
    return total_loss
 
############################# Evaluate the model ########################################

def evaluate(dataloader, model, loss_fn, device: str = 'cpu'):

    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, time, status in dataloader:
            X, time, status = X.to(device), time.to(
                device), status.to(device)  # x is covariates
            pred = model(X)
            test_loss += loss_fn(pred, time, status).item()

    test_loss /= num_batches
    return test_loss