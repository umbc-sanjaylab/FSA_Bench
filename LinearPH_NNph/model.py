from torch import nn

############################# LinearPH/NNPH Model Initializer #####################################

def init_nets(input_size,  n_parties, args): 
    nets = {net_i: None for net_i in range(n_parties)} 
    for net_i in range(n_parties):  
        if args.model == "CoxPH":
            net = LinearPH( dim_in=input_size, dim_out=args.num_duration, batch_norm=True)
        elif args.model == "NNph":
            net = NNPH( dim_in=input_size, hidden_sizes=[32, 32], dim_out=args.num_duration, batch_norm=True, dropout=args.dropout)        
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net 
    return nets 

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