from torch import nn 

############################# CoxPH Model Architecture #####################################

class CoxPHNetwork(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(CoxPHNetwork, self).__init__() 
        self.net =  nn.Linear(in_features, out_features, bias=False)
          
    def forward(self, x):
        return self.net(x) 

############################# CoxPH Model Initializer #####################################
def init_nets(input_size,  n_parties, args): 
    nets = {net_i: None for net_i in range(n_parties)}  
    for net_i in range(n_parties):  
        if args.model == "CoxPH":   
            net = CoxPHNetwork(in_features=input_size, out_features=1) 
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net 
    return nets 