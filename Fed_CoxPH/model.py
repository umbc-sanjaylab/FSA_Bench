
import torch 
from torch import nn

class VannilaMLPNetwork(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features=1, dropout=0.0, output_bias=False):
        super().__init__()
        layers = []
        prev = in_features  
        layers.append(nn.Linear(prev, out_features, bias=output_bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def init_nets(input_size, dropout_p, n_parties, args): 
    nets = {net_i: None for net_i in range(n_parties)} 

    for net_i in range(n_parties):

        output_size = 1
        hidden_sizes = [] 
         
        if args.model == "CoxPH":    
            net = VannilaMLPNetwork(
                in_features=input_size,
                hidden_sizes=hidden_sizes,
                out_features=output_size,
                dropout=dropout_p,
                output_bias=False )   
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net 
        
    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type
  