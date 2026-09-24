import torch
import torchtuples as tt   
from torch import nn 

class VannilaMLPNetwork(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features=1, dropout=0.0, output_bias=False):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h, bias=True))
            layers.append(nn.LayerNorm(h))       
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, out_features, bias=output_bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) 

class CauseSpecificNet(nn.Module):
    """
    Cause-specific network (DeepHit-style) WITHOUT residual connections,
    using custom VannilaMLPNetwork with LayerNorm (batch-size safe).
    """
    def __init__(  self,
        in_features, num_nodes_shared,  num_nodes_indiv,
        num_risks,  out_features,dropout=None,):
        super().__init__()

        # ---------- Shared network ----------
        # num_nodes_shared = [h1, h2, ..., h_last]
        self.shared_net = VannilaMLPNetwork(
            in_features=in_features,   hidden_sizes=num_nodes_shared[:-1],
            out_features=num_nodes_shared[-1],  dropout=dropout,  output_bias=True, )

        # ---------- Cause-specific (risk-specific) heads ----------
        self.risk_nets = nn.ModuleList()
        for _ in range(num_risks):
            net = VannilaMLPNetwork(
                in_features=num_nodes_shared[-1],
                hidden_sizes=num_nodes_indiv,
                out_features=out_features,
                dropout=dropout,
                output_bias=True,
            )
            self.risk_nets.append(net)

    def forward(self, x):
        """
        x: (batch_size, in_features)
        return: (batch_size, num_risks, out_features)
        """
        shared_out = self.shared_net(x)                     # (B, H)
        risk_outs = [net(shared_out) for net in self.risk_nets]
        out = torch.stack(risk_outs, dim=1)                 # (B, R, O)
        return out


def init_nets(input_size, dropout_p, n_parties, args):

    num_nodes_shared = [128, 64]
    num_nodes_indiv = [64, 32]
    num_risks=1                    
    
    output_size = args.num_duration 
    alpha, sigma = 0.5, 0.5 
    print("dataset is ", args.dataset)  
    
    nets = {net_i: None for net_i in range(n_parties)} 
    num_nodes_shared = [128, 64]
    num_nodes_indiv = [32] 
    alpha, sigma =   (0.8, 1.0)
            
    for net_i in range(n_parties):    
        if args.model == "DeepHit":
            net = CauseSpecificNet(input_size, num_nodes_shared, num_nodes_indiv, num_risks,
                       output_size, dropout_p)             
        else:
            print("not supported yet")
            exit(1)
            
        nets[net_i] = net   
        
    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    print(f"Value of alpha: {alpha} and value of sigma: {sigma} ") 
    return nets, model_meta_data, layer_type, alpha, sigma 


