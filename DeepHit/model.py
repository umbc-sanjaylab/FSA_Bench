import torch
from torch import nn

class CauseSpecificNet(nn.Module): 
    def __init__( self, in_features, num_nodes_shared, num_nodes_indiv, num_risks, out_features, dropout=0.0):
        super().__init__()

        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, num_nodes_shared[0]),
            nn.LayerNorm(num_nodes_shared[0]),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear( num_nodes_shared[0], num_nodes_shared[-1], bias=True)
        )

        # Concatenate original input with shared representation
        risk_in = num_nodes_shared[-1] + in_features
 
        layers = []
        for _ in range(num_risks):  
            layers.append(nn.Sequential(
                nn.Linear(risk_in, num_nodes_indiv[0]),
                nn.LayerNorm(num_nodes_indiv[0]),
                nn.ReLU(),
                nn.Dropout(dropout), 
                nn.Linear( num_nodes_indiv[0], out_features, bias=True )
            ))
            
        self.risk_nets = nn.ModuleList(layers)

    def forward(self, x): 
        shared_out = self.shared_net(x) 
        z = torch.cat([shared_out, x], dim=1)

        risk_outs = [net(z) for net in self.risk_nets] 
        return torch.stack(risk_outs, dim=1) 

def init_net(input_size, args): 
    if args.model == "DeepHit":
        net = CauseSpecificNet(in_features=input_size,
                num_nodes_shared=[128, 64],
                num_nodes_indiv=[32], num_risks=1,
                out_features=args.num_duration,
                dropout=args.dropout
              ) 
    else:
        raise ValueError("Model not supported yet")
    return net        