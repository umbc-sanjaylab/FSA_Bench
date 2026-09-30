import torch
from torch import nn


############################# Build NNnPH Model ###############################################

def build_nnnph(input_size, args):
    return NNnPHNetwork(
        dim_in=input_size,
        hidden_sizes=[32, 32],
        dim_out=args.num_duration,
        batch_norm=True,
        dropout=args.dropout
    )


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
