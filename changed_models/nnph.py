from torch import nn 

############################# Build NNPH Model ###############################################

def build_nnph(input_size, args):
    return NNPH(
        dim_in=input_size,
        hidden_sizes=[32, 32],
        dim_out=args.num_duration,
        batch_norm=True,
        dropout=args.dropout
    )


############################# NNPH Model Architecture ########################################

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