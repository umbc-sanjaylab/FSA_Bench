import torch
from torch import nn
torch.cuda.set_per_process_memory_fraction(0.15, 0)


def init_nets(input_size, dropout_p, n_parties, args, discretiser):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        hidden_sizes = [32, 32]
        batch_norm = True

        if args.model == "NNnPH":
            net = MLP(dim_in=input_size, num_nodes=hidden_sizes, dim_out=len(discretiser.cuts), batch_norm=batch_norm, dropout=dropout_p)
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


class DenseBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, batch_norm=True, dropout=0, activation=nn.ReLU,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias)
        # fill in self.linear.weight.data with kaiming normal
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()

        self.batch_norm = nn.LayerNorm(dim_out) if batch_norm else None
        # self.batch_norm = nn.BatchNorm1d(dim_out) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input

# output is unconstrained if output_activation not sigmoid


class MLP(nn.Module):
    def __init__(self, dim_in, num_nodes, dim_out, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        num_nodes = [dim_in] + num_nodes
        net = []
        for d_in, d_out in zip(num_nodes[:-1], num_nodes[1:]):
            net.append(DenseBlock(d_in, d_out, bias=True, batch_norm=batch_norm,
                                  dropout=dropout, activation=activation, w_init_=w_init_))
        # print(num_nodes[-1], dim_out, output_bias)
        # print(type(num_nodes[-1]), type(dim_out))
        net.append(nn.Linear(num_nodes[-1], dim_out, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)
