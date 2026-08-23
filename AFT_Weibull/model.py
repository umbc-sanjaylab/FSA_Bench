from typing import Callable,  Union
import torch.nn as nn
import torch
import gc
gc.collect()
torch.cuda.empty_cache()

ModuleType = Union[str, Callable[..., nn.Module]]
_INTERNAL_ERROR_MESSAGE = 'Internal error. Please, open an issue.'

############################# AFT_Weibull ############################################################################################

class WeibullTimeToEventModel(nn.Module):
    def __init__(self, num_input_features):
        super(WeibullTimeToEventModel, self).__init__()

        self.phi = nn.Parameter(torch.tensor([0.0], dtype=torch.float))

        # Important: negative sign because log_scale = -psi when phi=0 and linear≈0
        self.psi = nn.Parameter(
            torch.tensor([-torch.log(torch.tensor(100.0))], dtype=torch.float)
        )

        self.linear = nn.Linear(num_input_features, 1, bias=False)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        log_params = torch.zeros(
            batch_size, 2, dtype=torch.float, device=inputs.device
        )

        log_params[:, 1] = self.phi
        log_params[:, 0] = (
            -self.linear(inputs) - self.psi * torch.exp(-self.phi)
        ).view(-1)

        return log_params

################################################### Function For initializing networks ##################################################


def init_nets(input_size,  n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):

        if args.model == "AFT_Weibull":
            net = WeibullTimeToEventModel(input_size)
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    return nets
