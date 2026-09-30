import torch
from torch import nn 

############################# Build LinearPH Model ############################################

def build_linear_ph(input_size, args):
    return LinearPH( dim_in=input_size, dim_out=args.num_duration, batch_norm=True)


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