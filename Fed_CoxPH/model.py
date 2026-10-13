from torch import nn  

############################# CoxPH Model Initializer #####################################
def init_net(input_size,  args): 
    if args.model == "CoxPH":   
      net = CoxPH_Network(in_features=input_size, out_features=1) 
    else:
      raise ValueError("Model not supported")
    return net
############################# CoxPH Model Architecture #####################################

class CoxPH_Network(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(CoxPH_Network, self).__init__() 
        self.net =  nn.Linear(in_features, out_features, bias=False)
          
    def forward(self, x):
        return self.net(x) 

