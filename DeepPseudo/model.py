import torch 
import torch.nn as nn    

############################# DeepPseudo#####################################    
class DeepPseudo(nn.Module):
    def __init__(self, in_features, out_features, num_event, dropout):
        super().__init__()
        self.out_features = out_features #number of classes
        self.num_event = num_event #number of classes
        self.surv_net = nn.Sequential(
            nn.Linear(in_features, 128), nn.SELU(),nn.Dropout(p=dropout),       
            nn.Linear(128, 64), nn.SELU(),nn.Dropout(p=dropout),                      
            nn.Linear(64, 64), nn.SELU(),nn.Dropout(p=dropout),
            nn.Linear(64, 32), nn.SELU(),nn.Dropout(p=dropout),
            nn.Linear(32, 32), nn.SELU(),nn.Dropout(p=dropout),
            nn.Linear(32, out_features*num_event), nn.Sigmoid())
 
    def forward(self, input):
        output = self.surv_net(input)
        final_output=torch.reshape(output, (-1, self.num_event, self.out_features))
        return final_output

################################################### Function For initializing networks ##################################################

def init_nets(input_size, dropout_p, n_parties,  args): 
    nets = {net_i: None for net_i in range(n_parties)} 
    for net_i in range(n_parties): 
        output_size = args.pseudo_perc_interval
        num_event= 1 
            
        if args.model == "DeepPseudo":
            net = DeepPseudo(input_size, output_size, num_event, dropout_p) 
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net 
    return nets  


