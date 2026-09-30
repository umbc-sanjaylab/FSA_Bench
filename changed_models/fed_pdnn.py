import torch
import torch.nn as nn   
import os 
import numpy as np 
from utils import get_input_feature

############################# Build FedPDNN model ############################################ 

def build_fed_pdnn(input_size, args):
    return FedPDNN( in_features=input_size, out_features=9, dropout=args.dropout) 
     
############################# FedPDNN Model Architecture  #####################################    

class FedPDNN(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(FedPDNN, self).__init__() 

        self.surv_net = nn.Sequential(
            nn.Linear(in_features, 128), nn.SELU(),nn.Dropout(p=dropout),       
            nn.Linear(128, 64), nn.SELU(),nn.Dropout(p=dropout),                      
            nn.Linear(64, 64), nn.SELU(),nn.Dropout(p=dropout),
            nn.Linear(64, 32), nn.SELU(),nn.Dropout(p=dropout),
            nn.Linear(32, 32), nn.SELU(),nn.Dropout(p=dropout),
            nn.Linear(32, out_features), nn.Sigmoid()
        ) 
        
    def forward(self, input):
        output = self.surv_net(input)
        return output     