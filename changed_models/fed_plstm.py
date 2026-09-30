import torch
import torch.nn as nn   
import os 
import numpy as np 
from utils import get_input_feature
   
############################# Build FedPLSTM model ##################################################################################
  
def build_fed_plstm(input_size, args):
    return FedPLSTM( in_features=input_size, out_features=9, dropout=args.dropout, hidden_size=128, num_layers=2) 

############################### FedPLSTM Model Architecture  #########################################################################
 
class FedPLSTM(nn.Module):
    def __init__( self, in_features, out_features, dropout=0.1, hidden_size=128, num_layers=2):
        super(FedPLSTM, self).__init__()  
        
        self.out_features = out_features #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = in_features #input size
        self.hidden_size = hidden_size #hidden state

        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, hidden_size) #fully connected 1
        self.relu = nn.ReLU() 
               
        self.fc = nn.Linear(hidden_size, out_features) #fully connected last layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        x = torch.reshape( x, (-1, 1, x.shape[1])) 
        device = x.device

        h_0 = torch.zeros( self.num_layers, x.size(0), self.hidden_size, device=device) 
        c_0 = torch.zeros( self.num_layers, x.size(0), self.hidden_size, device=device) 
        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        output = output[:, -1, :] 
        out = self.fc_1(output)
        out = self.relu(out)

        out = self.fc_1(out)
        out = self.relu(out)

        out = self.fc(out)
        out = self.sigmoid(out) 
        return out