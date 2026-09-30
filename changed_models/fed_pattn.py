import torch
import torch.nn as nn   
import os 
import numpy as np  

############################# Build FedPAttn model ##################################################################################

def build_fed_pattn(input_size, args):
    return FedPAttn( in_features=input_size, out_features=9, dropout=args.dropout, hidden_size=128, num_layers=2)

############################### FedPAttn Model Architecture  #########################################################################
    
class FedPAttn(nn.Module):
      def __init__( self, in_features, out_features, dropout=0.1, hidden_size=128, num_layers=2):
        super(FedPAttn, self).__init__() 

        self.rnn = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.out_features = out_features #number of classes
        
        self.num_layers = num_layers #number of layers
        self.input_size = in_features #input size
        
        self.hidden_size = hidden_size #hidden state
        self.fc_1 =  nn.Linear(2*hidden_size, 2*hidden_size) #fully connected 1
        
        self.relu = nn.ReLU()        
        self.fc = nn.Linear(2*hidden_size, out_features) #fully connected last layer
        self.sigmoid=nn.Sigmoid()
     
      def attention(self, x): 
        device = x.device

        i_q = torch.normal( 0, 0.1, size=(self.input_size, self.hidden_size), device=device) 
        i_k = torch.normal( 0, 0.1, size=(self.input_size, self.hidden_size), device=device) 
        i_v = torch.normal( 0, 0.1, size=(self.input_size, self.hidden_size), device=device)

        i_w0 = torch.normal( 0, 0.1, size=(self.hidden_size, self.input_size), device=device)
        w0 = torch.tile( torch.reshape( i_w0, [-1, self.hidden_size, self.input_size] ), [x.shape[0], 1, 1] )
        
        q_trans = torch.tile( torch.reshape( i_q, [-1, self.input_size, self.hidden_size] ), [x.shape[0], 1, 1] ) 
        k_trans = torch.tile( torch.reshape( i_k, [-1, self.input_size, self.hidden_size] ), [x.shape[0], 1, 1] ) 
        v_trans = torch.tile( torch.reshape( i_v, [-1, self.input_size, self.hidden_size] ), [x.shape[0], 1, 1] )

        q = torch.matmul(x, q_trans)
        k = torch.matmul(x, k_trans)
        v = torch.matmul(x, v_trans)

        attention_scores = torch.matmul( q, torch.transpose(k, 2, 1) ) / 8 
        m = torch.matmul( attention_scores, v )
        m = torch.softmax(m, dim=2) 
        
        z = torch.matmul(m, w0) 
        z = torch.add(z, x)  
        return z      
      
      def forward(self, x): 
        x = torch.reshape( x, (-1, 1, x.shape[1]) ) 
        device = x.device

        attn_output = self.attention(x).clone() .detach()  
        h_0 = torch.zeros( 2 * self.num_layers, attn_output.size(0), self.hidden_size, device=device ) 
        c_0 = torch.zeros( 2 * self.num_layers, attn_output.size(0), self.hidden_size, device=device ) 
        
        output, (hidden, cell) = self.rnn( attn_output, (h_0, c_0) ) 
        out = self.fc(output) 
        
        out = torch.reshape(out, (-1, self.out_features))  
        out = self.sigmoid(out) 
        return out