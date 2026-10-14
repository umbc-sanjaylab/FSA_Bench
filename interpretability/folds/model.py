import torch
import torch.nn as nn   
import os 
# from fold.utils import get_input_feature
import torchtuples as tt
from torch import nn
import numpy as np 

##############################################################################################################################    
  
# def init_net(model, device, args): 
#     input_size = get_input_feature(args)
#     os.environ[f"{args.dataset}_input_size"] = str(input_size)
#     os.environ["DEVICE"] = device
    
#     in_features = input_size
#     out_features= 9  
    
#     if model=="FedPDNN":
#         return FedPDNN(dataset=args.dataset, in_features=in_features,out_features=out_features, dropout=args.dropout_p, hidden_size=0, num_layers=0)  
    
#     if model=="FedPLSTM":
#         return FedPLSTM(dataset=args.dataset, in_features=in_features,out_features=out_features,  dropout=args.dropout_p, hidden_size=128, num_layers=2) 
     
#     if model=="FedPAttn":
#         return FedPAttn(dataset=args.dataset, in_features=in_features,out_features=out_features,  dropout=args.dropout_p, hidden_size=128, num_layers=2)   

############################# FedPDNN  #####################################    

class FedPDNN(nn.Module):
    def __init__(self, dataset=None, in_features=None, out_features=None, dropout=0.1, hidden_size=128, num_layers=2):
        super(FedPDNN, self).__init__()
        
        if in_features is None or out_features is None:
            input_size = os.environ.get(f"{dataset}_input_size", "")
            in_features, out_features = int(input_size), 9

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


########################################## FedPLSTM #################################################

class FedPLSTM(nn.Module):
    def __init__(self, dataset=None,in_features=None, out_features=None, dropout=0.1, hidden_size=128, num_layers=2):
        super(FedPLSTM, self).__init__()  
        
        if in_features is None or out_features is None:
            input_size = os.environ.get(f"{dataset}_input_size", "")
            in_features, out_features = int(input_size), 9
          
        self.out_features = out_features #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = in_features #input size
        self.hidden_size = hidden_size #hidden state

        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, hidden_size) #fully connected 1
        
        self.relu = nn.ReLU()        
        self.fc = nn.Linear(hidden_size, out_features) #fully connected last layer
        self.sigmoid = nn.Sigmoid()

    def forward(self,x): 
        model_device = next(self.parameters()).device 
        if x.device != model_device:
            raise RuntimeError( f"FedPAttn input is on {x.device}, but model is on {model_device}" )
          
        x = torch.reshape(x, (-1, 1, x.shape[1]))
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(model_device) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(model_device) #internal state
        
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state        
        output = output[:, -1, :]
        out = self.fc_1(output) #first Dense
        out = self.relu(out) #relu 
        out = self.fc_1(out) #second Dense
        out = self.relu(out) #relu 
        out = self.fc(out) #Final Output
        out = self.sigmoid(out)
        return out


########################################## FedPAttn #################################################
    
class FedPAttn(nn.Module):
    def __init__(self, dataset=None, in_features=None, out_features=None, dropout=0.1, hidden_size=128, num_layers=2):
        super(FedPAttn, self).__init__()
        
        if in_features is None or out_features is None:
            input_size = os.environ.get(f"{dataset}_input_size", "")
            in_features, out_features = int(input_size), 9

        self.rnn = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.out_features = out_features #number of classes
        
        self.num_layers = num_layers #number of layers
        self.input_size = in_features #input size
        
        self.hidden_size = hidden_size #hidden state
        self.fc_1 =  nn.Linear(2*hidden_size, 2*hidden_size) #fully connected 1
        
        self.relu = nn.ReLU()        
        self.fc = nn.Linear(2*hidden_size, out_features) #fully connected last layer
        self.sigmoid=nn.Sigmoid()
     
    def attention(self, x, input_size=None, hidden_size=None):
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]

        i_q = torch.normal( mean=0.0, std=0.1, size=(self.input_size, self.hidden_size), device=device, dtype=dtype)
        i_k = torch.normal( mean=0.0, std=0.1, size=(self.input_size, self.hidden_size), device=device, dtype=dtype)
        i_v = torch.normal( mean=0.0, std=0.1, size=(self.input_size, self.hidden_size), device=device, dtype=dtype)
        i_w0 = torch.normal( mean=0.0, std=0.1, size=(self.hidden_size, self.input_size), device=device, dtype=dtype, )

        q_trans = i_q.unsqueeze(0).expand(batch_size, -1, -1)
        k_trans = i_k.unsqueeze(0).expand(batch_size, -1, -1)
        v_trans = i_v.unsqueeze(0).expand(batch_size, -1, -1)
        w0 = i_w0.unsqueeze(0).expand(batch_size, -1, -1)

        q = torch.matmul(x, q_trans)
        k = torch.matmul(x, k_trans)
        v = torch.matmul(x, v_trans)

        attention_scores = torch.matmul(q, k.transpose(1, 2)) / 8
        m = torch.softmax( torch.matmul(attention_scores, v), dim=2)

        return torch.matmul(m, w0) + x      
    def forward(self, x):
        model_device = next(self.parameters()).device 
        if x.device != model_device:
            raise RuntimeError( f"FedPAttn input is on {x.device}, but model is on {model_device}" )

        x = x.reshape(-1, 1, x.shape[1])
        attn_output = self.attention(x)

        h_0 = torch.zeros( 2 * self.num_layers, attn_output.size(0), self.hidden_size, device=model_device, dtype=attn_output.dtype)
        c_0 = torch.zeros( 2 * self.num_layers, attn_output.size(0), self.hidden_size, device=model_device, dtype=attn_output.dtype, )

        output, _ = self.rnn(attn_output, (h_0, c_0))
        out = self.fc(output)
        out = out.reshape(-1, self.out_features)

        return self.sigmoid(out)

############################# DeepPseudo ##################################### 

   
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
      

############################# AFT_Weibull ############################################# 

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

        log_params = torch.zeros( batch_size, 2, dtype=torch.float, device=inputs.device ) 
        log_params[:, 1] = self.phi
        log_params[:, 0] = (-self.linear(inputs) - self.psi * torch.exp(-self.phi)).view(-1)

        return log_params


############################# Fed. CoxPH ############################################# 

class Fed_CoxPH(nn.Module):
    def __init__(self, in_features,  out_features=1, output_bias=False):
        super().__init__()
        layers = []
        
        prev = in_features  
        layers.append(nn.Linear(prev, out_features, bias=output_bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)   
      
############################# NNPH ########################################################################  

class NNPH(nn.Module):
    def __init__(self, dim_in, num_nodes, dim_out, batch_norm=True, dropout=None, activation=nn.ReLU,
                output_activation=None, output_bias=True,
                w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        num_nodes = [dim_in] + num_nodes
        net = []
        
        for d_in, d_out in zip(num_nodes[:-1], num_nodes[1:]):
            net.append(DenseBlock(d_in, d_out, bias=True, batch_norm=batch_norm, dropout=dropout, activation=activation, w_init_=w_init_))
         
        net.append(PHBlock(num_nodes[-1], dim_out, batch_norm))
        if output_activation:
            net.append(output_activation)
        
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)
      

class DenseBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, batch_norm=True, dropout=0, activation=nn.ReLU, w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
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
      

############################################### LinearPH ######################################################


class LinearPH(nn.Module):
    def __init__(self, dim_in, dim_out, batch_norm=True, output_activation=None):
        super().__init__()
        net = []
        net.append(PHBlock(dim_in, dim_out, batch_norm))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input) 


class PHBlock(nn.Module):
    def __init__(self, dim_in, dim_out, batch_norm=True):
        super().__init__()
        self.dim_out = dim_out
        self.linear1 = nn.Linear(dim_in, 1, bias=False)
        self.batch_norm = nn.BatchNorm1d(1) if batch_norm else None
        # self.batch_norm = nn.BatchNorm1d(1) if batch_norm else None
        self.linear2 = nn.Linear(1, dim_out, bias=True)
        torch.nn.init.ones_(self.linear2.weight.data)
        torch.nn.init.normal_(self.linear2.bias.data, mean=0.0, std=0.5)
        self.linear2.weight.requires_grad = True 
 
    def forward(self, input):
        input = self.linear1(input) 
        input = self.linear2(input)
        return input
      

############################################### DeepHit ######################################################

class VannilaMLPNetwork(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features=1, dropout=0.0, output_bias=False):
        super().__init__()
        layers = []
        
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h, bias=True))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, out_features, bias=output_bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) 

class DeepHit_DNN(nn.Module):
    """
    Cause-specific network (DeepHit-style) WITHOUT residual connections,
    using custom VannilaMLPNetwork with LayerNorm (batch-size safe).
    """

    def __init__(self,
                 in_features, num_nodes_shared,  num_nodes_indiv,
                 num_risks,  out_features, dropout=None):
        super().__init__()

        # ---------- Shared network ----------
        # num_nodes_shared = [h1, h2, ..., h_last]
        self.shared_net = VannilaMLPNetwork(
            in_features=in_features,   hidden_sizes=num_nodes_shared[:-1],
            out_features=num_nodes_shared[-1],  dropout=dropout,  output_bias=True, )

        # ---------- Cause-specific (risk-specific) heads ----------
        self.risk_nets = nn.ModuleList()
        risk_in = num_nodes_shared[-1] + in_features
        
        for _ in range(num_risks):
            net = VannilaMLPNetwork(
                in_features=risk_in,
                hidden_sizes=num_nodes_indiv,
                out_features=out_features,
                dropout=dropout,
                output_bias=True,
            )
            self.risk_nets.append(net)

    def forward(self, x):
        """
        x: (batch_size, in_features)
        return: (batch_size, num_risks, out_features)
        """
        shared_out = self.shared_net(x)                     # (B, H)
        z = torch.cat([shared_out, x], dim = 1)
        risk_outs = [net(z) for net in self.risk_nets]
        out = torch.stack(risk_outs, dim=1)                 # (B, R, O)
        return out
      

############################################### NNnPH ###################################################### 

class NNnPH(nn.Module):
    def __init__(self, dim_in, num_nodes, dim_out, batch_norm=True, dropout=None, activation=nn.ReLU, output_activation=None, output_bias=True, w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        
        super().__init__()
        num_nodes = [dim_in] + num_nodes
        
        net = []
        for d_in, d_out in zip(num_nodes[:-1], num_nodes[1:]):
            net.append(DenseBlock(d_in, d_out, bias=True, batch_norm=batch_norm, dropout=dropout, activation=activation, w_init_=w_init_))
        
        # print(num_nodes[-1], dim_out, output_bias)
        # print(type(num_nodes[-1]), type(dim_out))
        net.append(nn.Linear(num_nodes[-1], dim_out, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)