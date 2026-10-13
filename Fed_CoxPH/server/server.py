import torch 
  
#################################### Federated Averaging ############################################################ 

def aggregate_fedavg(global_model, local_models, valid_clients):
    if not valid_clients:
        return False

    total_samples = sum(len(client.train.X) for client in valid_clients) 
    if total_samples == 0:
        return False

    aggregated_state = None 
    for client in valid_clients:
        local_state = local_models[ client.client_id ].cpu().state_dict() 
        weight = len(client.train.X) / total_samples  

        if aggregated_state is None:
            aggregated_state = {}

            for key, value in local_state.items():
                if torch.is_floating_point(value):
                    aggregated_state[key] =  value.clone() * weight 
                else:
                    aggregated_state[key] = value.clone()  
        else:
            for key, value in local_state.items():
                if torch.is_floating_point(value):
                    aggregated_state[key] +=  value * weight
                     

    if aggregated_state is None:
        return False

    global_model.load_state_dict(aggregated_state) 
    return True