import torch 

#################################### Federated Averaging ############################################################ 

def aggregate_fedavg(global_model, local_models, selected_clients):
    total_samples = sum(len(client.train.X) for client in selected_clients) 
    if total_samples <= 0:
        return False

    largest_client = max(selected_clients, key=lambda client: len(client.train.X)) 
    server_state = {
        key: value.detach().cpu().clone()
        for key, value in global_model.state_dict().items()
    }

    client_states = {} 
    for client in selected_clients: 
        client_states[client.client_id] = {key: value.detach().cpu().clone() for key, value in local_models[client.client_id].state_dict().items()} 

    new_global_state = {} 
    for key, reference_value in server_state.items(): 
        if  reference_value.is_floating_point() or reference_value.is_complex(): 
            aggregated_value = torch.zeros_like(reference_value) 
            
            for client in selected_clients: 
                weight =  len(client.train.X) / total_samples  
                aggregated_value +=  client_states[ client.client_id ][key] * weight  
            new_global_state[key] = aggregated_value 
        else: 
            new_global_state[key] = client_states[largest_client.client_id][key].clone() 
             
    global_model.load_state_dict(new_global_state, strict=True) 
    return True