def fedavg(client_models, client_sizes):
    total_samples = sum(client_sizes) 
    aggregated_state = None

    for model, n_samples in zip(client_models, client_sizes):
        weight = n_samples / total_samples
        state = model.cpu().state_dict()

        if aggregated_state is None: 
            aggregated_state = {
                name: tensor * weight
                for name, tensor in state.items()
            }

        else: 
            for name, tensor in state.items():
                aggregated_state[name] += tensor * weight
                 
    return aggregated_state