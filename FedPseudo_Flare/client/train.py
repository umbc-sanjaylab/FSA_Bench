import numpy as np, os, torch 
from torch import optim
from copy import deepcopy 
from .eval import Concordance
 
################################################## Method for training the model ##############################################
def train(dataloader, model, loss_fn, optimizer, device):
    # Total number of observation divided by batch size
    num_batches = len(dataloader)
    model.train()
    model.to(device)

    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # x is covariates and y is the pseudo values in the batch
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    total_loss /= num_batches
    return total_loss
 
 
################################################## Method for evaluating the model ##############################################
def evaluate(dataloader, model, loss_fn, device): 
    num_batches = len(dataloader)
    model.eval()

    # device="cuda:1"
    model.to(device)

    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            # x is covarites and y is the pseudo values in the batch
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += float(loss_fn(pred, y).item())
    test_loss /= num_batches

    return test_loss

################################################## Pseudo Value based Loss Function ##############################################
def pseudo_loss(output, target):
    loss = torch.mean(target*(1-2*output)+(output**2))
    return loss 
      
################  Training Weibull AFT Model with "FedAvg" framework in Local Client  #################################

def train_net(net_id, client_name, net, modeldir, dataset, train_dataloader, val_dataloader, train_time, train_status, X_val, val_time, val_status, evaltime, args, logger, device="cpu"):
  
    # logger initialization  
    logger.info(f"Client: {net_id} training is started")

    # validation data initialization
    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    val_time_array = np.asarray(val_time)
    val_status_array = np.asarray(val_status)

    # optimizer for model parameters udpate
    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters( )), lr=args.lr, weight_decay=args.reg)  # Default optimizer
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters( )), lr=args.lr, weight_decay=args.reg, amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter( lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)

    # learning scheduler
    if args.partition == 'centralized':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'min', patience=args.patience)  # Learning rate scheduler
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'min', patience=args.patience)  # Learning rate scheduler

    # epochs and patience
    patience = args.patience 

    # training and validation loss
    train_losses = []
    val_losses = []

    # concordance index
    cindex = []
    best_val_cindex = float("-inf")
    best_model_state = deepcopy(net.state_dict())

    # Pseudo value based loss function
    criterion = pseudo_loss

    es = 0
    for epoch in range(args.epochs):
        # Calculate the train loss
        train_loss = train(train_dataloader, net, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Calculate the validation loss
        val_loss = evaluate(val_dataloader, net, criterion, device)
        scheduler.step(val_loss)
        val_losses.append(val_loss)

        # Calculate the validation Cindex
        metrics = Concordance(net, X_val_tensor, val_time_array, val_status_array, evaltime, logger, device)
        cindex.append(metrics)

        # handle invalid concordance
        if np.sum(val_status_array == 1) == 0:
            continue  
        if metrics is None or not np.isfinite(metrics):
            logger.warning(
                f"Client: {net_id} invalid validation C-index at epoch {epoch}: {metrics}. "
                f"Skipping early-stopping update for this epoch."
            )
            es += 1
            if es >= patience:
                break
            continue  # Early Stopping Criteria

        elif metrics > best_val_cindex:
            best_val_cindex = metrics
            es = 0
            best_model_state = deepcopy(net.state_dict())
        else:
            es += 1
            if es >= patience:
                break

    # create directory if it does not exist
    checkpoint_dir = os.path.join(modeldir, dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # store and load the model parameters
    torch.save(best_model_state, f'{checkpoint_dir}/{args.model}_{args.run_id}_{client_name}.pt')
    net.load_state_dict(best_model_state)
    net.to(device) 

    return net