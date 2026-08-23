import pandas as pd
import numpy as np
from local_train import  local_partial_matrix, Pseudo_Evaltime, local_loo_survival_probability, federated_pseudo_values, train_net, evaluate_global_model_on_client
from model import init_net
from utils import DataBuilder, get_local_dataset, get_args,   write_logger_configuration
from torch.utils.data import DataLoader
import os
import nvflare.client as flare
import numpy as np
import logging
import logging.config
from pathlib import Path 

def main(args):

    # initializes NVFlare client API
    flare.init()
    site_name = flare.get_site_name()
    client_id = int(flare.get_site_name().split("-")[1]) - 1

    _, log_cfg, client_log_path = (
        write_logger_configuration(
            args=args,
            role=site_name,
            experiment_id=args.experiment_id,
            save_config=False,
        )
    )

    logging.config.dictConfig(log_cfg)

    # logger library initialization
    logger = logging.getLogger(f"fedPseudo.client{client_id}")
    logger.info(f"Starting FedPseudo Client: {flare.get_site_name()} and id is {client_id}")

    logger.info(f"model is {args.model} and device is {args.device}")
    logger.info(f"dataset is {args.dataset}")

    # change directory
    os.chdir(args.datadir)

    # covariates and kaplan-meier parameters
    X_train, X_valid, X_test = None, None, None
    train_pseudo_set, val_pseudo_set = None, None
    train_time, valid_time, test_time = None, None, None
    train_status, valid_status, test_status = None, None, None

    # read client dataset
    train_df, val_df, test_df, client_name, feature_list = get_local_dataset(logger, args.dataset, args.partition, client_id, args.run_id, args)
    features_for_standardization = [c for c in feature_list if c.startswith('num_')]

    # client covariates
    X_train = np.array(train_df.drop(columns=["time", "event"]), dtype='float32')
    X_valid = np.array(val_df.drop(columns=["time", "event"]), dtype='float32')
    X_test = np.array(test_df.drop(columns=["time", "event"]), dtype='float32')

    # kaplan-meier parameters
    train_time, train_status = train_df["time"], train_df["event"]
    valid_time, valid_status = val_df["time"], val_df["event"]
    test_time, test_status = test_df["time"], test_df["event"]

    # transform train_time into 1D vector
    train_time = np.asarray(train_time).reshape(-1)
    train_status = np.asarray(train_status).reshape(-1).astype(int)

    # transform valid_time into 1D vector
    valid_time = np.asarray(valid_time).reshape(-1)
    valid_status = np.asarray(valid_status).reshape(-1).astype(int)

    # transform test_time into 1D vector
    test_time = np.asarray(test_time).reshape(-1)
    test_status = np.asarray(test_status).reshape(-1).astype(int)

    # concatenation of training and validation time & status
    client_time = np.concatenate((train_time, valid_time), axis=0)
    client_status = np.concatenate((train_status, valid_status), axis=0)

    # client local matrix computation from Kaplan-Meier parameters
    local_matrix, columns = local_partial_matrix(client_time.copy(), client_status.copy(), sensitivity=args.sensitivity, epsilon=args.epsilon, is_DP=False, init_seed=args.init_seed)

    # client model initialization
    net = init_net(args.model, args.device, args)
    logger.info(f"client side model is {net}")

    # local matrix dimension log
    logger.info(f"Client {flare.get_site_name()}'s local matrix shape is {local_matrix.shape} and columns are {columns}")
    pseudo_evaltime = None

    # number of training and validation observation
    no_of_train_samples = train_time.shape[0]
    no_of_val_samples = valid_time.shape[0]
    no_of_test_samples = test_time.shape[0]
    
    X_valid_array = None
    X_test_array = None

    while flare.is_running():

        # receive model parameters from server and load in client model
        input_model = flare.receive()
        
        logger.info(f"Received model: round={input_model.current_round}, start_round={input_model.start_round}")
        logger.info(f"Received meta keys={list((input_model.meta or {}).keys())}")
        logger.info(f"Received phase={(input_model.meta or {}).get('phase', None)}")

        logger.info(f"Client model keys={list(net.state_dict().keys())[:5]}")
        logger.info(f"Server model keys={list(input_model.params.keys())[:5]}")
    
        net.load_state_dict(input_model.params)
        
        logger.info(f"current_round={input_model.current_round}")
        phase = (input_model.meta or {}).get("phase", None)
        
        # Bootstrap round must only exchange local matrix
        if phase == "bootstrap":
            if input_model.current_round == input_model.start_round:
              local_matrix32 = local_matrix.astype(np.float32)
              flare.send(flare.FLModel(params=net.cpu().state_dict(), meta={"local_matrix": local_matrix32, "columns": columns, "phase":"bootstrap"} )) 
              logger.info(f"local matrix is sent from client {client_id}. ")
            else: 
              logger.info(f"Bootstrap phase is only allowed in the first round, but current_round={input_model.current_round}")
              raise ValueError(f"Bootstrap phase is only allowed in the first round, but current_round={input_model.current_round}")
              
        # computation of local survival probabilities using parameters from server
        elif phase == "train":    
            meta = input_model.meta or {}
            required_keys = ("global_partial_matrix", "global_full_matrix", "global_surv")
            missing_keys = [key for key in required_keys if key not in meta]

            if missing_keys:
                raise RuntimeError( f"Train task is missing required metadata: {missing_keys}")
            
            global_partial_table = input_model.meta["global_partial_matrix"]
            global_full_table = input_model.meta["global_full_matrix"]
            global_surv = input_model.meta["global_surv"]
            
            global_partial_table = np.asarray(meta["global_partial_matrix"])
            global_full_table = np.asarray(meta["global_full_matrix"])
            global_surv = np.asarray(meta["global_surv"])

            # pseudo evaluation time computation
            pseudo_evaltime = Pseudo_Evaltime(global_full_table.copy())
            local_loo_surv = local_loo_survival_probability(global_partial_table.copy(), client_time.copy(), client_status.copy())

            # pseudo values computation
            pseudo_values = federated_pseudo_values( global_full_table.copy(), global_surv.copy(), local_loo_surv.copy(), pseudo_evaltime.copy(), args )
            pseudo_values = np.asarray(pseudo_values, dtype=np.float32)
            
            # logging of evaluation time point, training and validation pseudo values.
            logger.warning(f"Current_round={input_model.current_round} and pseudo value is calculated at round {input_model.start_round+1} ")
            logger.warning(f"pseudo_evaltime length: {pseudo_evaltime.shape}")
            
            # separate training pseudo values from validation pseudo values
            n_train = X_train.shape[0]
            train_pseudo_set = pseudo_values[:n_train, :]
            val_pseudo_set = pseudo_values[n_train:, :]
            logger.warning(f"training pseudo value: {train_pseudo_set.shape}, validation pseudo values: {val_pseudo_set.shape}") 
            
            train_x = DataBuilder(X_train, train_pseudo_set, feature_list, features_for_standardization=[]) 
            train_loader = DataLoader( dataset=train_x, batch_size=args.batch_size, shuffle=True, num_workers=1)
            
            val_x = DataBuilder(X_valid, val_pseudo_set, feature_list, features_for_standardization=[], scaler=None)
            val_loader = DataLoader(dataset=val_x, batch_size=args.batch_size, shuffle=False, num_workers=1)
                
            # evaluation of survival methods after training and validation 
            try:
                net = train_net(client_id, client_name, net, args.modeldir, args.dataset, train_loader, val_loader,  train_time, train_status, X_valid, valid_time, valid_status, pseudo_evaltime, args, logger, device=args.device)
            except Exception as e:
                logger.exception(f"train_net failed on {client_name}, dataset={args.dataset}")
                raise ValueError(f"train_net failed on {client_name}, dataset={args.dataset}")  
             
            # output_model = flare.FLModel(params=net.cpu().state_dict(), metrics={"phase": "train", "validation_c_index": -1})            
            # params = {k: v.detach().cpu() for k, v in net.state_dict().items()}

            output_model = flare.FLModel(
                params=net.cpu().state_dict(),
                metrics={},              # no string metrics here
                meta={"phase": "train"}  # phase belongs in meta
            )

            flare.send(output_model)
         
        elif phase == "eval": 
          try:
              meta = input_model.meta or {}
              received_evaltime = meta.get( "pseudo_evaltime")

              if received_evaltime is None:
                  raise RuntimeError( "Eval task does not contain meta['pseudo_evaltime'].")

              pseudo_evaltime = np.asarray( received_evaltime, dtype=np.float64)
              validation_c_index, test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = evaluate_global_model_on_client(net, train_time, train_status,  X_valid, valid_time,  valid_status, X_test, test_time,  test_status, pseudo_evaltime, logger, args.device)

              # log of client test and validation metrics result.
              logger.warning(f"client {client_name} validation_c_index is {validation_c_index}")
              logger.warning(f"client {client_name} test_cindex is {test_cindex} nad brier score is {test_brier}")   
              
              result_metrics = {
                  "validation_c_index": validation_c_index,
                  "no_of_train_samples": no_of_train_samples,
                  "no_of_val_samples": no_of_val_samples,
                  "no_of_test_samples": no_of_test_samples,
                  "test_c_index": test_cindex,
                  "test_brier": test_brier,
                  "test_nbll": test_nbll,
                  "test_p_value": test_p_value,
                  "test_weighted_MAE_PO":test_weighted_MAE_PO 
              }

              result_meta = {
                  "phase": "eval",
                  "eval_ok": True,
                  "client_name": client_name,
              }
          except Exception as exc: 
              logger.exception( "Evaluation failed on client=%s, round=%s", client_name, input_model.current_round)
              result_metrics = {
                  "validation_c_index": np.nan,
                  "no_of_train_samples": no_of_train_samples,
                  "no_of_val_samples": no_of_val_samples,
                  "no_of_test_samples": no_of_test_samples,
                  "test_c_index": np.nan,
                  "test_brier": np.nan,
                  "test_nbll": np.nan,
                  "test_p_value": np.nan,
                  "test_weighted_MAE_PO": np.nan,
              }

              result_meta = {
                  "phase": "eval",
                  "eval_ok": False,
                  "client_name": client_name,
                  "error_type": type(exc).__name__,
                  "error_message": str(exc)[:500],
              } 
          logger.warning(f"client {client_name} result metrics={result_metrics}")
          logger.warning(f"client {client_name} model param keys={list(net.state_dict().keys())[:5]}")
          flare.send(flare.FLModel(params=net.cpu().state_dict(), metrics=result_metrics,meta=result_meta))  
        
        else:
            raise RuntimeError( f"Unsupported client task phase: {phase!r}")
    
    logger.info("Finished Training & saved the trained model") 

if __name__ == "__main__":
    args = get_args()
    main(args)