import numpy as np, os 
import nvflare.client as flare 
import logging, logging.config
from nvflare.apis.fl_constant import FLMetaKey

from .train import train_net 
from datasets.data_class  import SurvivalDataset
from torch.utils.data import DataLoader

from datasets.data_processing import prepare_client_survival_data   
from .util import evaluate_global_model_on_client, local_partial_matrix 
from ..utils import write_logger_configuration
  
def main(args):
    flare.init()  
    
    # client name and id 
    site_name = flare.get_site_name() 
    client_id =  int( site_name.split("-")[-1] ) - 1 
    
    # logger configuration 
    _, log_cfg, _ = write_logger_configuration( args=args, role=site_name, experiment_id=args.experiment_id, save_config=False)  
    logging.config.dictConfig(log_cfg) 
    net = None 
    
    # logger library initialization
    logger = logging.getLogger(f"fedPseudo.client{client_id}")
    logger.info(f"Starting FedPseudo Client: {flare.get_site_name()} and id is {client_id}")

    logger.info(f"model is {args.model} and device is {args.device}")
    logger.info(f"dataset is {args.dataset}")
    logger.info(f"client name is {client_name}")

    # change directory
    os.chdir(args.datadir)
    
    # Load ONLY this client's data 
    client_data, _ = prepare_client_survival_data(args, client_id)  
    client_name = client_data.clients[0].client_name 
    
    # Features, time, and event status of training, validation and test data 
    X_train, X_valid, X_test = client_data.train.X, client_data.val.X, client_data.test.X
    train_time, valid_time, test_time = client_data.train.time, client_data.val.time, client_data.test.time
    train_status, valid_status, test_status = client_data.train.event, client_data.val.event, client_data.test.event 
    
    # concatenation of training and validation time & status
    client_time = np.concatenate((train_time, valid_time), axis=0)
    client_status = np.concatenate((train_status, valid_status), axis=0)
    
    # Training and Validation dataloader 
    train_dataset = SurvivalDataset(X_train, train_time, train_status)
    train_loader = DataLoader( dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    val_dataset = SurvivalDataset( X_valid, valid_time, valid_status)
    val_loader = DataLoader( dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    local_matrix, columns = local_partial_matrix(client_time.copy(), client_status.copy(), sensitivity=args.sensitivity, epsilon=args.epsilon, is_DP=False, init_seed=args.init_seed)
    
    # number of training and validation observation
    no_of_train_samples = train_time.shape[0]
    no_of_val_samples = valid_time.shape[0]
    no_of_test_samples = test_time.shape[0]

    # start federated experiment 
    while flare.is_running(): 
        input_model = flare.receive() 
        net.load_state_dict(input_model.params) 
        
        meta = input_model.meta or {} 
        phase = meta.get("phase")

        # Bootstrap evaluation times
        if phase == "bootstrap_eval_times": 
            local_unique_times = np.unique( np.concatenate([client_data.train.time, client_data.val.time]) )
            local_matrix32 = local_matrix.astype(np.float32) 
            
            flare.send(
                flare.FLModel(
                    params=net.cpu().state_dict(),
                    meta={
                        "phase": "bootstrap_eval_times", 
                        "local_matrix":  local_matrix32,
                        "local_unique_times":  local_unique_times,
                        "columns": columns
                    } 
                )
            )  
        # Local training 
        elif phase == "train":    
            try:
                net = train_net(client_id, client_name, net, args.modeldir, args.dataset, train_loader, val_loader,  train_time, train_status, X_valid, valid_time, valid_status, pseudo_evaltime, args, logger, device=args.device)
                
                output_model = flare.FLModel(
                      params={ name: tensor.detach().cpu() for name, tensor  in net.state_dict().items()},
                      meta={
                          "phase": "train",
                          "train_ok": True,
                          "client_name": site_name, 
                          FLMetaKey.NUM_STEPS_CURRENT_ROUND: len( X_train) 
                      } 
                )
            except Exception as e:
                logger.exception(f"train_net failed on {client_name}, dataset={args.dataset}")
                output_model = flare.FLModel(
                    params=input_model.params,
                    meta={
                        "phase": "train",
                        "train_ok": False,
                        "client_name": site_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e)[:500] 
                    } 
                ) 

            flare.send(output_model)

        # Global model evaluation at client level 
        elif phase == "eval":  
            try:
              meta = input_model.meta or {}
              received_evaltime = meta.get( "pseudo_evaltime")

              if received_evaltime is None:
                  raise RuntimeError( "Eval task does not contain meta['pseudo_evaltime'].")

              pseudo_evaltime = np.asarray( received_evaltime, dtype=np.float64)
              validation_c_index, test_cindex, test_brier, test_nbll, test_p_value, test_weighted_MAE_PO = evaluate_global_model_on_client(net, train_time, train_status,  X_valid, valid_time,  valid_status, X_test, test_time,  test_status, pseudo_evaltime, logger, args.device)

              # log of client test and validation metrics result.
              logger.warning(f"client {site_name} validation_c_index is {validation_c_index}")
              logger.warning(f"client {site_name} test_cindex is {test_cindex} nad brier score is {test_brier}")   
              
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
                  "client_name": client_name
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
                  "test_weighted_MAE_PO": np.nan
              }

              result_meta = {
                  "phase": "eval",
                  "eval_ok": False,
                  "client_name": client_name,
                  "error_type": type(exc).__name__,
                  "error_message": str(exc)[:500]
              } 
              
              logger.warning(f"client {client_name} result metrics={result_metrics}")
              logger.warning(f"client {client_name} model param keys={list(net.state_dict().keys())[:5]}")
              flare.send(flare.FLModel(params=net.cpu().state_dict(), metrics=result_metrics,meta=result_meta))

              flare.send(
                    flare.FLModel(
                        params=  net.cpu().state_dict(), 
                        metrics=result_metrics, 
                        meta=result_meta 
                    )
                )

        else: 
            raise RuntimeError(f"Unsupported phase: {phase!r}")