import numpy as np
import nvflare.client as flare 
from nvflare.apis.fl_constant import FLMetaKey

from federated.client.train import train_client
from models import MODEL_REGISTRY
from datasets.data_processing import SurvivalDataset
from torch.utils.data import DataLoader

from datasets.data_processing import prepare_client_survival_data   
from evaluation.metrics import compute_concordance, evaluate_survival_predictions 
  
def main(args):
    flare.init()  
    
    site_name = flare.get_site_name() 
    client_id =  int( site_name.split("-")[-1] ) - 1 
    
    # Load ONLY this client's data 
    client_data = prepare_client_survival_data(args, client_id) 
    model_spec = MODEL_REGISTRY[args.model]
     
    model_class = model_spec["model"] 
    input_dim = client_data.train.X.shape[1] 
    model = model_class(input_dim)

    while flare.is_running(): 
        input_model = flare.receive() 
        model.load_state_dict(input_model.params) 
        
        meta = input_model.meta or {} 
        phase = meta.get("phase")

        # Bootstrap evaluation times
        if phase == "bootstrap_eval_times": 
            local_unique_times = np.unique( np.concatenate([client_data.train.time, client_data.val.time]) )

            flare.send(
                flare.FLModel(
                    params=model.cpu().state_dict(),
                    meta={
                        "phase": "bootstrap_eval_times", 
                        "local_unique_times":  local_unique_times
                    } 
                )
            )

       
        # Local training 
        elif phase == "train": 
            train_dataset = SurvivalDataset(client_data.train.X, client_data.train.time, client_data.train.event)
            train_loader = DataLoader( dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

            val_dataset = SurvivalDataset( client_data.val.X, client_data.val.time, client_data.val.event)
            val_loader = DataLoader( dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=15)
            eval_times = np.asarray( meta["eval_times"], dtype=float)

 
            try: 
                model = train_client(
                    client_id=client_id,
                    model=model,
                    client_data=client_data,
                    train_dataloader=train_loader, 
                    val_dataloader=val_loader, 
                    model_spec=model_spec,
                    args=args,
                    device="cpu"
                ) 

                output_model = flare.FLModel(
                    params={ name: tensor.detach().cpu() for name, tensor  in model.state_dict().items()},
                    meta={
                        "phase": "train",
                        "train_ok": True,
                        "client_name": site_name, 
                        FLMetaKey.NUM_STEPS_CURRENT_ROUND: len( client_data.train.X) 
                    } 
                )

            except Exception as exc: 
                output_model = flare.FLModel(
                    params=input_model.params,
                    meta={
                        "phase": "train",
                        "train_ok": False,
                        "client_name": site_name,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500] 
                    } 
                )

            flare.send(output_model)

        # Global model evaluation
        elif phase == "eval": 
            eval_times = np.asarray( meta["eval_times"], dtype=float)

            try: 
                predict_survival =  model_spec[ "predict_survival" ]  
                
                # Validation 
                val_survival =  predict_survival(  model=model, X=client_data.val.X, eval_times=eval_times, device=args.device)  
                val_cindex = compute_concordance(
                    survival=val_survival,
                    eval_times=eval_times,
                    durations= client_data.val.time,
                    events= client_data.val.event 
                ) 

               
                # Test 
                test_survival = predict_survival(
                    model=model,
                    X=client_data.test.X,
                    eval_times=eval_times,
                    device=args.device
                )

                test_cindex, test_brier, test_nbll, test_p_value,\
                test_mae_po = evaluate_survival_predictions(
                    survival=test_survival,
                    eval_times=eval_times,
                    train_time= client_data.train.time,
                    train_event= client_data.train.event,
                    test_time= client_data.test.time,
                    test_event= client_data.test.event,
                )

                result_metrics = {
                    "no_of_val_samples": len(client_data.val.X),
                    "no_of_test_samples": len(client_data.test.X), 
                    
                    "validation_c_index": val_cindex,
                    "test_c_index": test_cindex,

                    "test_brier": test_brier, 
                    "test_nbll": test_nbll, 
                    
                    "test_p_value": test_p_value, 
                    "test_weighted_MAE_PO": test_mae_po 
                } 
                result_meta = {"phase": "eval", "eval_ok": True, "client_name": site_name}

            except Exception as exc: 
                result_metrics = {
                    "validation_c_index": np.nan, 
                    "no_of_val_samples": len(client_data.val.X), 
                    "no_of_test_samples": len(client_data.test.X), 
                    "test_c_index": np.nan, 
                    "test_brier": np.nan, 
                    "test_nbll": np.nan, 
                    "test_p_value": np.nan,  
                    "test_weighted_MAE_PO": np.nan 
                }

                result_meta = {
                    "phase": "eval",
                    "eval_ok": False,
                    "client_name": site_name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }

            flare.send(
                flare.FLModel(
                    params=  model.cpu().state_dict(), 
                    metrics=result_metrics, 
                    meta=result_meta 
                )
            )

        else: 
            raise RuntimeError(f"Unsupported phase: {phase!r}")