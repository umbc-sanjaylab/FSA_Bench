import contextlib
import os
import concurrent.futures
import multiprocessing as mp
import warnings
from typing import List, Optional, Sequence
from functools import partial
import numpy as np
import pandas as pd
import torch 
from survshap import SurvivalModelExplainer, ModelSurvSHAP
from fold.predict import predict_surv_matrix 
from functools import partial
from typing import Optional, Sequence
import concurrent.futures
import multiprocessing as mp
import warnings 
import numpy as np
import pandas as pd
import torch


_SURVSHAP_WORKER_STATE = {}


def _move_model_to_device(model, device: str):
    """
    Move a PyTorch module or a PyCox/torchtuples wrapper to one GPU.
    """
    torch.cuda.set_device(torch.device(device))

    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.eval()

    elif hasattr(model, "net") and isinstance(model.net, torch.nn.Module):
        model.net = model.net.to(device)
        model.net.eval()

        if hasattr(model, "device"):
            try:
                model.device = torch.device(device)
            except (AttributeError, TypeError):
                pass

    return model


def _move_model_to_cpu(model):
    """
    Ensure that the parent process passes a CPU model to spawned workers.
    """
    if isinstance(model, torch.nn.Module):
        model = model.cpu()
        model.eval()

    elif hasattr(model, "net") and isinstance(model.net, torch.nn.Module):
        model.net = model.net.cpu()
        model.net.eval()

        if hasattr(model, "device"):
            try:
                model.device = torch.device("cpu")
            except (AttributeError, TypeError):
                pass

    return model


# Each subprocess gets its own independent state.
_SURVSHAP_WORKER_STATE = {}  

def _initialize_survshap_worker( model, background: pd.DataFrame, y_background, times: np.ndarray, model_name: str, discretizer, device: str, method: str, B: int, random_state: int, ) -> None:
    """
    Initialize one persistent SurvSHAP worker on one GPU.
    """
    global _SURVSHAP_WORKER_STATE  
    device = torch.device(device)

    torch.manual_seed(random_state)
    np.random.seed(random_state) 

    model = _move_model_to_device(model=model, device=device)
    predict_function = partial( predict_surv_matrix, evaltime=times, model_name=model_name, discretizer=discretizer, device=device)
    explainer = SurvivalModelExplainer( model=model, data=background, y=y_background, predict_survival_function=predict_function)

    _SURVSHAP_WORKER_STATE = { "explainer": explainer, "times": times, "method": method, "B": B, "random_state": random_state, "feature_names": list(background.columns), "device": str(device)}

def _process_survshap_chunk( task: tuple[int, pd.DataFrame]) -> tuple[int, int, pd.Series]:
    """
    Process one SurvSHAP chunk.

    Returns
    -------
      chunk_id
      chunk_n
      weighted_importance
    """
    chunk_id, chunk = task
    state = _SURVSHAP_WORKER_STATE

    print( f"{state['device']} starting chunk {chunk_id} with {len(chunk)} observations", flush=True) 
    model_survshap = ModelSurvSHAP( calculation_method=state["method"],  aggregation_method="integral",  B=state["B"],  random_state=state["random_state"])

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore") 
    #     model_survshap.fit(explainer=state["explainer"], new_observations=chunk, timestamps=state["times"], save_individual_explanations=False)
    
    # Suppress SurvSHAP/tqdm terminal output.
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            with contextlib.redirect_stderr(devnull):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # track time 
                    model_survshap.fit(explainer=state["explainer"], new_observations=chunk, timestamps=state["times"], save_individual_explanations=False)
                    print(model_survshap.result.head(20))
                    
                    print(model_survshap.result.columns)
                    print(model_survshap.result.shape)

                    print(model_survshap.result.groupby("variable_name") .size())

    importance = (model_survshap.result.set_index("variable_name")["aggregated_change"].astype(float).reindex(state["feature_names"]))
    chunk_n = len(chunk)
    print( f"{state['device']} completed chunk {chunk_id}", flush=True) 

    return ( chunk_id, chunk_n, importance * chunk_n )

def global_importance(fold, model_name, model, X: pd.DataFrame, y, background: Optional[pd.DataFrame] = None, y_background=None, n_explain: Optional[int] = None, times: Optional[np.ndarray] = None, discretizer=None, B: int = 25, method: str = "sampling", random_state: int = 42, features: Optional[Sequence[str]] = None, chunk_size: int = 50, num_workers: int = 4, devices: Optional[Sequence[str]] = None) -> pd.Series:
    """
    Compute global SurvSHAP importance across multiple GPUs. One persistent subprocess is assigned to each GPU.
    """
    del fold, y

    devices = list(devices)
    feature_names = ( list(X.columns) if features is None else list(features))
    explain_X = ( X.loc[:, feature_names] .copy() .reset_index(drop=True))
    
    bg = (background.loc[:, feature_names] .copy() .reset_index(drop=True))
    y_bg = y_background
    
    times = np.asarray(times, dtype=float)
    chunks = [ explain_X.iloc[ start:start + chunk_size ].copy() for start in range( 0, len(explain_X), chunk_size) ]

    # One worker per active GPU.
    actual_workers = min(num_workers, len(devices), len(chunks)) 
    active_devices = devices[:actual_workers]

    # Do not pass an already-CUDA model between processes.
    model = _move_model_to_cpu(model)

    print( f"\nMulti-GPU SurvSHAP configuration: \n  Test instances: {len(explain_X)} \n  Background instances: {len(bg)} \n  Chunk size: {chunk_size} \n  Number of chunks: {len(chunks)} \n  GPU workers: {actual_workers} \n  Devices: {active_devices} \n  Method: {method} \n  B: {B}\n") 
    mp_context = mp.get_context("spawn")

    # One executor with one persistent process for each GPU.
    executors = []
    future_to_chunk_id = {}

    try:
        for device in active_devices:
            initializer_args = (model, bg, y_bg, times, model_name, discretizer, device, method, B, random_state)
            executor = concurrent.futures.ProcessPoolExecutor( 
                                                              max_workers=1, 
                                                              mp_context=mp_context, 
                                                              initializer=_initialize_survshap_worker, initargs=initializer_args
                                                              )
            executors.append(executor)

        for chunk_id, chunk in enumerate(chunks):
            executor_index = ( chunk_id % actual_workers )
            future = executors[ executor_index ].submit( _process_survshap_chunk, (chunk_id, chunk))
            future_to_chunk_id[future] = chunk_id

        chunk_results = [] 
        for future in concurrent.futures.as_completed(future_to_chunk_id):
            chunk_id = future_to_chunk_id[future]

            try:
                result = future.result()
                chunk_results.append(result) 
            except Exception as exc:
                raise RuntimeError( f"SurvSHAP chunk {chunk_id} failed." ) from exc

    finally:
        for executor in executors:
            executor.shutdown( wait=True, cancel_futures=False)

    total_observations = sum( chunk_n for _, chunk_n, _ in chunk_results) 
    weighted_importance = pd.concat( [ importance for _, _, importance in chunk_results ], axis=1, ).sum(axis=1)
    global_feature_importance = ( weighted_importance / total_observations)
    return global_feature_importance .sort_values(ascending=False).rename("global_importance")  