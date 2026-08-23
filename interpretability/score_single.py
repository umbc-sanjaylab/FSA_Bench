
import concurrent.futures
import multiprocessing as mp
import warnings
from typing import List, Optional, Sequence
from functools import partial
import numpy as np
import pandas as pd
import torch 
from survshap import SurvivalModelExplainer, ModelSurvSHAP
from predict import predict_surv_matrix 
from functools import partial
from typing import Optional, Sequence
import concurrent.futures
import multiprocessing as mp
import warnings 
import numpy as np
import pandas as pd
import torch

 
 

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

    print( f"{state['device']} starting chunk {chunk_id} with {len(chunk)} observations", flush=True) 
    model_survshap = ModelSurvSHAP( calculation_method=method,  aggregation_method="integral",  B=B,  random_state=random_state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        model_survshap.fit(explainer=state["explainer"], new_observations=chunk, timestamps=state["times"], save_individual_explanations=False)

    importance = (model_survshap.result.set_index("variable_name")["aggregated_change"].astype(float).reindex(state["feature_names"]))
    chunk_n = len(chunk)
    print( f"{state['device']} completed chunk {chunk_id}", flush=True, ) 

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
    chunks = [explain_X.iloc[ start:start + chunk_size ].copy() for start in range( 0, len(explain_X), chunk_size)]

    # One worker per active GPU.
    actual_workers = min(num_workers, len(devices), len(chunks)) 
    active_devices = devices[:actual_workers]
    
    model_survshap = ModelSurvSHAP( calculation_method=method,  aggregation_method="integral",  B=B,  random_state=random_state)
    predict_function = partial( predict_surv_matrix, evaltime=times, model_name=model_name, discretizer=discretizer, device=devices[0])
    
    explainer = SurvivalModelExplainer( model=model, data=background, y=y_background, predict_survival_function=predict_function)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        model_survshap.fit(explainer=explainer, new_observations=chunk, timestamps=times, save_individual_explanations=False)

    importance = (model_survshap.result.set_index("variable_name")["aggregated_change"].astype(float).reindex(feature_names))
    
    # One executor with one persistent process for each GPU.
    executors = []
    future_to_chunk_id = {}

    try:
        
        for chunk_id, chunk in enumerate(chunks):
            executor_index = ( chunk_id % actual_workers )
            future = executors[ executor_index ].submit( _process_survshap_chunk, (chunk_id, chunk))
            future_to_chunk_id[future] = chunk_id

        

            
 
   
    return global_feature_importance .sort_values(ascending=False).rename("global_importance")  