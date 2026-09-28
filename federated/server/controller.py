from copy import deepcopy
import logging
import os

import numpy as np
from scipy.stats import combine_pvalues 
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg 
from federated.server.utils import EarlyStopping

class FederatedSurvivalController(BaseFedAvg):
    """
    NVFLARE server controller for federated survival models.

    Workflow
    --------
    1. Bootstrap global evaluation times.
    2. Broadcast global model.
    3. Receive locally trained models.
    4. Weighted FedAvg.
    5. Broadcast aggregated model for evaluation.
    6. Aggregate client validation/test metrics.
    7. Apply server-side early stopping.
    """

    def __init__(self, *args, num_clients, num_rounds, patience, modeldir, model_name, dataset, run_id, pseudo_perc_interval=10, start_round=0, **kwargs): 
        super().__init__( *args, num_clients=num_clients, num_rounds=num_rounds,  start_round=start_round, **kwargs)

        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.patience = patience

        self.modeldir = modeldir
        self.model_name = model_name
        self.dataset = dataset
        self.run_id = run_id

        self.pseudo_perc_interval =  pseudo_perc_interval  
        self.eval_times = None
        
    
    def bootstrap_eval_times(self,  model, clients,  logger):
        
        """
        Ask each client for its unique train + validation times,
        then construct the global evaluation-time grid.

        This reproduces the logic previously implemented by
        compute_eval_times().
        """

        model.meta = model.meta or {} 
        model.meta.update({"phase": "bootstrap_eval_times"}) 
        results = self.send_model_and_wait(targets=clients, data=model)

        local_unique_times = [] 
        for result in results: 
            meta = result.meta or {}
            unique_times = meta.get("local_unique_times")

            if unique_times is None: 
                logger.warning("Client did not return local_unique_times.") 
                continue

            unique_times = np.asarray(unique_times, dtype=float) 
            if len(unique_times) > 0: 
                local_unique_times.append(unique_times)

        if not local_unique_times: 
            raise RuntimeError( "No client returned valid survival times during bootstrap.")

        global_unique_times = np.unique(np.concatenate(local_unique_times)) 
        self.eval_times = np.percentile(global_unique_times, np.arange(self.pseudo_perc_interval, 99, self.pseudo_perc_interval)) 
        logger.info(f"Global evaluation times: " f"{self.eval_times}")
        
    @staticmethod
    def _is_valid(value): 
        return value is not None and np.isscalar(value) and np.isfinite(float(value)) 

    def aggregate_evaluation_metrics(self, eval_results): 
        weighted_val_cindex = 0.0
        total_val_samples = 0

        weighted_test_cindex = 0.0
        cindex_samples = 0

        weighted_test_brier = 0.0
        brier_samples = 0

        weighted_test_nbll = 0.0
        nbll_samples = 0

        weighted_test_mae = 0.0
        mae_samples = 0

        client_p_values = []

        for result in eval_results: 
            meta = result.meta or {} 
            if not meta.get("eval_ok", True):
                continue

            metrics = result.metrics or {} 
            n_val = metrics.get( "no_of_val_samples", 0)  
            val_cindex = metrics.get( "validation_c_index")

            if self._is_valid(val_cindex) and n_val > 0: 
                weighted_val_cindex +=  float(val_cindex) * n_val
                total_val_samples += n_val
 
            # Test evaluation metrics
            n_test = metrics.get( "no_of_test_samples", 0) 
            test_cindex = metrics.get( "test_c_index") 
            test_brier = metrics.get( "test_brier") 
            
            test_nbll = metrics.get( "test_nbll")
            test_p_value = metrics.get("test_p_value")
            test_mae = metrics.get( "test_weighted_MAE_PO")
            
            if  self._is_valid(test_cindex) and n_test > 0: 
                weighted_test_cindex +=   float(test_cindex) * n_test  
                cindex_samples += n_test

            if self._is_valid(test_brier) and n_test > 0: 
                weighted_test_brier +=  float(test_brier) * n_test
                brier_samples += n_test

            if  self._is_valid(test_nbll) and n_test > 0: 
                weighted_test_nbll +=  float(test_nbll) * n_test
                nbll_samples += n_test

            if  self._is_valid(test_mae) and n_test > 0: 
                weighted_test_mae +=  float(test_mae) * n_test  
                mae_samples += n_test

            if self._is_valid(test_p_value):
                client_p_values.append(float(test_p_value))

        global_val_cindex =  weighted_val_cindex  / total_val_samples if total_val_samples > 0 else np.nan 
        test_cindex = weighted_test_cindex / cindex_samples if cindex_samples > 0 else np.nan   
        test_brier = weighted_test_brier / brier_samples if brier_samples > 0 else np.nan   
        test_nbll = weighted_test_nbll / nbll_samples if nbll_samples > 0 else np.nan   
        test_mae = weighted_test_mae / mae_samples if mae_samples > 0 else np.nan  

        test_p_value = np.nan
        if client_p_values: 
            test_p_value = combine_pvalues( client_p_values, method="fisher", )[1]
        return global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_mae 