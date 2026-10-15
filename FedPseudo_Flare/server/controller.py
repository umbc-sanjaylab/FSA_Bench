from copy import deepcopy  
import numpy as np, os, logging   
import pandas as pd 
from .utils import EarlyStopping, _is_valid,  save_results_to_csv 
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg
from .utils import global_survival_probability, Pseudo_Evaltime

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
        self.args = args
    
    def bootstrap_eval_times(self,  model, clients,  logger): 
        """
        Ask each client for its unique train + validation times,
        then construct the global evaluation-time grid.
        """

        model.meta = model.meta or {} 
        model.meta.update({"phase": "bootstrap_eval_times"}) 
        results = self.send_model_and_wait(targets=clients, data=model)
        
        # "phase": "bootstrap_eval_times", 
        # "local_matrix":  local_matrix32,
        # "local_unique_times":  local_unique_times,
        # "columns": columns

        local_unique_times = [] 
        global_partial_list = []
        
        for result in results:             
            local_matrix = (result.meta or {}).get("local_matrix")
            if local_matrix is None:
                logger.info(f"Bootstrap round expects every client to send meta['local_matrix']. Received meta keys: {list((result.meta or {}).keys())}")
            else: 
                global_partial_list.append(local_matrix)
 
        if len(global_partial_list) != len(results):
            raise RuntimeError(f"Bootstrap failed: expected local_matrix from {len(results)} clients,  but received {len(global_partial_list)}. Check failed client logs.")
        
        aggregated_matrix = pd.DataFrame(np.vstack(global_partial_list))
        aggregated_matrix.columns = ['time', 'time', 'nrisk', 'di', 'ci']
        aggregated_matrix = aggregated_matrix.iloc[:, 1:]

        global_partial_matrix = aggregated_matrix.groupby( 'time', as_index=False).sum()
        global_partial_matrix.loc[0, 'nrisk'] = np.sum(global_partial_matrix.loc[:, 'nrisk'])
        global_partial_matrix.loc[1:, 'nrisk'] = 0

        global_full_matrix, global_surv = global_survival_probability(global_partial_matrix, is_DP=False)
        aggregate_results = results[0]
        
        self.global_partial_matrix =  global_partial_matrix.to_numpy()  
        self.global_full_matrix = global_full_matrix
        self.global_surv = global_surv
        self.pseudo_evaltime = Pseudo_Evaltime(global_full_matrix.copy(), self.pseudo_perc_interval)

        global_unique_times = np.unique(np.concatenate(local_unique_times)) 
        self.eval_times = np.percentile(global_unique_times, np.arange(self.pseudo_perc_interval, 99, self.pseudo_perc_interval)) 
        logger.info(f"Global evaluation times: " f"{self.eval_times}")

    def run(self) -> None: 
        logger = logging.getLogger("fsa_bench.server") 
        logger.info("Starting federated survival training.") 
        
        # 1. Load initial global model 
        model = self.load_model() 
        model.meta = model.meta or {}

        model.start_round = self.start_round
        model.total_rounds = self.num_rounds 
        
        # 2. Select all clients 
        clients = self.sample_clients(self.num_clients) 
        if not clients:
            raise RuntimeError("No NVFLARE clients are available.")

        logger.info(f"Selected {len(clients)} clients.") 
        
        # 3. Bootstrap global evaluation times 
        self.bootstrap_eval_times( model=model, clients=clients, logger=logger)
 
        # 4. Server-side early stopping 
        checkpoint_dir = os.path.join( self.modeldir, self.dataset) 
        os.makedirs( checkpoint_dir, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=self.patience,
            verbose=True,
            directory=checkpoint_dir,
            checkpoint_pth=f"{self.model_name}_global_{self.run_id}.pt" 
        )  
 
        best_global_params = None
        comm_round_done = 0
        
        global_val_cindex, test_cindex, test_brier = np.nan, np.nan, np.nan 
        test_nbll, test_p_value, test_mae_po = np.nan, np.nan, np.nan  
        
        # 5. Federated communication rounds 
        for self.current_round in range( self.start_round, self.start_round + self.num_rounds): 
            logger.info( f"Communication round {self.current_round + 1} started.") 
            model.current_round = self.current_round
 
            # TRAIN PHASE 
            model.meta = model.meta or {} 
            model.meta.update({ "phase": "train", "eval_times": self.eval_times }) 
            train_results = self.send_model_and_wait( targets=clients, data=model) 
           
            # Remove failed clients before FedAvg 
            valid_train_results = [] 
            for result in train_results: 
              
                meta = result.meta or {} 
                if not meta.get("train_ok", True):

                    logger.warning(
                        "Skipping failed client training: "
                        f"{meta.get('client_name')}, "
                        f"{meta.get('error_type')}: "
                        f"{meta.get('error_message')}"
                    )

                    continue

                valid_train_results.append(result)

            if not valid_train_results: 
                logger.error( f"No valid client updates were received at round {self.current_round}.") 
                break

             
            # FedAvg  
            aggregated_result = self.aggregate(valid_train_results ) 
            model = self.update_model(model, aggregated_result) 
            logger.info(f"Aggregated {len(valid_train_results)} client models.") 
           
            # EVALUATION PHASE 
            model.meta = model.meta or {} 
            model.meta.update({
                "phase": "eval",
                "eval_times": self.eval_times 
            })

            eval_results = self.send_model_and_wait( targets=clients, data=model)
            global_val_cindex, test_cindex, test_brier, test_nbll, test_p_value, test_mae_po = self.aggregate_evaluation_metrics(eval_results)

            # log the evaluation metrics 
            logger.info(
                f"Round={self.current_round}, "
                f"val_cindex={global_val_cindex}, "
                f"test_cindex={test_cindex}, "
                f"IBS={test_brier}, "
                f"NBLL={test_nbll}, "
                f"p_value={test_p_value}, "
                f"MAE_PO={test_mae_po}"
            )

            
            # SERVER-SIDE EARLY STOPPING 
            if global_val_cindex is not None and np.isfinite(global_val_cindex): 
                current_global_params = deepcopy(model.params)

                improved = early_stopping(
                    global_val_cindex,
                    current_global_params,
                    round_idx=self.current_round,
                    test_metrics={
                        "test_c_index": test_cindex, 
                        "test_brier": test_brier, 
                        "test_nbll": test_nbll, 
                        "test_p_value": test_p_value, 
                        "test_MAE_PO": test_mae_po
                    } 
                )

                if improved: 
                    best_global_params = deepcopy(model.params) 
                    logger.info( f"New best global model selected at round {self.current_round}." )

            else: 
                logger.warning( f"Round {self.current_round}:  validation C-index unavailable;  early stopping skipped." )

            comm_round_done =  self.current_round + 1   
            if early_stopping.early_stop: 
                logger.info(f"Server-side early stopping  after {comm_round_done} rounds.") 
                break
 
        # 6. Restore the BEST global model 
        if best_global_params is not None: 
            model.params = deepcopy(best_global_params)

        # Persist the selected global model
        self.save_model(model)

        # final metrics 
        final_val_cindex = early_stopping.best_val_cindex if np.isfinite( early_stopping.best_val_cindex ) else global_val_cindex  
        final_test_cindex =  early_stopping.best_test_cindex if early_stopping.best_test_cindex is not None else test_cindex   
        final_test_brier =  early_stopping.best_test_brier if early_stopping.best_test_brier is not None else test_brier  
        
        final_test_nbll =   early_stopping.best_test_nbll if early_stopping.best_test_nbll is not None else test_nbll  
        final_test_p_value =  early_stopping.best_test_p_value if early_stopping.best_test_p_value is not None else test_p_value  
        final_test_mae_po =   early_stopping.best_test_MAE_PO if early_stopping.best_test_MAE_PO is not None else test_mae_po  
        
        # determine the best communication round 
        if early_stopping.best_round  is not None: 
            best_round = early_stopping.best_round
        elif comm_round_done > 0: 
             best_round = comm_round_done - 1
        else: 
            best_round = None
        
        save_results_to_csv(final_val_cindex, final_test_cindex, final_test_brier, final_test_nbll, final_test_p_value, final_test_mae_po, best_round, self.args)

        # log the final evaluation metrics 
        logger.info(
            "Federated training completed. "
            f"best_round={best_round}, "
            f"val_cindex={final_val_cindex}, "
            f"test_cindex={final_test_cindex}, "
            f"IBS={final_test_brier}, "
            f"NBLL={final_test_nbll}, "
            f"p_value={final_test_p_value}, "
            f"MAE_PO={final_test_mae_po}"
        )