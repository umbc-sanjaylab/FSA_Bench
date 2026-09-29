import numpy as np
import pandas as pd
from nvflare.app_common.workflows.fedavg import FedAvg
import os
from local_train import global_survival_probability, Pseudo_Evaltime
import logging
import time
import os
import psutil
from scipy.stats import combine_pvalues
from utils import EarlyStopping, save_results_to_csv, aggregate_model_with_finite_parameters, is_valid_number


class FederatedAvg(FedAvg):
    def __init__(self, *args,
        patience: int = 0,
        modeldir: str = '',
        reg: int = 0,
        lr: int = 0.001,
        epochs: int = 10,
        optimizer: str = '',
        partition: int = 0,
        batch_size: int = 0,
        modelName: str = None,
        device: str = None,
        dataset: str = None,
        num_clients: int = 3,
        num_rounds: int = 5,
        start_round: int = 0,
        seed: int = 0,
        logdir: str = '',
        run_id: int = 1,
        arguments=None,
        **kwargs,
    ):
        super().__init__(*args, num_clients=num_clients,
            num_rounds=num_rounds,
            start_round=start_round, **kwargs,
        )

        self.num_rounds = num_rounds
        self.patience = patience
        self.lr = round(float(lr), 4)
        self.epochs = epochs
        self.partition = partition
        self.batch_size = batch_size
        self.modelName  = modelName
        self.dataset = dataset
        self.device = device
        self.optimizer = optimizer
        self.reg = reg
        self.modeldir = modeldir
        self.seed = seed
        self.logdir = logdir
        self.run_id = run_id
        self.proc = psutil.Process(os.getpid())
        self.arguments = arguments

        print(f'Partition in Federated average constructor:  {self.partition}')
        print(f"Model in server: {self.modelName}")

    def mem_mb(self):
        return self.proc.memory_info().rss / (1024**2)
    
    def fix_cindex_direction(self, cindex):
        if np.isfinite(cindex) and cindex < 0.5:
            return 1.0 - cindex
        return cindex

    def run(self) -> None:
        logger = logging.getLogger("fedPseudo.server")
        logger.info("Start FedAvg.")
        
        logger.info(f"Initialized network: {self.modelName}")  
        logger.info(f"Partition: {self.partition}") 
        logger.info(f"device: {self.device}")

        model = self.load_model() 
        model.meta = model.meta or {}
        model.meta["phase"] = "bootstrap"
        
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        logger.info(f"dataset: {self.dataset}")
        start_counter = time.perf_counter()

        # create checkpoint directory
        checkpoint_dir = os.path.join(self.modeldir, self.dataset)
        os.makedirs(checkpoint_dir, exist_ok=True) 
        early_stopping = EarlyStopping(patience=self.patience,   verbose=True, directory=checkpoint_dir, checkpoint_pth=f"{self.modelName}_global_{self.run_id}.pt")

        # keep track of last completed communication round
        comm_round = self.start_round - 1

        # optional: keep latest metrics for debugging
        last_test_cindex = None
        last_test_brier = None
        last_test_nbll = None
        last_test_p_value = None
        last_test_MAE_PO = None 
        last_global_val_cindex = None

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            logger.info(f"Round {self.current_round} started.")
            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)
            results = self.send_model_and_wait(targets=clients, data=model)
            
            logger.info(f"local metrices obtained from all clients at bootstrap phase")
            model.meta = model.meta or {}

            ################# Bootstrap round  #####################################  
            if model.meta["phase"] == "bootstrap": 
                # self.current_round == self.start_round:
                global_partial_list = []

                for flModel in results:
                    local_matrix = (flModel.meta or {}).get("local_matrix")
                    if local_matrix is None:
                        logger.info(f"Bootstrap round expects every client to send meta['local_matrix']. Received meta keys: {list((flModel.meta or {}).keys())}")
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
                self.pseudo_evaltime = Pseudo_Evaltime(global_full_matrix.copy())
                
                aggregate_results.meta.update({
                    "phase": "train",
                    "global_partial_matrix": self.global_partial_matrix,
                    "global_full_matrix": self.global_full_matrix,
                    "global_surv": self.global_surv,
                    "pseudo_evaltime": self.pseudo_evaltime
                })   

                model = self.update_model(model, aggregate_results)
                self.save_model(model)
                logger.info( f"Bootstrap model saved at communication round: {self.current_round}")

                comm_round = self.current_round
                continue
            
            elif model.meta["phase"] == "train": 
                
                # aggregate client updates into the new global model
                # valid_results = aggregate_model_with_finite_parameters(results, logger)
                 
                # if not valid_results:
                #   logger.error(f"Round {self.current_round}: no finite client model updates. Skipping round.")

                #   model.meta.update({
                #       "phase": "train",
                #       "global_partial_matrix": self.global_partial_matrix,
                #       "global_full_matrix": self.global_full_matrix ,
                #       "global_surv": self.global_surv,
                #       "pseudo_evaltime": self.pseudo_evaltime,
                #   })

                #   continue
                
                 
                aggregate_results = self.aggregate(results)
                model = self.update_model(model, aggregate_results)
                logger.info( f"Aggregated local models at server: {self.current_round}")
                
                # 2. Send newly aggregated global model back for evaluation only  
                model.meta = model.meta or {}
                model.meta["phase"] = "eval"
                
                model.meta.update({
                    "phase": "eval",
                    "pseudo_evaltime": self.pseudo_evaltime,
                })
                
                logger.info( f"Sent global model for evaluation at round: {self.current_round}")
                eval_results = self.send_model_and_wait(targets=clients, data=model)
                
                successful_eval_results = []
                for fl_model in eval_results:
                    meta = fl_model.meta or {}

                    if not meta.get("eval_ok", True):
                        logger.error("Client evaluation failed:  client=%s, error=%s: %s", meta.get("client_name"), meta.get("error_type"), meta.get("error_message"))
                        continue

                    successful_eval_results.append(fl_model)
                
                global_val_cindex = 0.0
                total_val_samples = 0  
                client_p_values = []
                
                weighted_test_cindex = 0.0
                weighted_test_brier = 0.0
                weighted_test_nbll = 0.0
                weighted_test_p_value = 0.0
                weighted_test_MAE_PO = 0.0
                
                weighted_test_cindex_sample = 0.0
                weighted_test_brier_sample = 0.0
                weighted_test_nbll_sample = 0.0
                weighted_test_MAE_PO_sample = 0.0
                
                logger.warning(f"Number of received results: {len(results)}")
                logger.warning(f"Number of evaluation results: " f"{len(eval_results)}/{len(clients)}")
                
                for index, flModel in enumerate(successful_eval_results):
                    
                    logger.warning(f"Result {index}: metrics={flModel.metrics}, meta_keys={list((flModel.meta or {}).keys())}")
                    metrics = flModel.metrics or {}

                    ##################### validation ##################################
                    n_val = metrics.get("no_of_val_samples", 0)
                    valid_cindex = metrics.get("validation_c_index", None)

                    if valid_cindex is not None and np.isfinite(valid_cindex) and n_val > 0:
                        global_val_cindex += float(valid_cindex) * float(n_val)
                        total_val_samples += float(n_val) 

                    ##################### test  ################################## 
                    n_test = metrics.get("no_of_test_samples", 0)
                    test_cidx = metrics.get("test_c_index", None)
                    test_brier = metrics.get("test_brier", None)
                    test_nbll = metrics.get("test_nbll", None)
                    test_p_value = metrics.get("test_p_value", None)
                    test_MAE_PO = metrics.get("test_weighted_MAE_PO", None)
                    
                    if is_valid_number(test_cidx) and n_test > 0:
                          weighted_test_cindex +=  float(test_cidx) * n_test
                          weighted_test_cindex_sample += n_test

                    if is_valid_number(test_brier) and n_test > 0:
                          weighted_test_brier +=  float(test_brier) * n_test
                          weighted_test_brier_sample += n_test
                    
                    if is_valid_number(test_nbll) and n_test > 0:
                          weighted_test_nbll +=  float(test_nbll) * n_test
                          weighted_test_nbll_sample += n_test
                    
                    if is_valid_number(test_p_value):
                          client_p_values.append(float(test_p_value))
                    
                    if is_valid_number(test_MAE_PO) and n_test > 0:
                          weighted_test_MAE_PO +=  float(test_MAE_PO) * n_test
                          weighted_test_MAE_PO_sample += n_test 
                      
                if total_val_samples == 0:
                  logger.error( f"Round {self.current_round}: no valid  validation C-index was available.  Skipping server early stopping.")

                  model.meta.update({
                      "phase": "train",
                      "global_partial_matrix": self.global_partial_matrix,
                      "global_full_matrix": self.global_full_matrix,
                      "global_surv": self.global_surv,
                      "pseudo_evaltime": self.pseudo_evaltime,
                  })

                  continue   
                                      
                if len(client_p_values) > 0:
                    weighted_test_p_value = combine_pvalues(client_p_values, method="fisher")[1]
                else:
                    weighted_test_p_value = np.nan
                  
                global_val_cindex /= total_val_samples
                weighted_test_cindex = weighted_test_cindex / weighted_test_cindex_sample if weighted_test_cindex_sample > 0 else np.nan  
                weighted_test_brier = weighted_test_brier / weighted_test_brier_sample if weighted_test_brier_sample > 0 else np.nan  
                weighted_test_nbll = weighted_test_nbll / weighted_test_nbll_sample if weighted_test_nbll_sample > 0 else np.nan
                weighted_test_MAE_PO = weighted_test_MAE_PO / weighted_test_MAE_PO_sample if weighted_test_MAE_PO > 0 else np.nan   
                
                # early stopping selects the best checkpoint using validation C-index
                improved = early_stopping(
                    global_val_cindex,
                    model.params,
                    round_idx=self.current_round,
                    test_metrics={
                        "test_c_index": weighted_test_cindex,
                        "test_brier": weighted_test_brier,
                        "test_nbll": weighted_test_nbll,
                        "test_p_value": weighted_test_p_value,
                        "test_MAE_PO": weighted_test_MAE_PO,
                    },
                )
                
                if improved:
                    logger.info(
                        f"New best model selected at round {self.current_round}: "
                        f"val_cindex={global_val_cindex:.6f}, "
                        f"test_cindex={weighted_test_cindex:.6f}, "
                        f"test_brier={weighted_test_brier:.6f}, "
                        f"test_nbll={weighted_test_nbll:.6f}, "
                        f"test_p_value={weighted_test_p_value:.6f}, "
                        f"test_MAE_PO={weighted_test_MAE_PO:.6f}"
                    )
                
                model.meta = model.meta or {}
                model.meta.update({
                        "phase": "train",
                        "global_partial_matrix": self.global_partial_matrix,
                        "global_full_matrix": self.global_full_matrix,
                        "global_surv": self.global_surv,
                        "pseudo_evaltime": self.pseudo_evaltime,
                })

                comm_round = self.current_round
                if early_stopping.early_stop:
                    logger.info( f"Early stopping triggered at communication round {self.current_round}")
                    break
                  
        memory = self.mem_mb()
        end_counter = time.perf_counter()
        total_time_counter = end_counter - start_counter  

        # report the metrics from the best validation round, not the last round
        final_round = early_stopping.best_round if early_stopping.best_round is not None else comm_round
        final_val_cidx =  early_stopping.best_val_cindex  if np.isfinite(early_stopping.best_val_cindex)   else last_global_val_cindex 
        final_test_cidx =  early_stopping.best_test_cindex  if early_stopping.best_test_cindex is not None else last_test_cindex 
        final_test_brier =  early_stopping.best_test_brier  if early_stopping.best_test_brier is not None  else last_test_brier 
        final_test_nbll =  early_stopping.best_test_nbll  if early_stopping.best_test_nbll is not None  else last_test_nbll 
        final_test_p_value =  early_stopping.best_test_p_value  if early_stopping.best_test_p_value is not None  else last_test_p_value 
        final_test_MAE_PO =  early_stopping.best_test_MAE_PO  if early_stopping.best_test_MAE_PO is not None  else last_test_MAE_PO 
        
        save_results_to_csv(final_val_cidx, final_test_cidx, final_test_brier, final_test_nbll, final_test_p_value, final_test_MAE_PO, total_time_counter, memory, final_round, self.seed, self.run_id, self.dataset, self.partition, self.logdir, self.modelName, self.lr, self.batch_size, self.epochs, fileName="results_final.csv")

        logger.info(
            f"Finished FedAvg. Final selected round={final_round}, "
            f"best_val_cindex={final_val_cidx}, "
            f"reported_test_cindex={final_test_cidx}, "
            f"reported_test_brier={final_test_brier}, "
            f"reported_test_nbll={final_test_nbll}, "
            f"reported_test_p_value={final_test_p_value}, "
            f"reported_test_MAE_PO={final_test_MAE_PO}"
        )
