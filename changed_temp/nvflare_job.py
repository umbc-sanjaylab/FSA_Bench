import shlex
from models import init_nets
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from server.controller import FederatedSurvivalController
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor 

from federated.nvflare_job import build_nvflare_job, datetime, main
from federated.utils import write_logger_configuration
import os 
import json  
import shutil
import logging
from uuid import uuid4 
from config import get_args

def build_nvflare_job(args, experiment_id): 
    net = init_nets(args)
    persistor = PTFileModelPersistor(model=net)

    # Create BaseFedJob with initial model
    job = BaseFedJob(initial_model=net, min_clients=args.n_parties, key_metric=None, model_persistor=persistor, name=f"{args.dataset}_{args.model}_{args.run_id}_{args.partition}_fedavg")   
    controller = FederatedSurvivalController(
        patience=args.patience,
        modeldir=args.modeldir,  
        partition=args.partition, 
        modelName=args.model, 
        dataset=args.dataset,
        num_clients=args.n_parties,
        num_rounds=args.comm_round,
        seed=args.init_seed,
        logdir=args.logdir,
        run_id=args.run_id,
        arguments=args
    )   
    job.to_server(controller)

    # Share scriptRunner to all clients
        
    client_args = [
        "--model", args.model, 
        "--init_seed", str(args.init_seed),
        "--dataset", args.dataset,
        "--partition", args.partition,
        "--sensitivity", str(args.sensitivity),
        "--epsilon", str(args.epsilon),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--optimizer", args.optimizer,
        "--patience", str(args.patience),
        "--eval_every", str(args.eval_every),
        "--reg", str(args.reg),
        "--datadir", args.datadir,
        "--device", args.device,
        "--lr", str(args.lr),
        "--logdir", args.logdir,
        "--run_id", str(args.run_id),
        "--modeldir", args.modeldir,
        "--dropout_p", str(args.dropout_p),
        "--experiment_id", experiment_id
    ]

    my_args = shlex.join(client_args) 
    print(f"client argument: {my_args}") 
    
    runner = ScriptRunner(script="client_module.py", script_args=my_args, launch_external_process=False)
    job.to_clients(runner) 
    return job 
 
def run_simulator(job, args, config_path): 
    workdir_root = os.path.abspath( args.workdir_root) 
    os.makedirs( workdir_root, exist_ok=True)

    workspace = os.path.join(workdir_root, f"{args.dataset}_{args.model}_fold_{args.run_id}_{uuid4().hex[:8]}") 
    n_clients = args.n_parties 
    n_threads = min(args.simulator_threads, n_clients)

    try: 
        job.simulator_run(workspace, n_clients=n_clients,  threads=n_threads, log_config=os.path.abspath(config_path)) 
    finally: 
        if getattr(args, "cleanup_workdir", False): 
            if os.path.isdir(workspace):
                logging.info( f"Deleting NVFLARE workspace: {workspace}") 
                shutil.rmtree( workspace, ignore_errors=True)
                
def main(args): 
    experiment_id = datetime.now().strftime( "%Y%m%d_%H%M%S") 
    config_path, _, server_log_path = write_logger_configuration(args=args, role="server", experiment_id=experiment_id, save_config=True)  
    print( f"Server log file: {server_log_path}")

    # Save experiment configuration
    run_log_dir = os.path.dirname(server_log_path) 
    arguments_path = os.path.join( run_log_dir, "experiment_arguments.json")

    with open(arguments_path, "w", encoding="utf-8") as file: 
        json.dump( vars(args), file, indent=2, default=str)

    print( f"Experiment arguments: {arguments_path}")

    # Build NVFLARE job
    job = build_nvflare_job( args, experiment_id)

    # Run simulator
    run_simulator(job, args, config_path)


if __name__ == "__main__": 
    args = get_args() 
    main(args) 