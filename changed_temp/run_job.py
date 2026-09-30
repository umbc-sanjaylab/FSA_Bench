from federated.nvflare_job import build_nvflare_job, datetime, main
from federated.utils import write_logger_configuration
import os 
import json  
import shutil
import logging
from uuid import uuid4 
from config import get_args
 
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


