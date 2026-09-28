import logging
from server import FederatedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.script_runner import ScriptRunner
from model import init_net
from utils import get_args, write_logger_configuration,  get_model_spec
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
import os
import shutil
from datetime import datetime
import sys  
import json 

def main(args):
    experiment_id = datetime.now().strftime( "%Y%m%d_%H%M%S" )
    config_path, _, server_log_path = write_logger_configuration( args=args, role="server", experiment_id=experiment_id, save_config=True) 

    print(f"Server log file: {server_log_path}")
    net = init_net(args.model, args.device, args)
    model_spec = get_model_spec(args)
    persistor = PTFileModelPersistor(model=model_spec)

    # Create BaseFedJob with initial model
    job = BaseFedJob(
        name=f"{args.dataset}_{args.model}_{args.run_id}_{args.partition}_fedavg",
        initial_model=net,
        min_clients=args.n_parties,
        # key_metric="validation_c_index",
        key_metric=None,
        model_persistor=persistor,
    )
    # job.to_server(persistor, id="persistor")

    # Define the controller and send to server
    controller = FederatedAvg(
        patience=args.patience,
        modeldir=args.modeldir,
        reg=args.reg, lr=args.lr,
        optimizer=args.optimizer,
        epochs=args.epochs,
        partition=args.partition,
        batch_size=args.batch_size,
        modelName=args.model,
        device=args.device,
        dataset=args.dataset,
        num_clients=args.n_parties,
        num_rounds=args.comm_round,
        seed=args.init_seed,
        logdir=args.logdir,
        run_id=args.run_id,
        arguments=args,
    )  

    job.to_server(controller)

    # Share scriptRunner to all clients
    my_args = f"--model {args.model} --init_seed {args.init_seed} --dataset {args.dataset} --partition {args.partition} --sensitivity {args.sensitivity}\
          --epsilon {args.epsilon} --batch_size {args.batch_size} --epochs {args.epochs} --optimizer {args.optimizer} --patience {args.patience} --eval_every {args.eval_every} --reg {args.reg} --datadir {args.datadir}   --device {args.device} --lr {float(args.lr)} --logdir {args.logdir} --run_id {args.run_id} --modeldir {args.modeldir} --dropout_p {args.dropout_p}  --experiment_id {experiment_id}"

    print(f"client argument: {my_args}")

    train_script = "client_module.py"
    # runner = ScriptRunner(script=train_script, script_args=my_args)

    # runner = ScriptRunner(
    #     script=train_script,
    #     script_args=my_args,
    #     launch_external_process=True,
    #    command=f"{sys.executable} -u" 
    # )
    
    runner = ScriptRunner(
        script=train_script,
        script_args=my_args,
        launch_external_process=False
    )

    job.to_clients(runner) 
    
    run_log_dir = os.path.dirname( server_log_path )
    arguments_path = os.path.join( run_log_dir, "experiment_arguments.json", )

    with open( arguments_path, "w", encoding="utf-8", ) as file:
        json.dump( vars(args), file, indent=2, default=str)

    print( f"Experiment arguments: {arguments_path}" )

    workdir_root = "/home/il66977@ad.umbc.edu/Benchmarking_FSA/workdir"
    os.makedirs(workdir_root, exist_ok=True)

    last_error = None
    for attempt in range(3):
        ws = os.path.join( workdir_root,
            f"workdir_{args.dataset}_{args.model}_{args.run_id}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}",)
         
        n_clients =  min(1, args.n_parties) if args.dataset=="Dialysis" else args.n_parties
        try:
            job.simulator_run(ws,
                n_clients=n_clients,
                threads= args.n_parties,  
                #  threads=1, 
                log_config=os.path.abspath(config_path),
            )
            # Success — clean up and exit loop
            if os.path.isdir(ws):
                logging.info(f"Deleting NVFlare workspace: {ws}")
                shutil.rmtree(ws, ignore_errors=True)
            break
        except FileExistsError as e:
            last_error = e
            logging.warning(f"Workspace already exists, retrying: {ws}")
    else:
        raise RuntimeError(
            f"Failed to create a fresh simulator workspace after 3 attempts. Last error: {last_error}"
        )


if __name__ == "__main__":
    args = get_args()
    main(args)
