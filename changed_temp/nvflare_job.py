import shlex
from models import init_nets
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from server.controller import FederatedSurvivalController
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor 

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