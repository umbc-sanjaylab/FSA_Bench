from config import get_args 
from models import MODEL_REGISTRY
 
from datasets.data_processing import prepare_survival_data, compute_eval_times 
from CL_FL import run_centralized, run_federated 

def main(): 
    args = get_args()
    data = prepare_survival_data(args)
    
    eval_times = compute_eval_times(data, args.pseudo_perc_interval)
    model_spec = MODEL_REGISTRY[args.model]

    if args.alg == "all_in":
        results = run_centralized( args, data, model_spec, eval_times)

    elif args.alg == "fedavg":
        results = run_federated(args, data, model_spec, eval_times)

    else:
        raise ValueError(f"Unsupported algorithm: {args.alg}")

    # save_results(args,  results)


if __name__ == "__main__":
    main()