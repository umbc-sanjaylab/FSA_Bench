from config import get_args 
from models import MODEL_REGISTRY
 
from datasets.data_processing import prepare_survival_data, compute_eval_times 
from CL_FL import  run_federated 

def main(): 
    args = get_args()
    data = prepare_survival_data(args)
    
    eval_times = compute_eval_times(data, args.pseudo_perc_interval)
    model_spec = MODEL_REGISTRY[args.model] 
    
    results = run_federated(args, data, model_spec, eval_times) 
    # save_results(args,  results)

if __name__ == "__main__":
    main()