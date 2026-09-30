import pandas as pd
import numpy as np 
from models.aft_weibull import  train_epoch as AFT_train_epoch
from models.aft_weibull import validation_loss as AFT_val_loss, predict_survival as AFT_predict_survival  

from models.aft_weibull import build_aft_weibull 
from models.fed_pdnn import build_fed_pdnn
from models.fed_plstm import build_fed_plstm
from models.fed_pattn import build_fed_pattn
from models.deep_hit import build_deep_hit
from models.deep_pseudo import build_deep_pseudo
from models.linear_ph import build_linear_ph
from models.nnph import build_nnph
from models.nnnph import build_nnnph

################################################### Function For initializing networks ##################################################

MODEL_REGISTRY = {
    "AFT_Weibull": {
        "model": build_aft_weibull,
        "train_epoch": AFT_train_epoch,
        "validation_loss": AFT_val_loss,
        "predict_survival": AFT_predict_survival
    },  
    
    "FedPDNN": {
        "model": build_fed_pdnn  
    },

    "FedPLSTM": {
        "model": build_fed_plstm 
    },

    "FedPAttn": {
        "model": build_fed_pattn  
    }, 
    
    "DeepHit": {
        "model": build_deep_hit
    },
    
    "DeepPseudo": {
        "model": build_deep_pseudo
    }, 
    
    "LinearPH": {
        "model": build_linear_ph
    },
    "NNPH": {
        "model": build_nnph
    },
    "NNnPH": {
        "model": build_nnnph
    }
} 

################################################### Function For initializing networks ################################################## 

def init_net(args): 
    if args.model not in MODEL_REGISTRY:
        raise ValueError( f"Model '{args.model}' is not supported by FSA-Bench.")

    input_size = get_input_feature(args) 
    model_config = MODEL_REGISTRY[args.model] 
    
    if "model" not in model_config:
        raise ValueError(f"Model initialization is not defined for '{args.model}'.") 

    net = model_config["model"]( input_size, args)  
    return net

################################################### Method for getting feature Size ################################################## 

def get_input_feature(args):
    if args.partition == "federated":
        client_name = get_client_name_for_federated_datasets(args.dataset, client_id=0)
        dataset_directory = f"{args.datadir}/{args.dataset}/{args.partition}/{client_name}/{args.run_id}/"
    else:
        dataset_directory = f"{args.datadir}/{args.dataset}/{args.partition}/0/{args.run_id}/"

    train_df = pd.read_csv(dataset_directory + "train.csv")
    X_train = np.array(train_df.drop( columns=["time", "event"]), dtype='float32')
    return X_train.shape[1]

  
######################################### Region from Federated Dataset by Client ID ######################################## 

def get_client_name_for_federated_datasets(dataset, client_id):
    dataset_with_region = {
        "TCGA": ["South", "West", "Midwest", "Northeast", "Europe", "Canada", "Other"],
        "Aids2": ['NSW', 'Other', 'QLD', 'VIC'],
        "LeukSurv": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        "Pbc3": ['London', 'Copenhagen', 'Hvidovre', 'Barcelona', 'Munich', 'Lyon'],
        "Dialysis": [120, 128, 184, 217, 344, 561, 562, 641, 741, 743, 917, 918, 936, 954, 983, 984, 1029, 1039,
                     1048, 1051, 1053, 1061, 1064, 1070, 1071, 1072, 1093, 1100, 1159, 1372, 1437, 1654, 1692,
                     1699, 2026, 2210, 2211, 2242, 2320, 2351, 2611, 2632, 2811, 2844, 2987, 3064, 3596, 3675,
                     4122, 4157, 4163, 4779, 4825, 5089, 5135, 5419, 5423, 5432, 5481, 5495, 5513, 5514, 5543,
                     5653, 5681, 5692, 5768],
        "d.oropha.rec": [2, 5, 4, 6, 3, 1],
        "cgd": ['Europe:Amsterdam', 'Europe:other', 'US:NIH', 'US:other'],
        "uis": ["A", "B"],
        "scania":  [1, 2, 3, 4, 5],
        "seer": [1, 2, 3]
    }

    region_list = dataset_with_region[dataset]
    return region_list[client_id] 