import numpy as np
import pandas as pd 
import os
import json 
import builtins  
import re
from typing import Optional 

####################################### Logger configuration of Nvidia Flare ######################################################## 

def write_logger_configuration( args, role: str, experiment_id: str, save_config: bool = False):   
    safe_role = re.sub( r"[^A-Za-z0-9_.-]+", "_", str(role))
    run_name = f"{args.dataset}_{args.model}_" f"fold{args.run_id}_{experiment_id}"  
    
    run_log_dir = os.path.abspath( os.path.join(args.logdir, run_name) )
    os.makedirs( run_log_dir, exist_ok=True, )

    # Deterministic filename:
    log_file_path = os.path.join( run_log_dir, f"{safe_role}.log") 
    log_cfg = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "default": {
                "format": "%(asctime)s - %(process)d -  %(name)s - %(levelname)s -  %(message)s" 
            }
        },

        "handlers": {
            "console_info": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
                "stream": "ext://sys.stdout",
            },

            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": log_file_path,
                "level": "INFO",
                "encoding": "utf-8",

                # A restarted client continues in the same file.
                "mode": "a",
            }
        },

        "root": {
            "handlers": ["console_info", "file"],
            "level": "WARNING",
        },

        "loggers": { 
            "fedPseudo": {
                "handlers": [ "console_info", "file"],
                "level": "INFO",
                "propagate": False
            },

            # NVFlare log
            "nvflare": {
                "handlers": ["console_info", "file"],
                "level": "WARNING",
                "propagate": False 
            } 
        } 
    }

    config_path: Optional[str] = None 
    if save_config:
        config_path = os.path.join( run_log_dir, "server_logging_config.json")

        with open( config_path, "w", encoding="utf-8") as file:
            json.dump(log_cfg, file, indent=2)

    return config_path, log_cfg, log_file_path
 