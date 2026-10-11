from pathlib import Path
import logging

#################################### Experiment Logger ############################################################ 

def create_experiment_logger(logdir, setting, model, dataset, run_id):
    logger_name = f"{setting}_{model}_{dataset}_fold_{run_id}"
    log_path = Path(logdir) / f"{model}_{dataset}_fold_{run_id}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%m-%d %H:%M"
    )

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger