# bash scripts/run_fl.sh NNnPH Pbc3 0
# bash scripts/run_cl.sh NNnPH Pbc3 0 
# model → dataset → fold/run

DATASET_CONFIG = {
    "TCGA": {
        "folds": 5,
        "n_parties": 7,
        "batch_size": 128,
    },

    "Aids2": {
        "folds": 5,
        "n_parties": 4,
        "batch_size": 64,
    },

    "LeukSurv": {
        "folds": 3,
        "n_parties": 24,
        "batch_size": 32,
    },

    "Pbc3": {
        "folds": 3,
        "n_parties": 6,
        "batch_size": 32,
    },

    "d.oropha.rec": {
        "folds": 3,
        "n_parties": 6,
        "batch_size": 32,
    },

    "cgd": {
        "folds": 3,
        "n_parties": 4,
        "batch_size": 32,
    },

    "uis": {
        "folds": 5,
        "n_parties": 2,
        "batch_size": 64,
    },

    "scania": {
        "folds": 5,
        "n_parties": 5,
        "batch_size": 32,
    },

    "seer": {
        "folds": 5,
        "n_parties": 3,
        "batch_size": 128,
    },

    "Dialysis": {
        "folds": 3,
        "n_parties": 67,
        "batch_size": 32,
    },
}