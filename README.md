# FSA-Bench: Benchmarking Federated Survival Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![NVIDIA FLARE](https://img.shields.io/badge/NVIDIA%20FLARE-2.7+-orange.svg)](https://nvflare.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
 
## Overview

Paper • Installation • Required Packages • Running Experiments • Datasets • Evaluation Metrics • Experimental Settings • How to Cite

## Paper

This repository contains the official codebase for **[FSA-Bench: Benchmarking Federated Survival Models](https://proceedings.mlr.press/v340/ahmed26a.html)**, published at the **11th Machine Learning for Healthcare Conference (MLHC 2026)**.

Please cite the paper if you use FSA-Bench, its code, or datasets:

Sultan Ahmed and Sanjay Purushotham, "FSA-Bench: Benchmarking Federated Survival Models," in *Proceedings of the 11th Machine Learning for Healthcare Conference*, Proceedings of Machine Learning Research (PMLR), vol. 340, pp. 30–54, 2026.

### BibTeX

```bibtex
@InProceedings{pmlr-v340-ahmed26a,
  title     = {FSA-Bench: Benchmarking Federated Survival Models},
  author    = {Ahmed, Sultan and Purushotham, Sanjay},
  booktitle = {Proceedings of the 11th Machine Learning for Healthcare Conference},
  pages     = {30--54},
  year      = {2026},
  volume    = {340},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v340/ahmed26a.html}
}
```

## Installation

To run the experiments, first clone the repository:

```bash
git clone https://github.com/umbc-sanjaylab/FSA_Bench.git
cd FSA_Bench
```

Then configure the environment using one of the following options.

### Option 1: Install with pip

```bash
pip install -r requirements.txt
```

### Option 2: Install with Conda

```bash
conda create -n FSA_Bench python=3.10
conda activate FSA_Bench
pip install -r requirements.txt
``` 

## Requirements

The following Python packages are required for running our experiments. 

```text
jupyter==1.0.0
pycox==0.2.3
torch==2.0.1
scikit-learn==1.2.2
pandas==1.3.5
numpy==1.24.3
``` 

##  Running Experiments

The shell scripts in FSA-Bench framework are in the `scripts/` directory to reproduce the centralized and federated experiments.  From the repository root, we can execute any script as:

```bash
bash scripts/<script_name>.sh 
```

### Centralized Learning

The centralized experiments can be executed using the following scripts:

| Model | Command |
| --- | --- |
| AFT-Weibull | `bash scripts/AFT_Weibull_CL.sh` |
| DeepHit | `bash scripts/DeepHit_CL.sh` |
| DeepPseudo | `bash scripts/DeepPseudo_CL.sh` |
| Fed. CoxPH | `bash scripts/Fed_CoxPH_CL.sh` |
| Fed. Kaplan-Meier | `bash scripts/Fed_KM_CL.sh` |
| FedPDNN | `bash scripts/FedPseudo_CL.sh` |
| FedPLSTM | `bash scripts/FedPseudo_CL.sh` |
| FedPAttn | `bash scripts/FedPseudo_CL.sh` |
| LinearPH | `bash scripts/LinearPH_NNph_CL.sh` |
| NNPH | `bash scripts/LinearPH_NNph_CL.sh` |
| NNnPH | `bash scripts/NNnPH_CL.sh` |

### Federated Learning

The federated experiments can be executed using the following scripts:

| Model | Command |
| --- | --- |
| AFT-Weibull | `bash scripts/AFT_Weibull_FL.sh` |
| DeepHit | `bash scripts/DeepHit_FL.sh` |
| DeepPseudo | `bash scripts/DeepPseudo_FL.sh` |
| Fed. CoxPH | `bash scripts/Fed_CoxPH_FL.sh` |
| Fed. Kaplan-Meier | `bash scripts/Fed_KM_FL.sh` |
| FedPDNN | `bash scripts/FedPseudo_FL.sh` |
| FedPLSTM | `bash scripts/FedPseudo_FL.sh` |
| FedPAttn | `bash scripts/FedPseudo_FL.sh` |
| LinearPH | `bash scripts/LinearPH_NNPH_FL.sh` |
| NNPH | `bash scripts/LinearPH_NNPH_FL.sh` |
| NNnPH | `bash scripts/NNnPH_FL.sh` |

### Common Training Configuration

The deep-learning-based survival models use the following common experimental settings unless otherwise specified in the corresponding shell script:

| Parameter | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Maximum epochs | `250` |
| Weight regularization | `5e-5` |
| Dropout | `0.05` |
| Early-stopping patience | `25` |
| Client participation | `1.0` |
| Initialization seed | `5` |
| FL aggregation | FedAvg |
| FL communication rounds | `50` |

Model-specific and dataset-specific parameters are defined in the corresponding scripts in the `scripts/` directory.


## Dataset

The number of cross-validation folds, federated clients, and batch sizes vary across datasets.

### Dataset Summary

| Dataset | Samples | Variables | Event (%) | Censoring (%) | # Clients | Data Link |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **TCGA** | 6,084 | 283 | 30.50 | 69.50 | 7 | [Link](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) |
| **Aids2** | 2,839 | 3 | 62.03 | 37.97 | 4 | [Link](https://rdrr.io/cran/MASS/man/Aids2.html) |
| **LeukSurv** | 1,043 | 4 | 84.28 | 15.72 | 24 | [Link](https://rdrr.io/cran/spBayesSurv/man/LeukSurv.html) |
| **Pbc3** | 349 | 12 | 17.48 | 82.52 | 6 | [Link](https://www.rdocumentation.org/packages/pec/versions/2025.06.24/topics/Pbc3) |
| **Dialysis** | 6,805 | 2 | 23.56 | 76.44 | 67 | [Link](https://rdrr.io/cran/RcmdrPlugin.survival/man/Dialysis.html) |
| **Oropharyngeal Carcinoma** | 192 | 12 | 72.40 | 27.60 | 6 | [Link](https://rdrr.io/cran/invGauss/man/d.oropha.rec.html) |
| **cgd** | 128 | 8 | 34.38 | 65.62 | 4 | [Link](https://rdrr.io/cran/survival/man/cgd.html) |
| **uis** | 628 | 7 | 80.89 | 19.11 | 2 | [Link](https://github.com/graemeleehickey/hosmer-lemeshow/blob/master/edition2/uis.txt) |
| **scania** | 1,040 | 4 | 52.60 | 47.40 | 5 | [Link](https://cran.r-project.org/web/packages/eha/refman/eha.html#scania) |
| **SEER** | 25,749 | 11 | 25.36 | 74.64 | 3 | [Link](https://seer.cancer.gov/) |


> Dataset availability and configuration may vary across individual model scripts. 
 
## Interpretability

The interpretability scripts are available in the `interpretability/` directory.

From the repository root:

```bash
cd interpretability
```

### Fold-Level Interpretation

```bash
python fold_level.py
```

### Site-Level Interpretation

```bash
python site_level.py
```

### Partition-Level Interpretation

```bash
python CL_vs_FL.py
```

### Command Arguments

| Argument / Option | Description |
| --- | --- |
| `-d <dataset_id>` | Specifies the dataset ID used for the interpretability experiment. |
| `taskset -c <cores>` | Restricts the process to the specified CPU cores. The core IDs can be changed according to available compute resources. |
| `OMP_NUM_THREADS=1` | Limits OpenMP operations to one thread. |
| `MKL_NUM_THREADS=1` | Limits Intel MKL operations to one thread. |
| `OPENBLAS_NUM_THREADS=1` | Limits OpenBLAS operations to one thread. |
| `NUMEXPR_NUM_THREADS=1` | Limits NumExpr operations to one thread. |
| `VECLIB_MAXIMUM_THREADS=1` | Limits Apple Accelerate/VecLib operations to one thread. |

For example:

```bash
python fold_level.py -d 5
```

runs the fold-level interpretability script using dataset ID `5`. 

 
## License

This project is licensed under the [MIT License](LICENSE).

---

**FSA-Bench** — A comprehensive benchmark for federated survival models.
