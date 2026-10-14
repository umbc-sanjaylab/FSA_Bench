# FSA-Bench: Benchmarking Federated Survival Models

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![NVIDIA FLARE](https://img.shields.io/badge/NVIDIA%20FLARE-2.7+-orange.svg)](https://nvflare.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
 
## Overview

Paper • How to Cite • Installation • Requirements • Dataset • Dataset Preprocessing • Running Experiments • Interpretability • Contact • License

## Paper

This repository contains the official codebase for **[FSA-Bench: Benchmarking Federated Survival Models](https://proceedings.mlr.press/v340/ahmed26a.html)**, published at the **11th Machine Learning for Healthcare Conference (MLHC 2026)**.

## How to Cite

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
lifelines==0.30.0
numpy==2.5.4
nvflare==2.7.0
pandas==3.0.6
pycox==0.3.0
rpy2==3.6.4
rpy2_rinterface==3.6.3
rpy2_robjects==3.6.3
scikit_learn==1.9.1
scikit_survival==0.25.0
scipy==1.18.1
statsmodels==0.14.6
survivaleval==0.5.1
survshap==0.4.2
torch==2.9.0
torchsurv==0.2.0
torchtuples==0.2.2
transformers==4.57.3
```   

## Dataset

The number of cross-validation folds, federated clients, and batch sizes vary across datasets.

### Dataset Summary

| Dataset | Samples | Variables | Event (%) | Censoring (%) | # Clients |
| --- | ---: | ---: | ---: | ---: | ---: |
| TCGA | 6,084 | 283 | 30.50 | 69.50 | 7 |
| Aids2 | 2,839 | 3 | 62.03 | 37.97 | 4 |
| LeukSurv | 1,043 | 4 | 84.28 | 15.72 | 24 |
| Pbc3 | 349 | 12 | 17.48 | 82.52 | 6 |
| Dialysis | 6,805 | 2 | 23.56 | 76.44 | 67 |
| Oropharyngeal Carcinoma | 192 | 12 | 72.40 | 27.60 | 6 |
| cgd | 128 | 8 | 34.38 | 65.62 | 4 |
| uis | 628 | 7 | 80.89 | 19.11 | 2 |
| scania  | 1,040 | 4 | 52.60 | 47.40 | 5 |
| SEER  | 25,749 | 11 | 25.36 | 74.64 | 3 |

> **Dataset availability:** We do not distribute the exact datasets or processed versions used in this benchmark through this repository or via direct download links. Please contact us for additional information regarding data access and availability.

> Dataset configuration may vary across individual model scripts.
 
 ## Dataset Preprocessing

Before running the centralized or federated experiments, the datasets must be preprocessed and partitioned into the required train, validation, and test splits.

The preprocessing pipeline is implemented in the `datasets/` module. From the repository root, run:

```bash
python -m datasets.prepare_datasets
```

## Running Experiments

FSA-Bench provides unified scripts for running centralized learning (CL) and federated learning (FL) experiments. The scripts are located in the repository root.

Each experiment is specified using:

- **Model name**
- **Dataset name**
 
All commands should be executed from the repository root.

### Linux

To run a centralized learning experiment:

```bash
bash scripts/run_cl.sh <model> <dataset>
``` 

To run a federated experiment, use this command: 

```bash
# Federated Learning
bash scripts/run_fl.sh <model> <dataset> 
```

For example, to model Pbc3 dataset using NNnPH model, we can use this: 

```bash
bash scripts/run_cl.sh NNnPH Pbc3
bash scripts/run_fl.sh NNnPH Pbc3
```

### Windows

For native Windows PowerShell, use these PowerShell scripts: 

```bash
.\scripts\run_CL.ps1 <model> <dataset>
.\scripts\run_FL.ps1 <model> <dataset> 
```

For example, to model Pbc3 dataset using NNnPH model, we can use this: 

```bash
.\scripts\run_CL.ps1 NNnPH Pbc3
.\scripts\run_FL.ps1 NNnPH Pbc3
``` 

### Supported Models

| Argument | Description | Possible Values |
| --- | --- | --- |
| `<model>` | Specifies the survival model<br> used for the experiment. | `Fed. KM`, `AFT_Weibull`, `Fed_CoxPH`, `DeepPseudo`, `DeepHit`, <br> `LinearPH`, `NNPH`, `NNnPH`,  `FedPDNN`, `FedPLSTM`, `FedPAttn`|

### Supported Datasets

| Argument | Description | Possible Values |
| --- | --- | --- |
| `<dataset>` | Specifies the survival dataset<br> used for the experiment. | `TCGA`, `Aids2`, `LeukSurv`, `Pbc3`, `d.oropha.rec`,<br> `cgd`, `uis`, `scania`, `seer`, `Dialysis` |

## Interpretability

FSA-Bench provides fold-level, site-level, and partition-level interpretability analyses. The corresponding modules are available in the `interpretability/` directory.

Each interpretability experiment requires a dataset name specified using the `-d` or `--dataset` argument.

### Command Arguments

| Argument | Description | Possible Values |
| --- | --- | --- |
| `-d`, `--dataset` | Specifies the dataset used<br> for the interpretability experiment. | `TCGA`, `Aids2`, `LeukSurv`, `Pbc3`, `d.oropha.rec`,<br> `cgd`, `uis`, `scania`, `seer`, `Dialysis` |

For detailed information about these datasets, see the [Dataset](#dataset) section.

### Fold-Level Interpretation

```bash
python -m interpretability.fold_level -d <dataset>
```

### Site-Level Interpretation

```bash
python -m interpretability.site_level -d <dataset>
```

### Partition-Level Interpretation

```bash
python -m interpretability.CL_vs_FL -d <dataset>
```
 
For example, this command runs the fold-level interpretability analysis on the seer dataset:

```bash
python -m interpretability.fold_level -d seer
```  

## Contact

* Sultan Ahmed (sultan.ahmed@umbc.edu)
* Sanjay Purushotham (psanjay@umbc.edu)
 
## License

This project is licensed under the [MIT License](LICENSE).

---

**FSA-Bench** — A comprehensive benchmark for federated survival models.
