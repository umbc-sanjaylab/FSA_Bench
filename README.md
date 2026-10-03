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

## Running Experiments

FSA-Bench provides unified scripts for running centralized learning (CL) and federated learning (FL) experiments. The scripts are located in the repository root.

Each experiment is specified using:

- **Model name**
- **Dataset name**
 
All commands should be executed from the repository root.

### Linux

To run a centralized learning experiment:

```bash
bash scripts/run_fl.sh <model> <dataset>
``` 

To run a federated experiment, use this command: 

```bash
# Federated Learning
bash run_fl.sh <model> <dataset> 
```

For example, to model Pbc3 dataset using NNnPH model, we can use this: 

```bash
bash scripts/run_cl.sh NNnPH Pbc3
bash scripts/run_fl.sh NNnPH Pbc3
```

### Windows

For native Windows PowerShell, use this PowerShell scripts:

```bash
.\scripts\run_CL.ps1 <model> <dataset>
.\scripts\run_FL.ps1 <model> <dataset> 
```

For example, to model Pbc3 dataset using NNnPH model, we can use this: 

```bash
.\scripts\run_CL.ps1 NNnPH Pbc3
.\scripts\run_FL.ps1 NNnPH Pbc3
```

## Dataset

The number of cross-validation folds, federated clients, and batch sizes vary across datasets.

### Dataset Summary

| Dataset | Samples | Variables | Event (%) | Censoring (%) | # Clients | Data Link |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) | 6,084 | 283 | 30.50 | 69.50 | 7 | [TCGA Hub](https://xenabrowser.net/datapages/?hub=https://tcga.xenahubs.net:443) |
| [Aids2](https://rdrr.io/cran/MASS/man/Aids2.html) | 2,839 | 3 | 62.03 | 37.97 | 4 | [SurvSet](https://github.com/ErikinBC/SurvSet) |
| [LeukSurv](https://rdrr.io/cran/spBayesSurv/man/LeukSurv.html) | 1,043 | 4 | 84.28 | 15.72 | 24 | [SurvSet](https://github.com/ErikinBC/SurvSet) |
| [Pbc3](https://www.rdocumentation.org/packages/pec/versions/2025.06.24/topics/Pbc3) | 349 | 12 | 17.48 | 82.52 | 6 | [SurvSet](https://github.com/ErikinBC/SurvSet) |
| [Dialysis](https://rdrr.io/cran/RcmdrPlugin.survival/man/Dialysis.html) | 6,805 | 2 | 23.56 | 76.44 | 67 | [SurvSet](https://github.com/ErikinBC/SurvSetl) |
| [Oropharyngeal Carcinoma](https://rdrr.io/cran/invGauss/man/d.oropha.rec.html) | 192 | 12 | 72.40 | 27.60 | 6 | [SurvSet](https://github.com/ErikinBC/SurvSet) |
| [cgd](https://rdrr.io/cran/survival/man/cgd.html) | 128 | 8 | 34.38 | 65.62 | 4 | [SurvSet](https://github.com/ErikinBC/SurvSet) |
| [uis](https://github.com/graemeleehickey/hosmer-lemeshow/blob/master/edition2/uis.txt) | 628 | 7 | 80.89 | 19.11 | 2 | [SurvSet](https://github.com/ErikinBC/SurvSet) |
| [scania](https://cran.r-project.org/web/packages/eha/refman/eha.html#scania) | 1,040 | 4 | 52.60 | 47.40 | 5 | [SurvSet](https://github.com/ErikinBC/SurvSet) |
| [SEER](https://seer.cancer.gov/)  | 25,749 | 11 | 25.36 | 74.64 | 3 | [SEER](https://seer.cancer.gov/data/) |


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
| `-d <dataset_name>` | Specifies the dataset name used for the interpretability experiment. |

For example, this command runs the fold-level interpretability script using dataset name `seer`. 

```bash
python fold_level.py -d seer
```  

 
## License

This project is licensed under the [MIT License](LICENSE).

---

**FSA-Bench** — A comprehensive benchmark for federated survival models.
