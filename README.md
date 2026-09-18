# FSA-Bench: Benchmarking Federated Survival Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![NVIDIA FLARE](https://img.shields.io/badge/NVIDIA%20FLARE-2.7+-orange.svg)](https://nvflare.readthedocs.io/)
[![License: UMBC-Prof-Sanjay-Lab](https://img.shields.io/badge/License-UMBC_Prof_Sanjay_Lab-yellow.svg)](LICENSE)
 

## Overview

Paper • Installation • Required Packages • Running Experiments • Datasets •
Evaluation Metrics • Experimental Settings • How to Cite

## Paper

For more details, see our paper **FSA-Bench: Benchmarking Federated Survival Models**, accepted at the **Machine Learning for Healthcare (MLHC) 2026** conference.

> Paper link will be added upon publication.


## Installation

### From Source

```bash
git clone https://github.com/umbc-sanjaylab/FSA_Bench.git
cd FSA_Bench
pip install -r requirements.txt
```

### Configure the Environment

```bash
conda create -n FSA_Bench python=3.10
conda activate FSA_Bench
pip install -r requirements.txt
```

##  Running Experiments

FSA-Bench experiments are executed using the shell scripts provided in the `scripts/` directory. The scripts specify the model, dataset, learning configuration, cross-validation settings, and output directories.

From the repository root, we can execute any script as:

```bash
bash scripts/<script_name>.sh 
```

### Centralized Learning

Centralized learning experiments can be executed using this command. For example:

```bash
bash scripts/DeepPseudo_CL.sh
```

### Federated Learning

Federated learning experiments can be executed using this command. For example:

```bash
bash scripts/DeepPseudo_FL.sh
```

## Dataset

FSA-Bench comprehensively evaluates federated survival models on **10 real-world federated survival datasets** covering diverse clinical domains, patient populations, and federated client configurations. We preserve naturally occurring institutional or geographically defined partitions to reflect model evaluation under heterogeneous data distributions across clients at realistic federated healthcare environments.

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

The datasets span a wide range of federated settings, from **2 clients (UIS)** to **67 clients (Dialysis)**, and include both small cohorts such as **cgd (128 samples)** and large-scale datasets such as **SEER (25,749 samples)**.

## Survival Models

FSA-Bench evaluates the following survival models:

- **Fed. Kaplan-Meier (Fed. KM)**
- **Fed. AFT-Weibull**
- **Fed. CoxPH**
- **NNPH**
- **NNnPH**
- **LinearPH**
- **DeepPseudo**
- **DeepHit**
- **FedPDNN**
- **FedPLSTM**
- **FedPAttn**

## Experimental Setup

We use the following experimental setup during modeling of distributed survival data using federated survival models.

- **5-fold cross-validation**
- **Centralized Learning (CL)** and **Federated Learning (FL)**
- Up to **50 federated communication rounds** and **250 local epochs**
- **Early stopping** based on validation performance at both client level and server level  
- Consistent preprocessing and evaluation across datasets and models

## Evaluation
 
### Statistical Analysis

We perform pairwise Wilcoxon signed-rank statistical test on FedPAttn model with other federated survival baselines. Holm correction is applied across the comparisons.

### Explanation Stability

We perform explanation-stability analysis on federated survival models using **SurvSHAP** across folds, federated sites, and centralized/federated partitions.

Stability is measured using the following metrices:

- **Spearman's rank correlation**
- **Kendall's tau**
- **Jaccard similarity**

## Requirements

```text
jupyter==1.0.0
pycox==0.2.3
torch==2.0.1
scikit-learn==1.2.2
pandas==1.3.5
numpy==1.24.3
```

## How to Cite

If you use FSA-Bench, its code, or datasets in your research, please cite our paper:

```bibtex
@inproceedings{ahmed2026fsabench,
  title     = {FSA-Bench: Benchmarking Federated Survival Models},
  author    = {Ahmed, Sultan and Purushotham, Sanjay},
  year      = {2026}
}
```



## License

This project is licensed under the UMBC Professor Sanjay Lab license.

See the [LICENSE](LICENSE) file for details.

---

**FSA-Bench** — A comprehensive benchmark for federated survival models.
