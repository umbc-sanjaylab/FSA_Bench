# FSA-Bench: Benchmarking Federated Survival Models

## 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flare](https://img.shields.io/badge/Flare-2.7+-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License: UMBC-Prof-Sanjay-Lab](https://img.shields.io/badge/License-UMBC_Prof_Sanjay_Lab-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-20245xxxx-b31b1b.svg)](https://arxiv.org/abs/2024.xxxx)

## 🎯 Overview 
Recent efforts in the literature have explored survival
models, but several key gaps remain:

-   **Lack of comprehensive comparison**: No prior study has
    benchmarked a wide range of survival modeling methods within a
    single paper in federated settings. 
-   **Limited datasets** -- Most existing works have only been
    tested on a few datasets, restricting their applicability in
    different fields of research domains. 

### 📌 Our Contribution

In this work, we present:

-   A **comprehensive benchmarking analysis** of **Federated Survival
    Models** using the **SurvSet repository**.\
-   A **performance comparison** against **state-of-the-art neural
    network-based federated Cox proportional hazard models**.

This study provides deeper insights into the strengths and weaknesses of
different survival methods, helping researchers make **better-informed
decisions in healthcare applications**.

## 🏗️ Architecture Pipeline of FSA-Bench Framework

![STCRL Architecture](asset/FSA_Bench_Framework.png)


## 🚀 Installation
### From source
Download a local copy of Benchmark_Study_of_FSM and install from the directory:

```bash
	git clone https://github.com/sultanoid/Benchmark_Study_of_FSM.git
	cd Benchmark_Study_of_FSM
	pip install -r requirements.txt
```

### Configure the environement
```bash 
conda create -n benchmark_FSM python=3.10
conda activate benchmark_FSM 
pip install -r requirements.txt
``` 

## 📊 Dataset  

The **SurvSet repository** is designed to work with benchmarking analysis of different survival analysis models. The expected data format includes:  

- **Outcome**: Event status, Survival time  
- **Subject Information**: Covariates of participants for cross-subject analysis  

🔗 [SurvSet Repository](https://github.com/ErikinBC/SurvSet)


## 📈 Evaluation

### Comprehensive Metrics

- **Concordance Index**: Assesses how well a model can predict survival times close to the truth. 
- **Brier Score**: Measures the accuracy of the predicted survival
probability for a sample at a given time. 
 
## 📋 Requirements

```
jupyter==1.0.0
pycox==0.2.3
torch==2.0.1
scikit-learn==1.2.2
pandas==1.3.5
numpy==1.24.3 
```

## 📄 License

This project is licensed under the lab of UMBC Professor Sanjay Lab - see the [LICENSE](LICENSE) file for details.

**Benchmark_FSM** - A benchmark study of existing federated survival models. 
