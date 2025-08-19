<div align="center">

# AI reconstruction of European weather from the Euro-Atlantic regimes

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](https://img.shields.io/badge/arXiv-2506.13758-b31b1b.svg?style=flat&logo=arXiv)](https://arxiv.org/abs/2506.13758)
[![Zenodo](https://img.shields.io/badge/Zenodo-16751720-007ec6.svg?logo=zenodo)](https://zenodo.org/records/16751720)


</div>

<br>

## Description

Code release for the paper <b>"AI reconstruction of European weather from the Euro-Atlantic regimes"</b>

```
Camilletti, A., Tomasi, E., Franch, G., (2025). AI reconstruction of European weather from the Euro-Atlantic regimes. arXiv preprint 2506.13758.
```

<b>Preprint</b>: https://arxiv.org/abs/2506.13758

<b>Data & Models</b>: https://zenodo.org/records/16751720


## 📁 Folders
The project follows the folder structure of [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template):
  - configs: configs files (.yaml) for configuring the hyperparameters
  - data: containing the data used for training and testing the model. Initially is empty. Data can be downloaded from Zenodo: https://zenodo.org/records/16751720 
  - logs: training output, including the pretrained models. It will be created automatically during training.
  - notebooks: jupyter notebooks for analyzing the data and testing the model
  - scripts: python scripts for analyzing the data, testing the model and producing the plots
  - src: PyTorch implementation of the Dataloader and the Model

For a detailed description of the folder structure refer to [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

## 📄 Requirements:
Experiments were run in a Python 3.12 environment with the following packages:

  - python = "^3.12"
  - numpy = "^2.0.1"
  - torch = "^2.4.0"
  - lightning = "^2.3.3"
  - xarray = "^2024.6.0"
  - cfgrib = "^0.9.14.0"
  - ipykernel = "^6.29.5"
  - netcdf4 = "^1.7.1.post1"
  - matplotlib = "^3.9.1"
  - einops = "^0.8.0"
  - torchinfo = "^1.8.0"
  - hydra-core = "^1.3.2"
  - rootutils = "^1.0.7"
  - rich = "^13.7.1"
  - hydra-colorlog = "^1.2.0"
  - tensorboard = "^2.17.0"
  - xeofs = "^3.0.2"
  - xclim = "^0.57.0"
  - xsdba = "^0.5.0"
  - cartopy = "^0.25.0"


### 📦 Installation

If python3.12 is not already installed in your system, you can install it by running the following command:

```bash
# install python3.12 on ubuntu
bash install_python_ubuntu.sh
```

I suggest to create a new environment:

```bash
# create environment with poetry
bash create_environment.sh

# activate the environment
source .venv/bin/activate 
```

## ⬇️ Download data
To download the data from zenodo, run:

```bash
cd data

# download the dataset
python download_data.py
```

or download the data manually from Zenodo (https://zenodo.org/records/16751720)


## ⚙️ Pre-processing of the data
The raw data in the Zenodo dataset must be preprocessed to obtain the anomalies and the indices used to train and validate the model.

### Compute anomalies
To compute the anomalies and bias-correct the SEAS5 forecast, simply run:

```bash
cd scripts

# compute the anomalies and save them in the data folder
bash compute_anomalies.sh
```

The bias correction will take several minutes (~1h 30min).

### Compute indices
To compute the seven WR, four WR and NAO indices from the daily geopotential height anomalies, run:

```bash
cd scripts

# compute the indices and save them in the data folder
bash compute_z500_indices.sh
```

## 🤖 Train the model
To train models for both temperature and precipitation reconstruction with the same hyperparameters used in the paper:

```bash
# train all models used in the paper
bash scripts/schedule.sh
```

To change the hyperparameter you can create a new `configs/experiment/new-experiment.yaml` overriding the hyperparamenters you want to modify. Then you can run

```bash
# run a custom experiment
python train.py experiment=new-experiment
```

## 📊 Reproduce the plots
To reproduce the plots shown in the paper, run:

```bash
cd scripts

# create the plots
bash plot_results.sh
```

This will takes several hours (~3h).

## 📖 Citing

If you use this code or data, please cite: 

```
Camilletti, A., Franch, G., Tomasi, E., & Cristoforetti, M. (2025). AI reconstruction of European weather from the Euro-Atlantic regimes. ArXiv. https://doi.org/10.48550/arXiv.2506.13758
```