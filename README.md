<div align="center">

# AI reconstruction of European weather from the Euro-Atlantic regimes

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2407.02089-B31B1B.svg)](https://arxiv.org/abs/2407.02089)
[![Data](http://img.shields.io/badge/Data-Zenodo-4b44ce.svg)](https://doi.org/10.5281/zenodo.13692016)
[![Models](http://img.shields.io/badge/Models-Zenodo-4b44ce.svg)](https://doi.org/10.5281/zenodo.13594332)

</div>

<br>

## Description

Code release for the paper <b>"AI reconstruction of European weather from the Euro-Atlantic regimes"</b>

```
Camilletti, A., Tomasi, E., Franch, G., (2025). AI reconstruction of European weather from the Euro-Atlantic regimes. arXiv preprint arXiv:.
```

<b>preprint</b>: 

<b>data</b>: 

<b>models</b>: 

## Folders
The project follows the folder structure of [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template):
  - configs: configs files (.yaml) for training hyperparameters
  - data: containing the data used for training and testing the model
  - logs: training output, including the pretrained models
  - notebooks: jupyter notebooks for analyzing the data and testing the model
  - scripts: python scripts for analyzing the data, testing the model and producing the plots
  - src: PyTorch implementation of the AI-model

For a detailed description of the folder structure refer to [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

## Requirements:
Experiments were run in a Python 3.12 environment with the following packages:

  - numpy
  - xarray
  - netcdf4
  - xeofs
  - matplotlib
  - torch
  - lightning
  - torchinfo
  - hydra-core
  - einops
  - rootutils
  - rich
  - hydra-colorlog


### Installation

```bash
# install python3.12 on ubuntu
bash install_python_ubuntu.sh

# create environment with poetry
bash create_environment.sh

# activate the environment
source .venv/bin/activate 
```

## Download pre-trained model
To download the pre-trained model from zenodo, run:

```bash
python download_pre-trained_model.py
```

## Train your own model
To train models for both temperature and precipitation reconstruction with the same hyperparameters used in the paper:

```bash
# download the dataset
python data/download_data.py

# train all models used in the paper
bash scripts/schedule
```

## Reproduce the plots
To reproduce the plots shown in the paper, run:

```bash
cd scripts

bash plot_results.sh <model_7wr_temp> <model_7wr_prec> <model_4wr_temp> <model_4wr_prec> <seas5_temp> <seas5_prec> <Iwr_SEAS5>
```

where <texttt>\<path\></texttt> are the paths to:
  - model_7wr_temp: model trained to reconstruct the monthly mean two-meter temperature using seven weather regimes.
  - model_7wr_prec: model trained to reconstruct the monthly total precipitation using seven weather regimes.
  - model_4wr_temp: model trained to reconstruct the monthly mean two-meter  temperature using four weather regimes.
  - model_4wr_prec: model trained to reconstruct the monthly total precipitation using four weather regimes.
  - seas5_temp: NetCDF file containing SEAS5 hindcast and forecast of motlhy mean two-meter temperature.
  - seas5_prec: NetCDF file containing SEAS5 hindcast and forecast of motlhy total precipitation.
  - Iwr_SEAS5: NetCDF file containing the seven weather regime indices forecasted by SEAS5 (geopotential height at 500 hPa).

Pretrained models can be found in 
  - <code>logs/t2m_ERA5/multiruns</code> for two-meter temperature
  - <code>logs/tp_ERA5/multiruns</code> for total precipitation.

## Citing

If you use this code or data, please cite:
