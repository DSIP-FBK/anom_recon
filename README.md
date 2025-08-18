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


### 📦 Installation

```bash
# install python3.12 on ubuntu
bash install_python_ubuntu.sh

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
To compute the anomalies, edit the scripts/compute_anomalies.sh with the paths to the NetCDF files containing the relevant data:

  - `era5_z500_noa`: ERA5 daily geopotential height at 500 hPa in the Euro-Atlantic region.
  - `era5_t2m_eu`: ERA5 daily two-meter temperature in Europe
  - `era5_tp_eu`: ERA5 daily total precipitation in Europe

  - `seas5_z500_noa`: SEAS5 monthly forecast of geopotential height ad 500 hPa in the Euro-Atlantic sector
  - `seas5_t2m_eu`: SEAS5 monthly forecast of two-meter temperature in Europe
  - `seas5_tp_eu`: SEAS5 monthly forecast of total precipitation in Europe

### Compute indices
To compute the seven WR, four WR and NAO indices from the daily geopotential height anomalies, edit the paths in the `compute_z500_indices.py`:

  - `era5_daily_z500_anom`: ERA5 daily geopotential height anomalies at 500 hPa in the Euro-Atlantic sector
  - `seas5_monthly_z500_anom`: SEAS5 monthly geopotential height anomalies at 500 hPa in the Euro-Atlantic sector


## 🤖 Train the model
To train models for both temperature and precipitation reconstruction with the same hyperparameters used in the paper:

```bash
# train all models used in the paper
bash scripts/schedule
```

## 📊 Reproduce the plots
To reproduce the plots shown in the paper, run:

```bash
cd scripts

# create the plots
bash plot_results.sh \ 
    <model_7wr_temp> \           # model trained to reconstruct the temperature from the the 7WR
    <model_7wr_prec> \           # model trained to reconstruct the precipitation from the the 7WR
    \
    <model_4wr_temp_DJF> \       # model trained to reconstruct the temperature from the the 4WR in winter
    <model_4wr_temp_JJA> \       # model trained to reconstruct the temperature from the the 4WR in summer
    <model_4wr_prec_DJF> \       # model trained to reconstruct the precipitation from the the 4WR in winter
    <model_4wr_prec_JJA> \       # model trained to reconstruct the precipitation from the the 4WR in summer
    \
    <model_NAO_temp_DJF> \       # model trained to reconstruct the temperature from the the NAO in winter
    <model_NAO_temp_JJA> \       # model trained to reconstruct the temperature from the the NAO in summer
    <model_NAO_prec_DJF> \       # model trained to reconstruct the precipitation from the the NAO in winter
    <model_NAO_prec_JJA> \       # model trained to reconstruct the precipitation from the the NAO in summer
    \ 
    <model_no_index_temp> \      # model trained to reconstruct the temperature without WR information
    <model_no_index_prec> \      # model trained to reconstruct the temperature without WR information
    \
    <era5_daily_z500> \          # ERA5 daily geopotential height data at 500 hPa in the Euro-Atlantic region
    <era5_monthly_temp_anom> \   # ERA5 monthly two-meter temperature anomalies in Europe
    <era5_monthly_prec_anom> \   # ERA5 monthly precipitation anomaly in Europe
    \
    <seas5_monthly_temp_anom> \  # SEAS5 monthly two-meter temperature anomaly (forecast) in Europe
    <seas5_monthly_prec_anom> \  # SEAS5 monthly precipitation anomaly (forecast) in Europe
    <Iwr_SEAS5>                  # Index computed from SEAS5 geopotential height at 500 hPa in the Euro-Atlantic sector
```

where <texttt>\<path\></texttt> are the paths to:
  - <b> Models: </b>
    - `model_7wr_temp`: model trained to reconstruct the monthly mean two-meter temperature using seven weather regimes.
    - `model_7wr_prec`: model trained to reconstruct the monthly total precipitation using seven weather regimes.
    - `model_4wr_temp_DJF/JJA`: model trained to reconstruct the monthly mean two-meter temperature using four weather regimes in winter/summer.
    - `model_4wr_prec_DJF/JJA`: model trained to reconstruct the monthly total precipitation using four weather regimes in winter/summer.
    - `model_NAO_temp_DJF/JJA`: model trained to reconstruct the monthly mean two-meter temperature using the NAO index in winter/summer.
    - `model_NAO_prec_DJF/JJA`: model trained to reconstruct the monthly total precipitation using NAO the index in winter/summer
    - `model_no_index_temp`: model trained to reconstruct the monthly mean two-meter temperature without using WR indices
    - `model_no_index_prec`: model trained to reconstruct the monthly total precipitation without using WR indices
  - <b>ERA5 data:</b>
    - `era5_daily_z500`: NetCDF file containing ERA5 daily geopotential height data at 500 hPa in the Euro-Atlantic region
    - `era5_monthly_temp_anom`: NetCDF file containing ERA5 monthly two-meter temperature anomalies in Europe
    - `era5_monthly_prec_anom`: NetCDF file containing ERA5 monthly total precipitation anomaly in Europe
  - <b>SEAS5 data:</b>
    - `seas5_monthly_temp_anom`: NetCDF file containing SEAS5 hindcast and forecast of motlhy mean two-meter temperature.
    - `seas5_monthly_prec_anom`: NetCDF file containing SEAS5 hindcast and forecast of motlhy total precipitation.
    - `Iwr_SEAS5`: NetCDF file containing the seven weather regime indices forecasted by SEAS5 (geopotential height at 500 hPa).

Pretrained models can be found in 
  - <code>logs/t2m_ERA5/multiruns</code> for two-meter temperature
  - <code>logs/tp_ERA5/multiruns</code> for total precipitation.

## 📖 Citing

If you use this code or data, please cite: 

```
Camilletti, A., Franch, G., Tomasi, E., & Cristoforetti, M. (2025). AI reconstruction of European weather from the Euro-Atlantic regimes. ArXiv. https://doi.org/10.48550/arXiv.2506.13758
```